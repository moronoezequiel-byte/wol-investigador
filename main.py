
import os
import re
import json
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI
from google import genai


APP_VERSION = "0.3.1"

WOL_BASE = "https://wol.jw.org"
WOL_SEARCH = "https://wol.jw.org/es/wol/s/r4/lp-s"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

# Reintentos ante interrupciones temporales de Gemini.
GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_DELAYS = [2, 5]

app = FastAPI(title="WOL Investigador", version=APP_VERSION)

app.mount("/static", StaticFiles(directory="static"), name="static")


class SearchRequest(BaseModel):
    question: str = Field(min_length=4, max_length=1200)
    max_results: int = Field(default=10, ge=1, le=20)


class Article(BaseModel):
    title: str
    publication: str | None = None
    date: str | None = None
    url: str
    relevance: str | None = None


def parse_json(raw: str) -> dict:
    if not raw:
        raise ValueError("La IA devolvió una respuesta vacía.")

    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def is_temporary_gemini_error(error: Exception) -> bool:
    """
    Identifica errores temporales de disponibilidad.
    No reintenta errores de cuota que podrían requerir
    esperar mucho tiempo.
    """
    message = str(error).upper()

    temporary_markers = (
        "503",
        "UNAVAILABLE",
        "500 INTERNAL",
        "502",
        "504",
        "INTERNAL SERVER ERROR",
        "TEMPORARILY UNAVAILABLE",
        "OVERLOADED",
    )

    return any(marker in message for marker in temporary_markers)


def ask_ai(instructions: str, payload: dict[str, Any]) -> dict:
    content = json.dumps(payload, ensure_ascii=False)
    errors = []

    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # PRIMER MOTOR: GEMINI
    if gemini_key:
        client = None

        try:
            client = genai.Client(api_key=gemini_key)

            for attempt in range(GEMINI_MAX_ATTEMPTS):
                try:
                    print(
                        f"IA: intentando con Gemini ({GEMINI_MODEL}), "
                        f"intento {attempt + 1}/{GEMINI_MAX_ATTEMPTS}"
                    )

                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=content,
                        config={
                            "system_instruction": instructions,
                            "response_mime_type": "application/json",
                            "temperature": 0.1,
                            "max_output_tokens": 3000,
                        },
                    )

                    result = parse_json(response.text or "")

                    print("IA: Gemini respondió correctamente.")
                    return result

                except Exception as e:
                    message = f"{type(e).__name__}: {str(e)}"

                    print("ERROR GEMINI:", message)

                    # Solo reintentar errores temporales.
                    if (
                        is_temporary_gemini_error(e)
                        and attempt < GEMINI_MAX_ATTEMPTS - 1
                    ):
                        delay = GEMINI_RETRY_DELAYS[attempt]

                        print(
                            f"Gemini no está disponible temporalmente. "
                            f"Nuevo intento en {delay} segundos."
                        )

                        time.sleep(delay)
                        continue

                    errors.append("Gemini: " + message)
                    break

        except Exception as e:
            message = f"{type(e).__name__}: {str(e)}"
            errors.append("Gemini: " + message)
            print("ERROR INICIALIZANDO GEMINI:", message)

        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

    else:
        errors.append("Gemini: falta GEMINI_API_KEY")
        print("ERROR: falta GEMINI_API_KEY en Render.")

    # SEGUNDO MOTOR: OPENAI
    if openai_key:
        try:
            print(f"IA: intentando con OpenAI ({OPENAI_MODEL})")

            client = OpenAI(api_key=openai_key)

            try:
                response = client.responses.create(
                    model=OPENAI_MODEL,
                    instructions=instructions,
                    input=content,
                )

                result = parse_json(response.output_text)

                print("IA: OpenAI respondió correctamente.")
                return result

            finally:
                client.close()

        except Exception as e:
            message = f"{type(e).__name__}: {str(e)}"
            errors.append("OpenAI: " + message)

            print("ERROR OPENAI:", message)

    else:
        errors.append("OpenAI: falta OPENAI_API_KEY")
        print("ERROR: falta OPENAI_API_KEY en Render.")

    print("ERROR: ningún motor pudo procesar la solicitud.")

    for error in errors:
        print(error)

    raise HTTPException(
        status_code=503,
        detail=(
            "Los motores de inteligencia artificial no pudieron "
            "procesar la búsqueda. Intentá nuevamente más tarde."
        ),
    )


async def wol_search(client: httpx.AsyncClient, query: str):
    response = await client.get(
        WOL_SEARCH,
        params={
            "q": query,
            "r": "occ",
            "st": "e",
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results = {}

    for link in soup.select('a[href*="/wol/d/"]'):
        title = " ".join(link.get_text(" ", strip=True).split())

        if len(title) < 5:
            continue

        url = urljoin(WOL_BASE, link.get("href", ""))

        if urlparse(url).netloc == "wol.jw.org":
            results[url] = {
                "title": title,
                "url": url,
            }

    return list(results.values())


async def read_article(client: httpx.AsyncClient, item):
    try:
        response = await client.get(item["url"])
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        main = (
            soup.select_one("#article")
            or soup.select_one("main")
            or soup.body
        )

        body = " ".join(main.stripped_strings) if main else ""

        heading = soup.select_one("h1")

        title = (
            " ".join(heading.stripped_strings)
            if heading
            else item["title"]
        )

        publication = ""

        for selector in (
            ".publication",
            ".pubInfo",
            ".docSubTitle",
            ".context",
        ):
            node = soup.select_one(selector)

            if node:
                publication = " ".join(node.stripped_strings)[:180]
                break

        return {
            **item,
            "title": title,
            "body": re.sub(r"\s+", " ", body)[:3000],
            "publication": publication,
            "date": "",
        }

    except Exception as e:
        print(f"ERROR leyendo artículo {item.get('url')}: {e}")

        return {
            **item,
            "body": "",
            "publication": "",
            "date": "",
        }


@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "service": "wol-investigador",
    }


@app.post("/api/search")
async def search(req: SearchRequest):

    print("====================================")
    print("Nueva búsqueda:", req.question)
    print("====================================")

    # ETAPA 1: PLANIFICAR LA BÚSQUEDA

    plan = ask_ai(
        "Sos un planificador de búsquedas para wol.jw.org en español. "
        "Interpretá la intención conceptual. "
        "Devolvé SOLO JSON válido con las claves: "
        "concept (string), facets (array de strings), "
        "queries (array de 4 a 7 búsquedas breves). "
        "Incluí sinónimos y expresiones probables en publicaciones WOL. "
        "No respondas la pregunta. "
        "No inventes títulos ni fuentes.",

        {
            "question": req.question,
        },
    )

    queries = [
        str(query).strip()
        for query in plan.get("queries", [])
        if str(query).strip()
    ][:7]

    if not queries:
        queries = [req.question]

    # ETAPA 2: BUSCAR EN WOL

    async with httpx.AsyncClient(
        headers={
            "User-Agent": "WOLInvestigador/0.3.1",
        },
        timeout=25,
        follow_redirects=True,
    ) as client:

        found = {}

        for query in queries:
            try:
                articles = await wol_search(client, query)

                for article in articles:
                    found[article["url"]] = article

            except httpx.HTTPError as e:
                print(f"ERROR buscando en WOL: {e}")
                continue

        # Conservamos los 8 candidatos.
        candidates = list(found.values())[:8]

        print("Artículos encontrados:", len(candidates))

        if not candidates:
            return {
                "concept": plan.get("concept", ""),
                "articles": [],
                "message": (
                    "WOL no devolvió artículos para estas búsquedas."
                ),
            }

        # ETAPA 3: LEER ARTÍCULOS

        pages = await __import__("asyncio").gather(
            *(read_article(client, article) for article in candidates)
        )

    readable = [
        article
        for article in pages
        if article.get("body")
    ]

    print("Artículos que pudieron leerse:", len(readable))

    if not readable:
        return {
            "concept": plan.get("concept", ""),
            "articles": [],
            "message": (
                "No se pudo leer el contenido de los artículos encontrados."
            ),
        }

    # ETAPA 4: EVALUAR PERTINENCIA

    judged = ask_ai(
        "Evaluás artículos reales obtenidos de wol.jw.org. "
        "Seleccioná solo los que traten sustancialmente "
        "el concepto solicitado. "
        "No premies coincidencias de palabras sin pertinencia. "
        "Devolvé SOLO JSON con esta estructura: "
        "{articles:[{url,relevance,score}]}. "
        "score es un entero del 1 al 5. "
        "Incluí solamente artículos con puntuación 4 o 5. "
        "Utilizá las URLs exactas proporcionadas. "
        "No inventes ni modifiques títulos ni enlaces.",

        {
            "question": req.question,
            "concept": plan.get("concept", ""),
            "facets": plan.get("facets", []),

            "articles": [
                {
                    "title": article["title"],
                    "url": article["url"],
                    "publication": article.get("publication", ""),
                    "content": article["body"][:3000],
                }
                for article in readable
            ],
        },
    )

    # ETAPA 5: VALIDAR RESULTADOS

    by_url = {
        article["url"]: article
        for article in readable
    }

    selected = []

    for result in judged.get("articles", []):

        url = result.get("url")

        if url not in by_url:
            continue

        try:
            score = int(result.get("score", 0))
        except (ValueError, TypeError):
            continue

        if score < 4:
            continue

        article = by_url[url]

        selected.append(
            Article(
                title=article["title"],
                publication=article.get("publication") or None,
                date=article.get("date") or None,
                url=url,
                relevance=str(
                    result.get("relevance", "")
                )[:240],
            )
        )

    selected = selected[:req.max_results]

    print("Resultados pertinentes:", len(selected))
    print("Búsqueda finalizada.")

    return {
        "concept": plan.get("concept", ""),
        "articles": selected,
        "message": (
            None
            if selected
            else "No encontramos artículos que superen el criterio de pertinencia."
        ),
                    }
    
