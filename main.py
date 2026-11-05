

import os
import re
import json
import time
from pathlib import Path
from urllib.parse import urlencode, urljoin

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"

WOL_ORIGIN = "https://wol.jw.org"
WOL_SEARCH = "https://wol.jw.org/es/wol/s/r4/lp-s"
APP_VERSION = "0.4.0"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

app = FastAPI(title="WOL Investigador", version=APP_VERSION)

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


class SearchRequest(BaseModel):
    question: str = Field(min_length=4, max_length=1200)
    max_results: int = Field(default=8, ge=1, le=20)


def clean_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def is_quota_error(error):
    message = str(error).lower()
    return any(term in message for term in [
        "resource_exhausted",
        "quota exceeded",
        "quota_exceeded",
        "rate_limit_exceeded",
        "insufficient_quota",
        "tokens per minute",
        "requests per day",
        "429"
    ])


def is_temporary_error(error):
    message = str(error).lower()
    return any(term in message for term in [
        "503",
        "unavailable",
        "service unavailable",
        "overloaded",
        "temporarily unavailable"
    ])


def call_gemini(prompt):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text

    except Exception as exc:
        print("ERROR GEMINI:", repr(exc))

        if is_quota_error(exc):
            print("Gemini: cuota agotada. No se realizarán más intentos.")
        elif is_temporary_error(exc):
            print("Gemini: servicio temporalmente no disponible.")
        return None


def call_openai(prompt):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=key,
            timeout=35,
            max_retries=0
        )

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=1800
        )

        return response.output_text

    except Exception as exc:
        print("ERROR OPENAI:", repr(exc))

        if is_quota_error(exc):
            print("OpenAI: límite de uso alcanzado.")
        return None


def ask_ai(prompt):
    # Primero intenta Gemini. No repite solicitudes si falla.
    result = call_gemini(prompt)

    if result:
        return result, "Gemini"

    # Si Gemini no responde, intenta OpenAI una sola vez.
    result = call_openai(prompt)

    if result:
        return result, "OpenAI"

    return None, "local"


def fallback_plan(question):
    q = question.lower()

    if any(w in q for w in [
        "atalaya", "conducir", "dirigir", "presidir"
    ]):
        topic = "Conducción del Estudio de La Atalaya"
        queries = [
            "conductor del Estudio de La Atalaya",
            "preparación para conducir el estudio",
            "preguntas y participación Estudio de La Atalaya",
            "dirigir estudio congregación enseñanza",
            "preparar comentarios Estudio de La Atalaya"
        ]

    elif any(w in q for w in [
        "desanim", "triste", "abatid", "deprim"
    ]):
        topic = "Desánimo y cómo animar"
        queries = [
            "desánimo",
            "animar consolar fortalecer",
            "ayudar a alguien desanimado",
            "palabras de ánimo consuelo",
            "cómo recuperar el ánimo"
        ]

    elif any(w in q for w in [
        "vejez", "anciano", "envejecer"
    ]):
        topic = "Sentimientos y dificultades en la vejez"
        queries = [
            "sentimientos en la vejez",
            "personas mayores sentimientos",
            "afrontar la vejez",
            "ayudar a los mayores",
            "envejecer con alegría"
        ]

    else:
        words = [
            w.strip("¿?¡!.,;:()")
            for w in question.split()
            if len(w) > 3
        ]
        topic = " ".join(words[:7]) or question
        queries = [
            topic,
            "consejos " + topic,
            "ayuda " + topic,
            "orientación " + topic,
            "cómo afrontar " + topic
        ]

    return {
        "topic": topic,
        "queries": queries[:6]
    }


def interpret_question(question):
    prompt = f"""
Convertí esta pregunta en un plan de búsqueda para
la Biblioteca en Línea Watchtower en español.

No respondas la pregunta.
No redactes explicaciones religiosas.
Generá entre 4 y 6 consultas breves y diferentes.
Incluí sinónimos y conceptos relacionados.

Devolvé únicamente JSON con esta estructura:
{{"topic":"tema principal","queries":["consulta 1","consulta 2","consulta 3","consulta 4"]}}

Pregunta: {question}
"""

    result, provider = ask_ai(prompt)

    if result:
        try:
            result = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.strip())
            data = json.loads(result)
            queries = data.get("queries", [])

            if isinstance(queries, list) and queries:
                return {
                    "topic": data.get("topic", question),
                    "queries": queries[:6],
                    "provider": provider
                }
        except Exception as exc:
            print("Error interpretando JSON:", repr(exc))

    plan = fallback_plan(question)
    plan["provider"] = "local"
    return plan


def search_wol(query, limit=10):
    url = WOL_SEARCH + "?" + urlencode({"q": query})

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=20
        ) as client:
            response = client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 WOLInvestigador/0.4"
                }
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        found = []
        seen = set()

        for link in soup.select('a[href*="/wol/d/"]'):
            href = urljoin(WOL_ORIGIN, link.get("href", ""))
            title = clean_text(link.get_text(" ", strip=True))

            if not title or len(title) < 5:
                continue

            if not href.startswith(WOL_ORIGIN + "/"):
                continue

            if href in seen:
                continue

            seen.add(href)

            found.append({
                "title": title,
                "url": href,
                "search_query": query
            })

            if len(found) >= limit:
                break

        return found

    except Exception as exc:
        print("ERROR WOL:", repr(exc))
        return []


def read_article(article):
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=20
        ) as client:
            response = client.get(
                article["url"],
                headers={
                    "User-Agent": "Mozilla/5.0 WOLInvestigador/0.4"
                }
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for node in soup.select(
            "script, style, nav, footer, header"
        ):
            node.decompose()

        title_node = soup.find(["h1", "h2"])
        title = clean_text(
            title_node.get_text(" ", strip=True)
            if title_node else article["title"]
        )

        content = (
            soup.select_one("#article")
            or soup.select_one("#bodyContent")
            or soup.select_one("main")
            or soup.body
        )

        paragraphs = []

        if content:
            for node in content.find_all(["p", "h2", "h3"]):
                paragraph = clean_text(
                    node.get_text(" ", strip=True)
                )

                if len(paragraph) > 35:
                    paragraphs.append(paragraph)

                if sum(len(p) for p in paragraphs) >= 4500:
                    break

        publication_node = soup.select_one(
            ".publicationName, .pubName"
        )

        publication = (
            clean_text(publication_node.get_text(" ", strip=True))
            if publication_node else ""
        )

        return {
            **article,
            "title": title,
            "publication": publication,
            "text": " ".join(paragraphs)[:4500]
        }

    except Exception as exc:
        print("ERROR leyendo artículo:", repr(exc))
        return {
            **article,
            "publication": "",
            "text": ""
        }


def local_evaluation(question, articles):
    words = set(
        w for w in re.findall(
            r"[a-záéíóúñü]{4,}",
            question.lower()
        )
    )

    selected = []

    for article in articles:
        content = (
            article.get("title", "")
            + " "
            + article.get("text", "")
        ).lower()

        score = sum(1 for word in words if word in content)

        if score:
            selected.append({
                "url": article["url"],
                "relevant": score >= 2,
                "reason": "Coincidencia temática detectada en el artículo."
            })

    return selected


def evaluate_articles(question, articles):
    if not articles:
        return []

    payload = "\n\n".join(
        f"ID: {i}\n"
        f"TÍTULO: {a['title']}\n"
        f"URL: {a['url']}\n"
        f"CONTENIDO: {a['text'][:3500]}"
        for i, a in enumerate(articles)
    )

    prompt = f"""
Sos un evaluador de pertinencia para un buscador de WOL.

Pregunta original:
{question}

Evaluá los artículos según su contenido real.
No aceptes coincidencias superficiales de palabras.
No inventes información ni enlaces.
No respondas la pregunta.
No redactes resúmenes doctrinales.

Seleccioná solamente artículos claramente pertinentes.
Buscá diversidad de aspectos relacionados con la pregunta.
Devolvé únicamente JSON válido con esta estructura:

{{"results":[
{{"id":0,"relevant":true,"reason":"Nota breve de pertinencia"}}
]}}

Máximo 8 resultados.

ARTÍCULOS:
{payload}
"""

    result, provider = ask_ai(prompt)

    if result:
        try:
            result = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.strip())
            judged = json.loads(result).get("results", [])

            output = []
            used = set()

            for item in judged:
                idx = item.get("id")

                if not isinstance(idx, int):
                    continue

                if idx < 0 or idx >= len(articles):
                    continue

                if idx in used or not item.get("relevant"):
                    continue

                used.add(idx)
                article = articles[idx]

                output.append({
                    "title": article["title"],
                    "url": article["url"],
                    "publication": article.get("publication", ""),
                    "relevance": item.get("reason", "Relacionado"),
                    "need": item.get("reason", "Relacionado")
                })

            if output:
                return output[:8], provider

        except Exception as exc:
            print("ERROR evaluando JSON:", repr(exc))

    # Respaldo local: no realiza llamadas adicionales a la API.
    judged = local_evaluation(question, articles)
    output = []

    for item in judged:
        if not item["relevant"]:
            continue

        article = next(
            (a for a in articles if a["url"] == item["url"]),
            None
        )

        if not article:
            continue

        output.append({
            "title": article["title"],
            "url": article["url"],
            "publication": article.get("publication", ""),
            "relevance": item["reason"],
            "need": "Coincidencia temática"
        })

    return output[:8], "local"


@app.get("/")
def home():
    index = STATIC / "index.html"

    if not index.exists():
        raise HTTPException(
            status_code=404,
            detail="No se encontró la interfaz de la aplicación."
        )

    return FileResponse(index)


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "service": "wol-investigador"
    }


@app.post("/api/search")
def search(request: SearchRequest):
    question = clean_text(request.question)

    print("Nueva búsqueda:", question)

    plan = interpret_question(question)

    all_results = []
    seen = set()

    for query in plan["queries"][:6]:
        results = search_wol(query, limit=10)

        for article in results:
            if article["url"] not in seen:
                seen.add(article["url"])
                all_results.append(article)

        time.sleep(0.15)

    # Conservamos hasta 24 candidatos para leerlos.
    candidates = all_results[:24]

    print("Artículos candidatos:", len(candidates))

    articles = []

    for candidate in candidates:
        article = read_article(candidate)

        if article.get("text"):
            articles.append(article)

    print("Artículos que pudieron leerse:", len(articles))

    if not articles:
        return {
            "version": APP_VERSION,
            "articles": [],
            "message": "No encontramos artículos que pudiéramos leer. Intentá nuevamente.",
            "source": "WOL"
        }

    results, evaluation_provider = evaluate_articles(
        question,
        articles
    )

    print("Artículos pertinentes:", len(results))
    print("Evaluación realizada por:", evaluation_provider)

    return {
        "version": APP_VERSION,
        "question": question,
        "articles": results[:8],
        "articles_inspected": len(articles),
        "message": (
            ""
            if results
            else "No encontramos artículos que podamos confirmar como pertinentes. Probá reformular la consulta."
        ),
        "source": "WOL",
        "planning_provider": plan.get("provider", "local"),
        "evaluation_provider": evaluation_provider
        }
        
