
import os
import re
import json
import unicodedata
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ==========================================
# CONFIGURACION
# ==========================================

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"

WOL_ORIGIN = "https://wol.jw.org"
WOL_SEARCH = "https://wol.jw.org/es/wol/s/r4/lp-s"

APP_VERSION = "0.4.2"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

app = FastAPI(
    title="WOL Investigador",
    version=APP_VERSION
)

if STATIC.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC),
        name="static"
    )


# ==========================================
# SOLICITUD
# ==========================================

class SearchRequest(BaseModel):
    question: str = Field(
        min_length=4,
        max_length=1200
    )

    max_results: int = Field(
        default=8,
        ge=1,
        le=20
    )


# ==========================================
# FUNCIONES GENERALES
# ==========================================

def clean_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def normalize(value):
    value = (value or "").lower()

    value = unicodedata.normalize("NFD", value)

    value = "".join(
        char for char in value
        if unicodedata.category(char) != "Mn"
    )

    return re.findall(r"[a-z]{3,}", value)


def unique_list(items):
    result = []
    seen = set()

    for item in items:
        item = clean_text(item)

        key = item.lower()

        if item and key not in seen:
            seen.add(key)
            result.append(item)

    return result


# ==========================================
# INTELIGENCIA ARTIFICIAL
# ==========================================

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
        return None


def ask_ai(prompt):
    result = call_gemini(prompt)

    if result:
        return result, "Gemini"

    result = call_openai(prompt)

    if result:
        return result, "OpenAI"

    return None, "local"


def extract_json(text):
    if not text:
        return None

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.S)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


# ==========================================
# PLAN DE BUSQUEDA DE RESPALDO
# ==========================================

def fallback_plan(question):
    q = " ".join(normalize(question))

    if any(word in q for word in [
        "vejez", "anciano", "ancianos",
        "envejecer", "mayores", "envejecimiento"
    ]):
        topic = "Sentimientos y dificultades en la vejez"

        queries = [
            "sentimientos en la vejez",
            "personas mayores sentimientos emociones",
            "afrontar la vejez",
            "ayudar a las personas mayores",
            "envejecer con alegria",
            "soledad y vejez"
        ]

    elif any(word in q for word in [
        "desanimo", "triste", "abatido",
        "deprimido", "desalentado"
    ]):
        topic = "Desanimo y animo"

        queries = [
            "desanimo",
            "animar consolar fortalecer",
            "ayudar a alguien desanimado",
            "palabras de animo y consuelo",
            "recuperar el animo",
            "como afrontar el desanimo"
        ]

    elif any(word in q for word in [
        "atalaya", "conducir", "dirigir", "presidir"
    ]):
        topic = "Conduccion del Estudio de La Atalaya"

        queries = [
            "conductor del Estudio de La Atalaya",
            "preparacion para conducir el estudio",
            "preguntas y participacion Estudio de La Atalaya",
            "dirigir estudio congregacion",
            "preparar comentarios Estudio de La Atalaya"
        ]

    else:
        words = [
            word for word in question.split()
            if len(normalize(word)) > 0
        ]

        topic = " ".join(words[:8]) or question

        queries = [
            topic,
            "consejos sobre " + topic,
            "ayuda para " + topic,
            "orientacion sobre " + topic,
            "como afrontar " + topic
        ]

    return {
        "topic": topic,
        "queries": queries
    }


# ==========================================
# INTERPRETAR LA PREGUNTA
# ==========================================

def interpret_question(question):
    prompt = f"""
Tu tarea es preparar una busqueda en la Biblioteca
en Linea Watchtower, exclusivamente en espanol.

No respondas la pregunta.
No escribas explicaciones religiosas.
No inventes articulos.

Interpreta el significado de la pregunta.
Incluye sinonimos y conceptos relacionados.
Genera entre 4 y 6 consultas de busqueda diferentes.

Devuelve solamente JSON con esta estructura:

{{
  "topic": "tema principal",
  "queries": [
    "consulta 1",
    "consulta 2",
    "consulta 3",
    "consulta 4"
  ]
}}

Pregunta: {question}
"""

    result, provider = ask_ai(prompt)

    if result:
        data = extract_json(result)

        if isinstance(data, dict):
            queries = data.get("queries", [])

            if isinstance(queries, list):
                queries = unique_list(queries)

                if queries:
                    return {
                        "topic": clean_text(
                            data.get("topic", question)
                        ),
                        "queries": queries[:6],
                        "provider": provider
                    }

    plan = fallback_plan(question)
    plan["provider"] = "local"

    return plan


# ==========================================
# BUSQUEDA EN WOL
# ==========================================

def search_wol(query, limit=10):
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=20
        ) as client:

            response = client.get(
                WOL_SEARCH,
                params={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 WOLInvestigador/0.4"
                }
            )

            response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        found = []
        seen = set()

        for link in soup.select('a[href*="/wol/d/"]'):
            href = urljoin(
                WOL_ORIGIN,
                link.get("href", "")
            )

            title = clean_text(
                link.get_text(" ", strip=True)
            )

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


# ==========================================
# LECTURA DE ARTICULOS
# ==========================================

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

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for node in soup.select(
            "script, style, nav, footer, header"
        ):
            node.decompose()

        title_node = soup.find(["h1", "h2"])

        title = clean_text(
            title_node.get_text(" ", strip=True)
            if title_node else article["title"]
        )

        container = (
            soup.select_one("#article")
            or soup.select_one("main")
            or soup.body
        )

        if not container:
            return None

        paragraphs = []

        for paragraph in container.find_all(
            ["p", "h2", "h3"]
        ):
            text = clean_text(
                paragraph.get_text(" ", strip=True)
            )

            if len(text) > 35:
                paragraphs.append(text)

        content = clean_text(" ".join(paragraphs))

        if len(content) < 150:
            return None

        content = content[:4500]

        publication = ""
        date = ""

        metadata = soup.select(".publication, .date, .pubDate")

        if metadata:
            publication = clean_text(
                metadata[0].get_text(" ", strip=True)
            )

        return {
            "title": title,
            "url": article["url"],
            "content": content,
            "publication": publication,
            "date": date
        }

    except Exception as exc:
        print(
            "ERROR LEYENDO ARTICULO:",
            article.get("url"),
            repr(exc)
        )
        return None


# ==========================================
# EVALUACION LOCAL DE PERTINENCIA
# ==========================================

CONCEPT_GROUPS = {
    "vejez": [
        "vejez", "anciano", "ancianos", "mayores",
        "envejecer", "envejecimiento", "edad avanzada"
    ],
    "emociones": [
        "sentimientos", "emociones", "tristeza",
        "alegria", "animo", "soledad", "preocupacion",
        "miedo", "frustracion", "sentir"
    ],
    "familia": [
        "familia", "hijos", "padres", "matrimonio",
        "esposa", "esposo", "parientes"
    ],
    "espiritualidad": [
        "jehova", "oracion", "biblia", "espiritual",
        "fe", "confianza", "dios"
    ],
    "desanimo": [
        "desanimo", "desalentado", "triste",
        "abatido", "consuelo", "animar", "fortalecer"
    ]
}


def local_evaluation(question, article):
    question_words = set(normalize(question))
    article_words = set(normalize(
        article["title"] + " " + article["content"]
    ))

    if not question_words:
        return None

    direct_matches = question_words.intersection(
        article_words
    )

    score = len(direct_matches)

    matched_groups = 0

    for terms in CONCEPT_GROUPS.values():
        normalized_terms = set()

        for term in terms:
            normalized_terms.update(normalize(term))

        if question_words.intersection(normalized_terms):
            if article_words.intersection(normalized_terms):
                matched_groups += 1

    # Evita aceptar articulos por una coincidencia aislada.
    if score < 2 and matched_groups < 2:
        return None

    if score < 3 and matched_groups < 2:
        return None

    return {
        "score": score + matched_groups,
        "relevance": (
            "El contenido presenta coincidencias tematicas "
            "con la consulta. Conviene revisar el articulo original."
        )
    }


# ==========================================
# EVALUACION CON IA
# ==========================================

def ai_evaluation(question, articles):
    if not articles:
        return None, "local"

    compact_articles = []

    for index, article in enumerate(articles):
        compact_articles.append({
            "id": index,
            "title": article["title"],
            "content": article["content"][:3500]
        })

    prompt = f"""
Evalua la pertinencia de los siguientes articulos
de la Biblioteca en Linea Watchtower.

Pregunta del usuario:
{question}

Tu tarea NO es responder la pregunta.

Debes determinar si cada articulo trata realmente
el tema solicitado, considerando su contenido completo.

Reglas:
- No aceptes articulos por compartir solamente palabras.
- No confundas una mencion secundaria con el tema principal.
- Rechaza resultados que solo tengan una coincidencia superficial.
- No inventes informacion.
- No escribas explicaciones religiosas.
- Usa exclusivamente los articulos proporcionados.
- Si no hay articulos pertinentes, devuelve una lista vacia.

Devuelve solamente JSON:

{{
  "relevant": [
    {{
      "id": 0,
      "reason": "Motivo breve de pertinencia"
    }}
  ]
}}

Articulos:
{json.dumps(compact_articles, ensure_ascii=False)}
"""

    result, provider = ask_ai(prompt)

    if not result:
        return None, "local"

    data = extract_json(result)

    if not isinstance(data, dict):
        return None, "local"

    relevant = data.get("relevant")

    if not isinstance(relevant, list):
        return None, "local"

    accepted = []

    for item in relevant:
        if not isinstance(item, dict):
            continue

        try:
            index = int(item.get("id"))
        except (ValueError, TypeError):
            continue

        if index < 0 or index >= len(articles):
            continue

        reason = clean_text(item.get("reason", ""))

        if not reason:
            continue

        accepted.append({
            "index": index,
            "relevance": reason
        })

    return accepted, provider


# ==========================================
# INVESTIGACION
# ==========================================

def investigate(request):
    question = clean_text(request.question)

    plan = interpret_question(question)

    queries = unique_list(plan["queries"])

    candidates = []
    seen_urls = set()

    # Busqueda amplia: hasta 8 resultados por consulta.
    for query in queries:
        results = search_wol(query, limit=8)

        for item in results:
            if item["url"] in seen_urls:
                continue

            seen_urls.add(item["url"])
            candidates.append(item)

            if len(candidates) >= 24:
                break

        if len(candidates) >= 24:
            break

    print(
        "CONSULTAS:",
        queries,
        "CANDIDATOS:",
        len(candidates)
    )

    if not candidates:
        return {
            "topic": plan["topic"],
            "articles": [],
            "message": (
                "No se encontraron articulos para investigar. "
                "Intenta formular la pregunta de otra manera."
            )
        }

    articles = []

    for candidate in candidates:
        article = read_article(candidate)

        if article:
            articles.append(article)

    print(
        "ARTICULOS LEIDOS:",
        len(articles)
    )

    if not articles:
        return {
            "topic": plan["topic"],
            "articles": [],
            "message": (
                "Se encontraron resultados en WOL, "
                "pero no fue posible leer su contenido."
            )
        }

    evaluated, provider = ai_evaluation(
        question,
        articles
    )

    accepted = []

    if evaluated is not None:
        for item in evaluated:
            article = articles[item["index"]].copy()

            article["relevance"] = item["relevance"]

            article.pop("content", None)

            accepted.append(article)

    else:
        # Respaldo local si las API no responden.
        for article in articles:
            evaluation = local_evaluation(
                question,
                article
            )

            if evaluation:
                result = article.copy()

                result["relevance"] = evaluation["relevance"]

                result.pop("content", None)

                accepted.append(result)

        provider = "local"

    # Eliminar duplicados y ordenar.
    final_articles = []
    final_seen = set()

    for article in accepted:
        if article["url"] in final_seen:
            continue

        final_seen.add(article["url"])
        final_articles.append(article)

    final_articles = final_articles[:request.max_results]

    if final_articles:
        message = (
            f"Se encontraron {len(final_articles)} "
            "articulos que pueden ser pertinentes."
        )
    else:
        message = (
            "No encontramos articulos que superen "
            "la evaluacion de pertinencia. "
            "Prueba con otra formulacion."
        )

    return {
        "topic": plan["topic"],
        "articles": final_articles,
        "message": message,
        "provider": provider
    }


# ==========================================
# RUTAS
# ==========================================

@app.get("/")
def home():
    index_file = STATIC / "index.html"

    if not index_file.exists():
        return {
            "service": "WOL Investigador",
            "version": APP_VERSION
        }

    return FileResponse(index_file)


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "service": "wol-investigador"
    }


@app.post("/api/search")
def search(request: SearchRequest):
    try:
        return investigate(request)

    except Exception as exc:
        print(
            "ERROR GENERAL DE BUSQUEDA:",
            repr(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Ocurrio un error durante la investigacion. "
                "Intenta nuevamente mas tarde."
            )
        )
        
