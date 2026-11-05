
import os
import re
import json
import time
import unicodedata

from pathlib import Path
from urllib.parse import urlencode, urljoin

import httpx

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ==========================================
# CONFIGURACIÓN
# ==========================================

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"

WOL_ORIGIN = "https://wol.jw.org"
WOL_SEARCH = "https://wol.jw.org/es/wol/s/r4/lp-s"

APP_VERSION = "0.4.1"

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-6-luna"
)

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
# MODELO DE SOLICITUD
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
    return re.sub(
        r"\s+",
        " ",
        value or ""
    ).strip()


def normalize(text):

    text = (text or "").lower()

    text = unicodedata.normalize(
        "NFD",
        text
    )

    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )

    return re.findall(
        r"[a-z]{3,}",
        text
    )


# ==========================================
# CONTROL DE ERRORES DE API
# ==========================================

def is_quota_error(error):

    message = str(error).lower()

    terms = [
        "resource_exhausted",
        "quota exceeded",
        "quota_exceeded",
        "rate_limit_exceeded",
        "insufficient_quota",
        "tokens per minute",
        "requests per day",
        "429"
    ]

    return any(
        term in message
        for term in terms
    )


def is_temporary_error(error):

    message = str(error).lower()

    terms = [
        "503",
        "unavailable",
        "service unavailable",
        "overloaded",
        "temporarily unavailable"
    ]

    return any(
        term in message
        for term in terms
    )


# ==========================================
# GEMINI
# ==========================================

def call_gemini(prompt):

    key = os.getenv("GEMINI_API_KEY")

    if not key:
        return None

    try:

        from google import genai

        client = genai.Client(
            api_key=key
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text

    except Exception as exc:

        print(
            "ERROR GEMINI:",
            repr(exc)
        )

        if is_quota_error(exc):

            print(
                "Gemini: cuota agotada."
            )

        elif is_temporary_error(exc):

            print(
                "Gemini: servicio temporalmente no disponible."
            )

        return None


# ==========================================
# OPENAI
# ==========================================

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

        print(
            "ERROR OPENAI:",
            repr(exc)
        )

        if is_quota_error(exc):

            print(
                "OpenAI: límite de uso alcanzado."
            )

        return None


# ==========================================
# SELECCIÓN DE PROVEEDOR
# ==========================================

def ask_ai(prompt):

    result = call_gemini(prompt)

    if result:
        return result, "Gemini"

    result = call_openai(prompt)

    if result:
        return result, "OpenAI"

    return None, "local"


# ==========================================
# PLAN DE BÚSQUEDA LOCAL
# ==========================================

def fallback_plan(question):

    q = question.lower()

    if any(word in q for word in [
        "atalaya",
        "conducir",
        "dirigir",
        "presidir"
    ]):

        topic = "Conducción del Estudio de La Atalaya"

        queries = [
            "conductor del Estudio de La Atalaya",
            "preparación para conducir el estudio",
            "preguntas y participación Estudio de La Atalaya",
            "dirigir estudio congregación enseñanza",
            "preparar comentarios Estudio de La Atalaya"
        ]

    elif any(word in q for word in [
        "desanim",
        "triste",
        "abatid",
        "deprim"
    ]):

        topic = "Desánimo y cómo animar"

        queries = [
            "desánimo",
            "animar consolar fortalecer",
            "ayudar a alguien desanimado",
            "palabras de ánimo consuelo",
            "cómo recuperar el ánimo"
        ]

    elif any(word in q for word in [
        "vejez",
        "anciano",
        "envejecer",
        "mayores"
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
            word.strip("¿?¡!.,;:()")
            for word in question.split()
            if len(word) > 3
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


# ==========================================
# INTERPRETACIÓN DE LA PREGUNTA
# ==========================================

def interpret_question(question):

    prompt = f"""
Convertí esta pregunta en un plan de búsqueda
para la Biblioteca en Línea Watchtower en español.

No respondas la pregunta.

No redactes explicaciones religiosas.

Generá entre 4 y 6 consultas breves y diferentes.

Incluí sinónimos y conceptos relacionados.

Devolvé únicamente JSON con esta estructura:

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

        try:

            result = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                result.strip()
            )

            data = json.loads(result)

            queries = data.get(
                "queries",
                []
            )

            if isinstance(queries, list) and queries:

                return {
                    "topic": data.get(
                        "topic",
                        question
                    ),
                    "queries": queries[:6],
                    "provider": provider
                }

        except Exception as exc:

            print(
                "Error interpretando JSON:",
                repr(exc)
            )

    plan = fallback_plan(question)

    plan["provider"] = "local"

    return plan


# ==========================================
# BÚSQUEDA EN WOL
# ==========================================

def search_wol(query, limit=10):

    url = WOL_SEARCH + "?" + urlencode({
        "q": query
    })

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

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        found = []
        seen = set()

        for link in soup.select(
            'a[href*="/wol/d/"]'
        ):

            href = urljoin(
                WOL_ORIGIN,
                link.get("href", "")
            )

            title = clean_text(
                link.get_text(
                    " ",
                    strip=True
                )
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

        print(
            "ERROR WOL:",
            repr(exc)
        )

        return []


# ==========================================
# LECTURA DE ARTÍCULOS
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

        title_node = soup.find([
            "h1",
            "h2"
        ])

        title = clean_text(
            title_n
