import os, re, json, asyncio
from typing import Any
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

APP_VERSION = "0.1.0"
WOL_BASE = "https://wol.jw.org"
WOL_SEARCH = "https://wol.jw.org/es/wol/s/r4/lp-s"
MODEL = os.getenv("OPENAI_MODEL", "gpt-6")
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

def ask_ai(instructions: str, payload: dict[str, Any]) -> dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="Falta configurar OPENAI_API_KEY en el servidor.")
    response = OpenAI(api_key=key).responses.create(
        model=MODEL, instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False))
    raw = response.output_text.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.I).strip()
    return json.loads(raw)

async def wol_search(client: httpx.AsyncClient, query: str):
    r = await client.get(WOL_SEARCH, params={"q": query, "r": "occ", "st": "e"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = {}
    for a in soup.select('a[href*="/wol/d/"]'):
        title = " ".join(a.get_text(" ", strip=True).split())
        if len(title) < 5:
            continue
        url = urljoin(WOL_BASE, a.get("href", ""))
        if urlparse(url).netloc == "wol.jw.org":
            out[url] = {"title": title, "url": url}
    return list(out.values())

async def read_article(client: httpx.AsyncClient, item):
    try:
        r = await client.get(item["url"])
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        main = soup.select_one("#article") or soup.select_one("main") or soup.body
        body = " ".join(main.stripped_strings) if main else ""
        h1 = soup.select_one("h1")
        title = " ".join(h1.stripped_strings) if h1 else item["title"]
        pub = ""
        for selector in (".publication", ".pubInfo", ".docSubTitle", ".context"):
            node = soup.select_one(selector)
            if node:
                pub = " ".join(node.stripped_strings)[:180]
                break
        return {**item, "title": title, "body": re.sub(r"\s+", " ", body)[:12000],
                "publication": pub, "date": ""}
    except Exception:
        return {**item, "body": "", "publication": "", "date": ""}

@app.get("/")
async def home():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"ok": True, "version": APP_VERSION, "service": "wol-investigador"}

@app.post("/api/search")
async def search(req: SearchRequest):
    plan = ask_ai(
        "Sos un planificador de búsquedas para wol.jw.org en español. Interpretá la intención conceptual. "
        "Devolvé SOLO JSON válido con keys concept (string), facets (array de strings), queries (array de 4 a 7 búsquedas breves). "
        "Incluí sinónimos y expresiones probables en publicaciones WOL. No respondas la pregunta. No inventes títulos ni fuentes.",
        {"question": req.question})
    queries = [str(x).strip() for x in plan.get("queries", []) if str(x).strip()][:7] or [req.question]
    async with httpx.AsyncClient(headers={"User-Agent": "WOLInvestigador/0.1"},
                                 timeout=25, follow_redirects=True) as client:
        found = {}
        for query in queries:
            try:
                for a in await wol_search(client, query):
                    found[a["url"]] = a
            except httpx.HTTPError:
                continue
        candidates = list(found.values())[:35]
        if not candidates:
            return {"concept": plan.get("concept", ""), "articles": [],
                    "message": "WOL no devolvió artículos para estas búsquedas."}
        pages = await asyncio.gather(*(read_article(client, a) for a in candidates))
    readable = [a for a in pages if a.get("body")]
    if not readable:
        return {"concept": plan.get("concept", ""), "articles": [],
                "message": "No se pudo leer el contenido de los artículos encontrados."}
    judged = ask_ai(
        "Evaluás artículos reales obtenidos de wol.jw.org. Seleccioná solo los que traten sustancialmente el concepto solicitado. "
        "No premies coincidencias de palabras sin pertinencia. Devolvé SOLO JSON: {articles:[{url,relevance,score}]}. "
        "score es entero 1-5; incluí solo 4 o 5. URLs exactas, no inventes ni modifiques títulos o enlaces.",
        {"question": req.question, "concept": plan.get("concept", ""), "facets": plan.get("facets", []),
         "articles": [{"title": a["title"], "url": a["url"], "publication": a.get("publication", ""),
                      "content": a["body"][:8500]} for a in readable]})
    by_url = {a["url"]: a for a in readable}
    selected = []
    for x in judged.get("articles", []):
        url = x.get("url")
        if url in by_url and int(x.get("score", 0)) >= 4:
            a = by_url[url]
            selected.append(Article(title=a["title"], publication=a.get("publication") or None,
                date=a.get("date") or None, url=url, relevance=str(x.get("relevance", ""))[:240]))
    selected = selected[:req.max_results]
    return {"concept": plan.get("concept", ""), "articles": selected,
            "message": None if selected else "No encontramos artículos que superen el criterio de pertinencia."}
