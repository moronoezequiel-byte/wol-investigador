import os
import re
import json
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
WOL_ORIGIN = "https://wol.jw.org"
WOL_SEARCH = "https://wol.jw.org/es/wol/s/r4/lp-s"
APP_VERSION = "0.5.0"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
HTTP_TIMEOUT = httpx.Timeout(14.0, connect=7.0)
USER_AGENT = "Mozilla/5.0 (compatible; WOLInvestigador/0.5; +https://wol.jw.org/)"
STOPWORDS = {"al", "de", "el", "la", "en", "un", "es", "no", "si", "mi", "tu", "su", "ya", "me", "te", "se", "lo", "le", "ha", "he", "va", "ve", "da", "do", "hay", "y", "que", "como", "para", "por", "con", "sin", "una", "uno", "unos", "unas", "del", "las", "los", "les", "sus", "esto", "esta", "este", "esa", "ese", "son", "ser", "sea", "hay", "puede", "pueden", "cuando", "donde", "quien", "quienes", "cual", "cuales", "sobre", "entre", "desde", "hasta", "muy", "mas", "menos", "tambien", "porque", "pero", "ante", "tras", "cada", "todo", "toda", "todos", "todas", "algo", "alguien", "tener", "tiene", "tienen", "hacer", "hace", "hacen", "fue", "son", "era", "eran", "del", "les", "nos", "ellos", "ellas", "usted", "ustedes"}

app = FastAPI(title="WOL Investigador", version=APP_VERSION)
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


class SearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1200)
    max_results: int = Field(default=8, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value):
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Escribí un tema de al menos dos caracteres.")
        return value


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize(value):
    value = unicodedata.normalize("NFD", (value or "").lower())
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return [token for token in re.findall(r"[a-z0-9]{2,}", value) if token not in STOPWORDS]


def norm_key(value):
    return " ".join(normalize(value))


def unique_list(items, limit=None):
    result, seen = [], set()
    for item in items or []:
        item = clean_text(item)
        key = norm_key(item)
        if item and key and key not in seen:
            seen.add(key)
            result.append(item)
        if limit and len(result) >= limit:
            break
    return result


# Variantes concretas para mejorar el punto de partida; la IA agrega conceptos
# nuevos para cualquier otro tema. Estas listas nunca limitan la búsqueda.
SEED_PLANS = [
    (("vejez", "anciano", "ancianos", "envejecer", "mayores", "envejecimiento", "edad avanzada", "tercera edad", "adulto mayor", "ancianidad"),
     "La vejez y las personas mayores", [
         "sentimientos en la vejez", "personas mayores y emociones", "afrontar la vejez",
         "ayudar a las personas mayores", "soledad en la vejez", "envejecer con alegria"]),
    (("desanimo", "desalentado", "desaliento", "abatido", "abatimiento", "desanimado", "triste", "tristeza", "deprimido", "afligido"),
     "El desanimo y como afrontarlo", [
         "desanimo", "como vencer el desanimo", "que hacer en caso de desanimo",
         "personas desanimadas consuelo animo", "recuperar el animo", "fortalecer a quien esta triste"]),
    (("atalaya", "conducir", "dirigir", "presidir"),
     "Conducir el Estudio de La Atalaya", [
         "conductor del Estudio de La Atalaya", "preparacion para conducir el estudio",
         "preguntas y participacion Estudio de La Atalaya", "dirigir estudio congregacion"]),
]

SEMANTIC_GROUPS = {
    "desanimo": {"desanimo", "desanimado", "desanimada", "desanimarse", "desalentado", "desaliento", "abatido", "abatimiento", "afligido", "afliccion", "triste", "tristeza", "perder el animo", "falta de animo", "hundido"},
    "vejez": {"vejez", "anciano", "ancianos", "persona mayor", "personas mayores", "edad avanzada", "envejecimiento", "envejecer", "achaques de la edad"},
    "consuelo": {"consuelo", "consolar", "consuelo", "animar", "animo", "fortalecer", "dar fuerzas", "apoyo", "aliento"},
    "culpa": {"culpa", "sentirse culpable", "remordimiento", "arrepentimiento", "perdon", "perdonar"},
    "soledad": {"soledad", "aislamiento", "solo", "solitario", "falta de compania"},
    "ansiedad": {"ansiedad", "angustia", "preocupacion", "nerviosismo", "miedo", "temor"},
}


def fallback_plan(question):
    words = set(normalize(question))
    for triggers, topic, queries in SEED_PLANS:
        if words.intersection({w for t in triggers for w in normalize(t)}):
            return {"topic": topic, "queries": list(queries)}
    topic = clean_text(question.rstrip("?!. "))
    return {"topic": topic[:120], "queries": [
        topic, f"{topic} consejos", f"{topic} ayuda", f"{topic} ejemplos",
        f"como afrontar {topic}", f"que dice la Biblia sobre {topic}"
    ]}


def call_gemini(prompt):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=25000))
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=1400),
        )
        return getattr(response, "text", None)
    except Exception as exc:
        print("GEMINI no disponible:", type(exc).__name__)
        return None


def call_openai(prompt):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=25, max_retries=0)
        response = client.responses.create(
            model=OPENAI_MODEL, input=prompt, max_output_tokens=1400,
        )
        return response.output_text
    except Exception as exc:
        print("OPENAI no disponible:", type(exc).__name__)
        return None


def ask_ai(prompt):
    result = call_gemini(prompt)
    if result:
        return result, "Gemini"
    # OpenAI puede generar cargos. Solo se usa si el usuario lo habilita
    # expresamente mediante ENABLE_OPENAI_FALLBACK=true en el entorno.
    allow_openai = os.getenv("ENABLE_OPENAI_FALLBACK", "").strip().lower() in {"1", "true", "yes", "si"}
    if allow_openai:
        result = call_openai(prompt)
        if result:
            return result, "OpenAI"
    return None, "local"


def extract_json(raw):
    if not raw:
        return None
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    # Busca el primer objeto JSON equilibrado, sin tragarse texto posterior.
    start = text.find("{")
    if start < 0:
        return None
    depth, quoted, escaped = 0, False, False
    for pos in range(start, len(text)):
        char = text[pos]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        else:
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:pos + 1])
                    except ValueError:
                        return None
    return None


def interpret_question(question):
    backup = fallback_plan(question)
    prompt = f'''Prepara consultas para buscar artículos en la Biblioteca en Línea Watchtower en ESPAÑOL.
No respondas la pregunta ni des explicaciones religiosas. No inventes títulos ni enlaces.
Interpreta la intención, no repitas simplemente las mismas palabras. Propón sinónimos, expresiones equivalentes,
problemas relacionados, maneras habituales de formular el tema y conceptos específicos que puedan aparecer
en artículos. Mantén las consultas breves, naturales y distintas entre sí. No agregues conceptos solo por
ser religiosos: deben guardar relación real con la pregunta.
Devuelve SOLO JSON: {{"topic":"tema breve", "queries":["consulta 1","consulta 2","consulta 3","consulta 4","consulta 5","consulta 6"]}}
Pregunta: {question}'''
    raw, provider = ask_ai(prompt)
    data = extract_json(raw)
    ai_queries, topic = [], backup["topic"]
    if isinstance(data, dict):
        raw_queries = data.get("queries")
        if isinstance(raw_queries, list):
            ai_queries = [clean_text(x) for x in raw_queries if isinstance(x, str)]
        if isinstance(data.get("topic"), str) and clean_text(data["topic"]):
            topic = clean_text(data["topic"])[:120]
    # La IA nunca elimina el plan de respaldo. El total queda acotado.
    queries = unique_list([q[:120] for q in ai_queries[:7] if q] + [q[:120] for q in backup["queries"]], limit=10)
    return {"topic": topic, "queries": queries, "provider": provider}


def valid_article_url(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.netloc == "wol.jw.org" and bool(re.search(r"/es/wol/d/", parsed.path))
    except Exception:
        return False


def search_wol(query, limit=8, client=None):
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
    try:
        response = client.get(WOL_SEARCH, params={"q": query})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        found, seen = [], set()
        for link in soup.select('a[href*="/wol/d/"]'):
            href = urljoin(WOL_ORIGIN, link.get("href", "")).split("#", 1)[0]
            title = clean_text(link.get_text(" ", strip=True))
            if not title or len(title) < 5 or not valid_article_url(href):
                continue
            if href in seen:
                continue
            seen.add(href)
            found.append({"title": title, "url": href, "search_query": query})
            if len(found) >= limit:
                break
        return found
    except Exception as exc:
        print("ERROR WOL búsqueda:", clean_text(query), type(exc).__name__)
        return []
    finally:
        if own_client:
            client.close()


def read_article(article):
    try:
        response = httpx.get(article["url"], follow_redirects=True, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for node in soup.select("script, style, nav, footer, header, form, button, aside, .navigation, .breadcrumb, .sidebar"):
            node.decompose()
        title_node = soup.select_one("h1") or soup.select_one("#article h2")
        title = clean_text(title_node.get_text(" ", strip=True)) if title_node else article["title"]
        container = soup.select_one("#article") or soup.select_one("main") or soup.body
        if not container:
            return None
        # WOL article content is paragraph-based; preserve headings and paragraphs.
        blocks = []
        for node in container.find_all(["p", "h2", "h3"]):
            value = clean_text(node.get_text(" ", strip=True))
            if len(value) >= 24 and value not in blocks:
                blocks.append(value)
        content = clean_text(" ".join(blocks))
        if len(content) < 100:
            return None
        content = content[:6000]
        publication, date = "", ""
        # Recupera metadatos solo cuando están presentes en la página original.
        for selector in ["meta[name='citation_journal_title']", "meta[name='citation_publication_title']"]:
            node = soup.select_one(selector)
            if node and node.get("content"):
                publication = clean_text(node.get("content"))
                break
        if not publication:
            for selector in [".publication", ".pubName", ".publicationName", ".documentInfo"]:
                node = soup.select_one(selector)
                if node:
                    value = clean_text(node.get_text(" ", strip=True))
                    if value and len(value) < 180:
                        publication = value
                        break
        for selector in ["meta[name='citation_publication_date']", "meta[name='citation_date']", "meta[property='article:published_time']", "time[datetime]", ".pubDate", ".date"]:
            node = soup.select_one(selector)
            value = clean_text(node.get("content") or node.get("datetime") or node.get_text(" ", strip=True)) if node else ""
            if value:
                date = value[:40]
                break
        if not date and publication:
            year = re.search(r"\b(?:19|20)\d{2}\b", publication)
            if year:
                date = year.group(0)
        return {"title": title, "url": article["url"], "content": content, "publication": publication, "date": date}
    except Exception as exc:
        print("ERROR leyendo artículo:", article.get("url", ""), type(exc).__name__)
        return None


def expand_terms(question):
    terms = set(normalize(question))
    phrase = norm_key(question)
    for group in SEMANTIC_GROUPS.values():
        normalized_group = {norm_key(term) for term in group}
        if any(term and (term in phrase or phrase in term) for term in normalized_group):
            for term in group:
                terms.update(normalize(term))
    # Añade expresiones locales vinculadas a las variantes explícitas.
    for triggers, _, _ in SEED_PLANS:
        trigger_tokens = {token for trigger in triggers for token in normalize(trigger)}
        if terms.intersection(trigger_tokens):
            for trigger in triggers:
                terms.update(normalize(trigger))
    return terms


def local_evaluation(question, article):
    title_tokens = set(normalize(article.get("title", "")))
    body_tokens = set(normalize(article.get("content", "")))
    query_tokens = expand_terms(question)
    if not query_tokens or not (title_tokens or body_tokens):
        return None
    title_hits = query_tokens.intersection(title_tokens)
    body_hits = query_tokens.intersection(body_tokens)
    phrase = norm_key(question)
    title_norm = norm_key(article.get("title", ""))
    exact_phrase = bool(phrase and len(phrase) >= 2 and phrase in title_norm)
    # Términos semánticos tienen peso menor que el núcleo literal del tema.
    core = set(normalize(question))
    core_title = core.intersection(title_tokens)
    core_body = core.intersection(body_tokens)
    if exact_phrase or len(core_title) >= 1 or len(core_body) >= 2:
        score = len(title_hits) * 3 + len(body_hits) + (8 if exact_phrase else 0)
        if score >= 4:
            evidence = sorted(title_hits or body_hits)[:4]
            reason = "El título y/o el contenido aborda el tema de la consulta" + (f" y menciona {', '.join(evidence)}." if evidence else ".")
            return {"score": score, "relevance": reason}
    # Para una pregunta corta de concepto, exigir varias señales y que aparezca en el título.
    if len(title_hits) >= 2 and len(body_hits) >= 3:
        evidence = sorted(title_hits)[:4]
        return {"score": len(title_hits) * 3 + len(body_hits), "relevance": "El título y el desarrollo tratan el tema: " + ", ".join(evidence) + "."}
    return None


def strong_local_match(question, article, evaluation):
    """Respaldo conservador cuando la IA respondió válidamente que no hay resultados."""
    core = set(normalize(question))
    title = set(normalize(article.get("title", "")))
    body = set(normalize(article.get("content", "")))
    title_hits = core.intersection(title)
    body_hits = core.intersection(body)
    phrase = norm_key(question)
    title_phrase = norm_key(article.get("title", ""))
    if phrase and phrase in title_phrase:
        return True
    if len(core) == 1 and len(title_hits) == 1:
        return True
    return len(title_hits) >= 2 or (len(title_hits) >= 1 and len(body_hits) >= 2)


def ai_evaluation(question, articles):
    if not articles:
        return None, "local"
    accepted, provider_used, had_valid_response = [], "local", False
    # Lotes pequeños para limitar latencia y consumo de tokens.
    failed_indices = []
    for offset in range(0, len(articles), 12):
        batch = articles[offset:offset + 12]
        payload = [{"id": i + offset, "title": a["title"], "content": a["content"][:2400]} for i, a in enumerate(batch)]
        prompt = f'''Evalúa si cada artículo trata DE VERDAD el tema principal de la pregunta.
No respondas la pregunta. No juzgues por una palabra aislada ni por una mención secundaria.
Acepta artículos cuyo título o desarrollo sustancial se centre en el asunto, sus sinónimos directos
o una situación concreta claramente equivalente. Rechaza coincidencias incidentales y temas vecinos
que no respondan a la intención. Usa solo los artículos dados. No inventes datos.
Devuelve SOLO JSON con esta forma: {{"relevant":[{{"id":0,"reason":"Motivo concreto basado en el artículo"}}]}}
Si ninguno es pertinente, usa una lista vacía.
Pregunta: {question}
Artículos: {json.dumps(payload, ensure_ascii=False)}'''
        raw, provider = ask_ai(prompt)
        if not raw:
            failed_indices.extend(range(offset, offset + len(batch)))
            continue
        data = extract_json(raw)
        if not isinstance(data, dict) or not isinstance(data.get("relevant"), list):
            failed_indices.extend(range(offset, offset + len(batch)))
            continue
        had_valid_response = True
        provider_used = provider if provider != "local" else provider_used
        for item in data["relevant"]:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("id"))
            except (ValueError, TypeError):
                continue
            if offset <= index < offset + len(batch):
                reason = clean_text(item.get("reason", ""))
                if reason:
                    accepted.append({"index": index, "relevance": reason[:320]})
    if had_valid_response:
        # Si un lote falló, se evalúa localmente solo ese lote para no perder candidatos.
        for index in failed_indices:
            local = local_evaluation(question, articles[index])
            if local:
                accepted.append({"index": index, "relevance": local["relevance"]})
        return accepted, provider_used
    return None, "local"


def investigate(request):
    question = clean_text(request.question)
    plan = interpret_question(question)
    candidates, seen_urls = [], set()
    # Consultas en paralelo; map conserva el orden original del plan.
    with ThreadPoolExecutor(max_workers=5) as pool:
        query_results = list(pool.map(lambda query: search_wol(query, limit=8), plan["queries"]))
    for results in query_results:
        for item in results:
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            candidates.append(item)
            if len(candidates) >= 24:
                break
        if len(candidates) >= 24:
            break
    print("CONSULTAS:", plan["queries"], "CANDIDATOS:", len(candidates))
    if not candidates:
        return {"topic": plan["topic"], "articles": [], "message": "No se encontraron artículos en WOL con esas consultas. Probá reformular el tema o usar otras palabras.", "provider": plan["provider"]}

    articles = []
    # Lectura paralela limitada para no demorar una consulta artículo por artículo.
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(read_article, candidate) for candidate in candidates]
        for future in as_completed(futures):
            article = future.result()
            if article:
                articles.append(article)
    # Mantiene orden de búsqueda original, no el orden de finalización de la red.
    order = {item["url"]: i for i, item in enumerate(candidates)}
    articles.sort(key=lambda item: order.get(item["url"], 999))
    print("ARTÍCULOS LEÍDOS:", len(articles), "DE", len(candidates))
    if not articles:
        return {"topic": plan["topic"], "articles": [], "message": "WOL devolvió resultados, pero no se pudo recuperar suficiente texto de sus artículos. Intentá nuevamente más tarde.", "provider": plan["provider"]}

    evaluated, eval_provider = ai_evaluation(question, articles)
    selected = {}
    if evaluated is not None:
        for item in evaluated:
            selected[item["index"]] = item["relevance"]
        # Si la IA devuelve cero, rescata solo coincidencias locales fuertes (título o varias señales).
        if not selected:
            for index, article in enumerate(articles):
                local = local_evaluation(question, article)
                if local and strong_local_match(question, article, local):
                    selected[index] = local["relevance"]
    else:
        for index, article in enumerate(articles):
            local = local_evaluation(question, article)
            if local:
                selected[index] = local["relevance"]
        eval_provider = "local"

    final_articles = []
    for index, article in enumerate(articles):
        if index not in selected:
            continue
        result = {key: article[key] for key in ("title", "url", "publication", "date") if article.get(key)}
        result["relevance"] = selected[index]
        final_articles.append(result)
    final_articles = final_articles[:request.max_results]
    if final_articles:
        message = f"Se encontraron {len(final_articles)} artículos que pueden ser pertinentes. Abrí cada uno para consultar el contenido original."
    else:
        message = "No encontramos artículos que superen la evaluación de pertinencia. Probá con una formulación más concreta o con un sinónimo del tema."
    provider = eval_provider if eval_provider != "local" else plan["provider"]
    return {"topic": plan["topic"], "articles": final_articles, "message": message, "provider": provider}


@app.get("/")
def home():
    index_file = STATIC / "index.html"
    if not index_file.exists():
        return {"service": "WOL Investigador", "version": APP_VERSION}
    return FileResponse(index_file)


@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION, "service": "wol-investigador"}


@app.post("/api/search")
def search(request: SearchRequest):
    try:
        return investigate(request)
    except Exception as exc:
        print("ERROR GENERAL DE BÚSQUEDA:", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Ocurrió un error durante la investigación. Intentá nuevamente más tarde.")

