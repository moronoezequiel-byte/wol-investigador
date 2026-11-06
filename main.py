
import os, re, json, unicodedata
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
APP_VERSION = "0.6.0"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
HTTP_TIMEOUT = httpx.Timeout(16.0, connect=8.0)
USER_AGENT = "Mozilla/5.0 (compatible; WOLInvestigador/0.6.0; +https://wol.jw.org/)"

STOPWORDS = set("""
a al algo alguien ambos ante asi aun aunque bajo cada casi como con contra cual cuales
cuando cuanto de del desde donde dos durante e el ella ellas ello ellos en entre era eran
eres es esa esas ese eso esos esta estaba estaban estado estar estas este esto estos estoy
fue fueron fui ha habia han hasta hay la las le les lo los mas me mi mis mucha muchas
mucho muchos muy nada ni no nos nuestra nuestras nuestro nuestros o os otra otras otro
otros para pero poca pocas poco pocos por porque que quien quienes se sea sean segun ser
sera si sido sin sobre sois somos son su sus tambien te tener tiene tienen toda todas todo
todos tras tu tus un una unas uno unos usted ustedes va vais vamos van varias varios y ya yo
""".split())

app = FastAPI(title="WOL Investigador", version=APP_VERSION)
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


class SearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1200)
    max_results: int = Field(default=8, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Escribí un tema de al menos dos caracteres.")
        return value


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize(value):
    value = unicodedata.normalize("NFD", (value or "").lower())
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return [t for t in re.findall(r"[a-z0-9]{2,}", value) if t not in STOPWORDS]


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
        if limit is not None and len(result) >= limit:
            break
    return result


SEED_PLANS = [
    ({"vejez", "anciano", "ancianos", "envejecer", "mayores", "envejecimiento", "edad avanzada", "tercera edad", "adulto mayor", "ancianidad"},
     "La vejez y las personas mayores",
     ["sentimientos en la vejez", "personas mayores y emociones", "afrontar la vejez", "ayudar a las personas mayores", "soledad en la vejez", "envejecer con alegría"]),
    ({"desanimo", "desalentado", "desaliento", "abatido", "abatimiento", "desanimado", "triste", "tristeza", "deprimido", "afligido"},
     "El desánimo y cómo afrontarlo",
     ["desánimo", "cómo vencer el desánimo", "qué hacer en caso de desánimo", "personas desanimadas consuelo ánimo", "recuperar el ánimo", "fortalecer a quien está triste"]),
    ({"atalaya", "conducir", "dirigir", "presidir"},
     "Conducir el Estudio de La Atalaya",
     ["conductor del Estudio de La Atalaya", "preparación para conducir el estudio", "preguntas y participación Estudio de La Atalaya", "dirigir estudio congregación"]),
    ({"economia", "economico", "economica", "dinero", "finanzas", "financiero", "financiera", "inflacion", "deuda", "deudas", "desempleo", "gastos", "salario", "sueldo", "pobreza"},
     "Preocupaciones y dificultades económicas",
     ["problemas económicos", "preocupaciones por problemas económicos", "ansiedad por la economía", "cómo hacer frente a los problemas económicos", "cómo afrontar las dificultades económicas", "preocupación por el dinero", "falta de dinero y preocupaciones"]),
]

SEMANTIC_GROUPS = {
    "desanimo": {"desanimo", "desanimado", "desanimada", "desalentado", "desaliento", "abatido", "abatimiento", "afligido", "afliccion", "triste", "tristeza", "perder el animo", "falta de animo", "hundido"},
    "vejez": {"vejez", "anciano", "ancianos", "persona mayor", "personas mayores", "edad avanzada", "envejecimiento", "envejecer"},
    "economia": {"economia", "economico", "economica", "problemas economicos", "dificultades economicas", "crisis economica", "dinero", "finanzas", "deuda", "deudas", "desempleo", "gastos", "ingresos", "salario", "sueldo", "inflacion", "precios", "pobreza", "llegar a fin de mes", "apuros economicos"},
    "preocupacion": {"preocupacion", "preocupaciones", "ansiedad", "angustia", "estres", "agobiado", "agobio", "inquietud", "inquietudes", "miedo", "temor"},
    "consuelo": {"consuelo", "consolar", "animar", "animo", "fortalecer", "dar fuerzas", "apoyo", "aliento"},
    "culpa": {"culpa", "sentirse culpable", "remordimiento", "arrepentimiento", "perdon", "perdonar"},
    "soledad": {"soledad", "aislamiento", "solo", "solitario", "falta de compania"},
    "espirituales": {"criatura espiritual", "criaturas espirituales", "ser espiritual", "seres espirituales", "angel", "angeles", "demonio", "demonios", "satanas"},
}


def generic_queries(question):
    raw = clean_text(question).rstrip("?!. ")
    tokens = normalize(raw)
    starters = {"puede", "pueden", "podria", "podrian", "es", "son", "que", "como", "por", "cuando", "donde", "existe", "existen"}
    core = [t for t in tokens if t not in starters]
    phrase = " ".join(core[:8]) or " ".join(tokens[:8])
    queries = [raw, phrase]
    if len(core) >= 2:
        queries.extend([" ".join(core[:5]) + " Biblia", " ".join(core[:5]) + " explicación"])

    key = norm_key(raw)
    for group, terms in SEMANTIC_GROUPS.items():
        if any(norm_key(term) and (norm_key(term) in key or key in norm_key(term)) for term in terms):
            if group == "espirituales":
                queries.extend([
                    "destrucción de criaturas espirituales", "pueden morir los ángeles",
                    "muerte de los ángeles", "destrucción de Satanás y los demonios",
                    "destino de los demonios",
                ])
            elif group == "preocupacion":
                queries.extend(["cómo vencer la ansiedad", "cómo controlar las preocupaciones", "afrontar la angustia"])
            elif group == "culpa":
                queries.extend(["sentimientos de culpa", "perdón de los pecados", "valor del sacrificio de Jesús"])
            break
    return unique_list(queries, limit=10)


def fallback_plan(question):
    words = set(normalize(question))
    for triggers, topic, queries in SEED_PLANS:
        trigger_tokens = {t for phrase in triggers for t in normalize(phrase)}
        if words.intersection(trigger_tokens):
            return {"topic": topic, "queries": unique_list(list(queries) + generic_queries(question), limit=12)}
    return {"topic": clean_text(question.rstrip("?!. "))[:120], "queries": generic_queries(question)}


def safe_error(exc):
    message = clean_text(str(exc))[:350]
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        secret = os.getenv(name)
        if secret:
            message = message.replace(secret, "[CLAVE OCULTA]")
    return message or type(exc).__name__


def call_gemini(prompt):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("GEMINI no disponible: falta GEMINI_API_KEY")
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=25000))
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=1400))
        return getattr(response, "text", None)
    except Exception as exc:
        print("GEMINI ERROR:", type(exc).__name__, safe_error(exc))
        return None


def call_openai(prompt):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=25, max_retries=0)
        return client.responses.create(model=OPENAI_MODEL, input=prompt, max_output_tokens=1400).output_text
    except Exception as exc:
        print("OPENAI ERROR:", type(exc).__name__, safe_error(exc))
        return None


def ask_ai(prompt):
    result = call_gemini(prompt)
    if result:
        return result, "Gemini"
    enabled = os.getenv("ENABLE_OPENAI_FALLBACK", "").strip().lower() in {"1", "true", "yes", "si"}
    if enabled:
        result = call_openai(prompt)
        if result:
            return result, "OpenAI"
    return None, "local"


def extract_json(raw):
    if not raw:
        return None
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw).strip(), flags=re.I)
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        pass
    start = value.find("{")
    if start < 0:
        return None
    depth, quoted, escaped = 0, False, False
    for pos in range(start, len(value)):
        char = value[pos]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(value[start:pos + 1])
                except ValueError:
                    return None
    return None


def interpret_question(question):
    backup = fallback_plan(question)
    prompt = f"""Actúa como especialista en recuperación de información para buscar artículos en la Biblioteca en Línea Watchtower en ESPAÑOL.
No respondas la pregunta ni des explicaciones religiosas. No inventes títulos ni enlaces.
Identifica el asunto central y la necesidad concreta. Genera consultas breves con sinónimos, expresiones equivalentes y situaciones concretas.
No cambies solo el orden de las palabras ni amplíes a temas vecinos. Todas las consultas en español.
Devuelve SOLO JSON: {{"topic":"tema breve", "queries":["consulta 1","consulta 2","consulta 3","consulta 4","consulta 5","consulta 6"]}}
Pregunta: {question}"""
    raw, provider = ask_ai(prompt)
    data = extract_json(raw)
    ai_queries, topic = [], backup["topic"]
    if isinstance(data, dict):
        values = data.get("queries")
        if isinstance(values, list):
            ai_queries = [clean_text(x)[:120] for x in values if isinstance(x, str) and clean_text(x)]
        if isinstance(data.get("topic"), str) and clean_text(data["topic"]):
            topic = clean_text(data["topic"])[:120]
    return {"topic": topic, "queries": unique_list(ai_queries[:7] + backup["queries"], limit=12), "provider": provider}


def valid_article_url(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.netloc == "wol.jw.org" and bool(re.search(r"/es/wol/d/", parsed.path))
    except Exception:
        return False


def search_wol(query, limit=8):
    try:
        with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            response = client.get(WOL_SEARCH, params={"q": query})
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        found, seen = [], set()
        for link in soup.select('a[href*="/wol/d/"]'):
            href = urljoin(WOL_ORIGIN, link.get("href", "")).split("#", 1)[0]
            title = clean_text(link.get_text(" ", strip=True))
            if len(title) < 5 or not valid_article_url(href) or href in seen:
                continue
            seen.add(href)
            found.append({"title": title, "url": href, "search_query": query})
            if len(found) >= limit:
                break
        print("WOL_QUERY", repr(query), "RESULTADOS", len(found))
        return found
    except Exception as exc:
        print("ERROR WOL búsqueda:", clean_text(query), type(exc).__name__, safe_error(exc))
        return []


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
        blocks = []
        for node in container.find_all(["p", "h2", "h3"]):
            value = clean_text(node.get_text(" ", strip=True))
            if len(value) >= 24 and value not in blocks:
                blocks.append(value)
        content = clean_text(" ".join(blocks))[:7000]
        if len(content) < 100:
            return None
        publication, date = "", ""
        for selector in ["meta[name='citation_journal_title']", "meta[name='citation_publication_title']"]:
            node = soup.select_one(selector)
            if node and node.get("content"):
                publication = clean_text(node["content"])
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
            match = re.search(r"\b(?:19|20)\d{2}\b", publication)
            if match:
                date = match.group(0)
        return {"title": title, "url": article["url"], "content": content, "publication": publication, "date": date}
    except Exception as exc:
        print("ERROR leyendo artículo:", article.get("url", ""), type(exc).__name__, safe_error(exc))
        return None


def expand_terms(question):
    terms = set(normalize(question))
    words = set(terms)
    phrase = norm_key(question)
    for group in SEMANTIC_GROUPS.values():
        normalized = {norm_key(term) for term in group}
        if any(term and (term in phrase or phrase in term) for term in normalized):
            for term in group:
                terms.update(normalize(term))
    for triggers, _, queries in SEED_PLANS:
        trigger_tokens = {t for phrase in triggers for t in normalize(phrase)}
        if words.intersection(trigger_tokens):
            for query in queries:
                terms.update(normalize(query))
    return terms


def local_evaluation(question, article):
    title = set(normalize(article.get("title", "")))
    body = set(normalize(article.get("content", "")))
    core = set(normalize(question))
    expanded = expand_terms(question)
    if not expanded or not (title or body):
        return None
    title_hits, body_hits = expanded & title, expanded & body
    core_title, core_body = core & title, core & body
    phrase, title_phrase = norm_key(question), norm_key(article.get("title", ""))
    exact = bool(phrase and phrase in title_phrase)
    score = len(title_hits) * 4 + min(len(body_hits), 12) + len(core_title) * 2 + (10 if exact else 0)
    if exact:
        reason = "El título coincide directamente con el tema de la consulta."
    elif len(core_title) >= 2 and len(core_body) >= 2:
        reason = "El título y el contenido desarrollan el tema mediante: " + ", ".join(sorted(core_title | core_body)[:4]) + "."
    elif len(title_hits) >= 2 and len(body_hits) >= 3:
        reason = "El título y el desarrollo tratan aspectos relacionados directamente: " + ", ".join(sorted(title_hits | body_hits)[:4]) + "."
    elif len(core) == 1 and core_title and len(body_hits) >= 4:
        reason = "El concepto central aparece en el título y se desarrolla en el artículo."
    else:
        return None
    return {"score": score, "relevance": reason}


def strong_local_match(question, article, evaluation):
    core = set(normalize(question))
    title = set(normalize(article.get("title", "")))
    body = set(normalize(article.get("content", "")))
    expanded = expand_terms(question)
    title_hits, body_hits = expanded & title, expanded & body
    phrase, title_phrase = norm_key(question), norm_key(article.get("title", ""))
    return bool((phrase and phrase in title_phrase) or
                (len(core) >= 2 and len(core & title) >= 2 and len(core & body) >= 2) or
                (len(title_hits) >= 2 and len(body_hits) >= 4))


def ai_evaluation(question, articles):
    if not articles:
        return None, "local"
    accepted, provider_used, valid_response, failed = [], "local", False, []
    for offset in range(0, len(articles), 10):
        batch = articles[offset:offset + 10]
        payload = [{"id": i + offset, "title": a["title"], "content": a["content"][:2600]} for i, a in enumerate(batch)]
        prompt = f"""Selecciona solo artículos que traten REALMENTE el tema principal.
No respondas la pregunta. Rechaza menciones incidentales o palabras compartidas sin desarrollo sustancial.
Usa solo los textos suministrados y explica la evidencia concreta.
Devuelve SOLO JSON: {{"relevant":[{{"id":0,"reason":"Evidencia concreta del artículo"}}]}}
Pregunta: {question}
Artículos: {json.dumps(payload, ensure_ascii=False)}"""
        raw, provider = ask_ai(prompt)
        if not raw:
            failed.extend(range(offset, offset + len(batch)))
            continue
        data = extract_json(raw)
        if not isinstance(data, dict) or not isinstance(data.get("relevant"), list):
            failed.extend(range(offset, offset + len(batch)))
            continue
        valid_response = True
        if provider != "local":
            provider_used = provider
        for item in data["relevant"]:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
            except (ValueError, TypeError):
                continue
            if offset <= idx < offset + len(batch):
                reason = clean_text(item.get("reason", ""))
                if reason:
                    accepted.append({"index": idx, "relevance": reason[:320]})
    if valid_response:
        for idx in failed:
            local = local_evaluation(question, articles[idx])
            if local and strong_local_match(question, articles[idx], local):
                accepted.append({"index": idx, "relevance": local["relevance"]})
        return accepted, provider_used
    return None, "local"


def investigate(request):
    question = clean_text(request.question)
    plan = interpret_question(question)
    with ThreadPoolExecutor(max_workers=5) as pool:
        results_by_query = list(pool.map(search_wol, plan["queries"]))
    candidates, seen, counts = [], set(), []
    for query, results in zip(plan["queries"], results_by_query):
        counts.append((query, len(results)))
        for item in results:
            if item["url"] not in seen:
                seen.add(item["url"])
                candidates.append(item)
    candidates = candidates[:24]
    print("DIAGNOSTICO PLAN", json.dumps(counts, ensure_ascii=False), "CANDIDATOS_UNICOS", len(candidates))
    if not candidates:
        return {"topic": plan["topic"], "articles": [], "message": "No se encontraron artículos en WOL con las consultas realizadas. Probá reformular el tema o usar otras palabras.", "provider": plan["provider"]}

    articles = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(read_article, item) for item in candidates]
        for future in as_completed(futures):
            try:
                article = future.result()
                if article:
                    articles.append(article)
            except Exception as exc:
                print("ERROR lectura concurrente:", type(exc).__name__, safe_error(exc))
    order = {item["url"]: i for i, item in enumerate(candidates)}
    articles.sort(key=lambda a: order.get(a["url"], 999))
    print("DIAGNOSTICO LECTURA", len(articles), "DE", len(candidates))
    if not articles:
        return {"topic": plan["topic"], "articles": [], "message": "WOL devolvió resultados, pero no se pudo recuperar suficiente texto de los artículos. Intentá nuevamente más tarde.", "provider": plan["provider"]}

    evaluated, eval_provider = ai_evaluation(question, articles)
    selected = {}
    if evaluated is not None:
        for item in evaluated:
            selected[item["index"]] = item["relevance"]
        if not selected:
            for idx, article in enumerate(articles):
                local = local_evaluation(question, article)
                if local and strong_local_match(question, article, local):
                    selected[idx] = local["relevance"]
    else:
        for idx, article in enumerate(articles):
            local = local_evaluation(question, article)
            if local:
                selected[idx] = local["relevance"]
        eval_provider = "local"

    final = []
    for idx, article in enumerate(articles):
        if idx not in selected:
            continue
        item = {k: article[k] for k in ("title", "url", "publication", "date") if article.get(k)}
        item["relevance"] = selected[idx]
        final.append(item)
    final = final[:request.max_results]
    print("DIAGNOSTICO EVALUACION", "aceptados", len(selected), "devueltos", len(final), "proveedor", eval_provider)
    message = (f"Se encontraron {len(final)} artículos que pueden ser pertinentes. Abrí cada uno para consultar el contenido original."
               if final else "No encontramos artículos que superen la evaluación de pertinencia. Probá con una formulación más concreta o con un sinónimo del tema.")
    provider = eval_provider if eval_provider != "local" else plan["provider"]
    return {"topic": plan["topic"], "articles": final, "message": message, "provider": provider}


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
        print("ERROR GENERAL DE BÚSQUEDA:", type(exc).__name__, safe_error(exc))
        raise HTTPException(status_code=500, detail="Ocurrió un error durante la investigación. Intentá nuevamente más tarde.")
