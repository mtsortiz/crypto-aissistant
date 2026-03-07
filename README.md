# Crypto Agent

Asistente financiero cripto construido en Python con arquitectura modular, orientado a RAG + agentes con LangGraph + API productizable con FastAPI y Docker.

## Objetivo del proyecto

Construir un asistente que pueda:
- Consultar conocimiento técnico desde documentos (whitepapers en PDF) con RAG.
- Tomar decisiones de flujo con agentes en LangGraph.
- Integrar herramientas de mercado (precio BTC/ETH).
- Exponer un backend robusto para consumo externo.
- Ejecutarse de forma reproducible en contenedores.

## Arquitectura objetivo (roadmap)

- Día 1
- Módulo RAG (LangChain + Chroma + Embeddings + Reranker).
- Módulo de agentes con LangGraph.
- Validación y tipado estricto con Pydantic.

- Día 2
- Backend FastAPI (`/chat`).
- Dockerfile multietapa + `docker-compose` con persistencia.
- README final y demo end-to-end.

## Estado actual

### Completado

Se implementaron el **Punto 1, Punto 2 y Punto 3 del Dia 1**:

- `rag_pipeline.py`:
- Ingesta de PDFs desde `docs/`.
- Split con `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=100`).
- Embeddings con `GoogleGenerativeAIEmbeddings`.
- Vector store persistente con `Chroma` en `chroma_db/`.
- Reranking con `FlashRank` para top 3 resultados.

- `financial_graph.py`:
- Grafo de estado LangGraph para asesor financiero.
- Estado con `messages` y `needs_market_data`.
- Flujo implementado: `Start -> Agent -> Tools -> Agent -> End`.
- LLM configurable por entorno con `GOOGLE_MODEL`.
- Modelo por defecto: `gemini-2.5-flash`.
- Entrada y salida tipadas con Pydantic (`QueryInput`, `AgentResponse`).
- Respuesta estandarizada en JSON: `answer`, `sources`, `risk_level`.

- `market_tools.py`:
- Tool `get_crypto_prices_usd` con `yfinance`.
- Devuelve precios actuales en USD para `BTC`, `ETH` y `SOL`.

- `main.py`:
- API FastAPI con endpoint `POST /chat`.
- Validacion de request/response con Pydantic usando los modelos del agente.

### Pendiente

- Dockerizacion y demo end-to-end.

## Estructura del proyecto actual

```text
crypto-agent/
  docs/                 # PDFs fuente para RAG
  rag_pipeline.py       # Ingesta + embeddings + Chroma + reranking
  financial_graph.py    # Grafo LangGraph (asesor financiero)
  market_tools.py       # Tool de precios BTC/ETH/SOL
  main.py               # API FastAPI (/chat)
  requirements.txt      # Dependencias del proyecto
```

## Requisitos

- Python 3.11+
- Variable de entorno `GOOGLE_API_KEY`
- (Opcional) `GOOGLE_MODEL` para elegir el modelo Gemini

## Instalación

```bash
cd crypto-agent
py -m pip install -r requirements.txt
```

## Uso rápido del módulo RAG

1. Colocar PDFs en `docs/`.
2. Construir/actualizar índice vectorial persistente:

```bash
py rag_pipeline.py --docs-dir ./docs --persist-dir ./chroma_db
```

3. Ejecutar una consulta de prueba con reranking top 3:

```bash
py rag_pipeline.py --docs-dir ./docs --persist-dir ./chroma_db --query "What is Proof of Work in Bitcoin?"
```

## Uso rápido del módulo LangGraph (Día 1 - Punto 2)

```bash
py financial_graph.py --question "Should I buy BTC today and what are BTC ETH SOL prices now?"
```

El agente decide si necesita datos de mercado en tiempo real, llama la tool y vuelve al nodo `Agent` para entregar la respuesta final.

## Uso rápido de la API FastAPI

1. Crear `.env` con tu API key:

```env
GOOGLE_API_KEY=tu_api_key
GOOGLE_MODEL=gemini-2.5-flash
```

2. Levantar API:

```bash
cd crypto-agent
py -m uvicorn main:app --env-file .env --host 127.0.0.1 --port 8010
```

3. Probar endpoint:

```bash
curl -X POST "http://127.0.0.1:8010/chat" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"precio de BTC y ETH\"}"
```

Respuesta esperada:

```json
{
  "answer": "...",
  "sources": [],
  "risk_level": "low|medium|high"
}
```

## Notas técnicas

- Si hay PDFs en `docs/`, el script vuelve a construir el índice y lo persiste.
- Si no hay PDFs, intenta cargar un índice existente desde `persist_dir`.
- El reranker se aplica sobre candidatos recuperados de Chroma para mejorar precisión de contexto.

## Proximo paso recomendado

Avanzar con el **Dia 2**: dockerizacion, compose y demo final end-to-end.
