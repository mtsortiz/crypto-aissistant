# Crypto Agent

Asistente financiero cripto orientado a produccion, construido con Python y una arquitectura modular basada en:

- LangGraph para orquestacion de decisiones y uso de herramientas.
- RAG sobre whitepapers PDF (LangChain + Chroma + FlashRank).
- FastAPI para exponer una API estable (`POST /chat`).
- Docker y Docker Compose para ejecucion reproducible.

## Tabla de contenido

- [Vision general](#vision-general)
- [Arquitectura](#arquitectura)
- [Flujo de ejecucion](#flujo-de-ejecucion)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Configuracion](#configuracion)
- [Ejecucion local](#ejecucion-local)
- [Ejecucion con Docker](#ejecucion-con-docker)
- [Contrato de la API](#contrato-de-la-api)
- [Modulo RAG](#modulo-rag)
- [Observabilidad y manejo de errores](#observabilidad-y-manejo-de-errores)
- [Troubleshooting](#troubleshooting)
- [Roadmap sugerido](#roadmap-sugerido)

## Vision general

El proyecto responde preguntas sobre mercado cripto y fundamentos tecnicos usando dos capacidades complementarias:

- Datos de mercado en tiempo real para `BTC`, `ETH` y `SOL` via `yfinance`.
- Recuperacion de conocimiento desde whitepapers PDF en `docs/` mediante RAG.

La respuesta final siempre se normaliza con un esquema consistente:

```json
{
  "answer": "string",
  "sources": ["string"],
  "risk_level": "low|medium|high"
}
```

## Arquitectura

### 1) Capa API

- Archivo: `main.py`
- Framework: FastAPI
- Endpoint principal: `POST /chat`
- Validacion tipada con Pydantic (`QueryInput`, `AgentResponse`)
- Ejecucion del core en threadpool para no bloquear el loop async

### 2) Capa de orquestacion (Agente)

- Archivo: `financial_graph.py`
- Motor: LangGraph
- Estado del grafo:
- `messages`: historial de conversacion
- `needs_market_data`: bandera para decidir si pasa por herramientas

Grafo actual:

`Start -> Agent -> (Tools | End) -> Agent -> End`

Tools conectadas al agente:

- `get_crypto_prices_usd`
- `search_whitepapers`

### 3) Capa de herramientas

- Archivo: `market_tools.py`

Herramientas:

- `get_crypto_prices_usd`: consulta cotizaciones en USD de `BTC`, `ETH`, `SOL`.
- `search_whitepapers`: consulta conocimiento tecnico con RAG y fallback de busqueda por PDF.

### 4) Capa RAG

- Archivo: `rag_pipeline.py`
- Indexacion de PDFs con:
- `PyPDFLoader`
- `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=100`)
- `GoogleGenerativeAIEmbeddings`
- `Chroma` persistente (`chroma_db/`)
- Re-ranking con `FlashRank` (`top_n=3`)

## Flujo de ejecucion

1. Cliente envia una pregunta a `POST /chat`.
2. FastAPI valida payload y llama `run_query`.
3. El nodo `agent` decide si necesita tools.
4. Si hay tool calls:
- Ejecuta `ToolNode`.
- Retorna al `agent` para redactar respuesta final.
5. Se normalizan `sources` y se calcula `risk_level` por heuristica local.
6. API responde JSON estable.

Fallbacks relevantes:

- Si hay `429`/`RESOURCE_EXHAUSTED` del modelo, el sistema intenta responder con tools directas.
- Si la stack vectorial falla, `search_whitepapers` usa busqueda basica sobre texto de PDFs.

## Estructura del repositorio

```text
crypto-agent/
  docs/                    # Whitepapers PDF para RAG
  chroma_db/               # Persistencia local de Chroma
  main.py                  # FastAPI app
  financial_graph.py       # Grafo LangGraph + modelos Pydantic
  market_tools.py          # Herramientas de mercado y busqueda documental
  rag_pipeline.py          # Ingesta, embeddings, indexacion y reranking
  smoke_tests.py           # Pruebas basicas de humo
  requirements.txt         # Dependencias Python
  Dockerfile               # Imagen multietapa
  docker-compose.yml       # Orquestacion local
```

## Requisitos

- Python 3.11+
- Docker Desktop (opcional, para ejecucion contenedorizada)
- Clave valida para Gemini:
- `GOOGLE_API_KEY`

## Configuracion

Crear un archivo `.env` en la raiz de `crypto-agent/`:

```env
GOOGLE_API_KEY=tu_api_key
GOOGLE_MODEL=gemini-2.5-flash

# Opcional: modelo de embeddings
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001

# Opcional: controlar reconstruccion del indice
RAG_FORCE_REBUILD=false
RAG_INDEX_BATCH_SIZE=80
RAG_INDEX_BATCH_PAUSE_SECONDS=65
```

## Ejecucion local

1. Instalar dependencias:

```bash
cd crypto-agent
py -m pip install -r requirements.txt
```

2. Levantar API:

```bash
py -m uvicorn main:app --env-file .env --host 127.0.0.1 --port 8000
```

3. Probar endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"precio de BTC y ETH"}'
```

## Ejecucion con Docker

La imagen usa `Dockerfile` multietapa y `docker-compose.yml` con volumen persistente para Chroma.

1. Build + up:

```bash
cd crypto-agent
docker compose up --build -d
```

2. Ver logs:

```bash
docker compose logs -f
```

3. Probar API:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"explica proof of work"}'
```

4. Detener servicio:

```bash
docker compose down
```

5. Limpiar volumen de datos (opcional):

```bash
docker compose down -v
```

## Contrato de la API

### Endpoint

- Metodo: `POST`
- Ruta: `/chat`
- Content-Type: `application/json`

### Request

```json
{
  "question": "Should I buy BTC today?"
}
```

Regla de validacion:

- `question` es obligatorio y no puede ser vacio.

### Response 200

```json
{
  "answer": "...",
  "sources": ["docs/bitcoin.pdf"],
  "risk_level": "medium"
}
```

### Errores

- `400`: error de validacion o runtime controlado.
- `429`: cuota del modelo agotada.
- `500`: error interno no controlado.

## Modulo RAG

### Construir o actualizar indice

```bash
py rag_pipeline.py --docs-dir ./docs --persist-dir ./chroma_db
```

### Probar consulta RAG por CLI

```bash
py rag_pipeline.py --docs-dir ./docs --persist-dir ./chroma_db --query "What is Proof of Work in Bitcoin?"
```

### Notas operativas

- Si existe `chroma_db/chroma.sqlite3` y `RAG_FORCE_REBUILD=false`, se reutiliza el indice.
- La indexacion se hace en lotes para reducir riesgo de exceder cuota de embeddings.
- `search_whitepapers` intenta usar RAG y, si falla, aplica fallback por lectura directa de PDF.

## Observabilidad y manejo de errores

- Logging estructurado en `main.py` con `request_id` por llamada.
- Manejo explicito de errores de cuota (`429`) y errores internos (`500`).
- Respuesta de degradacion controlada cuando el LLM no esta disponible por cuota.

## Troubleshooting

### Error: `GOOGLE_API_KEY is required`

- Verificar archivo `.env` y variable exportada en entorno.

### Error: no responde Docker Desktop

- Confirmar que Docker Desktop esta iniciado.
- Validar con `docker version` que el daemon este disponible.

### No aparecen fuentes en `sources`

- Verificar que existan PDFs en `docs/`.
- Reconstruir indice con `rag_pipeline.py`.

### RAG lento al inicio

- Primer build puede tardar por embeddings e indexacion.
- Siguientes corridas reutilizan `chroma_db/`.

## Roadmap sugerido

- Agregar tests unitarios por modulo (tools, parser de sources, heuristica de riesgo).
- Agregar evaluacion automatizada de respuestas (quality harness).
- Exponer metricas (latencia, tasa de errores, fallback ratio).
- Implementar autenticacion y rate limiting para entorno productivo.
