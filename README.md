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

Se implementaron el **Punto 1 y Punto 2 del Día 1**:

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
- LLM: `Gemini 1.5 Pro` (`model="gemini-1.5-pro"`).

- `market_tools.py`:
- Tool `get_crypto_prices_usd` con `yfinance`.
- Devuelve precios actuales en USD para `BTC`, `ETH` y `SOL`.

### Pendiente

- Pydantic end-to-end en entradas/salidas.
- Respuesta estandarizada del agente en JSON (`answer`, `sources`, `risk_level`).
- API FastAPI, dockerización y demo.

## Estructura del proyecto actual

```text
crypto-agent/
  docs/                 # PDFs fuente para RAG
  rag_pipeline.py       # Ingesta + embeddings + Chroma + reranking
  financial_graph.py    # Grafo LangGraph (asesor financiero)
  market_tools.py       # Tool de precios BTC/ETH/SOL
  requirements.txt      # Dependencias del proyecto
```

## Requisitos

- Python 3.11+
- Variable de entorno `GOOGLE_API_KEY`

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

## Notas técnicas

- Si hay PDFs en `docs/`, el script vuelve a construir el índice y lo persiste.
- Si no hay PDFs, intenta cargar un índice existente desde `persist_dir`.
- El reranker se aplica sobre candidatos recuperados de Chroma para mejorar precisión de contexto.

## Próximo paso recomendado

Implementar el **Punto 3 del Día 1**: tipado y validación con Pydantic en entradas/salidas y formato de respuesta JSON estructurada.
