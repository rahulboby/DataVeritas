# DataVeritas

[Live demo](https://dataveritas.streamlit.app/)

DataVeritas is a Streamlit dashboard for profiling tabular datasets and auditing data quality.

## Install and run

DataVeritas can be installed from PyPI once a release has been published:

```bash
python -m pip install dataveritas
dataveritas
```

The `dataveritas` command starts the dashboard and opens it in your browser. The explicit form is also supported:

```bash
dataveritas serve
```

Streamlit server options can be passed through:

```bash
dataveritas serve --server.port 8502
```

The dashboard runs locally; it does not require a separate API server. The AI assistant tries Groq first using `GROQ_API_KEY` and falls back to local Ollama if Groq is unavailable. API keys are not included in the pip package; each user provides their own key in their environment or Streamlit secrets. For the fallback, install and start Ollama, then download the default model:

```bash
ollama pull llama3.2
```

Set `OLLAMA_MODEL` to use a different locally installed model, or `OLLAMA_HOST` if Ollama is running at a non-default address. If neither provider is available, the dashboard explains what needs to be configured.

## Install from a checkout

```bash
python -m pip install .
dataveritas
```

For development, install the declared dependencies and run Streamlit directly:

```bash
python -m pip install -r requirements.txt
streamlit run dashboard.py
```

## Package layout

- `dashboard.py` — Streamlit user interface
- `sections/` — dashboard views and interactions
- `core/` — data quality analysis and supporting logic
- `dataveritas/cli.py` — command-line launcher installed as `dataveritas`
