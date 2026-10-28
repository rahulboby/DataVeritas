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

The dashboard runs locally; it does not require a separate API server. To use the optional AI assistant, set the `GROQ_API_KEY` environment variable before launching the app.

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
