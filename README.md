Link: https://dataveritas.streamlit.app/

## Project Layout

```

dataveritas/
  dashboard.py
  apps/
    dashboard_update.py
    testing.py
  sections/
    cardinality.py
    consistency.py
    duplicates.py
    nulls.py
    outliers.py
    score.py
    value_distribution.py
  core/
    data/
    cardinality/
    consistency/
    downloads/
    duplicates/
    nulls/
    outliers/
    score/
    value_distribution/
  scripts/
    remove_duplicate_fuzzy.py
```

## Run

- Main dashboard: 
Run these in CMD or your terminal
  cd 'Path\to\dq_agent_streamlit_app'
  streamlit run dashboard.py
