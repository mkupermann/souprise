# Example Data

Two small synthetic datasets for trying Souprise without touching any real record. Generated with the shipped generators (seed 2026), no real customer information anywhere.

- `sample_invoices.csv` — 39 invoices as a plain table, the shape a typical ERP export has
- `sample_erp.jsonl` — 100 mixed records (invoices, orders, customers, products, KPIs, budgets) in the native entry format

## Ninety seconds to a working index

```bash
pip install -e .

# Build a persistent index from the CSV
souprise index build --from-csv examples/sample_invoices.csv \
    --id-column invoice_id --tag-columns status --output demo_index.db

# Search it, no model needed
souprise index query "overdue invoices in the EU" --path demo_index.db

# Ask with a local model (downloads a small model once; needs the mlx or torch extra)
souprise chat query "Which invoices are overdue?" --index demo_index.db
```

The JSONL works the same way with `--from-jsonl examples/sample_erp.jsonl`.
