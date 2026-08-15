# BENCH-10: Multi-tenant isolation

Pre-registered bars in `benchmarks/PROTOCOL.md` (committed before
implementation). Eval: `benchmarks/tenant_eval.py`. Two tenants built
from the same generator with different seeds (1,000 records each), so
entity names collide across tenants while values differ — the hardest
case for isolation.

## Results

| Check | Bar | Result | Verdict |
|---|---|---|---|
| T1 cross-tenant leaks (200 queries total) | = 0 | 0 | pass |
| T2 value accuracy, tenant acme | = 1.000 | 1.000 (100/100) | pass |
| T2 value accuracy, tenant globex | = 1.000 | 1.000 (100/100) | pass |
| T3 audit events separate and complete | true | 100/100 each | pass |
| T3 both logs append-only | true | true | pass |

## How isolation works

There is nothing to filter. Each tenant is a directory holding its own
index file, its own append-only audit log and its own policy files
(`souprise/core/tenants.py`). A query opens exactly one tenant's files;
no shared index exists, so a cross-tenant leak would require reading a
file the process never opens. Tenant names are validated as strict
slugs, which closes path-traversal routes, and the same rule guards
per-tenant policy lookups.

## One product fix that fell out

T2 started at 0.870/0.810 and exposed a routing bug unrelated to
tenancy: "How many units of Product_DQ are in stock?" was routed to the
count aggregation, which counted RECORDS matching the entity (1)
instead of returning the Stock value (1,287) — a wrong number shipped
as computed. A count question naming both a specific entity and a
numeric field is a field lookup; `parse_aggregate` now returns None for
that shape so the verified path answers it (issue #55). BENCH-9 and
BENCH-5 re-runs pass after the change.

## Known limits

- Tenant selection is an operator decision (`--tenant` on the CLI);
  per-user authentication arrives with the REST API (issue #29).
- Cross-tenant querying is intentionally impossible, not missing.
- Isolation is at the file level within one machine and one OS user;
  OS-level separation (separate accounts or containers per tenant) is
  the operator's call for hostile-tenant scenarios.
