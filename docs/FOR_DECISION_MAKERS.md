# Souprise for Decision Makers

One page, no jargon, honest about limits.

## The problem you already have

Your company runs on invoices, orders, customer records and KPIs. Getting a simple answer out of them, which invoices are overdue, what do we know about this customer, where is the budget drifting, means manual digging or a report request that takes days. Cloud AI could answer these questions, but it means handing your books to a third party, and for many businesses that is a compliance conversation nobody wants.

## What Souprise changes

Souprise puts an AI assistant on your own hardware that answers questions about your business records in plain language, with the source records attached to every answer. Nothing is uploaded. It works on an ordinary laptop, and it stays useful at scale, one million records answer in about a third of a second in our recorded tests.

## The cost argument

We won't invent an ROI percentage for you. The structural argument is simple enough to check yourself.

- **No usage fees.** Cloud AI bills per user or per query, forever. Souprise is Apache-2.0 open source, the software costs nothing, and it runs on hardware you already own.
- **No data processing agreement for your core records.** The data never leaves your infrastructure, so an entire class of GDPR, works-council and customer-confidentiality conversations does not need to happen.
- **No vendor lock-in.** Open source, open formats (SQLite, CSV), and any locally runnable language model. If you stop using it, you lose nothing.
- **Cheap to try.** A pilot costs one afternoon and one CSV export. No procurement, no account, no contract.

## What it can't do yet, stated plainly

- It is alpha software, version 0.2. Solid test coverage and honest recordings, but not a hardened product with support contracts.
- Local models are smaller than the biggest cloud models. For focused questions over your records they work well, and every answer shows its sources so wrong answers are catchable. For open-ended reasoning, cloud models are still ahead.
- SAP and DATEV connect today via CSV export. Native integration is on the roadmap, not in the product.
- There is no user management or access control yet. Today it is a single-user tool per machine.

## What a pilot looks like

1. Export a slice of your data as CSV (invoices are a good start).
2. One command builds the index, one command starts asking questions. The [examples folder](../examples/) walks through it in ninety seconds.
3. Nightly exports keep the index current, no retraining involved.
4. Judge it on your own data and your own hardware. The README publishes no benchmark tables on purpose; the tool measures itself on every query.

Questions: michael@kupermann.com
