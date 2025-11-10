Data Model Review Notes

- Naming and indexes
  - Ensure `systems.system_id`, `runs.run_id`, and `datasets.dataset_id` are indexed and unique (present in models).
  - Consider composite index on `dataset_items(dataset_id, item_index)` for upsert lookups.

- Foreign keys and cascades
  - Add FK from `runs.standard_code` → `standards.code`, `metric_results.metric_code` → `metrics.code`, `dataset_items.dataset_id` → `datasets.dataset_id` (present in models).
  - In Postgres, consider `ON DELETE CASCADE` for artifacts and metric_results when a run is deleted.

- JSON storage
  - SQLite stores JSON as TEXT; ensure consistent `json.dumps`/`json.loads` at boundaries.
  - In Postgres, migrate `metrics.applicability`, `metrics.config`, `runs.mapping_snapshot`, `runs.selected_metric_codes`, `datasets.profile`, `dataset_items.payload`, and `dataset_items.tags` to JSONB and add GIN indexes if needed for frequent querying.

- Security
  - Keep API keys in environment variables. Avoid persisting secrets in DB or artifacts.
  - Review dataset licensing before redistribution. Avoid storing PII in artifacts; redact if necessary.

- Observability
  - Include `run_id` in logs for correlation; capture request/response sizes and timing.
  - Consider a `jobs` table for async execution and status transitions.

- Pagination
  - For listing endpoints like `/runs`, add `limit`/`offset` params with sensible defaults.

- Migrations
  - Introduce Alembic for schema evolution. Provide migrations for new columns and tables. During dev with SQLite, additive columns can be applied via lightweight checks, but use real migrations moving forward.

