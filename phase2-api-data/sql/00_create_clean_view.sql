-- Question: What does "clean" mean?
-- Define the cleaning rules once so all analytical queries use the same dataset.

DROP VIEW IF EXISTS chat_logs_clean;

CREATE VIEW chat_logs_clean AS
SELECT DISTINCT
    id,
    timestamp,
    LOWER(TRIM(model)) AS model,
    LOWER(TRIM(category)) AS category,
    prompt,
    response,
    latency_ms,
    input_tokens,
    output_tokens,
    error,
    judge_score
FROM chat_logs
WHERE latency_ms IS NOT NULL
  AND latency_ms >= 0;
