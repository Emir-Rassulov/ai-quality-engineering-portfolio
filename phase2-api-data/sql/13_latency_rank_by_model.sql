-- Question: Which calls have the highest latency rank within each model?
-- RANK() assigns the same rank to tied latency values.
-- Only successful calls are ranked: non-empty response and no non-empty error.
-- Negative and missing latency values are already excluded by chat_logs_clean.

WITH ranked_calls AS (
    SELECT
        id,
        timestamp,
        model,
        category,
        latency_ms,
        RANK() OVER (
            PARTITION BY model
            ORDER BY latency_ms DESC
        ) AS latency_rank
    FROM chat_logs_clean
    WHERE response IS NOT NULL
      AND TRIM(response) <> ''
      AND (error IS NULL OR TRIM(error) = '')
)
SELECT
    id,
    model,
    category,
    latency_ms,
    latency_rank
FROM ranked_calls
WHERE latency_rank <= 5
ORDER BY model, latency_rank, id;
