-- Question: What are the five slowest successful calls?
-- Success means a non-empty response and no non-empty error message.
-- chat_logs_clean excludes rows with missing or negative latency.
-- These calls are classified as successful based on their response and error
-- fields, but IDs 213 (240625 ms), 412 (170620 ms), and 275 (90097.5 ms)
-- have unusually high latency. Logged errors elsewhere include a 30-second
-- TimeoutError, so these outliers warrant investigation for possible timing,
-- logging, or timeout-configuration inconsistencies. Do not assume the cause
-- from these records alone.

SELECT
    id,
    timestamp,
    model,
    category,
    latency_ms
FROM chat_logs_clean
WHERE response IS NOT NULL
  AND TRIM(response) <> ''
  AND (error IS NULL OR TRIM(error) = '')
ORDER BY latency_ms DESC
LIMIT 5;
