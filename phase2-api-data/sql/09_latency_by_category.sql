-- Question: What are the average and maximum latencies of successful calls by category?
-- chat_logs_clean excludes NULL and negative latency values.
-- Success requires a non-blank response and a blank error field.
-- Failed requests are excluded because timeout/error latency may distort
-- successful-call performance metrics. Investigate extreme successful-call
-- latencies separately before using these metrics for SLOs.

SELECT
    category,
    COUNT(*) AS successful_requests,
    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
    MAX(latency_ms) AS max_latency_ms
FROM chat_logs_clean
WHERE (error IS NULL OR TRIM(error) = '')
  AND response IS NOT NULL
  AND TRIM(response) <> ''
GROUP BY category
ORDER BY avg_latency_ms DESC;
