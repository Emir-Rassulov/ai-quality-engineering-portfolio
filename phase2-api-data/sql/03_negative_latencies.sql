-- Question: Are there any negative latency values?

SELECT
    id,
    latency_ms
FROM chat_logs
WHERE latency_ms < 0
ORDER BY latency_ms;
