-- Question: Which IDs appear more than once, and how many times?

SELECT
    id,
    COUNT(*) AS occurrences
FROM chat_logs
GROUP BY id
HAVING COUNT(*) > 1
ORDER BY occurrences DESC, id;
