-- Question: How many requests succeeded, failed silently, or logged an error?
-- A blank error field does not prove success: a response must also be present.
-- Silent failures are reported separately to avoid overstating successful calls.

SELECT
    CASE
        WHEN error IS NOT NULL AND TRIM(error) <> ''
            THEN TRIM(error)
        WHEN response IS NULL OR TRIM(response) = ''
            THEN 'Silent failure (no response, no logged error)'
        ELSE 'Successful call'
    END AS result_type,
    COUNT(*) AS request_count
FROM chat_logs_clean
GROUP BY
    CASE
        WHEN error IS NOT NULL AND TRIM(error) <> ''
            THEN TRIM(error)
        WHEN response IS NULL OR TRIM(response) = ''
            THEN 'Silent failure (no response, no logged error)'
        ELSE 'Successful call'
    END
ORDER BY request_count DESC, result_type;
