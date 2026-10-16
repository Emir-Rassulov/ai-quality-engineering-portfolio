-- Question: What is the error rate for each prompt category?
-- A request is an error when the error field is non-NULL and non-empty after trimming.

SELECT
    category,
    COUNT(*) AS total_requests,
    SUM(
        CASE
            WHEN error IS NOT NULL AND TRIM(error) <> '' THEN 1
            ELSE 0
        END
    ) AS error_count,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN error IS NOT NULL AND TRIM(error) <> '' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS error_rate_pct
FROM chat_logs_clean
GROUP BY category
ORDER BY category;
