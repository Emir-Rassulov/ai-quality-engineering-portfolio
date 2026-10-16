-- Question: Which categories have the highest average judge score?
-- Silent failures (empty response and empty error) are excluded because
-- their scores may be misleading. Only non-empty responses with a score count.

SELECT
    category,
    COUNT(judge_score) AS scored_requests,
    ROUND(AVG(judge_score), 2) AS avg_judge_score
FROM chat_logs_clean
WHERE response IS NOT NULL
  AND TRIM(response) <> ''
  AND judge_score IS NOT NULL
  AND TRIM(CAST(judge_score AS TEXT)) <> ''
  AND NOT (error IS NOT NULL AND TRIM(error) <> '')
GROUP BY category
ORDER BY avg_judge_score DESC, scored_requests DESC;
