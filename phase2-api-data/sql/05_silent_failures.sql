-- Question: Which rows have an empty response and no recorded error?
-- These are silent failures: neither a response nor an error was recorded.
-- Data-quality risk: 7 of the 10 silent failures have judge scores of 4 or 5.
-- These scores are unreliable because no response exists to evaluate.
-- Investigate the scoring pipeline and logging; do not treat these scores
-- as evidence of successful responses.

SELECT
    id,
    model,
    category,
    response,
    error,
    judge_score
FROM chat_logs
WHERE (response IS NULL OR TRIM(response) = '')
  AND (error IS NULL OR TRIM(error) = '')
ORDER BY id;
