-- Question: How many NULL and empty values exist in each column?

SELECT 'id' AS column_name,
       SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS null_count,
       SUM(CASE WHEN TRIM(CAST(id AS TEXT)) = '' THEN 1 ELSE 0 END) AS empty_count
FROM chat_logs

UNION ALL

SELECT 'timestamp',
       SUM(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN TRIM(timestamp) = '' THEN 1 ELSE 0 END)
FROM chat_logs

UNION ALL

SELECT 'model',
       SUM(CASE WHEN model IS NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN TRIM(model) = '' THEN 1 ELSE 0 END)
FROM chat_logs

UNION ALL

SELECT 'category',
       SUM(CASE WHEN category IS NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN TRIM(category) = '' THEN 1 ELSE 0 END)
FROM chat_logs

UNION ALL

SELECT 'prompt',
       SUM(CASE WHEN prompt IS NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN TRIM(prompt) = '' THEN 1 ELSE 0 END)
FROM chat_logs

UNION ALL

SELECT 'response',
       SUM(CASE WHEN response IS NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN TRIM(response) = '' THEN 1 ELSE 0 END)
FROM chat_logs

UNION ALL

SELECT 'latency_ms',
       SUM(CASE WHEN latency_ms IS NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN TRIM(CAST(latency_ms AS TEXT)) = '' THEN 1 ELSE 0 END)
FROM chat_logs

UNION ALL

SELECT 'input_tokens',
       SUM(CASE WHEN input_tokens IS NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN TRIM(CAST(input_tokens AS TEXT)) = '' THEN 1 ELSE 0 END)
FROM chat_logs

UNION ALL

SELECT 'output_tokens',
       SUM(CASE WHEN output_tokens IS NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN TRIM(CAST(output_tokens AS TEXT)) = '' THEN 1 ELSE 0 END)
FROM chat_logs

UNION ALL

SELECT 'error',
       SUM(CASE WHEN error IS NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN TRIM(error) = '' THEN 1 ELSE 0 END)
FROM chat_logs

UNION ALL

SELECT 'judge_score',
       SUM(CASE WHEN judge_score IS NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN TRIM(CAST(judge_score AS TEXT)) = '' THEN 1 ELSE 0 END)
FROM chat_logs;
