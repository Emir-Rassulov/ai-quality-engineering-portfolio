-- Question: Are there inconsistent casing or surrounding spaces in text columns?

SELECT
    'model' AS column_name,
    model AS original_value,
    COUNT(*) AS occurrences
FROM chat_logs
WHERE model <> TRIM(model)
   OR model <> LOWER(model)
GROUP BY model

UNION ALL

SELECT
    'category',
    category,
    COUNT(*)
FROM chat_logs
WHERE category <> TRIM(category)
   OR category <> LOWER(category)
GROUP BY category

ORDER BY column_name, original_value;
