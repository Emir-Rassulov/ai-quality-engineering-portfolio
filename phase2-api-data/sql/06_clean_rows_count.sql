-- Question: How many rows remain after applying the shared cleaning rules?
-- The cleaning logic is defined once in chat_logs_clean.

SELECT COUNT(*) AS clean_rows
FROM chat_logs_clean;
