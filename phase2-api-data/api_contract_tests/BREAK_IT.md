# Break-it log: can these tests fail?

A test that cannot fail proves nothing. For each break below I changed
the code on purpose, ran the tests, and restored the file.

## 1. Retries disabled

- **Broke:** `prompt_runner/runner.py`, `max_attempts = 3` changed to `max_attempts = 1`.
- **Red:** `test_retries_after_503_and_then_succeeds` (`assert 1 == 2`),
  `test_retries_after_rate_limit_and_then_succeeds` (`assert 1 == 2`),
  `test_gives_up_after_three_attempts_and_records_the_error` (`assert 1 == 3`).
- **Result:** 3 failed, 26 passed. After restoring the file: 29 passed.
- **Note:** `test_retry_logic.py` stayed green on purpose. It tests error
  classification, not the number of attempts.

## 2. Contract model no longer strict

- **Broke:** `api_contract_tests/models.py`, `strict=True` changed to `strict=False`.
- **Red:** `test_malformed_payload_is_rejected_for_the_right_reason[string-instead-of-bool]`,
  `test_strict_mode_rejects_a_value_that_lax_mode_would_quietly_convert`
  (both: `DID NOT RAISE ValidationError`).
- **Result:** 2 failed, 4 passed. After restoring the file: 6 passed.
- **Meaning:** with lax mode the string `"true"` is silently accepted as `True`,
  so a service that changes a field's type would go unnoticed.

---

## How to reproduce

Run from `phase2-api-data/` with the venv active.

### Break 1: retries disabled

```bash
cp ../exercise2_prompt_runner/prompt_runner/runner.py /tmp/runner.py.bak
sed -i '' 's/max_attempts = 3/max_attempts = 1/' ../exercise2_prompt_runner/prompt_runner/runner.py
python -m pytest -q api_contract_tests -m "not network"     # expect 3 failed
cp /tmp/runner.py.bak ../exercise2_prompt_runner/prompt_runner/runner.py
python -m pytest -q api_contract_tests -m "not network"     # expect 29 passed
```

### Break 2: contract model no longer strict

```bash
cp api_contract_tests/models.py /tmp/models.py.bak
sed -i '' 's/strict=True/strict=False/' api_contract_tests/models.py
python -m pytest -q api_contract_tests/test_pydantic_contract.py   # expect 2 failed
cp /tmp/models.py.bak api_contract_tests/models.py
python -m pytest -q api_contract_tests/test_pydantic_contract.py   # expect 6 passed
```

Note: `sed -i ''` is the macOS form. On Linux use `sed -i` without the quotes.
