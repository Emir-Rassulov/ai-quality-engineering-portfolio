"""Test helpers shared by the phase 3 tests."""
import pytest

import ml_data

needs_data = pytest.mark.skipif(
    not ml_data.DATA_PATH.exists(),
    reason="chat_logs_cleaned.csv is generated: run clean_data.py in phase2-api-data first",
)
