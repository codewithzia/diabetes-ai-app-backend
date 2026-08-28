"""Loading of predefined test patient cases."""

import json

from ..config import TEST_DATA_PATH


def load_test_patients() -> dict:
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_DATA_PATH}"
        )

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


TEST_PATIENTS = load_test_patients()
