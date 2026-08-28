# ============================================================
# BACKEND API TEST
# ============================================================

import requests
import json


BASE_URL = "http://127.0.0.1:5000"


def print_result(title, response):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("Status:", response.status_code)

    try:

        print(
            json.dumps(
                response.json(),
                indent=4
            )
        )

    except Exception:

        print(response.text)


# ============================================================
# 1. HEALTH
# ============================================================

response = requests.get(
    f"{BASE_URL}/api/health"
)

print_result(
    "1. HEALTH CHECK",
    response
)


# ============================================================
# 2. TEST CASES
# ============================================================

response = requests.get(
    f"{BASE_URL}/api/test/cases"
)

print_result(
    "2. TEST CASES",
    response
)


# ============================================================
# 3. TEST PREDICTIONS
# ============================================================

test_cases = [
    "TEST-001",
    "TEST-002",
    "TEST-003",
    "TEST-004",
    "TEST-005"
]

for test_case_id in test_cases:

    response = requests.post(
        f"{BASE_URL}/api/test/predict",
        json={
            "test_case_id": test_case_id
        }
    )

    print_result(
        f"3. PREDICTION: {test_case_id}",
        response
    )


# ============================================================
# 4. UNKNOWN TEST CASE
# ============================================================

response = requests.post(
    f"{BASE_URL}/api/test/predict",
    json={
        "test_case_id": "INVALID"
    }
)

print_result(
    "4. INVALID TEST CASE",
    response
)


# ============================================================
# 5. TEST ADAPTIVE UPDATE LOCK
# ============================================================

response = requests.post(
    f"{BASE_URL}/api/adaptive/update",
    json={
        "feedback_count": 10
    }
)

print_result(
    "5. ADAPTIVE UPDATE SAFETY TEST",
    response
)