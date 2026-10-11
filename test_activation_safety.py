"""Regression tests: activation safety in services/adaptive_learning.py.

Proves:
- VA-0008 and VA-0009 (status=rejected) cannot be activated.
- A rejected activation leaves V4 active and the registry unchanged.
- A candidate with failed quality-gate metadata cannot be activated.
- A valid candidate still follows the approval workflow (approved_by
  required; activation succeeds in production mode for eligible status).
- TEST MODE still blocks all activation.

Uses a temporary registry copied from the real one plus synthetic temp
candidate directories; the REAL registry, REAL candidate artifacts,
production database and active model are never modified.

Run:  python test_activation_safety.py
"""

import json
import os
import shutil
import tempfile

os.environ.setdefault("APP_MODE", "test")

from pathlib import Path

from app.config import ADAPTIVE_MODELS_DIR
from app.services import adaptive_learning as al

PASS, results = "PASS", []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    suffix = f"  ({detail})" if detail else ""
    print(f"[{PASS if condition else 'FAIL'}] {name}{suffix}")


REAL_REGISTRY = ADAPTIVE_MODELS_DIR / "registry.json"
registry_before = REAL_REGISTRY.read_text(encoding="utf-8")

# Build an isolated temp registry: real entries + two synthetic candidates.
temp_dir = Path(tempfile.mkdtemp(prefix="activation-safety-"))
registry = json.loads(registry_before)

real_model_src = ADAPTIVE_MODELS_DIR / "VA-0012" / "model.joblib"


def add_candidate(version, metadata_content, metadata_is_json=True):
    vdir = temp_dir / version
    vdir.mkdir()
    shutil.copy(real_model_src, vdir / "model.joblib")
    if metadata_content is not None:
        text = (json.dumps(metadata_content) if metadata_is_json
                else metadata_content)
        (vdir / "metadata.json").write_text(text, encoding="utf-8")
    registry["versions"][version] = {"path": str(vdir), "status": "candidate"}


add_candidate("VA-T-FAIL", {
    "version": "VA-T-FAIL", "status": "candidate",
    "quality_gates": {"valid": False, "degenerate": True,
                      "reasons": ["constant probability output"]}})
add_candidate("VA-T-OK", {
    "version": "VA-T-OK", "status": "candidate",
    "quality_gates": {"valid": True, "degenerate": False, "reasons": []}})
add_candidate("VA-T-NOMETA", None)               # metadata.json absent
add_candidate("VA-T-NOGATES", {"version": "VA-T-NOGATES"})  # no gates key
add_candidate("VA-T-BROKEN", "{not json", metadata_is_json=False)  # malformed

temp_registry = temp_dir / "registry.json"
temp_registry.write_text(json.dumps(registry, indent=2), encoding="utf-8")

original_path = al.ADAPTIVE_REGISTRY_PATH
original_mode = al.APP_MODE
al.ADAPTIVE_REGISTRY_PATH = temp_registry
al.APP_MODE = "production"  # exercise approval logic; TEST MODE checked last
try:
    # 1. VA-0008 / VA-0009 (status=rejected) cannot be activated.
    for version in ("VA-0008", "VA-0009"):
        try:
            al.activate_model(version, approved_by="safety-test")
            blocked, detail = False, "activation succeeded"
        except ValueError as e:
            blocked, detail = True, str(e)[:70]
        check(f"{version} (rejected) cannot be activated", blocked, detail)

    # 1b. VA-0006 / VA-0007 (status=candidate, legacy metadata without
    # quality gates) fail closed and cannot be activated.
    for version in ("VA-0006", "VA-0007"):
        try:
            al.activate_model(version, approved_by="safety-test")
            blocked, detail = False, "activation succeeded"
        except ValueError as e:
            blocked, detail = True, str(e)[:70]
        check(f"{version} (candidate, no valid gates) cannot be activated",
              blocked, detail)

    # 1c. Fail-closed variants: missing metadata, missing gates, malformed.
    for version in ("VA-T-NOMETA", "VA-T-NOGATES", "VA-T-BROKEN"):
        try:
            al.activate_model(version, approved_by="safety-test")
            blocked, detail = False, "activation succeeded"
        except ValueError as e:
            blocked, detail = True, str(e)[:70]
        check(f"{version} (fail-closed) cannot be activated",
              blocked, detail)

    # 2. Rejected activations leave V4 active + registry untouched.
    active = json.loads(temp_registry.read_text())["active_version"]
    check("rejected activation leaves V4 active", active == "V4")
    check("real registry file untouched",
          REAL_REGISTRY.read_text(encoding="utf-8") == registry_before)

    # 3. Failed quality gates block activation even with approval.
    try:
        al.activate_model("VA-T-FAIL", approved_by="safety-test")
        blocked = False
    except ValueError as e:
        blocked, detail = True, str(e)
    check("candidate with failed quality gates cannot be activated",
          blocked, "constant probability output" in detail)

    # 4. Approval still required (workflow ambiguity note: approved_by is
    #    a caller-supplied string; there is no auth layer — by design).
    try:
        al.activate_model("VA-T-OK", approved_by="")
        needs_approval = False
    except ValueError:
        needs_approval = True
    check("explicit approved_by still required", needs_approval)

    # 5. Valid candidate follows the existing workflow and activates.
    result = al.activate_model("VA-T-OK", approved_by="safety-test")
    active = json.loads(temp_registry.read_text())
    check("valid candidate activates with explicit approval (temp registry)",
          result["status"] == "activated"
          and active["active_version"] == "VA-T-OK"
          and active["versions"]["VA-T-OK"]["status"] == "active")

    # 6. TEST MODE still blocks activation before any approval logic.
    al.APP_MODE = "test"
    try:
        al.activate_model("VA-T-OK", approved_by="safety-test")
        test_blocked = False
    except RuntimeError:
        test_blocked = True
    check("TEST MODE blocks activation", test_blocked)
finally:
    al.ADAPTIVE_REGISTRY_PATH = original_path
    al.APP_MODE = original_mode
    shutil.rmtree(temp_dir, ignore_errors=True)

check("real registry still has V4 active after all checks",
      json.loads(REAL_REGISTRY.read_text())["active_version"] == "V4"
      and "VA-T-OK" not in json.loads(REAL_REGISTRY.read_text())["versions"])

failed = [name for name, ok in results if not ok]
print("=" * 70)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", *failed, sep="\n  - ")
    raise SystemExit(1)
print("ALL CHECKS PASSED")
