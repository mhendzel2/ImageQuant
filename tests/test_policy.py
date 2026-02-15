import pytest

from protocolquant.policy import PolicyError, can_export_results, load_policy, resolve_role_policy


def test_policy_student_blocks_warn_and_fail() -> None:
    policy = load_policy("configs/lab_policy.yaml")
    role = resolve_role_policy(policy, "student")

    ok_pass, _ = can_export_results(role_policy=role, qc_status="PASS")
    ok_warn, _ = can_export_results(role_policy=role, qc_status="WARN")
    ok_fail, _ = can_export_results(role_policy=role, qc_status="FAIL")

    assert ok_pass is True
    assert ok_warn is False
    assert ok_fail is False



def test_unknown_role_fails() -> None:
    policy = load_policy("configs/lab_policy.yaml")
    with pytest.raises(PolicyError):
        resolve_role_policy(policy, "unknown")
