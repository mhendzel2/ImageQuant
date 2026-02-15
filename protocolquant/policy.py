from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from protocolquant.models import QCStatus


class ExportRules(BaseModel):
    allowed_statuses: list[Literal["PASS", "WARN", "FAIL"]] = Field(default_factory=lambda: ["PASS"])
    allow_fail_override: bool = False
    require_override_reason: bool = True


class PolicyRole(BaseModel):
    allowed_protocols: list[str] = Field(default_factory=list)
    export_rules: ExportRules = Field(default_factory=ExportRules)
    qc_threshold_overrides: dict[str, Any] = Field(default_factory=dict)
    runner_editable_fields: list[str] = Field(default_factory=list)


class PolicySpec(BaseModel):
    roles: dict[Literal["student", "instructor", "research"], PolicyRole]


class PolicyError(ValueError):
    pass


def load_policy(path: str | Path) -> PolicySpec:
    path = Path(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise PolicyError(f"Failed to parse policy file '{path}': {exc}") from exc

    try:
        return PolicySpec.model_validate(payload)
    except ValidationError as exc:
        lines = [f"Policy validation failed for '{path}':"]
        for err in exc.errors():
            location = ".".join(str(part) for part in err.get("loc", []))
            lines.append(f"- {location}: {err.get('msg')}")
        raise PolicyError("\n".join(lines)) from exc


def resolve_role_policy(policy: PolicySpec, role: str) -> PolicyRole:
    if role not in policy.roles:
        raise PolicyError(f"Unknown role '{role}'. Available roles: {', '.join(policy.roles.keys())}")
    return policy.roles[role]  # type: ignore[index]


def can_export_results(
    *,
    role_policy: PolicyRole,
    qc_status: QCStatus,
    override_fail: bool = False,
    override_reason: str | None = None,
) -> tuple[bool, str | None]:
    if qc_status in role_policy.export_rules.allowed_statuses:
        return True, None

    if qc_status == "FAIL" and role_policy.export_rules.allow_fail_override and override_fail:
        if role_policy.export_rules.require_override_reason and not override_reason:
            return False, "FAIL override requires a reason under policy."
        return True, None

    return False, f"Export blocked by policy for QC status {qc_status}."


def enforce_protocol_allowed(role_policy: PolicyRole, protocol_id: str) -> None:
    if role_policy.allowed_protocols and protocol_id not in role_policy.allowed_protocols:
        raise PolicyError(f"Protocol '{protocol_id}' is not allowed for this role.")
