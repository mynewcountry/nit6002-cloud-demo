"""NIT6002 RQ2 exploratory response-policy V4: post-hoc rules, never executed.

Strictly synthetic operational context. No original target labels, sample IDs,
network interfaces, external security systems or executable response logic.
Every decision can contain *multiple* nonexclusive follow-up obligations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

KNOWN_ACTIONS = {
    "read_only_log_collection": "low",
    "temporary_endpoint_isolation": "medium",
    "block_production_subnet": "high",
    "disable_privileged_identity": "high",
}
IMPACT_ORDER = {"low": 0, "medium": 1, "high": 2}
# Illustrative protocol parameter, NOT a validated organisational safety threshold.
MAX_CONTAINMENT_MINUTES = 60


def bool_field(raw: Any, name: str) -> bool:
    if type(raw) is not bool:
        raise ValueError(f"{name} must be an explicit Boolean")
    return raw


@dataclass(frozen=True)
class Event:
    predicted_attack: bool
    all_zero: bool
    critical_asset: bool
    privileged_access: bool
    secondary_signal: bool
    nonfinite_features: bool = False
    independently_verified_ioc: bool = False
    # The three fields below refer to independently documented security evidence.
    signal_verified: bool | None = None
    signal_fresh: bool | None = None
    signal_provenance_independent: bool | None = None
    requested_action: str | None = None
    action_impact: str | None = None
    rollback_plan: bool | None = None
    rollback_validated: bool | None = None
    time_limit_minutes: int | None = None
    playbook_authorised: bool | None = None
    privacy_authorised: bool | None = None
    access_audit_enabled: bool | None = None

    @classmethod
    def from_scenario(cls, obj: Mapping[str, Any]) -> "Event":
        """Fail closed on unknown/misnested context and untrusted labels.

        Additional evidence must be explicitly nested. Never silently ignore
        requested_action or quality flags supplied at the wrong level.
        """
        if type(obj) is not dict:
            raise ValueError("Event must be an object")
        names = ("predicted_attack", "all_zero", "critical_asset", "privileged_access", "secondary_signal")
        allowed = {k for k in cls.__dataclass_fields__ if k not in names}
        blocked = {"supplied_label", "source_id", "label1", "target", "known_injected_flip"}
        if blocked.intersection(obj):
            raise ValueError("Evaluation labels and identifiers are not policy inputs")
        unknown_top = set(obj) - set(names) - {"additional_evidence"}
        if unknown_top:
            raise ValueError("Unknown top-level fields (put evidence under additional_evidence): " + ", ".join(sorted(unknown_top)))
        extra = obj.get("additional_evidence", {})
        if type(extra) is not dict:
            raise ValueError("additional_evidence must be an object")
        if blocked.intersection(extra):
            raise ValueError("Evaluation labels and identifiers are not policy inputs")
        unknown = set(extra) - allowed
        if unknown:
            raise ValueError("Unknown evidence keys: " + ", ".join(sorted(unknown)))
        base = {key:bool_field(obj[key],key) for key in names}
        result=cls(**base, **extra)
        result.validate()
        return result

    def validate(self) -> None:
        for key in ("predicted_attack", "all_zero", "critical_asset", "privileged_access", "secondary_signal", "nonfinite_features", "independently_verified_ioc"):
            bool_field(getattr(self,key), key)
        for key in ("signal_verified", "signal_fresh", "signal_provenance_independent", "rollback_plan", "rollback_validated", "playbook_authorised", "privacy_authorised", "access_audit_enabled"):
            value = getattr(self,key)
            if value is not None: bool_field(value,key)
        if self.requested_action is not None and (not isinstance(self.requested_action,str) or not self.requested_action):
            raise ValueError("requested_action must be a nonempty string or null")
        if self.requested_action is None and self.action_impact is not None:
            raise ValueError("action_impact requires requested_action")
        if self.action_impact is not None and self.action_impact not in IMPACT_ORDER:
            raise ValueError("action_impact must be low, medium or high")
        if self.requested_action in KNOWN_ACTIONS and self.action_impact is not None:
            if IMPACT_ORDER[self.action_impact] < IMPACT_ORDER[KNOWN_ACTIONS[self.requested_action]]:
                raise ValueError("Cannot downgrade known high-impact action")
        if self.time_limit_minutes is not None:
            if type(self.time_limit_minutes) is not int or self.time_limit_minutes <= 0:
                raise ValueError("time_limit_minutes must be positive integer")


@dataclass(frozen=True)
class Recommendation:
    request_source_evidence: bool
    SOC_review: bool
    human_approval_required: bool
    auto_collect_evidence: bool
    no_case: bool
    reason_codes: tuple[str,...]
    execution_authorised: bool = False  # permanent safety barrier

    def as_dict(self) -> dict[str, object]:
        return {
            "request_source_evidence":self.request_source_evidence,
            "SOC_review":self.SOC_review,
            "human_approval_required":self.human_approval_required,
            "auto_collect_evidence":self.auto_collect_evidence,
            "no_case":self.no_case,
            "reason_codes":list(self.reason_codes),
            "execution_authorised":self.execution_authorised,
        }


def decide(e: Event) -> Recommendation:
    """Return suggested workflow flags, never execute a response."""
    e.validate()
    request_source=False; soc=False; approval=False; auto=False; reasons=[]
    quality_bad=e.all_zero or e.nonfinite_features
    signal_observed=e.secondary_signal or e.independently_verified_ioc
    verified_independent_fresh=(signal_observed and e.signal_verified is True and
                               e.signal_fresh is True and e.signal_provenance_independent is True)

    if quality_bad:
        request_source=True
        reasons.append("UNRELIABLE_MODEL_FEATURES")
    if signal_observed and not verified_independent_fresh:
        soc=True
        reasons.append("SIGNAL_PROVENANCE_FRESHNESS_OR_VERIFICATION_UNCONFIRMED")
    if quality_bad and signal_observed:
        soc=True
        reasons.append("INDEPENDENT_SIGNAL_REQUIRES_SEPARATE_TRIAGE")

    # Treat model and contextual evidence independently. A model-only negative
    # cannot suppress an observed external signal, even if its provenance is weak.
    if not e.predicted_attack and signal_observed:
        soc=True
        reasons.append("BENIGN_MODEL_DISAGREES_WITH_SECURITY_SIGNAL")

    if not quality_bad and e.predicted_attack:
        if e.requested_action is None:
            soc=True
            reasons.append("NO_SPECIFIC_APPROVED_ACTION")
        else:
            impact=e.action_impact or KNOWN_ACTIONS.get(e.requested_action)
            if impact is None:
                soc=True
                reasons.append("UNKNOWN_ACTION_IMPACT")
            elif impact == "high":
                soc=True; approval=True
                reasons.append("HIGH_IMPACT_REQUIRES_HUMAN_APPROVAL")
            elif impact == "medium":
                soc=True
                ready=(e.rollback_plan is True and e.rollback_validated is True and
                       e.time_limit_minutes is not None and e.time_limit_minutes <= MAX_CONTAINMENT_MINUTES)
                if not ready:
                    approval=True
                    reasons.append("MEDIUM_RISK_ROLLBACK_NOT_VERIFIED")
                else:
                    reasons.append("MEDIUM_RISK_SOC_REVIEW_REQUIRED")
            else: # low
                ready=(e.requested_action == "read_only_log_collection" and
                       verified_independent_fresh and e.playbook_authorised is True and
                       e.privacy_authorised is True and e.access_audit_enabled is True)
                if ready:
                    auto=True
                    reasons.append("READ_ONLY_COLLECTION_RECOMMENDED")
                else:
                    soc=True
                    reasons.append("READ_ONLY_AUTHORISATION_OR_EVIDENCE_UNCONFIRMED")
    elif quality_bad and e.requested_action:
        # Quality failure blocks all action recommendations; a request for a
        # potentially disruptive action still requires investigation.
        soc=True
        reasons.append("ACTION_BLOCKED_BY_UNRELIABLE_MODEL_EVIDENCE")

    if e.critical_asset or e.privileged_access:
        if e.predicted_attack or signal_observed:
            soc=True
            reasons.append("CRITICAL_ASSET_OR_PRIVILEGED_ACCESS")

    # EXPLORATORY POST-HOC V4 REQUIREMENTS, ADDED ONLY AFTER INSPECTING
    # THE FROZEN V3 FAILURES N25/N27/N28. These do NOT constitute an
    # independently tested safety enhancement or action authorisation.
    if e.requested_action is not None:
        impact = e.action_impact or KNOWN_ACTIONS.get(e.requested_action)
        if impact == "high" and (e.predicted_attack or signal_observed):
            soc = True; approval = True
            reasons.append("PROPOSED_HIGH_IMPACT_ACTION_NEEDS_SEPARATE_APPROVAL")
        elif impact == "medium" and e.predicted_attack and e.playbook_authorised is not True:
            soc = True; approval = True
            reasons.append("MEDIUM_ACTION_PLAYBOOK_AUTHORITY_NOT_CONFIRMED")

    if auto and (request_source or approval):
        raise AssertionError("Automatic recommendation must never bypass quality/approval gates")
    no_case=not any((request_source,soc,approval,auto))
    if no_case: reasons.append("NO_QUALIFYING_EVIDENCE")
    return Recommendation(request_source,soc,approval,auto,no_case,tuple(dict.fromkeys(reasons)))
