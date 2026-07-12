from ..contracts.base import BaseRecord
from ..contracts.validation import (
    ContractValidator,
    ValidationContext,
    ValidationReport,
    ValidationFinding,
)


class BaseValidator:
    """Minimal validator implementing the ContractValidator Protocol."""

    def validate(
        self,
        record: BaseRecord,
        context: ValidationContext,
    ) -> ValidationReport:
        findings: list[ValidationFinding] = []
        if not record.record_id:
            findings.append(
                ValidationFinding(
                    severity="error",
                    code="MISSING_RECORD_ID",
                    message="record_id is empty",
                    field="record_id",
                )
            )
        decision = "pass" if not findings else "fail"
        return ValidationReport(
            schema_version="1.0",
            record_type="validation_report",
            target_record_type=record.record_type,
            target_record_id=record.record_id,
            findings=findings,
            decision=decision,  # type: ignore[arg-type]
        )
