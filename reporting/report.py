"""Build explainable, deterministic package-compliance reports.

Reports produced here are prototype decision-support artifacts, not official
government compliance certificates.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rules.engine import evaluate_compliance


REPORT_VERSION = "1.0"
ALLOWED_STATUSES = {"PASS", "FAIL", "REVIEW", "NOT_APPLICABLE"}
DISCLAIMER = (
    "Prototype engineering and rule-based decision-support report. This is not "
    "an official government compliance certificate or legal opinion."
)
EVIDENCE_KEYS = {
    "confidence", "source_text", "source_box", "source_image", "tokens",
    "source_layout", "label_text", "label_box", "source_lines", "method",
    "issues",
}
PRIVATE_PATH_KEYS = {"debug_image_path", "debug_overlay_path", "debug_image_saved"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if str(key) not in PRIVATE_PATH_KEYS
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _normalized_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        key: _json_safe(item)
        for key, item in value.items()
        if key not in EVIDENCE_KEYS and not key.startswith("source_")
        and key not in {"glyph_measurement", "measurement"}
    }


def _field_entry(field_name: str, value: Any) -> dict[str, Any]:
    normalized = _normalized_value(value)
    present = value is not None and (
        any(item is not None and str(item).strip() for item in normalized.values())
        if isinstance(normalized, dict)
        else not isinstance(normalized, str) or bool(normalized.strip())
    )
    if isinstance(value, dict):
        raw_text = value.get("source_text") or (
            "\n".join(value.get("source_lines") or []) or None
        )
        source_polygon = value.get("source_box")
        ocr_confidence = value.get("confidence")
        extraction_confidence = value.get("extraction_confidence", value.get("confidence"))
        method = value.get("extraction_method", value.get("method"))
        issues = list(value.get("issues") or [])
    else:
        raw_text = None
        source_polygon = None
        ocr_confidence = None
        extraction_confidence = None
        method = None
        issues = []
    return {
        "field_name": field_name,
        "present": present,
        "normalized_value": normalized,
        "raw_text": raw_text,
        "ocr_confidence": ocr_confidence,
        "extraction_confidence": extraction_confidence,
        "source_polygon": _json_safe(source_polygon),
        "extraction_method": method,
        "issues": issues,
    }


def canonicalize_extracted_fields(fields: dict[str, Any] | None) -> dict[str, Any]:
    source = fields or {}
    names = (
        "product", "net_quantity", "mrp", "manufacture_date", "manufacturer",
        "marketer", "packer", "importer", "consumer_care", "country_of_origin",
    )
    result = {name: _field_entry(name, source.get(name)) for name in names}
    ocr_items = source.get("ocr_evidence") or []
    for name in names:
        entry = result[name]
        if not entry["present"] or entry.get("source_polygon") is not None:
            continue
        value = source.get(name)
        if isinstance(value, dict):
            candidates = [str(item) for item in value.get("source_lines") or [] if item]
        elif value is not None:
            candidates = [str(value)]
        else:
            candidates = []
        if name == "country_of_origin" and candidates:
            candidates = [f"made in {candidates[0]}", f"country of origin {candidates[0]}"]
        normalized_candidates = {
            re.sub(r"[^a-z0-9]", "", candidate.lower()) for candidate in candidates
        }
        matches = []
        for item in ocr_items:
            text = str(item.get("normalized_text") or item.get("raw_text") or "")
            normalized_text = re.sub(r"[^a-z0-9]", "", text.lower())
            if any(
                candidate and (
                    candidate in normalized_text
                    or (
                        len(normalized_text) >= 4
                        and normalized_text in candidate
                        and len(normalized_text) / len(candidate) >= 0.5
                    )
                )
                for candidate in normalized_candidates
            ):
                matches.append(item)
        if matches:
            matched_text = list(dict.fromkeys(
                str(item.get("raw_text") or item.get("normalized_text") or "")
                for item in matches
            ))
            entry["raw_text"] = entry.get("raw_text") or "\n".join(matched_text)
            boxes = [item.get("box") for item in matches if item.get("box") is not None]
            entry["source_polygon"] = boxes[0] if len(boxes) == 1 else boxes or None
            confidences = [
                float(item["confidence"]) for item in matches
                if isinstance(item.get("confidence"), (int, float))
            ]
            if confidences and entry.get("ocr_confidence") is None:
                entry["ocr_confidence"] = round(min(confidences), 3)
                entry["extraction_confidence"] = round(min(confidences), 3)
    mrp = source.get("mrp") if isinstance(source.get("mrp"), dict) else {}
    care = source.get("consumer_care") if isinstance(source.get("consumer_care"), dict) else {}
    result["inclusive_of_all_taxes"] = _field_entry(
        "inclusive_of_all_taxes",
        {
            "value": mrp.get("inclusive_of_all_taxes"),
            "source_text": mrp.get("source_text"),
            "source_box": mrp.get("source_box"),
            "confidence": mrp.get("confidence"),
        } if mrp.get("inclusive_of_all_taxes") is not None else None,
    )
    for key in ("phone", "email"):
        result[f"consumer_care_{key}"] = _field_entry(
            f"consumer_care_{key}",
            {
                "value": care.get(key),
                "source_text": care.get("source_text"),
                "source_box": care.get("source_box"),
                "confidence": care.get("confidence"),
            } if care.get(key) is not None else None,
        )
    return result


def _field_for_rule(field_name: str) -> str | None:
    return {
        "manufacturer_name": "manufacturer",
        "manufacturer_address": "manufacturer",
        "commodity_name": "product",
        "net_quantity": "net_quantity",
        "month_year": "manufacture_date",
        "mrp": "mrp",
        "consumer_care": "consumer_care",
        "country_of_origin": "country_of_origin",
        "net_quantity_unit_scale": "net_quantity",
        "applicability_small_package": "net_quantity",
        "net_quantity_font_height": "net_quantity",
    }.get(field_name)


def _reason_codes(
    rule: dict[str, Any], batch_result: dict[str, Any], evidence: list[dict[str, Any]],
) -> list[str]:
    status = rule["status"]
    field_name = rule["field_name"]
    reason = str(rule.get("reason") or "").lower()
    if field_name == "mrp_netqty_contrast":
        code = {
            "PASS": "CONTRAST_CLEAR", "FAIL": "CONTRAST_LOW",
            "REVIEW": "CONTRAST_UNCERTAIN", "NOT_APPLICABLE": "NOT_APPLICABLE",
        }[status]
        codes = [code]
    elif field_name == "net_quantity_font_height":
        codes = [
            "NOT_APPLICABLE_QUANTITY_TYPE"
            if status == "NOT_APPLICABLE" else "MEASUREMENT_NOT_VALIDATED"
        ]
        if not (batch_result.get("aruco") or {}).get("detected"):
            codes.append("CALIBRATION_UNAVAILABLE")
    elif status == "NOT_APPLICABLE":
        codes = ["NOT_APPLICABLE_QUANTITY_TYPE"]
    elif status == "PASS":
        codes = ["FIELD_PRESENT"]
    elif status == "FAIL":
        codes = ["RULE_REQUIREMENT_NOT_MET"]
    elif "low ocr" in reason or "low confidence" in reason:
        codes = ["OCR_LOW_CONFIDENCE"]
    elif not evidence or all(item.get("value") is None for item in evidence):
        codes = ["FIELD_MISSING"]
    else:
        codes = ["EXTRACTION_UNCERTAIN"]
    quality_issues = set((batch_result.get("image_quality") or {}).get("issues") or [])
    if "HIGH_GLARE" in quality_issues:
        codes.append("HIGH_GLARE")
    if quality_issues:
        codes.append("IMAGE_QUALITY_LIMITATION")
    return list(dict.fromkeys(codes))


def _contrast_evidence(batch_result: dict[str, Any]) -> list[dict[str, Any]]:
    targets = (batch_result.get("contrast_evidence") or {}).get("targets") or {}
    evidence = []
    for target in ("NET_QUANTITY", "MRP"):
        item = targets.get(target)
        if not isinstance(item, dict):
            continue
        evidence.append({
            "evidence_type": "CONTRAST_MEASUREMENT",
            "target": target,
            "ocr_text": item.get("ocr_text"),
            "target_box": item.get("target_box"),
            "contrast_ratio": item.get("contrast_ratio"),
            "lab_difference": item.get("lab_color_difference"),
            "foreground_luminance": item.get("foreground_luminance"),
            "background_luminance": item.get("background_luminance"),
            "confidence": item.get("confidence"),
            "issues": item.get("issues") or [],
            "threshold_basis": "implementation-defined engineering thresholds; not statutory",
        })
    return evidence


def _measurement_evidence(batch_result: dict[str, Any]) -> dict[str, Any]:
    glyph = batch_result.get("glyph_measurement") or {}
    aruco = batch_result.get("aruco") or {}
    return {
        "evidence_type": "NUMERAL_HEIGHT_MEASUREMENT",
        "measurement_status": glyph.get("status", "REVIEW"),
        "estimated_numeral_height_mm": glyph.get("estimated_numeral_height_mm"),
        "measurement_confidence": glyph.get("measurement_confidence", glyph.get("confidence")),
        "confidence": glyph.get("measurement_confidence", glyph.get("confidence")),
        "calibration_detected": bool(aruco.get("detected")),
        "pixels_per_mm": aruco.get("pixels_per_mm"),
        "validation_status": batch_result.get("validation_status"),
        "unresolved_reason": (
            "Physical numeral-height measurement has not been independently validated"
        ),
    }


def _rule_evidence(
    rule: dict[str, Any], extracted: dict[str, Any], batch_result: dict[str, Any],
) -> list[dict[str, Any]]:
    field_name = rule["field_name"]
    if field_name == "mrp_netqty_contrast":
        return _contrast_evidence(batch_result)
    mapped_field = _field_for_rule(field_name)
    evidence = []
    if mapped_field and mapped_field in extracted:
        entry = extracted[mapped_field]
        evidence.append({
            "evidence_type": "EXTRACTED_FIELD",
            "field": mapped_field,
            "value": entry.get("normalized_value"),
            "raw_text": entry.get("raw_text"),
            "source_polygon": entry.get("source_polygon"),
            "confidence": entry.get("extraction_confidence"),
            "issues": entry.get("issues") or [],
        })
    if field_name == "net_quantity_font_height":
        evidence.append(_measurement_evidence(batch_result))
    return evidence


def _evidence_confidence(evidence: list[dict[str, Any]]) -> float | None:
    values = [
        float(item["confidence"])
        for item in evidence
        if isinstance(item.get("confidence"), (int, float))
    ]
    return round(min(values), 3) if values else None


def build_rule_results(
    fields: dict[str, Any], extracted: dict[str, Any], batch_result: dict[str, Any],
) -> list[dict[str, Any]]:
    compliance = evaluate_compliance(fields)
    results = []
    for rule in compliance["results"]:
        status = rule["status"]
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Unexpected rule status: {status}")
        evidence = _rule_evidence(rule, extracted, batch_result)
        evidence_issues = [issue for item in evidence for issue in item.get("issues", [])]
        results.append({
            "rule_id": rule["rule_id"],
            "description": rule["requirement"],
            "field_name": rule["field_name"],
            "legal_source": rule["legal_source"],
            "status": status,
            "applicable": status != "NOT_APPLICABLE",
            "reason_codes": _reason_codes(rule, batch_result, evidence),
            "reason": rule["reason"],
            "evidence": evidence,
            "confidence": _evidence_confidence(evidence),
            "issues": list(dict.fromkeys(evidence_issues)),
        })
    return results


def aggregate_package_status(
    rule_results: list[dict[str, Any]], *, critical_processing_uncertainty: bool = False,
) -> dict[str, Any]:
    counts = {
        "pass_count": sum(rule.get("status") == "PASS" for rule in rule_results),
        "fail_count": sum(rule.get("status") == "FAIL" for rule in rule_results),
        "review_count": sum(rule.get("status") == "REVIEW" for rule in rule_results),
        "not_applicable_count": sum(
            rule.get("status") == "NOT_APPLICABLE" for rule in rule_results
        ),
    }
    applicable = [rule for rule in rule_results if rule.get("status") != "NOT_APPLICABLE"]
    if counts["fail_count"]:
        overall = "FAIL"
        reason = "At least one applicable rule has a definitive FAIL result"
    elif counts["review_count"] or critical_processing_uncertainty:
        overall = "REVIEW"
        reason = (
            "At least one applicable rule or critical processing condition requires review"
        )
    elif applicable and all(rule.get("status") == "PASS" for rule in applicable):
        overall = "PASS"
        reason = "All applicable evaluated rules passed without critical uncertainty"
    elif rule_results and not applicable:
        overall = "NOT_APPLICABLE"
        reason = "No rule in this report is applicable"
    else:
        overall = "REVIEW"
        reason = "No conclusive applicable rule evaluation is available"
    return {"overall_status": overall, **counts, "reason": reason}


def _processing_warnings(batch_result: dict[str, Any]) -> list[dict[str, Any]]:
    quality = batch_result.get("image_quality") or {}
    warnings = []
    for issue in quality.get("issues") or []:
        warnings.append({
            "code": "IMAGE_QUALITY_LIMITATION",
            "source_code": issue,
            "severity": "REVIEW",
            "message": f"Image-quality analysis reported {issue}",
        })
    for warning in quality.get("warnings") or []:
        warnings.append({
            "code": "IMAGE_QUALITY_WARNING",
            "source_code": warning,
            "severity": "WARNING",
            "message": f"Image-quality analysis reported {warning}",
        })
    if batch_result.get("failure_stage"):
        warnings.append({
            "code": "PROCESSING_STAGE_UNCERTAINTY",
            "source_code": batch_result.get("failure_stage"),
            "severity": "REVIEW",
            "message": batch_result.get("reason") or "A processing stage was incomplete",
        })
    return warnings


def _canonical_ocr_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = _json_safe(items)
    for item in evidence:
        source_image = item.get("source_image")
        if source_image:
            item["source_image"] = Path(str(source_image)).name
    return evidence


def build_package_report(
    batch_result: dict[str, Any], *, processing_timestamp: str | None = None,
) -> dict[str, Any]:
    fields = dict(batch_result.get("extracted_fields") or {})
    if not fields:
        fields = {
            "net_quantity": batch_result.get("net_quantity"),
            "mrp": batch_result.get("mrp"),
        }
    fields["mrp_netqty_contrast"] = batch_result.get("contrast_evidence") or {
        "targets": {}
    }
    extracted = canonicalize_extracted_fields(fields)
    quality = batch_result.get("image_quality") or {}
    rule_results = build_rule_results(fields, extracted, batch_result)
    warnings = _processing_warnings(batch_result)
    critical_uncertainty = (
        quality.get("usable") is False
        or (batch_result.get("ocr") or {}).get("success") is False
        or any(item["severity"] == "REVIEW" for item in warnings)
    )
    summary = aggregate_package_status(
        rule_results, critical_processing_uncertainty=critical_uncertainty
    )
    image_path = Path(str(batch_result.get("image") or ""))
    ocr_items = fields.get("ocr_evidence") or []
    report = {
        "report_version": REPORT_VERSION,
        "disclaimer": DISCLAIMER,
        "image": {
            "filename": image_path.name,
            "path": image_path.name,
            "width": quality.get("width"),
            "height": quality.get("height"),
            "processing_timestamp": processing_timestamp or datetime.now(timezone.utc).isoformat(),
            "quality_status": "USABLE" if quality.get("usable") else "REVIEW",
        },
        "quality": {
            "usable": quality.get("usable"),
            "blur_score": quality.get("blur_score"),
            "brightness": quality.get("brightness"),
            "glare_ratio": quality.get("glare_ratio"),
            "issues": quality.get("issues") or [],
            "warnings": quality.get("warnings") or [],
            "threshold_basis": "prototype engineering thresholds; not statutory",
        },
        "ocr": {
            **(batch_result.get("ocr") or {}),
            "evidence": _canonical_ocr_evidence(ocr_items),
        },
        "extracted_fields": extracted,
        "rule_results": rule_results,
        "summary": summary,
        "evidence": {
            "contrast": _json_safe(batch_result.get("contrast_evidence") or {}),
            "numeral_height": _measurement_evidence(batch_result),
            "calibration": _json_safe(batch_result.get("aruco") or {}),
        },
        "warnings": warnings,
        "evidence_images": _json_safe(batch_result.get("evidence_images") or []),
    }
    return _json_safe(report)


def _display_value(entry: dict[str, Any]) -> str:
    value = entry.get("normalized_value")
    if value is None:
        return "Unknown"
    if isinstance(value, dict):
        meaningful = [f"{key}={item}" for key, item in value.items() if item is not None]
        return ", ".join(meaningful) if meaningful else "Unknown"
    return str(value)


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Package Compliance Report", "", f"**Overall result: {report['summary']['overall_status']}**",
        "", report["summary"]["reason"], "", f"> {report['disclaimer']}", "",
        "## Image and quality", "",
        f"- File: `{report['image']['filename']}`",
        f"- Dimensions: {report['image'].get('width')} × {report['image'].get('height')}",
        f"- Quality status: {report['image']['quality_status']}", "",
        "## Extracted declarations", "",
    ]
    for name, entry in report["extracted_fields"].items():
        if name in {"consumer_care", "inclusive_of_all_taxes"} or entry.get("present"):
            lines.append(f"- **{name.replace('_', ' ').title()}:** {_display_value(entry)}")
    lines.extend(["", "## Rule results", ""])
    for rule in report["rule_results"]:
        lines.extend([
            f"### {rule['rule_id']} — {rule['status']}", "",
            rule["description"], "",
            f"Reason codes: `{', '.join(rule['reason_codes'])}`", "",
            f"Explanation: {rule['reason']}", "",
            f"Evidence items: {len(rule['evidence'])}", "",
        ])
    lines.extend(["## Processing warnings", ""])
    if report["warnings"]:
        lines.extend(
            f"- `{warning['code']}`: {warning['message']}" for warning in report["warnings"]
        )
    else:
        lines.append("None.")
    lines.extend(["", "## Summary counts", ""])
    summary = report["summary"]
    lines.append(
        f"PASS {summary['pass_count']} · FAIL {summary['fail_count']} · "
        f"REVIEW {summary['review_count']} · NOT_APPLICABLE {summary['not_applicable_count']}"
    )
    return "\n".join(lines).rstrip() + "\n"


def save_package_report(
    report: dict[str, Any], json_path: str | Path, markdown_path: str | Path,
) -> None:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown_report(report), encoding="utf-8")
