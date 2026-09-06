import csv
import re
from pathlib import Path


RULES_FILE = Path(__file__).resolve().parent.parent / "data" / "prototype-rules.csv"


def load_rules():
    rules = []

    with open(RULES_FILE, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rules.append(row)

    return rules


def get_nested_value(data, field_name):
    """
    Maps rule field names to extracted JSON values.
    """

    mapping = {
        "manufacturer_name": (
            data.get("manufacturer", {}).get("name")
            if isinstance(data.get("manufacturer"), dict)
            else data.get("manufacturer")
        ),
        "manufacturer_address": (
            data.get("manufacturer", {}).get("address")
            if isinstance(data.get("manufacturer"), dict)
            else data.get("manufacturer_address")
        ),
        "commodity_name": data.get("product"),
        "net_quantity": data.get("net_quantity"),
        "month_year": data.get("manufacture_date"),
        "mrp": data.get("mrp"),
        "consumer_care": data.get("consumer_care"),
        "country_of_origin": data.get("country_of_origin"),
        "mrp_netqty_contrast": data.get("mrp_netqty_contrast"),
    }

    return mapping.get(field_name)


def is_present(value):
    if value is None:
        return False

    if isinstance(value, str):
        return len(value.strip()) > 0

    if isinstance(value, dict):
        return any(
            v is not None and str(v).strip()
            for v in value.values()
        )

    return True


def has_low_confidence(value, threshold=0.5):
    """Only reject confidence when the extractor explicitly supplies it."""
    if not isinstance(value, dict):
        return False
    confidence = value.get("confidence")
    return isinstance(confidence, (int, float)) and confidence < threshold


def validate_mrp(value):
    if not isinstance(value, dict):
        return "REVIEW", "MRP could not be reliably determined"

    amount = value.get("value")

    if amount is None:
        return "REVIEW", "MRP amount could not be reliably determined"

    if has_low_confidence(value):
        return "REVIEW", "MRP candidate has low OCR/extraction confidence"

    if not isinstance(amount, (int, float)) or amount <= 0:
        return "FAIL", "Detected MRP must be greater than zero"

    return "PASS", f"MRP detected: ₹{amount}"


def validate_net_quantity(value):
    if not isinstance(value, dict):
        return "REVIEW", "Net quantity could not be reliably determined"

    quantity = value.get("value")
    unit = value.get("unit")

    if quantity is None:
        return "REVIEW", "Net quantity value could not be reliably determined"

    if not isinstance(quantity, (int, float)) or quantity <= 0:
        return "FAIL", "Detected net quantity must be positive"

    if not unit:
        return "REVIEW", "Net quantity unit could not be reliably determined"

    if has_low_confidence(value):
        return "REVIEW", "Net quantity candidate has low OCR/extraction confidence"

    allowed_units = {
        "G",
        "KG",
        "ML",
        "L",
        "N",
        "U",
        "CM",
        "M"
    }

    if unit.upper() not in allowed_units:
        return "FAIL", f"Detected quantity has unsupported unit: {unit}"

    return "PASS", f"Net quantity detected: {quantity} {unit}"


def validate_consumer_care(value):
    if not isinstance(value, dict):
        return "REVIEW", "Consumer-care data could not be reliably determined"

    phone = value.get("phone")
    email = value.get("email")

    if not phone:
        return "REVIEW", "Consumer-care phone number could not be reliably determined"

    if has_low_confidence(value):
        return "REVIEW", "Consumer-care details have low OCR/extraction confidence"

    phone_digits = re.sub(r"\D", "", phone)

    if len(phone_digits) < 8:
        return "FAIL", "Detected consumer-care phone number appears invalid"

    if email:
        email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

        if not re.match(email_pattern, email):
            return "FAIL", "Detected consumer-care email appears invalid"

    return "PASS", "Consumer-care details detected"


def validate_month_year(value):
    if not value:
        return "REVIEW", "Manufacture date could not be reliably determined"

    if isinstance(value, dict):
        if has_low_confidence(value):
            return "REVIEW", "Manufacture date has low OCR/extraction confidence"
        normalized = value.get("normalized")
        date_type = value.get("type")
        if normalized and date_type in {"manufacture_date", "manufacture_month_year"}:
            return "PASS", f"Manufacture date detected: {normalized}"
        return "REVIEW", "Manufacture date structure is incomplete"

    pattern = (
        r"^(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}$"
    )

    if not re.match(pattern, value, re.IGNORECASE):
        return "FAIL", f"Detected month/year has unrecognized format: {value}"

    return "PASS", f"Manufacture date detected: {value}"


def validate_mrp_netqty_contrast(value):
    """Evaluate deterministic prototype contrast evidence for both declarations.

    These cut-offs are implementation-defined engineering thresholds. They are
    not represented as statutory Legal Metrology thresholds.
    """
    if not isinstance(value, dict):
        return "REVIEW", "MRP/net-quantity contrast evidence is unavailable"

    targets = value.get("targets")
    if not isinstance(targets, dict):
        return "REVIEW", "Contrast evidence does not contain target measurements"

    required = ("NET_QUANTITY", "MRP")
    missing = [name for name in required if not isinstance(targets.get(name), dict)]
    if missing:
        return "REVIEW", f"Contrast evidence is missing: {', '.join(missing)}"

    measurements = [targets[name] for name in required]
    unreliable = [
        measurement.get("target", name)
        for name, measurement in zip(required, measurements)
        if measurement.get("status") != "OK"
        or not isinstance(measurement.get("confidence"), (int, float))
        or measurement["confidence"] < 0.65
    ]
    if unreliable:
        return (
            "REVIEW",
            "Contrast evidence is not reliable enough for: " + ", ".join(unreliable),
        )

    missing_metrics = [
        measurement.get("target", name)
        for name, measurement in zip(required, measurements)
        if not isinstance(measurement.get("contrast_ratio"), (int, float))
        or not isinstance(measurement.get("lab_color_difference"), (int, float))
    ]
    if missing_metrics:
        return (
            "REVIEW",
            "Contrast metrics are incomplete for: " + ", ".join(missing_metrics),
        )

    strong = [
        measurement["contrast_ratio"] >= 3.0
        or measurement["lab_color_difference"] >= 35.0
        for measurement in measurements
    ]
    low = [
        measurement["contrast_ratio"] < 1.5
        and measurement["lab_color_difference"] < 12.0
        and measurement.get("confidence", 0) >= 0.75
        for measurement in measurements
    ]
    if all(strong):
        return (
            "PASS",
            "Both declarations meet the implementation-defined engineering "
            "contrast threshold (not a statutory threshold)",
        )
    if any(low):
        failed = [
            measurement.get("target", required[index])
            for index, measurement in enumerate(measurements)
            if low[index]
        ]
        return (
            "FAIL",
            "Low contrast detected for " + ", ".join(failed)
            + " under implementation-defined engineering thresholds "
            "(not statutory thresholds)",
        )
    return (
        "REVIEW",
        "Contrast is borderline under implementation-defined engineering "
        "thresholds; human review is required",
    )


def validate_contrast_target(value, target_name=None):
    """Classify one contrast measurement using prototype engineering cut-offs."""
    name = target_name or (value.get("target") if isinstance(value, dict) else None) or "target"
    if not isinstance(value, dict):
        return "REVIEW", f"Contrast evidence is unavailable for {name}"
    confidence = value.get("confidence")
    if value.get("status") != "OK" or not isinstance(confidence, (int, float)) or confidence < 0.65:
        return "REVIEW", f"Contrast evidence is not reliable enough for {name}"
    ratio = value.get("contrast_ratio")
    color_difference = value.get("lab_color_difference")
    if not isinstance(ratio, (int, float)) or not isinstance(color_difference, (int, float)):
        return "REVIEW", f"Contrast metrics are incomplete for {name}"
    if ratio >= 3.0 or color_difference >= 35.0:
        return (
            "PASS",
            f"{name} meets an implementation-defined engineering contrast threshold "
            "(not a statutory threshold)",
        )
    if ratio < 1.5 and color_difference < 12.0 and confidence >= 0.75:
        return (
            "FAIL",
            f"{name} is below implementation-defined engineering contrast thresholds "
            "(not statutory thresholds)",
        )
    return "REVIEW", f"{name} contrast is borderline; human review is required"

def evaluate_small_package_exemption(value):
    if not isinstance(value, dict):
        return (
            "REVIEW",
            "Net quantity unavailable; exemption cannot be determined"
        )

    quantity = value.get("value")
    unit = str(value.get("unit", "")).upper()

    if quantity is None:
        return "REVIEW", "Quantity unavailable"

    if unit == "G" and quantity <= 10:
        return (
            "PASS",
            "Package falls within <=10 g exemption condition"
        )

    if unit == "ML" and quantity <= 10:
        return (
            "PASS",
            "Package falls within <=10 ml exemption condition"
        )

    return (
        "NOT_APPLICABLE",
        "Package does not meet small-package exemption condition"
    )

def evaluate_font_height_applicability(fields):
    quantity = fields.get("net_quantity")

    if not isinstance(quantity, dict):
        return (
            "REVIEW",
            "Net quantity unavailable"
        )

    unit = str(quantity.get("unit", "")).upper()

    if unit in {"N", "U"}:
        return (
            "NOT_APPLICABLE",
            "This prototype font-height rule applies to weight/volume quantities"
        )

    if unit in {"G", "KG", "ML", "L"}:
        return (
            "REVIEW",
            "Physical font measurement requires calibrated image analysis"
        )

    return (
        "REVIEW",
        f"Unknown quantity unit: {unit}"
    )

def evaluate_rule(rule, fields):
    field_name = rule["field_name"]

    value = get_nested_value(fields, field_name)

    # -----------------------------
    # Basic presence rules
    # -----------------------------

    if field_name in {
        "manufacturer_name",
        "manufacturer_address",
        "commodity_name",
        "country_of_origin"
    }:

        organization = fields.get("manufacturer") if field_name.startswith("manufacturer_") else None
        if is_present(value) and not has_low_confidence(organization):
            return {
                "status": "PASS",
                "reason": f"{field_name} detected",
                "value": value
            }

        reason = (
            f"{field_name} has low OCR/extraction confidence"
            if is_present(value)
            else f"{field_name} could not be reliably determined"
        )
        return {
            "status": "REVIEW",
            "reason": reason,
            "value": value
        }

    # -----------------------------
    # Specific validators
    # -----------------------------

    if field_name == "mrp":
        status, reason = validate_mrp(value)

    elif field_name == "net_quantity":
        status, reason = validate_net_quantity(value)

    elif field_name == "consumer_care":
        status, reason = validate_consumer_care(value)

    elif field_name == "month_year":
        status, reason = validate_month_year(value)

    elif field_name == "net_quantity_unit_scale":

        status, reason = validate_unit_scale(
        fields.get("net_quantity")
        )

        return {
        "status": status,
        "reason": reason,
        "value": fields.get("net_quantity")
        }
    elif field_name == "applicability_small_package":

        status, reason = evaluate_small_package_exemption(
        fields.get("net_quantity")
        )

        return {
        "status": status,
        "reason": reason,
        "value": fields.get("net_quantity")
        }
    elif field_name == "net_quantity_font_height":

        status, reason = evaluate_font_height_applicability(fields)

        return {
        "status": status,
        "reason": reason,
        "value": fields.get("net_quantity")
        }
    elif field_name == "mrp_netqty_contrast":
        status, reason = validate_mrp_netqty_contrast(value)
    else:
        return {
            "status": "REVIEW",
            "reason": (
                f"Automatic validator not implemented "
                f"for field: {field_name}"
            ),
            "value": value
        }

    return {
        "status": status,
        "reason": reason,
        "value": value
    }

def validate_unit_scale(value):
    if not isinstance(value, dict):
        return "REVIEW", "Net quantity not available"

    quantity = value.get("value")
    unit = str(value.get("unit", "")).upper()

    if quantity is None or not unit:
        return "REVIEW", "Quantity or unit missing"

    # Count-based commodity
    if unit in {"N", "U"}:
        return (
            "NOT_APPLICABLE",
            "Weight/volume unit-scale rule does not apply to count quantity"
        )

    # Weight
    if unit == "G":
        if quantity < 1000:
            return "PASS", "Gram unit appropriate below 1 kg"
        return "FAIL", "Quantity >= 1000 g should generally use kg"

    if unit == "KG":
        return "PASS", "Kilogram unit accepted"

    # Volume
    if unit == "ML":
        if quantity < 1000:
            return "PASS", "Millilitre unit appropriate below 1 litre"
        return "FAIL", "Quantity >= 1000 ml should generally use litre"

    if unit == "L":
        return "PASS", "Litre unit accepted"

    return "REVIEW", f"Unit-scale validation not implemented for {unit}"

def evaluate_compliance(fields):
    rules = load_rules()

    results = []

    for rule in rules:

        if str(rule.get("prototype_supported", "")).lower() != "true":
            continue

        evaluation = evaluate_rule(rule, fields)

        results.append({
            "rule_id": rule["rule_id"],
            "rule_number": rule["rule_number"],
            "requirement": rule["requirement_name"],
            "field_name": rule["field_name"],
            "legal_source": rule["legal_source"],
            "status": evaluation["status"],
            "reason": evaluation["reason"],
            "value": evaluation["value"]
        })

    summary = {
        "total": len(results),
        "passed": sum(r["status"] == "PASS" for r in results),
        "failed": sum(r["status"] == "FAIL" for r in results),
        "review": sum(r["status"] == "REVIEW" for r in results),
        "not_applicable": sum(
        r["status"] == "NOT_APPLICABLE"
        for r in results
        )
    }

    if summary["failed"] > 0:
        overall_status = "NON_COMPLIANT"
    elif summary["review"] > 0:
        overall_status = "MANUAL_REVIEW"
    else:
        overall_status = "COMPLIANT"

    return {
        "overall_status": overall_status,
        "summary": summary,
        "results": results
    }
