"""
Low-intrusion accounting and state invariant checks.

The checks in this module are intentionally defensive: callers may pass a live
EconomicCenter/ProductMarket-like object, a snapshot dict, or lightweight test
objects. The module does not import Ray or trigger actor calls.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEPARTMENT_FLOW_SECTORS: Tuple[str, ...] = (
    "household",
    "firm",
    "government",
    "bank_credit",
    "market_or_external",
    "unknown",
)


@dataclass
class AccountingInvariantIssue:
    """A structured invariant finding."""

    code: str
    message: str
    subject: Optional[str] = None
    severity: str = "error"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccountingInvariantResult:
    """Result returned by accounting invariant checks."""

    errors: List[AccountingInvariantIssue] = field(default_factory=list)
    warnings: List[AccountingInvariantIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(
        self,
        code: str,
        message: str,
        *,
        subject: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.errors.append(
            AccountingInvariantIssue(
                code=code,
                message=message,
                subject=subject,
                severity="error",
                details=details or {},
            )
        )

    def add_warning(
        self,
        code: str,
        message: str,
        *,
        subject: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.warnings.append(
            AccountingInvariantIssue(
                code=code,
                message=message,
                subject=subject,
                severity="warning",
                details=details or {},
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "metrics": dict(self.metrics),
            "ok": self.ok,
        }


@dataclass(frozen=True)
class _NormalizedTransaction:
    tx_id: Optional[str]
    tx_type: str
    sender_id: Optional[str]
    receiver_id: Optional[str]
    amount: Optional[float]
    month: Optional[int]
    metadata: Dict[str, Any]
    source: str


def check_accounting_invariants(
    economic_center: Any = None,
    product_market: Any = None,
    abstract_resource_market: Any = None,
    *,
    ledger: Any = None,
    transactions: Optional[Iterable[Any]] = None,
    firm_ids: Optional[Iterable[Any]] = None,
    household_ids: Optional[Iterable[Any]] = None,
    government_ids: Optional[Iterable[Any]] = None,
    bank_ids: Optional[Iterable[Any]] = None,
    loan_balances: Optional[Mapping[Any, Any]] = None,
    loan_transactions: Optional[Iterable[Any]] = None,
    month: Optional[int] = None,
    balance_tolerance: float = 1e-6,
    inventory_tolerance: float = 1e-6,
) -> AccountingInvariantResult:
    """
    Run accounting/state invariant diagnostics on lightweight snapshots.

    Args:
        economic_center: EconomicCenter-like object or dict snapshot.
        product_market: ProductMarket-like object or dict snapshot.
        abstract_resource_market: AbstractResourceMarket-like object or dict.
        ledger: Optional explicit ledger mapping, overriding center.ledger.
        transactions: Optional explicit transaction iterable.
        firm_ids: Optional firm id iterable. If omitted, common center fields are
            used and finally firm-like ledger id prefixes are inferred.
        household_ids: Optional household id iterable for flow diagnostics.
        government_ids: Optional government id iterable for flow diagnostics.
        bank_ids: Optional bank id iterable for flow diagnostics.
        loan_balances: Optional mapping of firm id to outstanding loan/debt.
        loan_transactions: Optional extra transactions used only as loan
            explanations.
        month: Optional month filter for transaction-based checks.
        balance_tolerance: Numeric tolerance for cash checks.
        inventory_tolerance: Numeric tolerance for inventory checks.

    Returns:
        AccountingInvariantResult with ``errors``, ``warnings``, and ``metrics``.
    """
    result = AccountingInvariantResult(metrics={})

    ledger_mapping = _coerce_mapping(
        ledger if ledger is not None else _get_field(economic_center, "ledger", "accounts", "balances")
    )
    balances = _check_ledger_balances(
        result,
        ledger_mapping,
        balance_tolerance=abs(float(balance_tolerance or 0.0)),
    )

    firm_id_set = _collect_agent_ids(
        explicit_ids=firm_ids,
        owner=economic_center,
        field_names=("firm_id", "firm_ids", "firms", "firm_by_id"),
        ledger_ids=balances.keys(),
        infer_prefixes=("firm_", "mfg_", "ret_", "svc_"),
    )
    household_id_set = _collect_agent_ids(
        explicit_ids=household_ids,
        owner=economic_center,
        field_names=("household_id", "household_ids", "households", "household_by_id"),
        ledger_ids=(),
        infer_prefixes=(),
    )
    government_id_set = _collect_agent_ids(
        explicit_ids=government_ids,
        owner=economic_center,
        field_names=("government_id", "government_ids", "governments"),
        ledger_ids=(),
        infer_prefixes=(),
    )
    bank_id_set = _collect_agent_ids(
        explicit_ids=bank_ids,
        owner=economic_center,
        field_names=("bank_id", "bank_ids", "banks"),
        ledger_ids=(),
        infer_prefixes=(),
    )
    registered_household_id_set = _collect_agent_ids(
        explicit_ids=None,
        owner=economic_center,
        field_names=("household_id", "household_ids", "households", "household_by_id"),
        ledger_ids=(),
        infer_prefixes=(),
    )
    registered_government_id_set = _collect_agent_ids(
        explicit_ids=None,
        owner=economic_center,
        field_names=("government_id", "government_ids", "governments"),
        ledger_ids=(),
        infer_prefixes=(),
    )
    registered_bank_id_set = _collect_agent_ids(
        explicit_ids=None,
        owner=economic_center,
        field_names=("bank_id", "bank_ids", "banks"),
        ledger_ids=(),
        infer_prefixes=(),
    )

    txs = _collect_transactions(
        economic_center=economic_center,
        explicit_transactions=transactions,
        month=month,
    )
    normalized_txs = [_normalize_transaction(tx, "economic_center") for tx in txs]
    normalized_txs = [tx for tx in normalized_txs if tx is not None]

    loan_txs = list(loan_transactions or [])
    normalized_loan_txs = [_normalize_transaction(tx, "loan_transactions") for tx in loan_txs]
    normalized_loan_txs = [tx for tx in normalized_loan_txs if tx is not None]
    loan_amount_by_firm = _loan_explanations_by_firm(
        [*normalized_txs, *normalized_loan_txs],
        explicit_loan_balances=loan_balances,
        ledger_mapping=ledger_mapping,
    )

    _check_negative_cash(
        result,
        balances=balances,
        firm_ids=firm_id_set,
        non_firm_registered_ids=(
            set(registered_household_id_set)
            | set(registered_government_id_set)
            | set(registered_bank_id_set)
        ),
        loan_amount_by_firm=loan_amount_by_firm,
        balance_tolerance=abs(float(balance_tolerance or 0.0)),
    )

    _check_inventory_non_negative(
        result,
        product_market=product_market,
        inventory_tolerance=abs(float(inventory_tolerance or 0.0)),
    )

    _check_duplicate_resource_purchases(
        result,
        transactions=normalized_txs,
        month=month,
    )

    result.metrics["department_flow_summary"] = summarize_department_flows(
        normalized_txs,
        firm_ids=firm_id_set,
        household_ids=household_id_set,
        government_ids=government_id_set,
        bank_ids=bank_id_set,
        month=month,
    )

    local_resource_txs = _collect_local_resource_transactions(abstract_resource_market, month=month)
    result.metrics["resource_market_local_transactions"] = len(local_resource_txs)

    result.metrics["errors"] = len(result.errors)
    result.metrics["warnings"] = len(result.warnings)
    return result


def run_accounting_invariant_checks(*args: Any, **kwargs: Any) -> AccountingInvariantResult:
    """Alias for callers that prefer a verb-style entry point."""

    return check_accounting_invariants(*args, **kwargs)


def summarize_department_flows(
    transactions: Iterable[Any],
    *,
    firm_ids: Optional[Iterable[Any]] = None,
    household_ids: Optional[Iterable[Any]] = None,
    government_ids: Optional[Iterable[Any]] = None,
    bank_ids: Optional[Iterable[Any]] = None,
    month: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Summarize bilateral money flows by broad SFC-style department.

    This is diagnostic-only: every finite transaction amount is counted as an
    outflow from the sender department and an inflow to the receiver department.
    Unknown parties stay in ``unknown`` instead of being inferred as firms.
    """
    normalized: List[_NormalizedTransaction] = []
    source_transactions = transactions if transactions is not None else []
    for raw_tx in source_transactions:
        tx = raw_tx if isinstance(raw_tx, _NormalizedTransaction) else _normalize_transaction(raw_tx, "flow_summary")
        if tx is None:
            continue
        if month is not None and tx.month != month:
            continue
        normalized.append(tx)

    context = {
        "firm": set(_ids_from_container(firm_ids)),
        "household": set(_ids_from_container(household_ids)),
        "government": set(_ids_from_container(government_ids)),
        "bank_credit": set(_ids_from_container(bank_ids)),
    }
    departments: Dict[str, Dict[str, float]] = {
        sector: {"inflow": 0.0, "outflow": 0.0, "net_flow": 0.0}
        for sector in DEPARTMENT_FLOW_SECTORS
    }
    unknown_party_ids: set[str] = set()
    market_or_external_party_ids: set[str] = set()
    skipped_transaction_count = 0
    non_positive_amount_count = 0

    for tx in normalized:
        amount = _to_finite_float(tx.amount)
        if amount is None:
            skipped_transaction_count += 1
            continue
        if amount <= 0.0:
            non_positive_amount_count += 1

        sender_sector = _classify_flow_party(tx.sender_id, tx=tx, side="sender", context=context)
        receiver_sector = _classify_flow_party(tx.receiver_id, tx=tx, side="receiver", context=context)

        departments[sender_sector]["outflow"] += amount
        departments[receiver_sector]["inflow"] += amount

        if sender_sector == "unknown" and tx.sender_id:
            unknown_party_ids.add(tx.sender_id)
        if receiver_sector == "unknown" and tx.receiver_id:
            unknown_party_ids.add(tx.receiver_id)
        if sender_sector == "market_or_external" and tx.sender_id:
            market_or_external_party_ids.add(tx.sender_id)
        if receiver_sector == "market_or_external" and tx.receiver_id:
            market_or_external_party_ids.add(tx.receiver_id)

    total_inflow = 0.0
    total_outflow = 0.0
    for rec in departments.values():
        rec["inflow"] = float(rec["inflow"])
        rec["outflow"] = float(rec["outflow"])
        rec["net_flow"] = float(rec["inflow"] - rec["outflow"])
        total_inflow += rec["inflow"]
        total_outflow += rec["outflow"]

    return {
        "departments": departments,
        "transaction_count": len(normalized),
        "skipped_transaction_count": skipped_transaction_count,
        "non_positive_amount_transaction_count": non_positive_amount_count,
        "total_inflow": float(total_inflow),
        "total_outflow": float(total_outflow),
        "total_net_flow_residual": float(total_inflow - total_outflow),
        "unknown_party_ids": sorted(unknown_party_ids),
        "market_or_external_party_ids": sorted(market_or_external_party_ids),
    }


def _classify_flow_party(
    agent_id: Optional[str],
    *,
    tx: _NormalizedTransaction,
    side: str,
    context: Mapping[str, set[str]],
) -> str:
    party_id = str(agent_id) if agent_id is not None else ""

    if party_id:
        for sector in ("household", "firm", "government", "bank_credit"):
            if party_id in context.get(sector, set()):
                return sector

    metadata_sector = _flow_department_from_metadata(tx.metadata, side)
    if metadata_sector is not None:
        return metadata_sector

    id_sector = _flow_department_from_account_id(party_id)
    if id_sector is not None:
        return id_sector

    if tx.tx_type == "credit_draw" and side == "sender":
        return "bank_credit"

    return "unknown"


def _flow_department_from_metadata(metadata: Mapping[str, Any], side: str) -> Optional[str]:
    if side == "sender":
        keys = (
            "sender_department",
            "sender_sector",
            "sender_type",
            "payer_department",
            "payer_sector",
            "payer_type",
            "buyer_department",
            "buyer_sector",
            "buyer_type",
            "source_department",
            "source_sector",
        )
    else:
        keys = (
            "receiver_department",
            "receiver_sector",
            "receiver_type",
            "payee_department",
            "payee_sector",
            "payee_type",
            "seller_department",
            "seller_sector",
            "seller_type",
            "destination_department",
            "destination_sector",
        )

    for key in keys:
        sector = _flow_department_from_hint(_get_field(metadata, key, default=None))
        if sector is not None:
            return sector
    return None


def _flow_department_from_hint(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        for key in ("department", "sector", "type", "agent_type"):
            sector = _flow_department_from_hint(_get_field(value, key, default=None))
            if sector is not None:
                return sector
        return None
    if value is None:
        return None

    hint = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not hint:
        return None
    if hint in DEPARTMENT_FLOW_SECTORS:
        return hint

    tokens = set(filter(None, hint.split("_")))
    if hint in {"households", "consumer", "consumers", "worker", "workers", "person", "family"}:
        return "household"
    if hint in {"firms", "business", "businesses", "company", "manufacturer", "retailer", "employer", "producer"}:
        return "firm"
    if hint in {"gov", "public", "public_sector", "tax_authority", "treasury"}:
        return "government"
    if hint in {"bank", "banks", "credit", "creditor", "lender", "financial", "financial_sector"}:
        return "bank_credit"
    if hint in {"market", "markets", "external", "external_sector", "outside", "outside_sector"}:
        return "market_or_external"

    if tokens & {"household", "households", "consumer", "worker"}:
        return "household"
    if tokens & {"firm", "firms", "business", "company", "manufacturer", "retailer", "employer", "producer"}:
        return "firm"
    if tokens & {"government", "gov", "public", "treasury"}:
        return "government"
    if tokens & {"bank", "banks", "credit", "creditor", "lender", "financial"}:
        return "bank_credit"
    if tokens & {"market", "markets", "external", "outside"}:
        return "market_or_external"
    if "unknown" in tokens:
        return "unknown"

    return None


def _flow_department_from_account_id(agent_id: str) -> Optional[str]:
    account_id = str(agent_id or "").strip()
    if not account_id:
        return None
    lower = account_id.lower()

    if lower in {"bank", "central_bank", "bank_credit_system", "credit_facility"}:
        return "bank_credit"
    if lower.startswith(("bank_", "central_bank", "bank_credit", "credit_")):
        return "bank_credit"
    if "bank_credit" in lower or "credit_system" in lower:
        return "bank_credit"

    if lower in {"government", "gov", "gov_main_simulation", "tax_authority", "treasury"}:
        return "government"
    if lower.startswith(("gov_", "government_")) or lower.endswith("_government"):
        return "government"

    if lower.startswith(("household_", "hh_")):
        return "household"

    if lower.startswith(("firm_", "manufacturer_", "retailer_", "mfg_", "ret_", "svc_", "company_")):
        return "firm"

    if (
        lower in {"market", "external", "outside", "resource_market", "product_market", "supplier_market"}
        or "market" in lower
        or "external" in lower
        or lower.startswith(("outside_", "abstract_resource"))
    ):
        return "market_or_external"

    return None


def _check_ledger_balances(
    result: AccountingInvariantResult,
    ledger_mapping: Mapping[Any, Any],
    *,
    balance_tolerance: float,
) -> Dict[str, float]:
    balances: Dict[str, float] = {}
    total_cash = 0.0
    min_balance: Optional[float] = None
    non_finite_count = 0

    for raw_agent_id, entry in ledger_mapping.items():
        agent_id = str(raw_agent_id)
        amount = _read_numeric_field(entry, ("amount", "balance", "cash", "current_balance"))
        if amount is None:
            non_finite_count += 1
            result.add_error(
                "ledger_invalid_cash_balance",
                f"Ledger entry for {agent_id} has no finite cash balance.",
                subject=agent_id,
                details={"entry_type": type(entry).__name__},
            )
            continue

        balances[agent_id] = amount
        total_cash += amount
        min_balance = amount if min_balance is None else min(min_balance, amount)

    negative_accounts = [agent_id for agent_id, amount in balances.items() if amount < -balance_tolerance]

    result.metrics.update(
        {
            "ledger_account_count": len(ledger_mapping),
            "ledger_valid_account_count": len(balances),
            "ledger_invalid_account_count": non_finite_count,
            "ledger_total_cash": total_cash,
            "ledger_min_balance": min_balance if min_balance is not None else 0.0,
            "ledger_negative_account_count": len(negative_accounts),
        }
    )
    return balances


def _check_negative_cash(
    result: AccountingInvariantResult,
    *,
    balances: Mapping[str, float],
    firm_ids: Sequence[str],
    non_firm_registered_ids: Sequence[str],
    loan_amount_by_firm: Mapping[str, float],
    balance_tolerance: float,
) -> None:
    firm_id_set = set(firm_ids)
    non_firm_id_set = set(non_firm_registered_ids)
    negative_firms = []
    negative_without_loan = []
    negative_with_loan = []

    for firm_id in sorted(firm_id_set):
        if firm_id not in balances:
            continue
        balance = balances[firm_id]
        if balance >= -balance_tolerance:
            continue

        negative_firms.append(firm_id)
        explained_amount = float(loan_amount_by_firm.get(firm_id, 0.0) or 0.0)
        details = {"balance": balance, "loan_or_debt_explanation": explained_amount}
        if explained_amount > balance_tolerance:
            negative_with_loan.append(firm_id)
            result.add_warning(
                "firm_negative_cash",
                f"Firm {firm_id} has negative cash with a loan/debt explanation.",
                subject=firm_id,
                details=details,
            )
        else:
            negative_without_loan.append(firm_id)
            result.add_error(
                "negative_cash_without_loan",
                f"Firm {firm_id} has negative cash without a loan/debt explanation.",
                subject=firm_id,
                details=details,
            )

    for agent_id in sorted(non_firm_id_set):
        balance = balances.get(agent_id)
        if balance is not None and balance < -balance_tolerance:
            result.add_error(
                "non_firm_negative_cash",
                f"Registered non-firm {agent_id} has negative cash.",
                subject=agent_id,
                details={"balance": balance},
            )

    result.metrics.update(
        {
            "registered_firm_count": len(firm_id_set),
            "negative_firm_cash_count": len(negative_firms),
            "negative_firm_cash_without_loan_count": len(negative_without_loan),
            "negative_firm_cash_with_loan_count": len(negative_with_loan),
            "registered_non_firm_negative_cash_count": sum(
                1
                for agent_id in non_firm_id_set
                if balances.get(agent_id, 0.0) < -balance_tolerance
            ),
        }
    )


def _check_inventory_non_negative(
    result: AccountingInvariantResult,
    *,
    product_market: Any,
    inventory_tolerance: float,
) -> None:
    checked_records = 0
    negative_records = 0
    seen_product_fields: set[Tuple[str, str]] = set()

    for product_key, product in _iter_products(product_market):
        product_id = str(
            _get_field(product, "product_id", "sku_id", "id", default=product_key)
            or product_key
            or "<unknown_product>"
        )
        for field_name in ("available_stock", "reserved_stock", "stock", "inventory", "quantity"):
            if not _has_field(product, field_name):
                continue
            seen_key = (product_id, field_name)
            if seen_key in seen_product_fields:
                continue
            seen_product_fields.add(seen_key)
            value = _read_numeric_field(product, (field_name,))
            checked_records += 1
            if value is None:
                continue
            if value < -inventory_tolerance:
                negative_records += 1
                result.add_error(
                    "negative_inventory",
                    f"Product {product_id} has negative {field_name}.",
                    subject=product_id,
                    details={"field": field_name, "value": value},
                )

    for retailer_id, product_id, stock in _iter_retailer_inventory(product_market):
        checked_records += 1
        if stock < -inventory_tolerance:
            negative_records += 1
            subject = f"{retailer_id}/{product_id}"
            result.add_error(
                "negative_inventory",
                f"Retailer inventory {subject} is negative.",
                subject=subject,
                details={
                    "retailer_id": retailer_id,
                    "product_id": product_id,
                    "field": "retailer_inventory",
                    "value": stock,
                },
            )

    result.metrics.update(
        {
            "inventory_records_checked": checked_records,
            "negative_inventory_count": negative_records,
        }
    )


def _check_duplicate_resource_purchases(
    result: AccountingInvariantResult,
    *,
    transactions: Sequence[_NormalizedTransaction],
    month: Optional[int],
) -> None:
    groups: Dict[Tuple[Optional[int], str, str], List[_NormalizedTransaction]] = defaultdict(list)

    for tx in transactions:
        if tx.tx_type != "resource_purchase":
            continue
        if month is not None and tx.month != month:
            continue
        buyer_id = str(
            tx.sender_id
            or _get_field(tx.metadata, "buyer", "buyer_id", "firm_id", default="")
            or ""
        )
        resource = str(
            _get_field(tx.metadata, "industry_code", "resource", "resource_id", "industry", default="")
            or ""
        )
        if not buyer_id or not resource:
            continue
        groups[(tx.month, buyer_id, resource)].append(tx)

    duplicate_groups = 0
    exact_duplicate_groups = 0
    duplicate_tx_count = 0

    for (tx_month, buyer_id, resource), group in sorted(groups.items(), key=_resource_group_sort_key):
        if len(group) <= 1:
            continue

        duplicate_groups += 1
        duplicate_tx_count += len(group)
        signatures = Counter(_resource_purchase_signature(tx) for tx in group)
        exact_count = max(signatures.values()) if signatures else 0
        is_exact_duplicate = exact_count > 1
        if is_exact_duplicate:
            exact_duplicate_groups += 1

        result.add_warning(
            "duplicate_resource_purchase",
            (
                f"{len(group)} resource_purchase transactions share firm/resource/month "
                f"({buyer_id}, {resource}, {tx_month})."
            ),
            subject=f"{buyer_id}:{resource}:{tx_month}",
            details={
                "month": tx_month,
                "buyer_id": buyer_id,
                "industry_code": resource,
                "count": len(group),
                "tx_ids": [tx.tx_id for tx in group],
                "amounts": [tx.amount for tx in group],
                "quantities": [_read_numeric_field(tx.metadata, ("quantity",)) for tx in group],
                "unit_prices": [_read_numeric_field(tx.metadata, ("unit_price",)) for tx in group],
                "has_exact_duplicate_signature": is_exact_duplicate,
            },
        )

    result.metrics.update(
        {
            "resource_purchase_transactions_checked": sum(
                1 for tx in transactions if tx.tx_type == "resource_purchase"
            ),
            "resource_purchase_duplicate_groups": duplicate_groups,
            "resource_purchase_exact_duplicate_groups": exact_duplicate_groups,
            "resource_purchase_duplicate_transaction_count": duplicate_tx_count,
        }
    )


def _resource_group_sort_key(item: Tuple[Tuple[Optional[int], str, str], List[_NormalizedTransaction]]) -> Tuple[int, int, str, str]:
    tx_month, buyer_id, resource = item[0]
    return (1 if tx_month is None else 0, tx_month if tx_month is not None else -1, buyer_id, resource)


def _resource_purchase_signature(tx: _NormalizedTransaction) -> Tuple[Any, ...]:
    return (
        tx.sender_id,
        _get_field(tx.metadata, "industry_code", "resource", "resource_id", "industry", default=None),
        _round_for_signature(tx.amount),
        _round_for_signature(_read_numeric_field(tx.metadata, ("quantity",))),
        _round_for_signature(_read_numeric_field(tx.metadata, ("unit_price",))),
        tx.receiver_id,
    )


def _round_for_signature(value: Any) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return value


def _collect_transactions(
    *,
    economic_center: Any,
    explicit_transactions: Optional[Iterable[Any]],
    month: Optional[int],
) -> List[Any]:
    if explicit_transactions is not None:
        return list(explicit_transactions)

    tx_history = _get_field(economic_center, "tx_history", "transactions", default=None)
    if tx_history is not None:
        return list(tx_history)

    tx_by_month = _get_field(economic_center, "tx_by_month", default=None)
    tx_by_month_mapping = _coerce_mapping(tx_by_month)
    if tx_by_month_mapping:
        if month is not None:
            return list(tx_by_month_mapping.get(month, tx_by_month_mapping.get(str(month), [])) or [])
        transactions: List[Any] = []
        for month_txs in tx_by_month_mapping.values():
            transactions.extend(list(month_txs or []))
        return transactions

    return []


def _collect_local_resource_transactions(
    abstract_resource_market: Any,
    *,
    month: Optional[int],
) -> List[_NormalizedTransaction]:
    raw_txs = _get_field(abstract_resource_market, "transactions", default=None)
    normalized = []
    for raw_tx in list(raw_txs or []):
        tx = _normalize_transaction(raw_tx, "abstract_resource_market")
        if tx is None:
            continue
        if tx.tx_type != "resource_purchase":
            continue
        if month is not None and tx.month != month:
            continue
        normalized.append(tx)
    return normalized


def _normalize_transaction(raw_tx: Any, source: str) -> Optional[_NormalizedTransaction]:
    if raw_tx is None:
        return None

    metadata = _get_field(raw_tx, "metadata", "meta", default=None)
    if not isinstance(metadata, Mapping):
        metadata = {}
    else:
        metadata = dict(metadata)

    tx_type = str(_get_field(raw_tx, "type", "tx_type", "transaction_type", default="") or "")
    sender_id = _get_field(raw_tx, "sender_id", "sender", "buyer_id", "buyer", default=None)
    receiver_id = _get_field(raw_tx, "receiver_id", "receiver", "seller_id", "seller", default=None)
    amount = _read_numeric_field(raw_tx, ("amount", "total_cost", "total_spent", "value"))
    tx_month = _read_int_field(raw_tx, ("month", "period", "time", "tick"))
    tx_id = _get_field(raw_tx, "id", "tx_id", "transaction_id", default=None)

    # AbstractResourceMarket keeps local backup records without a "type" field.
    resource = _get_field(raw_tx, "resource", "industry_code", default=None)
    if not tx_type and resource is not None:
        tx_type = "resource_purchase"
    if resource is not None and "industry_code" not in metadata:
        metadata["industry_code"] = resource
    for field_name in ("quantity", "unit_price", "unit", "base_price", "budget"):
        value = _get_field(raw_tx, field_name, default=None)
        if value is not None and field_name not in metadata:
            metadata[field_name] = value

    return _NormalizedTransaction(
        tx_id=str(tx_id) if tx_id is not None else None,
        tx_type=tx_type,
        sender_id=str(sender_id) if sender_id is not None else None,
        receiver_id=str(receiver_id) if receiver_id is not None else None,
        amount=amount,
        month=tx_month,
        metadata=metadata,
        source=source,
    )


def _loan_explanations_by_firm(
    transactions: Sequence[_NormalizedTransaction],
    *,
    explicit_loan_balances: Optional[Mapping[Any, Any]],
    ledger_mapping: Mapping[Any, Any],
) -> Dict[str, float]:
    loan_amounts: Dict[str, float] = defaultdict(float)

    for raw_firm_id, amount in (explicit_loan_balances or {}).items():
        numeric = _to_finite_float(amount)
        if numeric and numeric > 0:
            loan_amounts[str(raw_firm_id)] += numeric

    for raw_firm_id, entry in ledger_mapping.items():
        debt = _read_numeric_field(
            entry,
            (
                "debt",
                "loan_balance",
                "loans",
                "outstanding_loan",
                "outstanding_debt",
                "credit_used",
            ),
        )
        if debt and debt > 0:
            loan_amounts[str(raw_firm_id)] += debt

    for tx in transactions:
        if not _looks_like_loan_transaction(tx):
            continue
        if not tx.receiver_id:
            continue
        amount = tx.amount or _read_numeric_field(tx.metadata, ("amount", "loan_amount", "principal"))
        if amount and amount > 0:
            loan_amounts[str(tx.receiver_id)] += amount

    return dict(loan_amounts)


def _looks_like_loan_transaction(tx: _NormalizedTransaction) -> bool:
    haystack_values: List[str] = [tx.tx_type]
    for key in ("category", "purpose", "kind", "subtype", "transaction_type", "memo", "description"):
        value = _get_field(tx.metadata, key, default=None)
        if value is not None:
            haystack_values.append(str(value))
    haystack = " ".join(haystack_values).lower()
    return any(keyword in haystack for keyword in ("loan", "borrow", "credit", "debt"))


def _iter_products(product_market: Any) -> Iterable[Tuple[str, Any]]:
    if product_market is None:
        return []

    products_by_id = _coerce_mapping(_get_field(product_market, "products_by_id", "skus_by_id", default=None))
    products = _get_field(product_market, "products", "skus", "items", default=None)

    yielded_ids: set[str] = set()
    result: List[Tuple[str, Any]] = []

    for product_id, product in products_by_id.items():
        product_key = str(product_id)
        yielded_ids.add(product_key)
        result.append((product_key, product))

    if products is not None and not isinstance(products, (str, bytes)):
        for idx, product in enumerate(list(products or [])):
            product_id = str(_get_field(product, "product_id", "sku_id", "id", default=idx))
            if product_id in yielded_ids:
                continue
            yielded_ids.add(product_id)
            result.append((product_id, product))

    if not result and isinstance(product_market, Mapping):
        for key, value in product_market.items():
            if isinstance(value, Mapping) and any(
                field_name in value
                for field_name in ("available_stock", "reserved_stock", "stock", "inventory", "quantity")
            ):
                result.append((str(key), value))

    return result


def _iter_retailer_inventory(product_market: Any) -> Iterable[Tuple[str, str, float]]:
    inventory = _coerce_mapping(_get_field(product_market, "retailer_inventory", default=None))
    records: List[Tuple[str, str, float]] = []
    for raw_retailer_id, product_stock in inventory.items():
        nested = _coerce_mapping(product_stock)
        for raw_product_id, raw_stock in nested.items():
            stock = _to_finite_float(raw_stock)
            if stock is not None:
                records.append((str(raw_retailer_id), str(raw_product_id), stock))
    return records


def _collect_agent_ids(
    *,
    explicit_ids: Optional[Iterable[Any]],
    owner: Any,
    field_names: Sequence[str],
    ledger_ids: Iterable[str],
    infer_prefixes: Sequence[str],
) -> List[str]:
    ids: set[str] = set()

    if explicit_ids is not None:
        ids.update(_ids_from_container(explicit_ids))

    for field_name in field_names:
        value = _get_field(owner, field_name, default=None)
        ids.update(_ids_from_container(value))

    for agent_id in ledger_ids:
        if any(str(agent_id).startswith(prefix) for prefix in infer_prefixes):
            ids.add(str(agent_id))

    ids.discard("")
    ids.discard("None")
    return sorted(ids)


def _ids_from_container(value: Any) -> set[str]:
    ids: set[str] = set()
    if value is None:
        return ids
    if isinstance(value, Mapping):
        ids.update(str(key) for key in value.keys())
        return ids
    if isinstance(value, (str, bytes)):
        ids.add(str(value))
        return ids
    try:
        iterator = iter(value)
    except TypeError:
        iterator = iter((value,))
    for item in iterator:
        if item is None:
            continue
        if isinstance(item, (str, bytes, int)):
            ids.add(str(item))
            continue
        item_id = _get_field(item, "firm_id", "household_id", "government_id", "bank_id", "agent_id", "id", default=None)
        if item_id is not None:
            ids.add(str(item_id))
    return ids


def _coerce_mapping(value: Any) -> Mapping[Any, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "items"):
        try:
            return dict(value.items())
        except Exception:
            return {}
    return {}


def _get_field(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        try:
            return getattr(obj, name)
        except Exception:
            continue
    return default


def _has_field(obj: Any, name: str) -> bool:
    if obj is None:
        return False
    if isinstance(obj, Mapping):
        return name in obj
    try:
        getattr(obj, name)
        return True
    except Exception:
        return False


def _read_numeric_field(obj: Any, names: Sequence[str]) -> Optional[float]:
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return _to_finite_float(obj)
    value = _get_field(obj, *names, default=None)
    return _to_finite_float(value)


def _read_int_field(obj: Any, names: Sequence[str]) -> Optional[int]:
    value = _get_field(obj, *names, default=None)
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return None
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _to_finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
