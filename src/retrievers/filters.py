"""Shared metadata filter builders for retrievers.

Provides:
- validate_meta_key(key)
- build_where_clauses(filters, params) -> List[str]

The callers should pass a mutable params list which will be extended with parameter values
in the correct order for parameterized SQL execution.
"""

import re
from typing import List, Dict, Any

_meta_key_re = re.compile(r"^[A-Za-z0-9_]+$")


def validate_meta_key(key: str) -> None:
    """Validate metadata key to avoid SQL injection via keys."""
    if not isinstance(key, str) or not _meta_key_re.match(key):
        raise ValueError(f"Invalid metadata key: {key}")


def build_where_clauses(filters: Dict[str, Any], params: List[Any]) -> List[str]:
    """Build SQL WHERE clauses from metadata filters.

    Args:
        filters: dict of filters (same shape as BM25 retriever supports)
        params: list that will be extended with parameter values in order

    Returns:
        List of SQL clause strings (without the leading 'WHERE')
    """
    if not filters:
        return []

    where_clauses: List[str] = []

    for key, cond in filters.items():
        validate_meta_key(key)

        if isinstance(cond, dict):
            op = cond.get("op", "is").lower()
            val = cond.get("value")
        else:
            op = "is"
            val = cond

        accessor = f"(metadata->>'{key}')"

        if op == "is":
            where_clauses.append(f"{accessor} = %s")
            params.append(str(val))

        elif op == "is not":
            where_clauses.append(f"{accessor} <> %s")
            params.append(str(val))

        elif op == "contains":
            where_clauses.append(f"{accessor} ILIKE %s")
            params.append(f"%{val}%")

        elif op == "does not contain":
            where_clauses.append(f"NOT ({accessor} ILIKE %s)")
            params.append(f"%{val}%")

        elif op == "is one of":
            if not isinstance(val, (list, tuple)):
                raise ValueError("'is one of' operator requires a list/tuple of values")
            placeholders = []
            for v in val:
                placeholders.append(f"{accessor} = %s")
                params.append(str(v))
            where_clauses.append("(" + " OR ".join(placeholders) + ")")

        elif op in (">", "<"):
            try:
                num = float(val)
            except Exception:
                raise ValueError(f"Numeric comparison requires a numeric value for key '{key}'")
            where_clauses.append(f"{accessor}::numeric {op} %s")
            params.append(num)

        else:
            raise ValueError(f"Unsupported filter operator: {op}")

    return where_clauses

