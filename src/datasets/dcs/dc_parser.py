import re
from collections import namedtuple

Predicate = namedtuple("Predicate", ["attr1", "op", "attr2", "is_value", "tuple1", "tuple2"])


def _normalize_constraints_string(constraints_string: str) -> str:
    normalized = constraints_string.strip()
    return normalized[4:-1] if normalized.startswith("not(") and normalized.endswith(")") else normalized


def _get_raw_predicates(constraints_string: str) -> list[str]:
    return [p.strip() for p in constraints_string.split("&") if p.strip()]


def _parse_predicate(raw_predicate: str) -> Predicate:
    # Try to match two-tuple predicate: t1.attr1 op t2.attr2
    two_tuple_pattern = re.compile(
        r'^t(\d+)\.([A-Za-z_]\w*)\s*(=|!=|<=|>=|<|>)\s*t(\d+)\.([A-Za-z_]\w*)$'
    )
    m = two_tuple_pattern.match(raw_predicate)
    if m:
        return Predicate(
            attr1=m.group(2),
            op=m.group(3),
            attr2=m.group(5),
            is_value=False,
            tuple1=int(m.group(1)),
            tuple2=int(m.group(4))
        )

    single_tuple_pattern = re.compile(
        r'^t(\d+)\.([A-Za-z_]\w*)\s*(=|!=|<=|>=|<|>)\s*([\'\"].*?[\'\"]|[\d.]+)$'
    )
    m = single_tuple_pattern.match(raw_predicate)
    if m:
        return Predicate(
            attr1=m.group(2),
            op=m.group(3),
            attr2=m.group(4),
            is_value=True,
            tuple1=int(m.group(1)),
            tuple2=None
        )

    raise ValueError(f"Invalid predicate format: {raw_predicate}")


def parse_denial_constraint(constraint: str) -> list[Predicate]:
    normalized = _normalize_constraints_string(constraint)
    raw_predicates = _get_raw_predicates(normalized)
    return [_parse_predicate(p) for p in raw_predicates]
