from src.datasets.dcs.dc_parser import parse_denial_constraint


class DenialConstraint:
    def __init__(self, constraint: str):
        self.predicates = parse_denial_constraint(constraint)

    @property
    def as_sql(self) -> str:
        def format_predicate(p):
            if p.is_value:
                return f"t{p.tuple1}.{p.attr1}{p.op}{p.attr2}"
            else:
                return f"t{p.tuple1}.{p.attr1}{p.op}t{p.tuple2}.{p.attr2}"

        return " AND ".join(format_predicate(p) for p in self.predicates)
