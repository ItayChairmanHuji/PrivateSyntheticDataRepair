from typing import Any


class MockLogger:
    def log(self, metric: str, value: Any):
        print(f"Logging {metric}: {value}")
