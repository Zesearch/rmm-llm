from dataclasses import dataclass


@dataclass
class RMMConfig:
    """Runtime configuration for RMM attention pruning.

    ``dimension_keep_ratio`` controls the Q/K feature dimensions retained per
    batch item and attention head. ``token_keep_ratio`` controls the attention
    columns and corresponding V tokens retained in the output multiplication.
    """

    dimension_keep_ratio: float = 1.0
    token_keep_ratio: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self):
        for name, value in (
            ("dimension_keep_ratio", self.dimension_keep_ratio),
            ("token_keep_ratio", self.token_keep_ratio),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")

    @property
    def is_baseline(self):
        return (
            not self.enabled
            or (
                self.dimension_keep_ratio >= 1.0
                and self.token_keep_ratio >= 1.0
            )
        )

    def set_keep_ratio(self, keep_ratio):
        keep_ratio = float(keep_ratio)
        self.dimension_keep_ratio = keep_ratio
        self.token_keep_ratio = keep_ratio
        self.enabled = keep_ratio < 1.0
        self.validate()

