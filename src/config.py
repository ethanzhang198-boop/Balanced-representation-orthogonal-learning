from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root / "data"
    output_dir: Path = root / "outputs"
    search_file: str = "table_search_with_controls.xlsx"
    innovation_file: str = "table_innovation_with_controls.xlsx"
    outcome: str = "Patent1"
    treatment: str = "SVI_code_year"
    train_end: int = 2021
    valid_year: int = 2022
    seed: int = 42
    cpu: bool = True
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    folds: int = 3
    trim_rates: tuple = (0.05, 0.10, 0.15)
    cf_n_estimators: int = 160
    cf_min_samples_leaf: int = 40
    cf_max_depth: int = 10
    repr_hidden_dim: int = 64
    repr_latent_dim: int = 16
    repr_dropout: float = 0.10
    repr_lr: float = 0.003
    repr_weight_decay: float = 1e-4
    repr_epochs: int = 25
    repr_batch_size: int = 512
    repr_patience: int = 5
    bootstrap_reps: int = 200
    controls: list = field(
        default_factory=lambda: [
            "SOE",
            "Size",
            "Lev",
            "ROA",
            "Growth",
            "Board",
            "Indep",
            "Dual",
            "Top1",
            "TobinQ",
            "ListAge",
            "Big4",
            "Ofee",
            "Mfee",
            "ATO",
            "Occupy",
            "Inst",
            "HighTech",
        ]
    )
    search_core: list = field(
        default_factory=lambda: ["search_indcode", "SVI_code_year", "SVI_All_year", "Kwdnum"]
    )
    innovation_candidates: list = field(
        default_factory=lambda: ["RD1", "Patent1", "Patent_Award1", "InnoEff1"]
    )

    @property
    def search_path(self) -> Path:
        return self.data_dir / self.search_file

    @property
    def innovation_path(self) -> Path:
        return self.data_dir / self.innovation_file
