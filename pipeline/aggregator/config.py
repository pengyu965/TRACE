"""Pipeline configuration. Override any field via the CLI or a YAML/JSON config file."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # Inputs
    per_video_dir: Path                       # directory of *_results.json (one per video)
    queries_file: Path                        # queries.jsonl
    topic_video_mapping: Path | None = None   # optional, only used for a sanity log

    # Outputs (under work_dir)
    work_dir: Path = Path("./run")

    # Submission metadata
    team_id: str = "123456"
    task: str = "oracle"
    run_id_prefix: str = "penkil"

    # Embedding stage
    embed_model: str = "Qwen/Qwen3-Embedding-8B"
    embed_max_len: int = 8192
    cluster_tau: float = 0.9
    embed_gpu_mem_util: float = 0.92

    # LLM aggregation stage. Matches the TRACE paper: Qwen3-30B-A3B-Instruct-2507.
    llm_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    llm_max_model_len: int = 32768
    llm_max_new_tokens: int = 8000
    llm_gpu_mem_util: float = 0.92
    llm_tensor_parallel: int = 1              # set >1 for multi-GPU
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_seed: int = 0
    llm_max_retries: int = 3

    # Which methods to run. Method A (embed + per-cluster LLM verify) is the
    # headline method from the paper. Method B (pure-LLM clustering) is the
    # ablation baseline and is OFF by default; enable with --run-method-b.
    run_method_a: bool = True
    run_method_b: bool = False

    @property
    def per_query_dir(self) -> Path: return self.work_dir / "per_query"
    @property
    def clusters_dir(self) -> Path: return self.work_dir / "clusters"
    @property
    def method_a_out(self) -> Path: return self.work_dir / "agent_io/method_a/output"
    @property
    def method_b_out(self) -> Path: return self.work_dir / "agent_io/method_b/output"
    @property
    def submission_dir(self) -> Path: return self.work_dir / "submission"
    @property
    def raw_a_dir(self) -> Path: return self.work_dir / "raw_llm/method_a"
    @property
    def raw_b_dir(self) -> Path: return self.work_dir / "raw_llm/method_b"

    def ensure_dirs(self):
        for d in (self.per_query_dir, self.clusters_dir, self.method_a_out,
                  self.method_b_out, self.submission_dir,
                  self.raw_a_dir, self.raw_b_dir):
            d.mkdir(parents=True, exist_ok=True)
