import pickle
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ExperimentConfig:
    """
    Config for attacks
    """
    experiment_name: str
    """Name for the experiment"""
    base_model: str
    """Base model name"""
    dataset_member: str
    """Dataset source for members"""
    dataset_nonmember: str
    """Dataset source for nonmembers"""
    output_name: Optional[str] = None
    """Output name for sub-directory."""
    dataset_nonmember_other_sources: Optional[List[str]] = field(
        default_factory=lambda: None
    )
    """Dataset sources for nonmembers for which metrics will be computed, using the thresholds 
    derived from the main member/nonmember datasets"""
    pretokenized: Optional[bool] = False
    """Is the data already pretokenized"""
    revision: Optional[str] = None
    """Model revision to use"""
    presampled_dataset_member: Optional[str] = None
    """Path to presampled dataset source for members"""
    presampled_dataset_nonmember: Optional[str] = None
    """Path to presampled dataset source for non-members"""
    token_frequency_map: Optional[str] = None
    """Path to a pre-computed token frequency map"""
    dataset_key: Optional[str] = None
    """Dataset key"""
    specific_source: Optional[str] = None
    """Specific sub-source to focus on. Only valid for the_pile"""
    full_doc: Optional[bool] = False
    """Determines whether MIA will be performed over entire doc or not"""
    max_substrs: Optional[int] = 20
    """If full_doc, determines the maximum number of sample substrs to evaluate on"""
    dump_cache: Optional[bool] = False
    """Dump data to cache? Exits program after dumping"""
    load_from_cache: Optional[bool] = False
    """Load data from cache?"""
    load_from_hf: Optional[bool] = True
    """Load data from HuggingFace?"""
    blackbox_attacks: Optional[List[str]] = field(
        default_factory=lambda: None
    )
    """List of attacks to evaluate"""
    tokenization_attack: Optional[bool] = False
    """Run tokenization attack?"""
    quantile_attack: Optional[bool] = False
    """Run quantile attack?"""
    n_samples: Optional[int] = 200
    """Number of records (member and non-member each) to run the attack(s) for"""
    max_tokens: Optional[int] = 512
    """Consider samples with at most these many tokens"""
    max_data: Optional[int] = 5_000
    """Maximum samples to load from data before processing. Helps with efficiency"""
    min_words: Optional[int] = 100
    """Consider documents with at least these many words"""
    max_words: Optional[int] = 200
    """Consider documents with at most these many words"""
    max_words_cutoff: Optional[bool] = True
    """Is max_words a selection criteria (False), or a cutoff added on text (True)?"""
    batch_size: Optional[int] = 50
    """Batch size"""
    chunk_size: Optional[int] = 20
    """Chunk size"""
    scoring_model_name: Optional[str] = None
    """Scoring model (if different from base model)"""
    top_k: Optional[int] = 40
    """Consider only top-k tokens"""
    do_top_k: Optional[bool] = False
    """Use top-k sampling?"""
    top_p: Optional[float] = 0.96
    """Use tokens (minimal set) with cumulative probability of <=top_p"""
    do_top_p: Optional[bool] = False
    """Use top-p sampling?"""
    pre_perturb_pct: Optional[float] = 0.0
    """Percentage of tokens to perturb before attack"""
    pre_perturb_span_length: Optional[int] = 5
    """Span length for pre-perturbation"""
    tok_by_tok: Optional[bool] = False
    """Process data token-wise?"""
    fpr_list: Optional[List[float]] = field(default_factory=lambda: [0.001, 0.01])
    """FPRs at which to compute TPR"""
    random_seed: Optional[int] = 0
    """Random seed"""
    ref_config: Optional[str] = None
    """Reference model config"""
    recall_config: Optional[str] = None
    """ReCaLL attack config"""
    neighborhood_config: Optional[str] = None
    """Neighborhood attack config"""
    env_config: Optional[str] = None
    """Environment config"""
    openai_config: Optional[str] = None
    """OpenAI config"""

    def __post_init__(self):
        if self.dump_cache and (self.load_from_cache or self.load_from_hf):
            raise ValueError("Cannot dump and load cache at the same time")

    def save(self, filepath: str) -> None:
        """Saves the config to a file using pickle."""
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath: str) -> 'ExperimentConfig':
        """Loads the config from a file using pickle."""
        with open(filepath, 'rb') as file:
            return pickle.load(file)
