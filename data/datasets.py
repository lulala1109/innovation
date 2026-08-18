"""Dataset loading and normalization for safety-state experiments.

Local tabular files are the primary interface.  Hugging Face loading is an
optional convenience and is imported lazily so importing this module (or
asking a CLI for ``--help``) never requires ``datasets`` or network access.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable, Union

import pandas as pd


TablePath = Union[str, Path]
HuggingFaceLoader = Callable[..., Any]


class DatasetSchemaError(ValueError):
    """Raised when an input table cannot be normalized safely."""


# Keep the ordering stable: ``assign_stratum`` intentionally returns the first
# matching category, matching the behavior of the legacy AdvBench utility.
STRATA_PATTERNS: dict[str, tuple[str, ...]] = {
    "violence": (
        r"\b(kill|murder|assassinat(e|ion)|bomb|violence|bombs|disrupt(?:ing)?|weapon|weapons|explosive|shoot|gun|firearm|terrorist|massacre|assault|hit-and-run)\b",
        r"\b(physical harm|hijack.*car|plant a bomb|poison\w* (a )?(person|someone|individual|people|water supply)|sabotage|mass shooting|terrorist attack|violent crime)\b",
        r"\b(disrupt.*\b(systems?|infrastructures?)|eco-terrorism|harm animals|start a fire|abduction|torture)\b",
        r"\b(interfere with.*\b(systems?|infrastructures?))\b",
    ),
    "illicit_cyber": (
        r"\b(hack|phish|malware|virus|ransomware|backdoor|crack(?:ing)? passwords?|passwords|keylogger|spyware|botnet|DDoS|spoof|cyber attack)\b",
        r"\b(steal.*(data|information|credentials))|(exploit.*vulnerabilit)|(hijack.*(car|drone|system|website|webcam))\b",
        r"\b(bypass security|disable antivirus|secure network|government database|spy on people|(?:track|monitor|spy on|surveil)\b.*\b(location|whereabouts|movements)|collects personal data)\b",
        r"\b(leaks confidential data|accesses and manipulates.*data|crash a computer system|denies access to a website|deletes files from.*computer)\b",
        r"(tutorial|guide|instructions|automates).*(hacking|virus|cyberattack)",
        r"\b(spam|harvest.*data|circumvent.*(censorship|surveillance|digital rights management))\b",
    ),
    "harmful_speech": (
        r"\b(hate speech|defam\w*|slander|libel|propaganda|incit(e|ing) violence|radicalization|glorif\w* (?:violence|war|gang violence))\b",
        r"\b(manipulat(e|ion)|trick people|cyberbully|harass|stalk|threaten|blackmail|false accusations|bullying|inflammatory comments)\b",
        r"\b(dangerous (?:behaviors|stunts|challenge)|(?:drug|alcohol) use|reckless driving|eating disorder)\b",
        r"\b(exploit.*(vulnerable populations|insecurities|fears)|promotes.*(drugs|steroids|dangerous dieting)|encourage.*commit crimes)\b",
        r"(teach|guide|tutorial|instructions).*(cyberbullying|cyberstalking)",
    ),
    "hate": (
        r"\b(racist|nazi|supremacist|bigot|slur|xenophobic|homophobic|sexist)\b",
        r"\b(violence against.*(group|minority|women))|(oppression of.*group)|(dehumanization)|(inferior to others)|(threatening letter to.*minority)\b",
    ),
    "fraud": (
        r"\b(counterfeit|insider trading|steal.*(?:money|identity)|fraud|phishing|scam)\b",
        r"\b(credit card fraud|financial crimes|tax fraud|insurance fraud|mail fraud|falsify.*records)\b",
        r"(tutorial|guide|instructions|how to|educate).*(identity theft|credit card)",
        r"\b(?:steal|theft|obtain|use).*credit card(?: information)?\b",
        r"\bmanipulate.*financial market\b",
    ),
    "deception_rule_evasion": (
        r"\b(cheat on.*(exam|test))|(impersonat)|(fake (review|ID|identity|news|online store|product review))\b",
        r"\b(evad(e|ing) (?:law|taxes|detection|security))|(get away with)|(cover your tracks|concealing criminal activity)\b",
        r"\b(lie to|mislead|cheat(?:ing)|deceive|trick|review.*(not even used|without.*(using|testing))|manipulates customer ratings|automates.*fake reviews)\b",
        r"\b(circumvent.*government censorship)\b",
    ),
    "illegal_acts_goods": (
        r"\b(illegal drugs|smuggl(e|ing)|trafficking|poach|fake id|counterfeit money|illegal download|piracy|pirate|pirated content)\b",
        r"\b(steal.*(?:car|store|charity))|(pick a lock|vandalize|trespass|rob a bank|escape from.*(prison|correctional facility))|(infiltrate.*facility)\b",
        r"\b(synthesized at home|homemade firearms|date rape drug|acquire.*firearms illegally|teach.*firearms)\b",
        r"(tutorial|guide|instructions).*(illegal.*download|counterfeit.*money|fake id)",
    ),
    "self_harm": (
        r"\b(kill myself|suicide|self[- ]harm|end my life|how to die|anorexia|bulimia|eating disorder)\b",
    ),
    "sexual_exploit": (
        r"\b(sexual.*minors?|child(?:ren)? porn|child pornography|exploit(?:ing)? minor|groom|grooming|sex|exploitation|molest|child abuse)\b",
        r"\b(acquire.*child porn|solicit sexual conversations.*minor|promotes.*child pornography)\b",
    ),
    "misinformation": (
        r"\b(fake news|false information|misinformation|conspiracy theor(y|ies))\b",
        r"(deadly side effects.*vaccine)|(climate change.*hoax)",
        r"\b(mislead.*public about.*(climate change|healthcare))\b",
    ),
}

COMPILED_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    name: tuple(re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns)
    for name, patterns in STRATA_PATTERNS.items()
}


def assign_stratum(prompt: object) -> str:
    """Return the first matching AdvBench harm stratum, or ``"other"``."""

    text = "" if prompt is None or pd.isna(prompt) else str(prompt)
    for name, patterns in COMPILED_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            return name
    return "other"


def read_table(path: TablePath) -> pd.DataFrame:
    """Read a local CSV/TSV/JSON(L)/Parquet table.

    The format is selected from the filename suffix.  A copy is returned so
    callers can normalize it without mutating a cached DataFrame.
    """

    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(f"Input table does not exist: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(table_path)
    elif suffix in {".tsv", ".tab"}:
        frame = pd.read_csv(table_path, sep="\t")
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(table_path, lines=True)
    elif suffix == ".json":
        frame = pd.read_json(table_path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(table_path)
    else:
        raise ValueError(
            f"Unsupported table format {suffix!r}; use CSV, TSV, JSONL, JSON, or Parquet"
        )

    if frame.empty:
        raise DatasetSchemaError(f"Input table is empty: {table_path}")
    return frame.copy()


def write_table(frame: pd.DataFrame, path: TablePath) -> None:
    """Write a DataFrame using the same local formats accepted by ``read_table``."""

    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(table_path, index=False)
    elif suffix in {".tsv", ".tab"}:
        frame.to_csv(table_path, sep="\t", index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        frame.to_json(table_path, orient="records", lines=True, force_ascii=False)
    elif suffix == ".json":
        frame.to_json(table_path, orient="records", indent=2, force_ascii=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(table_path, index=False)
    else:
        raise ValueError(
            f"Unsupported table format {suffix!r}; use CSV, TSV, JSONL, JSON, or Parquet"
        )


def _resolve_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    columns = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        resolved = columns.get(candidate.casefold())
        if resolved is not None:
            return resolved
    return None


def normalize_prompt_table(
    frame: pd.DataFrame,
    *,
    default_source: str,
    infer_strata: bool = True,
) -> pd.DataFrame:
    """Normalize common AdvBench/JBB column names to a canonical prompt table.

    The returned columns are ``prompt``, ``stratum``, ``target``, ``behavior``,
    and ``source``.  Optional values are represented by empty strings rather
    than pandas NaNs so manifests serialize predictably.
    """

    if frame.empty:
        raise DatasetSchemaError("Prompt table is empty")

    prompt_column = _resolve_column(frame, ("prompt", "harmful_text", "goal"))
    if prompt_column is None:
        raise DatasetSchemaError(
            "Prompt table needs one of: prompt, harmful_text, Goal"
        )

    stratum_column = _resolve_column(frame, ("stratum", "category"))
    target_column = _resolve_column(frame, ("target",))
    behavior_column = _resolve_column(frame, ("behavior",))
    source_column = _resolve_column(frame, ("source",))

    output = pd.DataFrame(index=frame.index)
    output["prompt"] = frame[prompt_column].fillna("").astype(str).str.strip()
    if (output["prompt"] == "").any():
        bad_rows = output.index[output["prompt"] == ""].tolist()[:5]
        raise DatasetSchemaError(f"Prompt is blank at row(s): {bad_rows}")

    if stratum_column is not None:
        output["stratum"] = (
            frame[stratum_column].fillna("").astype(str).str.strip()
        )
    else:
        output["stratum"] = ""
    if infer_strata:
        missing = output["stratum"] == ""
        output.loc[missing, "stratum"] = output.loc[missing, "prompt"].map(
            assign_stratum
        )
    elif (output["stratum"] == "").any():
        raise DatasetSchemaError("Prompt table is missing a stratum/category value")

    output["target"] = (
        frame[target_column].fillna("").astype(str).str.strip()
        if target_column is not None
        else ""
    )
    output["behavior"] = (
        frame[behavior_column].fillna("").astype(str).str.strip()
        if behavior_column is not None
        else ""
    )
    output["source"] = (
        frame[source_column].fillna("").astype(str).str.strip()
        if source_column is not None
        else default_source
    )
    output.loc[output["source"] == "", "source"] = default_source
    return output.reset_index(drop=True)


def _default_hf_loader() -> HuggingFaceLoader:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Hugging Face loading requires the optional 'datasets' package; "
            "local CSV/TSV/JSONL inputs do not require it"
        ) from exc
    return load_dataset


def _hf_to_frame(dataset: Any) -> pd.DataFrame:
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    try:
        return pd.DataFrame(dataset)
    except Exception as exc:  # pragma: no cover - defensive for third-party types
        raise DatasetSchemaError("Could not convert Hugging Face dataset to a table") from exc


def load_advbench(
    local_path: TablePath | None = None,
    *,
    split: str = "train",
    hf_loader: HuggingFaceLoader | None = None,
) -> pd.DataFrame:
    """Load AdvBench from a local table, or optionally from Hugging Face."""

    if local_path is not None:
        raw = read_table(local_path)
    else:
        loader = hf_loader or _default_hf_loader()
        resolved_split = "train+test+validation" if split == "all" else split
        raw = _hf_to_frame(loader("walledai/AdvBench", split=resolved_split))
    return normalize_prompt_table(
        raw, default_source="walledai/AdvBench", infer_strata=True
    )


def load_jbb(
    local_path: TablePath | None = None,
    *,
    split: str = "harmful",
    hf_loader: HuggingFaceLoader | None = None,
) -> pd.DataFrame:
    """Load JBB Behaviors from a local table, or optionally from Hugging Face."""

    if split not in {"harmful", "benign"}:
        raise ValueError("JBB split must be 'harmful' or 'benign'")
    if local_path is not None:
        raw = read_table(local_path)
    else:
        loader = hf_loader or _default_hf_loader()
        raw = _hf_to_frame(
            loader("JailbreakBench/JBB-Behaviors", "behaviors", split=split)
        )
    return normalize_prompt_table(
        raw,
        default_source=f"JailbreakBench/JBB-Behaviors:{split}",
        infer_strata=True,
    )


__all__ = [
    "COMPILED_PATTERNS",
    "DatasetSchemaError",
    "STRATA_PATTERNS",
    "assign_stratum",
    "load_advbench",
    "load_jbb",
    "normalize_prompt_table",
    "read_table",
    "write_table",
]
