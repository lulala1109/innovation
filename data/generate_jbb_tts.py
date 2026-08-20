"""
Generate TTS audio for JBB Stage1 measurement data.

Default behavior:
    Generate only the first 2 pairs (4 WAV files) for testing.

Input:
    dataset/processed/stage1/jbb_pairs_split.csv

Output:
    dataset/derived/jbb_audio/benign/
    dataset/derived/jbb_audio/harmful_clean/

After generation:
    dataset/processed/stage1/jbb_pairs_audio.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import argparse
import os
import sys
import time
import pandas as pd
import soundfile as sf
from tqdm import tqdm


# ============================================================
# Make project root importable even when running:
# python .\data\generate_jbb_tts.py
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.audio import DEFAULT_SAMPLE_RATE, generate_tts


# ============================================================
# Paths
# ============================================================

INPUT_CSV = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "stage1"
    / "jbb_pairs_split.csv"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "stage1"
    / "jbb_pairs_audio.csv"
)

AUDIO_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "derived"
    / "jbb_audio"
)

BENIGN_DIR = AUDIO_ROOT / "benign"
HARMFUL_DIR = AUDIO_ROOT / "harmful_clean"


SAMPLE_RATE = DEFAULT_SAMPLE_RATE

# gTTS is an online service, so a failed request can be retried.
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2


# ============================================================
# Utility functions
# ============================================================

def generate_one_audio(
    text: str,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """
    Generate one 16 kHz mono PCM16 WAV file using the project's
    existing core.audio.generate_tts().
    """

    if output_path.exists() and not overwrite:
        validate_audio(output_path)
        return

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty TTS text for: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Existing project function:
            # gTTS -> temporary MP3 -> librosa -> mono waveform -> normalization
            wav = generate_tts(
                text=text.strip(),
                sample_rate=SAMPLE_RATE,
                output_path=None,
            )

            wav_np = wav.squeeze(0).detach().cpu().numpy()

            # Explicitly save as WAV PCM16.
            sf.write(
                str(output_path),
                wav_np,
                SAMPLE_RATE,
                subtype="PCM_16",
            )

            validate_audio(output_path)
            return

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                print(
                    f"\nRetry {attempt}/{MAX_RETRIES} "
                    f"for {output_path.name}: {exc}"
                )
                time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(
        f"Failed to generate {output_path} "
        f"after {MAX_RETRIES} attempts."
    ) from last_error


def validate_audio(path: Path) -> None:
    """
    Validate generated WAV properties.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Audio was not created: {path}")

    info = sf.info(str(path))

    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"{path.name}: expected {SAMPLE_RATE} Hz, "
            f"got {info.samplerate} Hz"
        )

    if info.channels != 1:
        raise ValueError(
            f"{path.name}: expected mono audio, "
            f"got {info.channels} channels"
        )

    if info.frames <= 0:
        raise ValueError(
            f"{path.name}: audio contains no samples"
        )


def relative_path(path: Path) -> str:
    """
    Store paths relative to project root using forward slashes.
    """
    return path.relative_to(PROJECT_ROOT).as_posix()


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate gTTS audio for JBB measurement pairs."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help=(
            "Number of JBB pairs to generate. "
            "Default: 2 pairs = 4 audio files."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all JBB pairs.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate WAV files that already exist.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 1. Check input
    # --------------------------------------------------------

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Input CSV does not exist:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "pair_id",
        "jbb_index",
        "benign_text",
        "harmful_text",
        "measurement_split",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair_id found.")

    if df["benign_text"].isna().any():
        raise ValueError("Missing benign_text found.")

    if df["harmful_text"].isna().any():
        raise ValueError("Missing harmful_text found.")

    # Keep stable JBB order.
    df = df.sort_values(
        "jbb_index"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 2. Decide how many pairs to generate
    # --------------------------------------------------------

    if args.all:
        selected = df.copy()
    else:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than 0.")

        selected = df.head(args.limit).copy()

    BENIGN_DIR.mkdir(parents=True, exist_ok=True)
    HARMFUL_DIR.mkdir(parents=True, exist_ok=True)

    print("=== JBB TTS generation ===")
    print("Input:", INPUT_CSV)
    print("Pairs selected:", len(selected))
    print("Expected WAV files:", len(selected) * 2)
    print("Sample rate:", SAMPLE_RATE)
    print("Benign dir:", BENIGN_DIR)
    print("Harmful dir:", HARMFUL_DIR)
    print()

    # --------------------------------------------------------
    # 3. Generate audio
    # --------------------------------------------------------

    failures = []

    for _, row in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc="Generating JBB audio",
    ):
        pair_id = str(row["pair_id"])

        benign_path = BENIGN_DIR / f"{pair_id}.wav"
        harmful_path = HARMFUL_DIR / f"{pair_id}.wav"

        try:
            generate_one_audio(
                text=str(row["benign_text"]),
                output_path=benign_path,
                overwrite=args.overwrite,
            )

            generate_one_audio(
                text=str(row["harmful_text"]),
                output_path=harmful_path,
                overwrite=args.overwrite,
            )

        except Exception as exc:
            failures.append(
                {
                    "pair_id": pair_id,
                    "error": str(exc),
                }
            )

            print(
                f"\nFAILED: {pair_id}\n"
                f"{exc}"
            )

    # --------------------------------------------------------
    # 4. Build/update audio manifest
    # --------------------------------------------------------

    df["benign_audio_path"] = ""
    df["harmful_audio_path"] = ""
    df["tts_model"] = "gTTS-en"
    df["audio_sample_rate"] = SAMPLE_RATE
    df["tts_status"] = "not_generated"

    for index, row in df.iterrows():
        pair_id = str(row["pair_id"])

        benign_path = BENIGN_DIR / f"{pair_id}.wav"
        harmful_path = HARMFUL_DIR / f"{pair_id}.wav"

        benign_exists = benign_path.is_file()
        harmful_exists = harmful_path.is_file()

        if benign_exists:
            df.at[index, "benign_audio_path"] = relative_path(
                benign_path
            )

        if harmful_exists:
            df.at[index, "harmful_audio_path"] = relative_path(
                harmful_path
            )

        if benign_exists and harmful_exists:
            df.at[index, "tts_status"] = "complete"

        elif benign_exists or harmful_exists:
            df.at[index, "tts_status"] = "partial"

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 5. Final report
    # --------------------------------------------------------

    completed_pairs = (
        df["tts_status"] == "complete"
    ).sum()

    print("\n=== TTS generation finished ===")
    print("Complete pairs:", completed_pairs)
    print("Failed pairs:", len(failures))
    print("Manifest:", OUTPUT_MANIFEST)

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(
                failure["pair_id"],
                "->",
                failure["error"],
            )
        raise SystemExit(1)

    print("\nPASS: selected JBB audio generated successfully.")


if __name__ == "__main__":
    main()