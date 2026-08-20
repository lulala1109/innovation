import pandas as pd
import numpy as np
from pathlib import Path


#阶段一数据集划分为训练集和测试集



INPUT_PATH = Path(
    "dataset/processed/stage1/jbb_pairs_base.csv"
)

OUTPUT_PATH = Path(
    "dataset/processed/stage1/jbb_pairs_split.csv"
)

SEED = 42
VAL_PER_CATEGORY = 2


# =========================
# 1. 读取数据
# =========================
df = pd.read_csv(INPUT_PATH)

print("Total rows:", len(df))
print("Unique pair_id:", df["pair_id"].nunique())


# =========================
# 2. 基础检查
# =========================
assert len(df) == 100, "Expected 100 JBB pairs."
assert df["pair_id"].nunique() == 100, "pair_id is not unique."
assert df["category"].notna().all(), "Missing category found."


print("\n=== Category counts before split ===")
print(df["category"].value_counts().sort_index())


# =========================
# 3. 按 Category 分层划分
# =========================
rng = np.random.default_rng(SEED)

df["measurement_split"] = "measurement_train"

for category, group in df.groupby("category", sort=True):

    if len(group) < VAL_PER_CATEGORY:
        raise ValueError(
            f"Category '{category}' has too few samples: {len(group)}"
        )

    val_indices = rng.choice(
        group.index.to_numpy(),
        size=VAL_PER_CATEGORY,
        replace=False,
    )

    df.loc[
        val_indices,
        "measurement_split"
    ] = "measurement_val"


# =========================
# 4. 检查整体数量
# =========================
split_counts = df["measurement_split"].value_counts()

print("\n=== Overall split ===")
print(split_counts)

assert split_counts["measurement_train"] == 80
assert split_counts["measurement_val"] == 20


# =========================
# 5. 检查每个 Category 是否 8/2
# =========================
category_split = pd.crosstab(
    df["category"],
    df["measurement_split"]
)

print("\n=== Split by category ===")
print(category_split)

assert (
    category_split["measurement_train"] == 8
).all()

assert (
    category_split["measurement_val"] == 2
).all()


# =========================
# 6. 检查 pair_id 无重复
# =========================
train_ids = set(
    df.loc[
        df["measurement_split"] == "measurement_train",
        "pair_id"
    ]
)

val_ids = set(
    df.loc[
        df["measurement_split"] == "measurement_val",
        "pair_id"
    ]
)

assert train_ids.isdisjoint(val_ids)


# =========================
# 7. 保存
# =========================
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nPASS: measurement split created successfully.")
print("Output:", OUTPUT_PATH)