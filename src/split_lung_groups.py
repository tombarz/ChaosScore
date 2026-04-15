from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("data") / "processed" / "lung"


def format_age_label(age_cutoff: float) -> str:
    return str(int(age_cutoff)) if float(age_cutoff).is_integer() else str(age_cutoff).replace(".", "_")


def build_healthy_mask(
    obs: pd.DataFrame,
    *,
    age_cutoff: float,
    age_column: str,
    smoking_column: str,
    bmi_column: str,
    disease_column: str,
    never_smoking_value: str,
    normal_disease_value: str,
    bmi_low: float,
    bmi_high: float,
) -> pd.Series:
    age = pd.to_numeric(obs[age_column], errors="coerce")
    bmi = pd.to_numeric(obs[bmi_column], errors="coerce")
    smoking = obs[smoking_column].astype("string").str.strip().str.lower()
    disease = obs[disease_column].astype("string").str.strip().str.lower()

    return (
        age.le(age_cutoff)
        & smoking.eq(never_smoking_value.strip().lower())
        & bmi.ge(bmi_low)
        & bmi.lt(bmi_high)
        & disease.eq(normal_disease_value.strip().lower())
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_subset(adata: ad.AnnData, output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / "subset.h5ad"
    adata.write_h5ad(subset_path)
    write_json(output_dir / "summary.json", summary | {"output_h5ad": str(subset_path.resolve())})


def split_lung_groups(
    *,
    input_h5ad: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    age_cutoff: float = 30,
    age_column: str = "age_or_mean_of_age_range",
    smoking_column: str = "smoking_status",
    bmi_column: str = "BMI",
    disease_column: str = "disease",
    never_smoking_value: str = "never",
    normal_disease_value: str = "normal",
    bmi_low: float = 18.5,
    bmi_high: float = 25.0,
) -> None:
    input_path = Path(input_h5ad)
    if not input_path.exists():
        raise FileNotFoundError(f"Input H5AD file not found: {input_path}")

    output_root = Path(output_root)
    age_label = format_age_label(age_cutoff)
    healthy_dir = output_root / f"{age_label}_non_smoking_normal_healthy"
    risk_dir = output_root / f"older_than_{age_label}_smoked_normal_risk_group"

    adata = ad.read_h5ad(input_path)
    healthy_mask = build_healthy_mask(
        adata.obs,
        age_cutoff=age_cutoff,
        age_column=age_column,
        smoking_column=smoking_column,
        bmi_column=bmi_column,
        disease_column=disease_column,
        never_smoking_value=never_smoking_value,
        normal_disease_value=normal_disease_value,
        bmi_low=bmi_low,
        bmi_high=bmi_high,
    )
    risk_mask = ~healthy_mask

    healthy_adata = adata[healthy_mask.to_numpy()].copy()
    risk_adata = adata[risk_mask.to_numpy()].copy()

    criteria = {
        "age_column": age_column,
        "age_cutoff": age_cutoff,
        "smoking_column": smoking_column,
        "never_smoking_value": never_smoking_value,
        "bmi_column": bmi_column,
        "bmi_low": bmi_low,
        "bmi_high_exclusive": bmi_high,
        "disease_column": disease_column,
        "normal_disease_value": normal_disease_value,
    }

    save_subset(
        healthy_adata,
        healthy_dir,
        {
            "group": "healthy",
            "cells": int(healthy_adata.n_obs),
            "genes": int(healthy_adata.n_vars),
            "input_h5ad": str(input_path.resolve()),
            "criteria": criteria,
        },
    )
    save_subset(
        risk_adata,
        risk_dir,
        {
            "group": "risk_group",
            "cells": int(risk_adata.n_obs),
            "genes": int(risk_adata.n_vars),
            "input_h5ad": str(input_path.resolve()),
            "selection_rule": "complement of healthy group",
            "criteria": criteria,
        },
    )

    print("Saved lung subsets")
    print(f"Healthy group: {healthy_dir / 'subset.h5ad'} ({healthy_adata.n_obs} cells)")
    print(f"Risk group: {risk_dir / 'subset.h5ad'} ({risk_adata.n_obs} cells)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a lung .h5ad into a young non-smoking normal-BMI healthy subset and its complement."
    )
    parser.add_argument("--input_h5ad", required=True, help="Input .h5ad file")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--age_cutoff", type=float, default=30)
    parser.add_argument("--age_column", default="age_or_mean_of_age_range")
    parser.add_argument("--smoking_column", default="smoking_status")
    parser.add_argument("--bmi_column", default="BMI")
    parser.add_argument("--disease_column", default="disease")
    parser.add_argument("--never_smoking_value", default="never")
    parser.add_argument("--normal_disease_value", default="normal")
    parser.add_argument("--bmi_low", type=float, default=18.5)
    parser.add_argument("--bmi_high", type=float, default=25.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_lung_groups(
        input_h5ad=args.input_h5ad,
        output_root=args.output_root,
        age_cutoff=args.age_cutoff,
        age_column=args.age_column,
        smoking_column=args.smoking_column,
        bmi_column=args.bmi_column,
        disease_column=args.disease_column,
        never_smoking_value=args.never_smoking_value,
        normal_disease_value=args.normal_disease_value,
        bmi_low=args.bmi_low,
        bmi_high=args.bmi_high,
    )


if __name__ == "__main__":
    main()
