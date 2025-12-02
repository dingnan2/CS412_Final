"""
Unified entry point for the CS 412 Research Project pipeline.

This script provides a simple way to run the full pipeline or selected phases.

Phases overview:
  1. Data preprocessing        -> clean raw Yelp JSON to CSV
  2. Exploratory data analysis -> EDA plots + EDA report
  3. Feature engineering       -> temporal features + user credibility (2012–2020)
  4. Temporal validation       -> infer labels + temporal model evaluation (12m)
  5. Baseline models           -> Logistic/DT/RF with temporal split
  6. Advanced models           -> XGBoost/LightGBM/MLP + ensembles
  7. Ablation study            -> feature category ablation/additive
  8. Case study                -> TP/TN/FP/FN case analysis
  9. Parameter study           -> hyperparameter sensitivity analysis
 10. Final report generation   -> aggregate all results into docs/*

IMPORTANT: All modeling phases (5-9) use the SAME data file and split configuration
           from config.py to ensure consistent and comparable results.

Usage examples (run from project root):
  - Run the full pipeline:
      python main.py
      python main.py --all

  - Run specific phases (e.g., 3 -> 4 -> 6 only):
      python main.py --phase 3 4 6
"""

from __future__ import annotations

import sys, io, os

if os.name == "nt":  # 只在 Windows 下改
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

# Ensure src/ is importable as top-level package (data_processing, models, etc.)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import unified configuration
from config import DATA_PATHS, SPLIT_CONFIG, RANDOM_STATE


def run_phase_1() -> None:
    """Phase 1: Data preprocessing."""
    from data_processing.data_preprocessing import main as preprocessing_main

    preprocessing_main()


def run_phase_2() -> None:
    """Phase 2: Exploratory data analysis (EDA)."""
    from data_processing.EDA_analysis import main as eda_main

    eda_main()


def run_phase_3() -> None:
    """Phase 3: Feature engineering (temporal mode, 2012–2020)."""
    from feature_engineering.feature_eng import FeatureEngineer

    engineer = FeatureEngineer(
        use_temporal_validation=True,
        prediction_years=list(range(2012, 2021)),
    )
    engineer.run_pipeline()


def run_phase_4() -> None:
    """Phase 4: Temporal validation (12‑month prediction window).
    
    This phase generates labels and saves to business_features_temporal_labeled_12m.csv
    which is then used by all subsequent modeling phases.
    """
    from models.temporal_validation import TemporalValidator

    validator = TemporalValidator(
        features_path=DATA_PATHS['features_temporal']
    )
    validator.run_pipeline(prediction_window_months=12)


def run_phase_5() -> None:
    """Phase 5: Baseline models (Logistic/DT/RF with temporal split).
    
    CRITICAL: Uses the SAME labeled data file as Phase 6-9 for consistency.
    Split configuration comes from config.SPLIT_CONFIG.
    """
    from models.baseline_models import BaselineModelPipeline

    pipeline = BaselineModelPipeline(
        data_path=DATA_PATHS['model_data'],  # UNIFIED: Same as Phase 6-9
        output_path="src/models",
        random_state=RANDOM_STATE,
        use_temporal_split=True,
    )
    pipeline.run_pipeline()


def run_phase_6() -> None:
    """Phase 6: Advanced models (XGBoost/LightGBM/MLP + ensembles).
    
    Uses the same data and split as Phase 5 for fair comparison.
    """
    from models.advanced_models import AdvancedModelPipeline

    pipeline = AdvancedModelPipeline(
        data_path=DATA_PATHS['model_data'],  # UNIFIED: Same as Phase 5
        output_path="src/models/advanced_models",
        random_state=RANDOM_STATE,
        use_temporal_split=True,
        handle_covid=True,
        tune_hyperparameters=True,  # Set to False for faster runs; True for best results
    )
    pipeline.run_pipeline()


def run_phase_7() -> None:
    """Phase 7: Ablation study (feature categories, user credibility).
    
    Uses the same data and split as Phase 5-6 for consistency.
    """
    from evaluation.ablation_study import AblationStudy

    study = AblationStudy(
        data_path=DATA_PATHS['model_data'],  # UNIFIED
        output_path="src/evaluation/ablation_study",
        random_state=RANDOM_STATE,
    )
    study.run_pipeline()


def run_phase_8() -> None:
    """Phase 8: Case study (TP/TN/FP/FN detailed analysis).
    
    Uses the same data and split as Phase 5-7 for consistency.
    """
    from evaluation.case_study import CaseStudyAnalyzer

    analyzer = CaseStudyAnalyzer(
        data_path=DATA_PATHS['model_data'],  # UNIFIED
        business_path=DATA_PATHS['business_clean'],
        model_path=None,
        output_path="src/evaluation/case_study",
        random_state=RANDOM_STATE,
        use_shap=False,  # Disabled: SHAP is too slow (25+ min) for large test sets
                         # Using deviation-based case-specific analysis instead
    )
    analyzer.run_pipeline(n_cases_per_type=5)


def run_phase_9() -> None:
    """Phase 9: Parameter study (hyperparameter sensitivity analysis).
    
    Uses the same data and split as Phase 5-8 for consistency.
    """
    from evaluation.parameter_study import ParameterStudy

    study = ParameterStudy(
        data_path=DATA_PATHS['model_data'],  # UNIFIED
        output_path="src/evaluation/parameter_study",
        random_state=RANDOM_STATE,
    )
    study.run_pipeline()


def run_phase_10() -> None:
    """Phase 10: Final report generation (docs/final_report.* + figures)."""
    from reporting.generate_final_report import FinalReportGenerator

    generator = FinalReportGenerator(
        output_path="docs",
        figures_path="docs/figures",
    )
    generator.run_pipeline()


PHASE_RUNNERS: Dict[int, Callable[[], None]] = {
    1: run_phase_1,
    2: run_phase_2,
    3: run_phase_3,
    4: run_phase_4,
    5: run_phase_5,
    6: run_phase_6,
    7: run_phase_7,
    8: run_phase_8,
    9: run_phase_9,
    10: run_phase_10,
}

PHASE_DESCRIPTIONS: Dict[int, str] = {
    1: "Data preprocessing (clean raw Yelp JSON to CSV)",
    2: "Exploratory data analysis (EDA)",
    3: "Feature engineering (temporal, user credibility, feature interactions, 2012–2020)",
    4: "Temporal validation (label inference + temporal split + temporal CV, 12m)",
    5: "Baseline models (Logistic/DecisionTree/RandomForest)",
    6: "Advanced models (XGBoost/LightGBM/MLP + ensembles + statistical testing)",
    7: "Ablation study (feature category, VIF analysis, correlation analysis)",
    8: "Case study (TP/TN/FP/FN examples + quantitative error analysis)",
    9: "Parameter study (hyperparameter sensitivity analysis)",
    10: "Final report generation (docs/final_report.* and figures)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CS 412 Research Project pipeline (full or selected phases)."
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        type=int,
        choices=range(1, 11),
        help="Phase numbers to run (1–10). If omitted, all phases run in order.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all phases (1–9) in order (default if --phase not specified).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.phase:
        phases: List[int] = sorted(set(args.phase))
    else:
        # Default: run all phases if nothing specified or --all provided
        phases = list(range(1, 11))

    print("=" * 70)
    print("CS 412 RESEARCH PROJECT - UNIFIED PIPELINE RUNNER")
    print("=" * 70)
    print("Selected phases:\n")
    for p in phases:
        desc = PHASE_DESCRIPTIONS.get(p, "")
        print(f"  {p}: {desc}")
    print("")

    for p in phases:
        runner = PHASE_RUNNERS.get(p)
        if runner is None:
            print(f"[WARN] No runner defined for phase {p}, skipping.")
            continue

        print("\n" + "=" * 70)
        print(f"PHASE {p}: {PHASE_DESCRIPTIONS.get(p, '')}")
        print("=" * 70)
        try:
            runner()
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Phase {p} failed with exception: {e}")
            # Stop the pipeline on first hard failure
            break

    print("\n" + "=" * 70)
    print("PIPELINE RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


