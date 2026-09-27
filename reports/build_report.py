"""Builds reports/freight_rate_prediction_report.pdf.

This is a report-generation utility only - it does not touch any ML
code, model, or prediction file. Every number and image used here comes
from README.md, the eda/plots/ charts, and scorer_results/candidate_december.png
(the official chart produced by score.py) - nothing is invented.
"""
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).parent.parent
EDA_PLOTS = PROJECT_ROOT / "eda" / "plots"
SCORER_RESULTS = PROJECT_ROOT / "scorer_results"
OUTPUT_PATH = Path(__file__).parent / "freight_rate_prediction_report.pdf"

PAGE_WIDTH, PAGE_HEIGHT = letter
CONTENT_WIDTH = PAGE_WIDTH - 1.3 * inch

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", fontSize=22, leading=26, spaceAfter=6,
                           textColor=colors.HexColor("#0B2E3C"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="ReportSubtitle", fontSize=12, leading=16, spaceAfter=18,
                           textColor=colors.HexColor("#4A5D63")))
styles.add(ParagraphStyle(name="H1", fontSize=15, leading=19, spaceBefore=14, spaceAfter=8,
                           textColor=colors.HexColor("#0B2E3C"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="H2", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4,
                           textColor=colors.HexColor("#0B2E3C"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14, spaceAfter=8,
                           textColor=colors.HexColor("#1E1E1E")))
styles.add(ParagraphStyle(name="Caption", fontSize=8.5, leading=11, spaceAfter=12,
                           textColor=colors.HexColor("#4A5D63"), fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle(name="BulletItem", fontSize=10, leading=14, spaceAfter=4,
                           leftIndent=14, bulletIndent=2, textColor=colors.HexColor("#1E1E1E")))

TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2E3C")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2D4")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F7")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
])


def para(text, style="Body"):
    return Paragraph(text, styles[style])


def bullets(items):
    return [Paragraph(f"- {item}", styles["BulletItem"]) for item in items]


def sized_image(path: Path, max_width=CONTENT_WIDTH, max_height=4.6 * inch):
    with PILImage.open(path) as im:
        w, h = im.size
    aspect = h / w
    width = max_width
    height = width * aspect
    if height > max_height:
        height = max_height
        width = height / aspect
    return Image(str(path), width=width, height=height)


def chart_block(path: Path, title: str, explanation: str, max_height=3.1 * inch):
    return [
        para(title, "H2"),
        sized_image(path, max_height=max_height),
        para(explanation, "Caption"),
    ]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#7A8B90"))
    canvas.drawString(0.65 * inch, 0.5 * inch, "Freight Rate Prediction Challenge")
    canvas.drawRightString(PAGE_WIDTH - 0.65 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


cell_header_style = ParagraphStyle(name="CellHeader", fontSize=9, leading=11.5,
                                    textColor=colors.white, fontName="Helvetica-Bold")
cell_body_style = ParagraphStyle(name="CellBody", fontSize=9, leading=11.5,
                                  textColor=colors.HexColor("#1E1E1E"), fontName="Helvetica")


def metric_table(rows, col_widths=None):
    # Wrap every cell in a Paragraph so long text wraps within its column
    # instead of overflowing into the next one.
    wrapped_rows = []
    for r_index, row in enumerate(rows):
        style = cell_header_style if r_index == 0 else cell_body_style
        wrapped_rows.append([Paragraph(str(v), style) for v in row])
    t = Table(wrapped_rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TABLE_STYLE)
    return t


def build():
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.65 * inch, rightMargin=0.65 * inch,
        title="Freight Rate Prediction - Final Report",
    )

    story = []

    # ---------------- Title ----------------
    story.append(para("Freight Rate Prediction Challenge", "ReportTitle"))
    story.append(para("Final Report - Model Development, Validation, and Results", "ReportSubtitle"))

    # ---------------- 1. Executive Summary ----------------
    story.append(para("1. Executive Summary", "H1"))
    story.append(para(
        "This project builds a machine learning model to predict truckload freight rates "
        "(<b>posted_rate</b>) from load characteristics such as pickup/delivery location, "
        "distance, equipment type, weight, and date.", "Body"))
    story.append(metric_table([
        ["Item", "Value"],
        ["Training/development rows", "48,000 (data/train-test.csv)"],
        ["Validation rows to predict", "12,000 (data/validation.csv)"],
        ["Final model", "HistGradientBoostingRegressor (tuned), log1p target"],
        ["Final validation metrics (avg. of 3 time windows)", "MAE 119.40 | RMSE 627.96 | R2 0.8260"],
        ["Final deliverables", "validation_predictions.csv, data/december_predictions.csv, "
                                "scorer_results/candidate_december.png"],
    ], col_widths=[2.3 * inch, 4.0 * inch]))

    # ---------------- 2. Problem Statement ----------------
    story.append(para("2. Problem Statement", "H1"))
    story.append(para(
        "The task is to predict the freight rate (<b>posted_rate</b>) for each of the 12,000 "
        "loads in data/validation.csv, using a model trained on 48,000 historical loads with "
        "known rates. Predictions must be delivered as validation_predictions.csv with exactly "
        "the columns load_id,predicted_rate. A second, smaller task requires predicting rates "
        "for a fixed December route across all 31 days of the month, used to visually check "
        "how the model's predictions behave over time.", "Body"))

    # ---------------- 3. Dataset Overview ----------------
    story.append(para("3. Dataset Overview", "H1"))
    story.append(metric_table([
        ["File", "Rows", "Purpose"],
        ["data/train-test.csv", "48,000", "Labeled data (has posted_rate) for training/validation"],
        ["data/validation.csv", "12,000", "Unlabeled loads to predict for final submission"],
        ["data/december-chart-inputs.csv", "31", "One fixed lane, one row per December day"],
    ], col_widths=[2.3 * inch, 0.9 * inch, 3.1 * inch]))
    story.append(Spacer(1, 6))
    story.append(para("Key columns", "H2"))
    story.append(Paragraph(
        "<br/>".join([
            "- <b>pickup / delivery</b>: origin and destination city",
            "- <b>pickup_lat/lon, delivery_lat/lon</b>: coordinates",
            "- <b>distance</b>: shipment distance",
            "- <b>equipment</b>: Dry Van, Reefer, or Flatbed",
            "- <b>weight</b>: load weight",
            "- <b>date</b>: shipment date",
            "- <b>market_index, quote_signal</b>: market-related numeric signals",
            "- <b>posted_rate</b>: the target variable (training data only)",
        ]), styles["Body"]))

    story.append(PageBreak())

    # ---------------- 4. EDA ----------------
    story.append(para("4. Exploratory Data Analysis", "H1"))
    story.append(para(
        "The charts below were generated directly from the training data "
        "(eda/generate_plots.py) using the same logic and findings from the original "
        "exploratory analysis (eda/explore_data.py).", "Body"))

    story.extend(chart_block(
        EDA_PLOTS / "distance_vs_posted_rate.png",
        "Distance vs Posted Rate",
        "Distance shows a clear, strong positive relationship with posted_rate "
        "(correlation approximately 0.91) - the strongest individual relationship found "
        "during EDA, and the main driver of model accuracy."))

    story.extend(chart_block(
        EDA_PLOTS / "posted_rate_distribution.png",
        "Posted Rate Distribution",
        "posted_rate is right-skewed: most loads sit at lower rates, with a long tail of "
        "rare, high-value loads. This motivated testing a log1p(posted_rate) target "
        "transform during modeling."))

    story.extend(chart_block(
        EDA_PLOTS / "posted_rate_by_equipment.png",
        "Posted Rate by Equipment",
        "Rate distributions differ modestly by equipment type, with overlapping ranges "
        "across Dry Van, Reefer, and Flatbed. Equipment was retained as a feature."))

    story.extend(chart_block(
        EDA_PLOTS / "weight_vs_posted_rate.png",
        "Weight vs Posted Rate",
        "The direct relationship between weight and posted_rate is weak. This chart also "
        "makes a data-quality issue clearly visible: a cluster of negative weight values, "
        "which do not make physical sense and were treated as data-entry errors."))

    story.extend(chart_block(
        EDA_PLOTS / "market_index_vs_posted_rate.png",
        "Market Index vs Posted Rate",
        "No strong direct linear relationship is visible between market_index and "
        "posted_rate, consistent with a low measured correlation. The feature was kept "
        "since tree-based models can still use non-linear signal."))

    story.extend(chart_block(
        EDA_PLOTS / "quote_signal_vs_posted_rate.png",
        "Quote Signal vs Posted Rate",
        "Similarly, quote_signal shows a weak direct linear relationship with "
        "posted_rate and was retained for the same reason."))

    story.append(PageBreak())

    story.extend(chart_block(
        EDA_PLOTS / "market_index_train_vs_validation.png",
        "Market Index: Train vs Validation Distribution",
        "Validation's market_index values are noticeably lower than training's. This is "
        "a genuine distribution shift, not a data error, and was documented rather than "
        "corrected (see Section 5)."))

    story.append(para("Key EDA findings", "H2"))
    story.extend(bullets([
        "Distance is the strongest individual predictor found in the data.",
        "posted_rate is right-skewed with a long high-rate tail.",
        "Equipment type is associated with modest differences in typical rate.",
        "Negative weight values were identified as a data-quality issue, not a real category.",
        "market_index shows a notable distribution shift between training and validation periods.",
        "Validation contains cities not present in training, so raw city names were not "
        "relied on as model features.",
    ]))

    story.append(PageBreak())

    # ---------------- 5. Data Quality and Preprocessing ----------------
    story.append(para("5. Data Quality and Preprocessing", "H1"))
    story.append(para(
        "All cleaning decisions were made using statistics computed from the training "
        "data only, then applied identically to validation and December data, to avoid "
        "leaking information from data the model should not have access to.", "Body"))
    story.extend(bullets([
        "No duplicate rows were found in the training or validation data.",
        "Missing weight values were filled using the training-data median.",
        "Negative weight values were converted using abs(weight).",
        "A weight_was_negative flag column was created to preserve this information for the model.",
        "A weight_was_missing flag column was created for originally missing weight values.",
        "Missing market_index values were filled using the training-data median.",
        "A market_index_was_missing flag column was created.",
        "Date features (month, day of week, and a cyclical day-of-year encoding) were created.",
        "Raw pickup/delivery city names were dropped from model features, since validation "
        "contains cities never seen in training.",
        "Pickup/delivery coordinates (latitude/longitude) were retained, since they are "
        "well-defined for any city, seen or unseen.",
        "Raw input data files (data/train-test.csv, data/validation.csv, "
        "data/december-chart-inputs.csv) were never modified.",
    ]))
    story.append(para(
        "The negative-weight treatment (abs(weight) + flag) was not just a modeling "
        "assumption - it was experimentally compared against two alternatives (treating "
        "negatives as missing, and leaving them unchanged) using the same time-based "
        "validation windows described in Section 7. The abs(weight) approach produced the "
        "lowest, most consistent error across all three windows and was kept for the final "
        "model.", "Body"))

    # ---------------- 6. Feature Engineering ----------------
    story.append(para("6. Feature Engineering", "H1"))
    story.append(para(
        "Feature engineering focused on turning cleaned columns into numeric signals a "
        "tree-based model can use directly, while keeping the feature set simple.", "Body"))
    story.extend(bullets([
        "<b>Equipment one-hot encoding</b>: Dry Van / Reefer / Flatbed converted to three binary columns.",
        "<b>delta_lat, delta_lon</b>: difference in latitude/longitude between pickup and delivery.",
        "<b>haversine_distance</b>: straight-line distance between pickup and delivery, computed "
        "purely from coordinates so it works for any city, including ones unseen in training.",
        "<b>log_distance</b>: log(1 + distance), to reduce the influence of very long hauls.",
        "<b>route_directness</b>: actual distance divided by straight-line distance - a higher "
        "value means a more indirect route.",
        "<b>distance x equipment interactions</b>: lets each equipment type have its own "
        "rate-per-mile relationship rather than assuming one shared slope.",
        "<b>Calendar features</b>: month, day of week, and a sine/cosine pair representing day "
        "of year, so that December 31 and January 1 are treated as numerically close.",
    ]))

    story.append(PageBreak())

    # ---------------- 7. Validation Strategy ----------------
    story.append(para("7. Validation Strategy", "H1"))
    story.append(para(
        "Freight rates are time-dependent, and the actual task is to predict a future "
        "period (validation.csv begins in November) using only past data (training data "
        "runs January-October). For this reason, a random train/test split was "
        "deliberately avoided, since it would let the model be evaluated on dates earlier "
        "than some of its own training data - not a fair test of forecasting ability.", "Body"))
    story.append(para(
        "Instead, a chronological split was used, and the evaluation was repeated across "
        "three separate time windows to confirm the result was stable rather than "
        "specific to one arbitrary cutoff:", "Body"))
    story.append(metric_table([
        ["Window", "Training period", "Validation period"],
        ["Window 1", "Jan 1 - Jun 30", "Jul 1 - Aug 31"],
        ["Window 2", "Jan 1 - Jul 31", "Aug 1 - Sep 30"],
        ["Window 3", "Jan 1 - Aug 31", "Sep 1 - Oct 31"],
    ], col_widths=[1.3 * inch, 2.6 * inch, 2.6 * inch]))
    story.append(Spacer(1, 6))
    story.append(para(
        "Measuring performance across all three windows (rather than a single split) "
        "gives a more reliable estimate of how the model will perform going forward: a "
        "model that only does well on one specific window could simply be lucky on that "
        "slice of data. In this project, MAE remained reasonably stable across all three "
        "windows, with the best-to-worst window differing by about $13, which "
        "supports treating the final metrics as a realistic estimate of performance rather "
        "than an optimistic one. data/validation.csv itself was never used for training, "
        "tuning, or model selection at any point - it has no true rate to compare "
        "against and is reserved solely for the final submission.", "Body"))

    # ---------------- 8. Model Development ----------------
    story.append(para("8. Model Development", "H1"))
    story.append(para(
        "Models were developed through a sequence of controlled experiments, each "
        "changing exactly one variable and evaluated across the same three time windows. "
        "Full experiment code is in the experiments/ folder.", "Body"))
    story.append(metric_table([
        ["Step", "Approach", "MAE", "RMSE", "R2", "Outcome"],
        ["1", "Random Forest (raw target)", "186.29", "686.16", "0.7978", "Baseline"],
        ["2", "Random Forest + log1p target", "169.41", "656.64", "0.8149", "Improved, kept"],
        ["3", "HistGradientBoosting + log1p", "128.98", "639.57", "0.8244", "Improved, kept"],
        ["4", "Multi-window stability check", "-", "-", "-", "Confirmed stable across windows"],
        ["5", "HGB hyperparameter tuning", "119.40", "627.96", "0.8260", "Final model"],
        ["6", "XGBoost (baseline config)", "145.07", "638.03", "0.8203", "Did not beat HGB"],
        ["7", "Feature experiments (interactions)", "118.9-121.3", "-", "-", "None improved consistently; dropped"],
        ["8", "Negative-weight treatment test", "119.40 (best)", "-", "-", "Confirmed current approach best"],
        ["9", "CatBoost (baseline config)", "119.87", "627.94", "0.8260", "Tied HGB, not better"],
        ["10", "Local HGB refinement", "119.40 (best)", "-", "-", "No nearby config improved consistently"],
    ], col_widths=[0.5 * inch, 1.85 * inch, 0.85 * inch, 0.85 * inch, 0.65 * inch, 1.7 * inch]))
    story.append(Spacer(1, 6))
    story.append(para(
        "Each step's improvement (or rejection) was judged on consistency across all "
        "three time windows, not on average performance alone - a change that helped one "
        "window while hurting another was not kept, even if its average looked favorable.", "Body"))

    # ---------------- 9. Final Model ----------------
    story.append(para("9. Final Model", "H1"))
    story.append(para(
        "The final model is a tuned <b>HistGradientBoostingRegressor</b>. It was selected "
        "because it consistently outperformed Random Forest, untuned XGBoost, and "
        "CatBoost across all three validation windows, and a further local search around "
        "its tuned parameters and several engineered features failed to find a consistent "
        "improvement - indicating this configuration is a stable, well-supported choice "
        "rather than a result of chance.", "Body"))
    story.append(metric_table([
        ["Parameter", "Value"],
        ["learning_rate", "0.05"],
        ["max_iter", "300"],
        ["max_leaf_nodes", "15"],
        ["max_depth", "10"],
        ["min_samples_leaf", "50"],
        ["l2_regularization", "1.0"],
    ], col_widths=[2.3 * inch, 1.5 * inch]))
    story.append(Spacer(1, 6))
    story.append(para(
        "The model is trained on log1p(posted_rate) rather than the raw rate, since "
        "posted_rate is right-skewed; predictions are converted back to dollars using "
        "expm1(). This was tested and confirmed to improve accuracy over training on the "
        "raw target directly.", "Body"))
    story.append(para("Final average metrics across the 3 validation windows", "H2"))
    story.append(metric_table([
        ["Metric", "Value", "Meaning"],
        ["MAE", "$119.40", "On a typical load, the prediction is off by about $119"],
        ["RMSE", "$627.96", "Larger than MAE because a small number of large errors "
                             "(rare high-rate loads) weigh heavily in this metric"],
        ["R2", "0.8260", "The model explains approximately 82.6% of the variance in "
                          "posted_rate on unseen, later dates"],
    ], col_widths=[0.8 * inch, 1.1 * inch, 4.2 * inch]))

    # ---------------- 10. Error Analysis ----------------
    story.append(para("10. Error Analysis", "H1"))
    story.append(para(
        "Errors from the final model configuration were analyzed by group, pooling "
        "holdout predictions across all three validation windows.", "Body"))
    story.append(metric_table([
        ["Group", "MAE"],
        ["Distance 0-500 miles", "$41.32"],
        ["Distance 2,500+ miles", "$233.52"],
        ["Equipment: Dry Van", "$116.20"],
        ["Equipment: Flatbed", "$119.84"],
        ["Equipment: Reefer", "$126.24"],
        ["Loads with posted_rate > $6,000 (269 rows, 0.9% of data)", "$3,871.67"],
        ["All other loads (posted_rate <= $6,000)", "$83.80"],
    ], col_widths=[4.3 * inch, 1.8 * inch]))
    story.append(Spacer(1, 6))
    story.append(para(
        "Negative-weight rows and missing-weight rows were also checked specifically: "
        "neither group showed higher error than the rest of the data, so they are not a "
        "meaningful source of remaining error.", "Body"))
    story.append(para(
        "The clearest pattern is that error is concentrated in a small number of rare, "
        "high-rate, long-distance loads. This is a genuine limitation of the current "
        "model, not something that has been resolved - addressing it further would likely "
        "require a different modeling approach for that segment, which was outside the "
        "scope of this project.", "Body"))

    # ---------------- 11. Final Validation Predictions ----------------
    story.append(para("11. Final Validation Predictions", "H1"))
    story.append(para(
        "The final model was retrained on all 48,000 rows of the training/development "
        "data (not just one time window), using the same tuned configuration, and used to "
        "predict every row of data/validation.csv.", "Body"))
    story.append(metric_table([
        ["Check", "Result"],
        ["Output file", "validation_predictions.csv"],
        ["Row count", "12,000"],
        ["Unique load_id values", "12,000"],
        ["Missing predicted_rate values", "0"],
        ["Negative predicted_rate values", "0"],
        ["Columns", "load_id, predicted_rate (exact match to required format)"],
    ], col_widths=[2.6 * inch, 3.7 * inch]))

    # ---------------- 12. Official December Prediction ----------------
    story.append(para("12. Official December Prediction", "H1"))
    story.append(para(
        "The same final model was used to predict all 31 rows of "
        "data/december-chart-inputs.csv, a fixed Lexington to Fort Wayne route repeated "
        "once for each day of December 2025. All 31 predictions are positive and fall "
        "within a narrow, plausible range, since only the date changes row to row.", "Body"))
    story.append(para(
        "The chart below is the <b>official</b> chart, produced directly by the "
        "provided score.py after validating the required output format. score.py does "
        "not calculate model accuracy - it only validates that the submission files meet "
        "the required structure and generates this chart.", "Body"))
    story.append(sized_image(SCORER_RESULTS / "candidate_december.png", max_height=3.6 * inch))
    story.append(para(
        "Official chart: scorer_results/candidate_december.png, generated by "
        "`python score.py --predictions validation_predictions.csv "
        "--december-predictions data/december_predictions.csv`.", "Caption"))

    # ---------------- 13. Limitations and Future Improvements ----------------
    story.append(para("13. Limitations and Future Improvements", "H1"))
    story.extend(bullets([
        "High-rate, long-distance loads remain the model's weakest segment (Section 10) "
        "and account for a disproportionate share of total error.",
        "The December chart inputs require the fixed assessment format (one lane, one "
        "row per day), which limits how much can be inferred about general "
        "time-based behavior from that chart alone.",
        "A specialized approach for the high-rate tail (for example, a separate model "
        "or error-weighted training) remains untested and could be investigated further.",
    ]))
    story.append(para(
        "After the final model was selected, 4 further ideas were tested to see if it "
        "could still be improved: predicting rate-per-mile instead of the raw rate, "
        "extra coordinate-derived geometry features, historical average rate per route, "
        "and explicit long-haul/high-rate flags. All four were rejected using the same "
        "3-window consistency rule used throughout this project - none improved results, "
        "and historical route averages made results notably worse (too noisy with only "
        "64 distinct routes in the data). This is evidence the tuned model is close to "
        "the practical ceiling of what this feature set and dataset can support, rather "
        "than a model that was only lightly checked before finalizing. Full details are "
        "in experiments/extras/.", "Body"))

    # ---------------- 14. Reproducibility ----------------
    story.append(para("14. Reproducibility", "H1"))
    story.append(para("Exact commands to reproduce the final pipeline and outputs:", "Body"))
    repro_lines = [
        "python -m pip install -r requirements.txt",
        "python preprocessing/run_preprocessing.py",
        "python feature_engineering/run_feature_engineering.py",
        "python modeling/train_final.py",
        "python modeling/predict_december.py",
        "python score.py --predictions validation_predictions.csv "
        "--december-predictions data/december_predictions.csv",
    ]
    code_style = ParagraphStyle(name="Code", fontName="Courier", fontSize=8.5, leading=12,
                                 textColor=colors.HexColor("#0B2E3C"),
                                 backColor=colors.HexColor("#F2F6F7"), leftIndent=6,
                                 spaceAfter=4, borderPadding=4)
    for line in repro_lines:
        story.append(Paragraph(line, code_style))

    story.append(Spacer(1, 8))
    story.append(para(
        "The repository also includes an automated test suite (<b>tests/</b>, run with "
        "<b>pytest tests/</b>) covering the preprocessing and feature engineering logic "
        "plus the final submission file formats, a GitHub Actions workflow that runs "
        "these tests on every push, and a small command-line demo "
        "(<b>modeling/predict_single_load.py</b>) that predicts a rate for any single "
        "load using the exact final model.", "Body"))

    # ---------------- 15. Conclusion ----------------
    story.append(para("15. Conclusion", "H1"))
    story.append(para(
        "This project delivers a tuned HistGradientBoostingRegressor trained on cleaned, "
        "leak-free features, validated with a chronological, multi-window strategy "
        "appropriate for a forecasting task. Every modeling decision - target "
        "transform, negative-weight handling, feature choices, model family, and "
        "hyperparameters - was tested through controlled experiments and kept only when "
        "it improved results consistently, not just on average. The final model achieves "
        "an average MAE of $119.40 and R2 of 0.8260 across three independent validation "
        "windows, with the main known limitation being reduced accuracy on rare, "
        "high-rate, long-distance loads. Final predictions for both validation.csv and "
        "the December chart inputs have been generated and validated against the "
        "official scorer.", "Body"))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
    print(f"Report written to: {OUTPUT_PATH}")
