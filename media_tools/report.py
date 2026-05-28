# =========================================================
# OpenSSL / hashlib compatibility patch (MUST be first)
# =========================================================
import hashlib

_orig_md5 = hashlib.md5

def _patched_md5(*args, **kwargs):
    kwargs.pop("usedforsecurity", None)
    return _orig_md5(*args, **kwargs)

hashlib.md5 = _patched_md5

# =========================================================
# Imports
# =========================================================
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.graphics.charts.barcharts import VerticalBarChart

# =========================================================
# CONFIG
# =========================================================
output_pdf = "DDRNet23_slim_ONNX_vs_GCNet_ONNX_Report.pdf"
superclass_names = ["Field", "Grass", "Windrow", "Unused objects"]

# =========================================================
# DATA (ONNX RESULTS ONLY)
# =========================================================

# DDRNet23-slim ONNX
ddrnet23_slim = {
    "train": {
        "Pixel Acc": None,
        "Mean Acc": None,
        "mIoU": 0.899977,
        "Class IoU": [0.902710, 0.942180, 0.774415, 0.980602],
    },
    "val": {
        "Pixel Acc": None,
        "Mean Acc": None,
        "mIoU": 0.800016,
        "Class IoU": [0.823152, 0.921295, 0.493712, 0.961905],
    },
    "test": {
        "Pixel Acc": None,
        "Mean Acc": None,
        "mIoU": 0.884040,
        "Class IoU": [0.886047, 0.934121, 0.735623, 0.980369],
    },
}

# GCNet ONNX (your real results)
gcnet = {
    "train": {
        "Pixel Acc": 0.9607,
        "Mean Acc": 0.9245,
        "mIoU": 0.8742,
        "Class IoU": [0.8804, 0.9327, 0.7082, 0.9754],
    },
    "val": {
        "Pixel Acc": 0.9549,
        "Mean Acc": 0.8714,
        "mIoU": 0.7847,
        "Class IoU": [0.8313, 0.9218, 0.4271, 0.9584],
    },
    "test": {
        "Pixel Acc": 0.9496,
        "Mean Acc": 0.9078,
        "mIoU": 0.8451,
        "Class IoU": [0.8534, 0.9262, 0.6282, 0.9726],
    },
}

# =========================================================
# TABLE HELPER
# =========================================================
def build_table(data):
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    return table

# =========================================================
# CHARTS
# =========================================================
def miou_chart():
    d = Drawing(460, 260)
    d.add(String(230, 240, "mIoU Comparison",
                 textAnchor="middle", fontSize=12))

    chart = VerticalBarChart()
    chart.x = 70
    chart.y = 50
    chart.height = 160
    chart.width = 320

    chart.data = [
        [ddrnet23_slim[s]["mIoU"] for s in ["train", "val", "test"]],
        [gcnet[s]["mIoU"] for s in ["train", "val", "test"]],
    ]

    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 1
    chart.categoryAxis.categoryNames = ["Train", "Val", "Test"]

    chart.bars[0].fillColor = colors.darkred
    chart.bars[1].fillColor = colors.darkgreen

    d.add(chart)
    d.add(String(110, 25, "DDRNet23-slim ONNX", fontSize=9))
    d.add(String(260, 25, "GCNet ONNX", fontSize=9))

    return d


def per_class_iou_chart(split):
    d = Drawing(460, 260)
    d.add(String(
        230, 240,
        f"Per-class IoU – {split.capitalize()}",
        textAnchor="middle",
        fontSize=12
    ))

    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 50
    chart.height = 160
    chart.width = 360

    chart.data = [
        ddrnet23_slim[split]["Class IoU"],
        gcnet[split]["Class IoU"],
    ]

    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 1
    chart.categoryAxis.categoryNames = superclass_names

    chart.bars[0].fillColor = colors.darkred
    chart.bars[1].fillColor = colors.darkgreen

    d.add(chart)
    return d

# =========================================================
# BUILD REPORT
# =========================================================
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(output_pdf, pagesize=A4)
story = []

# Title
story.append(Paragraph(
    "<b>DDRNet23-slim ONNX vs GCNet ONNX Report<br/>RSM Dataset</b>",
    styles["Title"]
))
story.append(Spacer(1, 16))

# Table
table = [
    ["Split", "Model", "Pixel Acc", "Mean Acc", "mIoU"] +
    [f"IoU {c}" for c in superclass_names]
]

for split in ["train", "val", "test"]:
    for label, src in [
        ("DDRNet23-slim ONNX", ddrnet23_slim),
        ("GCNet ONNX", gcnet),
    ]:
        v = src[split]
        table.append([
            split,
            label,
            "-" if v["Pixel Acc"] is None else f"{v['Pixel Acc']:.4f}",
            "-" if v["Mean Acc"] is None else f"{v['Mean Acc']:.4f}",
            f"{v['mIoU']:.4f}",
            *[f"{x:.4f}" for x in v["Class IoU"]],
        ])

story.append(build_table(table))
story.append(Spacer(1, 24))

# Charts
story.append(miou_chart())
story.append(Spacer(1, 24))

for split in ["train", "val", "test"]:
    story.append(per_class_iou_chart(split))
    story.append(Spacer(1, 24))

# Save
doc.build(story)
print(f"Report saved to: {output_pdf}")