# -*- coding: utf-8 -*-
# Assignment 2 — nanoGPT Data Parallelism on Shakespeare (experiments-only)
# Fixes symbol rendering issues by:
#   1) Preferring a wide-Unicode TrueType font if available (e.g., DejaVuSans/NotoSans in ./fonts/)
#   2) Sanitizing problematic Unicode to ASCII fallbacks (arrows, times sign, subscripts, dashes, etc.)
# Also applies your layout alignment (margins/styles), larger fonts, global Settings section,
# per-experiment specific settings, concise figure captions (x=..., y=...), and bullet lists.

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    Flowable, PageBreak, ListFlowable
)

OUTPUT_PDF = "nanoGPT-DP-report.pdf"

# --------------------------
# Figure registry — measured PNGs will be auto-inserted if present
# --------------------------
FIGS = {
    # Exp A: 1 GPU vs 4 GPUs (fixed global batch)
    "A_runtime_loss": "/u/ddong/workspace/nanoGPT/assignment2/results/1u_vs_4u/loss_time.png",
    "A_gpu_util": "/u/ddong/workspace/nanoGPT/assignment2/results/1u_vs_4u/gpu_util.png",
    # Exp B: 1 GPU vs 4 GPUs vs 8 GPUs (fixed global batch)
    "B_runtime_loss": "/u/ddong/workspace/nanoGPT/assignment2/results/1u_vs_8u/loss_time.png",
    "B_gpu_util": "/u/ddong/workspace/nanoGPT/assignment2/results/1u_vs_8u/gpu_util.png",
    # Exp C: 1 GPU scaling
    "C_runtime_loss": "/u/ddong/workspace/nanoGPT/assignment2/results/1u_scale/loss_time.png",
    "C_step_loss": "/u/ddong/workspace/nanoGPT/assignment2/results/1u_scale/loss_step.png",
    "C_gpu_util": "/u/ddong/workspace/nanoGPT/assignment2/results/1u_scale/gpu_util.png",
    # Exp D: 4 GPUs warmup
    "D_step_loss": "/u/ddong/workspace/nanoGPT/assignment2/results/4u_warmup/loss_step.png",
}


# --------------------------
# Font registration & text sanitization
# --------------------------

def register_preferred_body_font() -> str:
    """
    Try to register a broad-Unicode TTF if available in ./fonts/.
    Fallback to STSong-Light (CID) which is robust for CJK/ASCII.
    """
    candidates = [
        ("DejaVuSans", "fonts/DejaVuSans.ttf"),
        ("NotoSans", "fonts/NotoSans-Regular.ttf"),
        ("FreeSans", "fonts/FreeSans.ttf"),
        ("ArialUnicodeMS", "fonts/ArialUnicodeMS.ttf"),
    ]
    for name, path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                pass
    # Fallback: CID font (covers wide range; keep text ASCII-sanitized to avoid rare glyphs)
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    return 'STSong-Light'


BODY_FONT = register_preferred_body_font()

# Map fancy Unicode to ASCII-safe equivalents (minimize '?' in PDFs)
REPLACE_MAP = {
    "—": "-", "–": "-", "-": "-", "‒": "-", "…": "...",
    "×": "x", "÷": "/", "·": ".", "•": "-", "‧": ".",
    "→": "->", "←": "<-", "↔": "<->", "⇒": "=>", "⇐": "<=", "≈": "~", "≃": "~",
    "≥": ">=", "≤": "<=", "≠": "!=", "±": "+/-",
    "“": '"', "”": '"', "„": '"', "‟": '"', "‘": "'", "’": "'", "‚": "'", "`": "'",
    "β": "beta", "α": "alpha", "γ": "gamma", "δ": "delta",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4", "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9", "⁰": "^0",
    "µ": "u", "°": " deg",
}


def sanitize_text(s: str) -> str:
    if not s:
        return s
    out = []
    for ch in s:
        out.append(REPLACE_MAP.get(ch, ch))
    return "".join(out)


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(sanitize_text(text), style)


# --------------------------
# Styles (aligned with your previous template; larger fonts)
# --------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CodeMono', fontName='Courier', fontSize=10, leading=12))
styles.add(ParagraphStyle(name='SectionHdr', fontName=BODY_FONT, fontSize=16, leading=20,
                          spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1f2937")))
styles.add(ParagraphStyle(name='TitleBig2', fontName=BODY_FONT, fontSize=22, leading=26,
                          spaceAfter=12, alignment=1, textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name='Body', fontName=BODY_FONT, fontSize=13, leading=19,
                          textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name='Caption', fontName=BODY_FONT, fontSize=11, leading=14,
                          textColor=colors.HexColor("#374151"), alignment=1))


# --------------------------
# Helpers: figure blocks & bullet lists
# --------------------------
class FigurePlaceholder(Flowable):
    def __init__(self, w, h, title, desc):
        super().__init__()
        self.w, self.h = w, h
        self.title = title
        self.desc = desc

    def wrap(self, availWidth, availHeight):
        return (self.w, self.h)

    def draw(self):
        c = self.canv
        c.saveState()
        c.setStrokeColor(colors.lightgrey)
        c.setDash(3, 3)
        c.rect(0, 0, self.w, self.h)
        c.setDash()
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        y = self.h - 12
        for line in (f"[Placeholder] {sanitize_text(self.title)}", sanitize_text(self.desc)):
            c.drawString(6, y, line)
            y -= 12
        c.restoreState()


def image_or_placeholder(path, width, height, title):
    if path and os.path.exists(path):
        return Image(path, width=width, height=height)
    desc = f"Expected: {os.path.basename(path) if path else ''}  |  Size: {int(width)}x{int(height)}"
    return FigurePlaceholder(width, height, title, desc)


def figure_block(path, width, height, title, caption):
    fig = image_or_placeholder(path, width, height, title)
    cap = P(caption, styles["Caption"])
    t = Table([[fig], [cap]], colWidths=[width])
    t.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def two_col(fig_left, fig_right, total_w=10.2 * inch, gap=18):
    table = Table([[fig_left, fig_right]], colWidths=[(total_w - gap) / 2, (total_w - gap) / 2])
    table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def bulleted(items, left_indent=14, bullet_indent=0, bullet_font_size=13):
    # Use ASCII dash as bulletChar to avoid Unicode bullet issues.
    paras = [P(text, styles["Body"]) for text in items]
    return ListFlowable(
        paras,
        bulletType='bullet',
        bulletChar='-',
        leftIndent=left_indent,
        bulletIndent=bullet_indent,
        bulletFontName=BODY_FONT,
        bulletFontSize=bullet_font_size,
        bulletColor=colors.HexColor("#111827"),
        spaceBefore=0,
        spaceAfter=6
    )


# --------------------------
# Document (margins aligned with your template)
# --------------------------
doc = SimpleDocTemplate(
    OUTPUT_PDF,
    pagesize=landscape(letter),
    rightMargin=18, leftMargin=18, topMargin=54, bottomMargin=54
)
story = []

# Title
title = "Assignment 2 — nanoGPT Data Parallelism on Shakespeare"
author = "Daize Dong"
date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
story.append(P(title, styles['TitleBig2']))
story.append(P(f"{author} · Generated {date_str}", styles['Body']))
story.append(Spacer(1, 0.25 * inch))

# Common figure sizes
w2 = 5.0 * inch
h2 = 2.8 * inch

# =========================
# Global Settings (shared)
# =========================
story.append(P("Global Settings", styles["SectionHdr"]))
global_settings = [
    "Dataset: Shakespeare (vocabulary size 65)",
    "Model: nanoGPT (~10.65M parameters)",
    "Optimizer: AdamW",
    "Global batch size: 64 for Experiments A, B, and D; 64/128/256/512/1024 for Experiment C",
    "Sequence length: 256",
    "Gradient accumulation: 1",
    "Fixed random seeds for reproducibility",
    "Cluster (DeltaAI): 4 GH200 GPUs on each node",
]
story.append(bulleted(global_settings))
story.append(Spacer(1, 0.12 * inch))

# =========================
# EXPERIMENT A
# =========================
story.append(P("Experiment A — Fixed Global Batch: 1 GPU vs 4 GPUs", styles["SectionHdr"]))
story.append(P("Experiment-specific settings", styles["Body"]))
story.append(bulleted([
    "Devices: 1 GPU vs 4 GPUs on a single node, data parallel",
    "Per-GPU microbatch on GPUs: 16",
]))
A1 = figure_block(
    FIGS["A_runtime_loss"], w2, h2,
    "Fig A1 - 1 GPU vs 4 GPUs: time-loss",
    "Loss curve by time (s)"
)
A2 = figure_block(
    FIGS["A_gpu_util"], w2, h2,
    "Fig A2 - 1 GPU vs 4 GPUs: GPU utilization",
    "GPU utilization (%) by steps"
)
story.append(two_col(A1, A2))
story.append(Spacer(1, 0.12 * inch))
story.append(P("Findings", styles["Body"]))
story.append(bulleted([
    "The 4-GPU run reaches the same loss in fewer seconds (left-shifted time-loss), but the speedup is less than 4x.",
    "The validation loss matches the 1-GPU run when compared at the best checkpoint."
]))
story.append(P("Analysis", styles["Body"]))
story.append(bulleted([
    "Small per-GPU microbatches increase the fraction of communication overheads, resulting in sublinear scaling. 4 GPUs achieve only ~1.2x speedup over 1 GPU in wall-clock time.",
    "The 4-GPU run shows lower GPU utilization, indicating that GPUs are not compute-bound and all-reduce and launch latencies dominate a larger share of step time."
]))
story.append(PageBreak())

# =========================
# EXPERIMENT B
# =========================
story.append(P("Experiment B — Fixed Global Batch: 1 GPU vs 4 GPUs vs 8 GPUs", styles["SectionHdr"]))
story.append(P("Experiment-specific settings", styles["Body"]))
story.append(bulleted([
    "Devices: 1 GPU vs 4 GPUs (1 node) vs 8 GPUs (2 nodes x 4 GPUs)",
    "Per-GPU microbatch on 8 GPUs: 8",
]))
B1 = figure_block(
    FIGS["B_runtime_loss"], w2, h2,
    "Fig B1 - 1 GPU vs 4 GPUs vs 8 GPUs: time-loss",
    "Loss curve by time (s)"
)
B2 = figure_block(
    FIGS["B_gpu_util"], w2, h2,
    "Fig B2 - 1 GPU vs 4 GPUs vs 8 GPUs: GPU utilization",
    "GPU utilization (%) by steps"
)
story.append(two_col(B1, B2))
story.append(Spacer(1, 0.12 * inch))
story.append(P("Findings", styles["Body"]))
story.append(bulleted([
    "The 8-GPU (multi-node) run fails to beat 4 GPUs and can even trail the 1-GPU baseline in wall-clock time.",
    "GPU utilization on 8 GPUs is lower and more erratic."
]))
story.append(P("Analysis", styles["Body"]))
story.append(bulleted([
    "Inter-node bandwidth and latency on DeltaAI are inferior to intra-node NVLink, making multi-node scaling inefficient.",
    "The 8-GPU run shows lower GPU utilization, indicating that GPUs are not compute-bound."
]))
story.append(PageBreak())

# =========================
# EXPERIMENT C
# =========================
story.append(P("Experiment C — 1 GPU Batch Scaling", styles["SectionHdr"]))
story.append(P("Experiment-specific settings", styles["Body"]))
story.append(bulleted([
    "Devices: 1 GPU",
    "Per-GPU microbatch: 64",
    "Increase per-step batch (64 -> 128 -> 256 -> 512 -> 1024)",
    "I went beyond from the requested 4x to 16x!!!",
]))
C1 = figure_block(
    FIGS["C_runtime_loss"], w2, h2,
    "Fig C1 - 1 GPU scaling: time-loss",
    "Loss curve by time (s)"
)
C2 = figure_block(
    FIGS["C_step_loss"], w2, h2,
    "Fig C2 - 1 GPU scaling: step-loss",
    "Loss curve by steps"
)
story.append(two_col(C1, C2))
story.append(Spacer(1, 0.12 * inch))
C3 = figure_block(
    FIGS["C_gpu_util"], 5.1 * inch, 2.5 * inch,
    "Fig C3 - 1 GPU scaling: GPU utilization",
    "GPU utilization (%) by steps"
)
story.append(C3)
story.append(Spacer(1, 0.12 * inch))
story.append(P("Findings", styles["Body"]))
story.append(bulleted([
    "Increasing batch size lifts utilization and shifts the time–loss curve left—optimization advances faster per minute.",
    "The best validation loss remains essentially unchanged across batch sizes."
]))
story.append(P("Analysis", styles["Body"]))
story.append(bulleted([
    "Larger batches improve arithmetic intensity and kernel efficiency, reducing the pipeline overheads.",
    "Since data and model are held constant, the attainable optimum (validation) is similar."
]))
story.append(PageBreak())

# =========================
# EXPERIMENT D
# =========================
story.append(P("Experiment D — Four GPUs with Learning-Rate Warmup", styles["SectionHdr"]))
story.append(P("Experiment-specific settings", styles["Body"]))
story.append(bulleted([
    "Devices: 4 GPUs",
    "Per-GPU microbatch: 32",
    "Linear warmup for the first 100 steps vs no warmup (baseline)",
]))
D1 = figure_block(
    FIGS["D_step_loss"], 5.1 * inch, 2.7 * inch,
    "Fig D1 - 4 GPUs with warmup: step-loss",
    "Loss curve by steps"
)
story.append(D1)
story.append(Spacer(1, 0.12 * inch))
story.append(P("Findings", styles["Body"]))
story.append(bulleted([
    "Learning-rate warmup smooths the first few hundred steps and lead s to a slightly better final loss."
]))
story.append(P("Analysis", styles["Body"]))
story.append(bulleted([
    "Early oversized updates can overshoot before AdamW’s moments settle down, and warmup mitigates this risk."
]))

# Build
doc.build(story)
print(f"[OK] Exported: {OUTPUT_PDF}")
