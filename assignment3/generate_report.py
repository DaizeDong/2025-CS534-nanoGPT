import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, legal
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    Flowable, PageBreak, ListFlowable
)

# =========================
# Output & Figure paths
# =========================
OUTPUT_PDF = "nanoGPT-MP-report.pdf"

# Update these paths to your actual images. If a file is missing, a placeholder is rendered.
FIGS = {
    # ---------- Task 1 (MP2 vs Baseline) ----------
    "T1_split_MP2": "/u/ddong/workspace/nanoGPT/assignment3/results/best_split/train_large_bs_mp2.png",
    "T1_loss_steps_MP2v1": "/u/ddong/workspace/nanoGPT/assignment3/results/task1/loss_step.png",
    "T1_loss_time_MP2v1": "/u/ddong/workspace/nanoGPT/assignment3/results/task1/loss_time.png",
    "T1_mem_bar_MP2v1": "/u/ddong/workspace/nanoGPT/assignment3/results/task1/gpu_memory_avg.png",
    "T1_runtime_bar_MP2v1": "/u/ddong/workspace/nanoGPT/assignment3/results/task1/final_time_comparison.png",

    # ---------- Task 2 (MP2/4/6 vs Baseline) ----------
    "T2_split_MP4": "/u/ddong/workspace/nanoGPT/assignment3/results/best_split/train_large_bs_mp4.png",
    "T2_split_MP6": "/u/ddong/workspace/nanoGPT/assignment3/results/best_split/train_large_bs_mp6.png",
    "T2_loss_steps_1v2v4v6": "/u/ddong/workspace/nanoGPT/assignment3/results/task2/loss_step.png",
    "T2_loss_time_1v2v4v6": "/u/ddong/workspace/nanoGPT/assignment3/results/task2/loss_time.png",
    "T2_mem_bar_1v2v4v6": "/u/ddong/workspace/nanoGPT/assignment3/results/task2/gpu_memory_avg.png",
    "T2_runtime_bar_1v2v4v6": "/u/ddong/workspace/nanoGPT/assignment3/results/task2/final_time_comparison.png",
}


# =========================
# Fonts & text sanitization
# =========================
def register_preferred_body_font() -> str:
    """Prefer a wide-Unicode TTF; fallback to STSong-Light (CID)."""
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
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    return 'STSong-Light'


BODY_FONT = register_preferred_body_font()

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
    return "".join(REPLACE_MAP.get(ch, ch) for ch in s)


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(sanitize_text(text), style)


# =========================
# Styles
# =========================
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='SectionHdr', fontName=BODY_FONT, fontSize=16, leading=20,
                          spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#1f2937")))
styles.add(ParagraphStyle(name='SubHdr', fontName=BODY_FONT, fontSize=14, leading=18,
                          spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name='TitleBig2', fontName=BODY_FONT, fontSize=22, leading=26,
                          spaceAfter=8, alignment=1, textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name='Body', fontName=BODY_FONT, fontSize=13, leading=19,
                          textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name='Caption', fontName=BODY_FONT, fontSize=11, leading=14,
                          textColor=colors.HexColor("#374151"), alignment=1))


# =========================
# Helpers: image scaling (fixed height, keep AR), placeholders, blocks, columns
# =========================
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
        for line in (f"[Missing] {sanitize_text(self.title)}",
                     sanitize_text(self.desc)):
            c.drawString(6, y, line)
            y -= 12
        c.restoreState()


def scaled_img_fixed_height(path: str, max_col_width: float, fixed_height: float, title: str):
    """Create an Image flowable scaled to a fixed height, preserving aspect ratio (no stretch).
       The cell width (col width) can be wider than the image; image is centered."""
    if path and os.path.exists(path):
        img = Image(path)
        iw, ih = float(getattr(img, "imageWidth", 1.0)), float(getattr(img, "imageHeight", 1.0))
        if ih <= 0 or iw <= 0:
            return FigurePlaceholder(max_col_width, fixed_height, title, "Invalid image size")
        aspect = iw / ih
        img.drawHeight = fixed_height
        img.drawWidth = fixed_height * aspect  # width derived from fixed height (keeps AR)
        return img
    else:
        # Placeholder takes the full column width at the fixed height
        return FigurePlaceholder(max_col_width, fixed_height, title, os.path.basename(path) if path else "")


def figure_block_fit_h(path, max_col_width, fixed_height, title, caption):
    """Wrap the image + a short caption in a single-column table (width equals column width)."""
    fig = scaled_img_fixed_height(path, max_col_width, fixed_height, title)
    cap = P(caption, styles["Caption"])
    t = Table([[fig], [cap]], colWidths=[max_col_width])
    t.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def two_col(fig_left, fig_right, total_w, gap=18):
    """Place two figures side-by-side. Each figure should already be sized to column height."""
    col_w = (total_w - gap) / 2.0
    table = Table([[fig_left, fig_right]], colWidths=[col_w, col_w])
    table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def bulleted(items, left_indent=14, bullet_indent=0, bullet_font_size=13):
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


# =========================
# Document (landscape legal for wider columns)
# =========================
doc = SimpleDocTemplate(
    OUTPUT_PDF,
    pagesize=landscape(legal),
    rightMargin=18, leftMargin=18, topMargin=36, bottomMargin=36
)
story = []
CONTENT_WIDTH = doc.width
GAP = 18

# Common figure heights (fixed), columns computed from CONTENT_WIDTH
h2 = 2.8 * inch  # two-column figure height
h1 = 2.7 * inch  # single-wide figure height
col_w = (CONTENT_WIDTH - GAP) / 2.0

# =========================
# Title
# =========================
title = "Assignment 3 - nanoGPT Model Parallelism on Shakespeare"
author = "Daize Dong"
date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
story.append(P(title, styles['TitleBig2']))
story.append(P(f"{author} · Generated {date_str}", styles['Body']))
story.append(Spacer(1, 0.22 * inch))

# =========================
# Global Settings (shared)
# =========================
story.append(P("Global Settings", styles["SectionHdr"]))
global_settings = [
    "Global batch size: 512",
    "Sequence length: 256",
    "Gradient accumulation: 1",
    "Training Steps: 1000",
    "Warmup Steps: 20",
    "Optimizer: AdamW",
    "Cluster (Delta): Max 6 A100 GPUs",
]
story.append(bulleted(global_settings))
story.append(P(
    "Note that I changed the global batch_size from 64 to 512 to better utilize the GPUs in model parallelism settings. "
    "Also, the training steps were reduced from 5000 to 1000 to limit the total runtime for all experiments.",
    styles["Body"],
))
story.append(Spacer(1, 0.18 * inch))

# =========================
# Global Settings (shared)
# =========================
story.append(P("Global Settings (Model Parallel)", styles["SectionHdr"]))
global_settings = [
    "Search Split Size: 1,2,4,8,16,32,64,128,256,512",
    "Search Warmup Steps: 10",
    "Search Profile Steps: 20",
]
story.append(bulleted(global_settings))
story.append(Spacer(1, 0.18 * inch))

story.append(PageBreak())

# ============================================================
# Task 1: 2-GPU Model Parallelism (MP2)
# ============================================================
story.append(P("Task 1: 2-GPU Model Parallelism (MP2)", styles["SectionHdr"]))

# MP2 split curve (best split = 64)
story.append(P("MP2 Split-size Sweep", styles["SubHdr"]))
t1_split = figure_block_fit_h(
    FIGS["T1_split_MP2"], max_col_width=CONTENT_WIDTH, fixed_height=h1,
    title="MP2 split sweep", caption="Best split size: 64"
)
story.append(t1_split)
story.append(P(
    "The best split size is 64 for MP=2. "
    "From the line we can see that very small splits (1,2,4) and very large splits (256,512) both lead to higher step times due to overheads. "
    "And the overheads at small splits are more pronounced because of the increased communication frequency.",
    styles["Body"],
))

# MP2 vs Baseline: loss (steps & time)
story.append(P("MP2 vs Baseline: Loss Curves", styles["SubHdr"]))
t1_ls = figure_block_fit_h(
    FIGS["T1_loss_steps_MP2v1"], max_col_width=col_w, fixed_height=h2,
    title="Loss vs Steps (1-GPU vs MP2)", caption="Loss vs steps"
)
t1_lt = figure_block_fit_h(
    FIGS["T1_loss_time_MP2v1"], max_col_width=col_w, fixed_height=h2,
    title="Loss vs Time (1-GPU vs MP2)", caption="Loss vs time"
)
story.append(two_col(t1_ls, t1_lt, total_w=CONTENT_WIDTH, gap=GAP))
# story.append(Spacer(1, 0.10 * inch))
story.append(P(
    "With the same optimizer and schedule, MP2 reaches the same loss earlier than the single-GPU baseline. "
    "When plotted by steps, the two curves closely overlap. This indicates that the optimization is identical.",
    styles["Body"],
))

# MP2 vs Baseline: memory & runtime bars
story.append(P("MP2 vs Baseline: Memory and Runtime", styles["SubHdr"]))
t1_mb = figure_block_fit_h(
    FIGS["T1_mem_bar_MP2v1"], max_col_width=col_w, fixed_height=h2,
    title="Memory (1-GPU vs MP2)", caption="Memory comparison"
)
t1_rb = figure_block_fit_h(
    FIGS["T1_runtime_bar_MP2v1"], max_col_width=col_w, fixed_height=h2,
    title="Runtime (1-GPU vs MP2)", caption="Runtime comparison"
)
story.append(two_col(t1_mb, t1_rb, total_w=CONTENT_WIDTH, gap=GAP))
# story.append(Spacer(1, 0.10 * inch))
story.append(P(
    "Distributing parameters across 2 GPUs lowers the per-GPU memory peak by half."
    "This leads to a notable reduction in average step time compared to the single-GPU baseline.",
    styles["Body"],
))

story.append(PageBreak())

# ============================================================
# Task 2: Configurable MP (MP4 & MP6) vs Baseline
# ============================================================
story.append(P("Task 2: MP4 & MP6 vs Baseline", styles["SectionHdr"]))

# MP4 & MP6 split curves (best split = 64)
story.append(P("MP4 & MP6 Split-size Sweeps", styles["SubHdr"]))
t2_s4 = figure_block_fit_h(
    FIGS["T2_split_MP4"], max_col_width=col_w, fixed_height=h2,
    title="MP4 split sweep", caption="Best split size: 64"
)
t2_s6 = figure_block_fit_h(
    FIGS["T2_split_MP6"], max_col_width=col_w, fixed_height=h2,
    title="MP6 split sweep", caption="Best split size: 64"
)
story.append(two_col(t2_s4, t2_s6, total_w=CONTENT_WIDTH, gap=GAP))
# story.append(Spacer(1, 0.10 * inch))
story.append(P(
    "The best split size remains 64 when scaling to 4 and 6 stages. "
    "This consistency suggests the micro-batch splitting overheads are well-matched at that granularity on this cluster.",
    styles["Body"],
))

# 1-GPU vs MP2/4/6: loss (steps & time)
story.append(P("Baseline vs MP2/4/6: Loss Curves", styles["SubHdr"]))
t2_ls = figure_block_fit_h(
    FIGS["T2_loss_steps_1v2v4v6"], max_col_width=col_w, fixed_height=h2,
    title="Loss vs Steps (1-GPU vs MP2/4/6)", caption="Loss vs steps"
)
t2_lt = figure_block_fit_h(
    FIGS["T2_loss_time_1v2v4v6"], max_col_width=col_w, fixed_height=h2,
    title="Loss vs Time (1-GPU vs MP2/4/6)", caption="Loss vs time"
)
story.append(two_col(t2_ls, t2_lt, total_w=CONTENT_WIDTH, gap=GAP))
# story.append(Spacer(1, 0.10 * inch))
story.append(P(
    "Moving from 2 to 4 stages pushes the loss–time curve further left, but with sublinear returns relative to stage count. "
    "The communication overheads are quite large, as the improvements from 2 to 4 and 6 are quite small.",
    styles["Body"],
))

# 1-GPU vs MP2/4/6: memory & runtime bars
story.append(P("Baseline vs MP2/4/6: Memory and Runtime", styles["SubHdr"]))
t2_mb = figure_block_fit_h(
    FIGS["T2_mem_bar_1v2v4v6"], max_col_width=col_w, fixed_height=h2,
    title="Memory (1-GPU vs MP2/4/6)", caption="Memory comparison"
)
t2_rb = figure_block_fit_h(
    FIGS["T2_runtime_bar_1v2v4v6"], max_col_width=col_w, fixed_height=h2,
    title="Runtime (1-GPU vs MP2/4/6)", caption="Runtime comparison"
)
story.append(two_col(t2_mb, t2_rb, total_w=CONTENT_WIDTH, gap=GAP))
# story.append(Spacer(1, 0.10 * inch))
story.append(P(
    "Deeper model parallelism further reduces per-GPU memory pressure and reduces training time.",
    styles["Body"],
))

story.append(PageBreak())

# =========================
# Build PDF
# =========================
doc.build(story)
print(f"[OK] Exported: {OUTPUT_PDF}")
