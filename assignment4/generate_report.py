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
OUTPUT_PDF = "nanoGPT-TP-report.pdf"

# Update these paths to your actual images. If a file is missing, a placeholder is rendered.
FIGS = {
    # ---------- Task 1 & 2 (TP4 vs DP4) ----------
    "T2_loss_steps_tp4vdp4": "assignment4/results/task2/loss_step.png",
    "T2_loss_time_tp4vdp4": "assignment4/results/task2/loss_time.png",
    "T2_runtime_bar_tp4vdp4": "assignment4/results/task2/runtime_comparison.png",
    "T2_mem_bar_tp4vdp4": "assignment4/results/task2/memory_comparison.png",

    # ---------- Task 3 & 4 (DP2+TP4 vs DP8) ----------
    "T4_loss_steps_dp2tp4vdp8": "assignment4/results/task4/loss_step.png",
    "T4_loss_time_dp2tp4vdp8": "assignment4/results/task4/loss_time.png",
    "T4_runtime_bar_dp2tp4vdp8": "assignment4/results/task4/runtime_comparison.png",
    "T4_mem_bar_dp2tp4vdp8": "assignment4/results/task4/memory_comparison.png",
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
    """: '"', """: '"', "„": '"', "‟": '"', "'": "'", "'": "'", "‚": "'", "`": "'",
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
title = "Assignment 4 - nanoGPT Tensor Parallelism on Shakespeare"
author = "Daize Dong"
date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
story.append(P(title, styles['TitleBig2']))
story.append(P(f"{author} · Generated {date_str}", styles['Body']))
story.append(Spacer(1, 0.22 * inch))

# =========================
# Global Settings
# =========================
story.append(P("Global Settings", styles["SectionHdr"]))
global_settings = [
    "Global batch size: 512",
    "Sequence length: 256",
    "Training Steps: 1000",
    "Warmup Steps: 20",
    "Optimizer: AdamW",
    "Cluster (Delta): 8 A100 GPUs",
]
story.append(bulleted(global_settings))
story.append(P(
    "Note that I changed the global batch_size from 64 to 512 to better utilize the GPUs in model parallelism settings. "
    "Also, the training steps were reduced from 5000 to 1000 to limit the total runtime for all experiments.",
    styles["Body"],
))
story.append(Spacer(1, 0.18 * inch))

story.append(PageBreak())

# ============================================================
# Task 1 & 2: TP4 vs DP4 Comparison
# ============================================================
story.append(P("Task 1 & 2: TP4 vs DP4 Comparison", styles["SectionHdr"]))

# Loss curves comparison
story.append(P("TP4 vs DP4: Loss Curves", styles["SubHdr"]))
t2_ls = figure_block_fit_h(
    FIGS["T2_loss_steps_tp4vdp4"], max_col_width=col_w, fixed_height=h2,
    title="Loss vs Steps (TP4 vs DP4)", caption="Loss vs steps"
)
t2_lt = figure_block_fit_h(
    FIGS["T2_loss_time_tp4vdp4"], max_col_width=col_w, fixed_height=h2,
    title="Loss vs Time (TP4 vs DP4)", caption="Loss vs time"
)
story.append(two_col(t2_ls, t2_lt, total_w=CONTENT_WIDTH, gap=GAP))
# story.append(P(
#     "TP4 and DP4 achieve similar loss curves when plotted by steps, but TP4 shows different execution time characteristics due to communication patterns.",
#     styles["Body"],
# ))

# Memory and runtime comparison
story.append(P("TP4 vs DP4: Memory and Runtime", styles["SubHdr"]))
t2_mb = figure_block_fit_h(
    FIGS["T2_mem_bar_tp4vdp4"], max_col_width=col_w, fixed_height=h2,
    title="Memory (TP4 vs DP4)", caption="Memory comparison"
)
t2_rb = figure_block_fit_h(
    FIGS["T2_runtime_bar_tp4vdp4"], max_col_width=col_w, fixed_height=h2,
    title="Runtime (TP4 vs DP4)", caption="Runtime comparison"
)
story.append(two_col(t2_mb, t2_rb, total_w=CONTENT_WIDTH, gap=GAP))
# story.append(P(
#     "TP4 distributes model parameters across GPUs, reducing per-GPU memory, while DP4 replicates the model and requires more memory per GPU.",
#     styles["Body"],
# ))

story.append(PageBreak())

# ============================================================
# Task 3 & 4: DP2+TP4 vs DP8 Comparison
# ============================================================
story.append(P("Task 3 & 4: DP2+TP4 vs DP8 Comparison", styles["SectionHdr"]))

# Loss curves comparison
story.append(P("DP2+TP4 vs DP8: Loss Curves", styles["SubHdr"]))
t4_ls = figure_block_fit_h(
    FIGS["T4_loss_steps_dp2tp4vdp8"], max_col_width=col_w, fixed_height=h2,
    title="Loss vs Steps (DP2+TP4 vs DP8)", caption="Loss vs steps"
)
t4_lt = figure_block_fit_h(
    FIGS["T4_loss_time_dp2tp4vdp8"], max_col_width=col_w, fixed_height=h2,
    title="Loss vs Time (DP2+TP4 vs DP8)", caption="Loss vs time"
)
story.append(two_col(t4_ls, t4_lt, total_w=CONTENT_WIDTH, gap=GAP))
# story.append(P(
#     "DP2+TP4 and DP8 achieve similar loss curves when plotted by steps, but show different execution time characteristics due to communication patterns.",
#     styles["Body"],
# ))

# Runtime and memory comparison
story.append(P("DP2+TP4 vs DP8: Memory and Runtime", styles["SubHdr"]))
t4_mb = figure_block_fit_h(
    FIGS["T4_mem_bar_dp2tp4vdp8"], max_col_width=col_w, fixed_height=h2,
    title="Memory (DP2+TP4 vs DP8)", caption="Memory comparison"
)
t4_rb = figure_block_fit_h(
    FIGS["T4_runtime_bar_dp2tp4vdp8"], max_col_width=col_w, fixed_height=h2,
    title="Runtime (DP2+TP4 vs DP8)", caption="Runtime comparison"
)
story.append(two_col(t4_mb, t4_rb, total_w=CONTENT_WIDTH, gap=GAP))
# story.append(P(
#     "DP2+TP4 reduces per-GPU memory usage compared to DP8, but may have different execution time due to cross-node communication overheads in TP.",
#     styles["Body"],
# ))

story.append(PageBreak())

# =========================
# Build PDF
# =========================
doc.build(story)
print(f"[OK] Exported: {OUTPUT_PDF}")

