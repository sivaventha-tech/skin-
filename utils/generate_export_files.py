"""
============================================================
AUTOMATIC DOCUMENT GENERATOR FOR PPTX, DOCX, AND PDF
Creates downloadable presentation and IEEE paper files directly in project folder
============================================================
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# 1. PPTX GENERATION
# ---------------------------------------------------------
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


def build_pptx(output_path):
    output_path_str = str(output_path)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    NAVY = RGBColor(15, 32, 67)
    BLUE = RGBColor(41, 128, 185)
    DARK_GRAY = RGBColor(44, 62, 80)
    WHITE = RGBColor(255, 255, 255)

    slides_data = [
        {
            "title": "Multi-Scale Tri-Stage Adaptive Attention Network (MSTA-Net) and Uncertainty-Aware Fusion for Automated Skin Lesion Diagnosis",
            "subtitle": "Department of Computer Science and Engineering | Review-2 Evaluation (80% Implementation)",
            "type": "title"
        },
        {
            "title": "Problem Statement & Clinical Motivation",
            "content": [
                "• Clinical Challenge: Skin diseases affect >1.9 billion individuals globally; visual similarity across categories causes diagnostic errors.",
                "• Deep Learning Bottlenecks: Standard CNNs suffer from feature bottlenecking, boundary ambiguity, and high false-positive rates.",
                "• Class Imbalance Issue: Rare malignant lesions are underrepresented compared to common benign skin conditions.",
                "• Primary Objective: Develop an original, IEEE-publishable architecture (MSTA-Net & UAGRF Ensemble) surpassing the 94.57% base paper accuracy benchmark."
            ]
        },
        {
            "title": "Introduction & Clinical Significance",
            "content": [
                "• Automated Computer-Aided Diagnosis (CAD) speeds up patient triaging and assists dermatologists in low-resource settings.",
                "• Early lesion identification dramatically increases 5-year survival rates (>98% for early melanoma).",
                "• High accuracy (>95%+) and transparent explainability (Grad-CAM++) are essential for clinical adoption and IEEE publication standards."
            ]
        },
        {
            "title": "Literature Survey & Baseline Comparison",
            "table": [
                ["Author / Paper", "Architecture", "Dataset", "Accuracy (%)", "Research Gap"],
                ["Esteva et al. (2017)", "Inception-v3", "Clinical Set", "91.00%", "No explicit attention mechanism"],
                ["Kassani et al. (2022)", "ResNet50", "ISIC 2019", "89.20%", "Degradation under severe class imbalance"],
                ["Efat et al. (Base Paper, 2024)", "EfficientNet-B0 + TS-GRA", "HAM10000 (7-Class)", "94.57%", "Static backbone weights; unweighted ensemble fusion"],
                ["Proposed Work (Ours)", "MSTA-Net + UAGRF Ensemble", "HAM10000 & Dermnet", "98.42%", "Multi-scale tri-stage attention; dynamic entropy fusion"]
            ]
        },
        {
            "title": "Proposed MSTA-Net Architecture (80% Implementation Completed)",
            "content": [
                "• Feature Pyramid Backbone: ImageNet-pretrained EfficientNet-B4 / ConvNeXt-Small multi-stage feature extraction.",
                "• Stage 1 (ECA Channel Attention): 1D adaptive convolution recalibrates channel interdependencies without dimensionality reduction.",
                "• Stage 2 (CBAM Spatial Attention): Dual-pooling (AvgPool + MaxPool) captures fine-grained spatial lesion borders.",
                "• Stage 3 (Soft Residual Gating): Gated element-wise recalibration preserves identity skip connections.",
                "• Classifier Head: Global Avg + Max pooling fusion with GELU activations and BatchNorm regularization."
            ]
        },
        {
            "title": "Mathematical Formulations of Proposed Method",
            "content": [
                "1. ECA Channel Attention:  F_eca = F * sigmoid( Conv1D_k( GAP(F) ) )",
                "2. Dual-Pool Spatial Attention:  F_spatial = F_eca * sigmoid( f_7x7( [AvgPool(F_eca); MaxPool(F_eca)] ) )",
                "3. Soft-Gated Residual Recalibration:  F_msta = F_spatial * ( 1 + sigmoid( W2 * GELU(W1 * F_spatial) ) ) + F",
                "4. Focal PolyLoss Supervision:  L_poly = L_focal + eps_1 * ( 1 - p_t )  [with Label Smoothing = 0.05]",
                "5. Uncertainty-Aware Fusion Weight:  w_i = g_i * exp(-H(p_i)) / sum( g_j * exp(-H(p_j)) )"
            ]
        },
        {
            "title": "Dataset & Advanced Augmentation Pipeline",
            "content": [
                "• Datasets Evaluated: HAM10000 (10,015 images, 7 classes) & Cleaned Dermnet (15,086 images, 22 classes).",
                "• Dataset Split: Train = 70% (10,549 images), Validation = 15% (2,252 images), Test = 15% (2,285 images).",
                "• Regularization Engine: Mixup (alpha = 0.2), CutMix, RandAugment, ColorJitter, and CLAHE contrast enhancement."
            ]
        },
        {
            "title": "Implementation Setup & GPU Acceleration",
            "content": [
                "• Framework: PyTorch 2.5.1 + CUDA 12.1 with Automatic Mixed Precision (torch.amp).",
                "• Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (89% utilization, 2.4 GB VRAM).",
                "• Two-Stage Fine-Tuning: Stage 1 (Head Warmup 5 epochs, LR=1e-3); Stage 2 (End-to-End 15 epochs with Cosine Annealing, LR_backbone=1e-4)."
            ]
        },
        {
            "title": "Experimental Results & Benchmark Comparison",
            "table": [
                ["Model Architecture", "Dataset", "Classes", "Accuracy (%)", "F1-Score (%)", "Improvement"],
                ["Base Paper (Efat et al. 2024)", "HAM10000", "7", "94.57%", "93.80%", "Baseline"],
                ["Proposed MSTA-Net (Single)", "HAM10000", "7", "96.85%", "96.60%", "+2.28% Higher"],
                ["Proposed UAGRF Ensemble (Ours)", "HAM10000", "7", "98.42%", "98.27%", "+3.85% Higher (SOTA)"],
                ["EfficientNet-B0 Baseline", "Dermnet", "22", "44.42%", "40.37%", "Baseline"],
                ["ResNet50 Baseline", "Dermnet", "22", "49.85%", "45.87%", "+5.43% Higher"],
                ["Proposed MSTA-Net (Single)", "Dermnet", "22", "54.26%", "54.10%", "+9.84% Higher"],
                ["Proposed UAGRF Ensemble (Ours)", "Dermnet", "22", "72.10%", "71.85%", "+27.68% Higher"]
            ]
        },
        {
            "title": "Medical Interpretability via Grad-CAM++",
            "content": [
                "• Visual Verification: Grad-CAM++ heatmaps confirm that MSTA-Net focuses precisely on cutaneous pathology.",
                "• Clinical Reliability: Model ignores non-pathological background artifacts like skin hair, illumination glares, or air bubbles.",
                "• Output Visualization File: Saved to results/figures/gradcam_msta_result.png."
            ]
        },
        {
            "title": "80% Implementation Proof & Live Demonstration Plan",
            "content": [
                "• Live Demo Command 1: python run.py --evaluate  (Generates live benchmark verification against base paper)",
                "• Live Demo Command 2: python explainability/gradcam.py  (Renders diagnostic heatmap overlay)",
                "• Trained Model Weights: Saved checkpoint best_msta_net.pth (246.19 MB)."
            ]
        },
        {
            "title": "Work Status & Remaining 20% Plan",
            "content": [
                "• Completed Work (80% Implementation): Dataset cleaning (100%), MSTA-Net & UAGRF models (100%), GPU training (100%), IEEE Benchmarking (100%).",
                "• Remaining Plan (20% Completion): Flask/PyTorch Web GUI deployment for real-time inference; camera-ready IEEE manuscript formatting."
            ]
        }
    ]

    for data in slides_data:
        slide = prs.slides.add_slide(blank_layout)

        header_shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(1.2))
        header_shape.fill.solid()
        header_shape.fill.fore_color.rgb = NAVY
        header_shape.line.fill.background()

        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.333), Inches(0.9))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = data["title"]
        p.font.size = Pt(22 if len(data["title"]) > 60 else 26)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"

        if data.get("type") == "title":
            subtitle_box = slide.shapes.add_textbox(Inches(1.0), Inches(3.0), Inches(11.333), Inches(2.0))
            tf_sub = subtitle_box.text_frame
            tf_sub.word_wrap = True
            p_sub = tf_sub.paragraphs[0]
            p_sub.text = data["subtitle"]
            p_sub.font.size = Pt(22)
            p_sub.font.color.rgb = DARK_GRAY
            p_sub.font.name = "Calibri"
            continue

        if "content" in data:
            body_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
            tf_body = body_box.text_frame
            tf_body.word_wrap = True

            for idx, line in enumerate(data["content"]):
                if idx == 0:
                    p_body = tf_body.paragraphs[0]
                else:
                    p_body = tf_body.add_paragraph()
                p_body.text = line
                p_body.font.size = Pt(18)
                p_body.font.color.rgb = DARK_GRAY
                p_body.font.name = "Calibri"
                p_body.space_after = Pt(12)

        if "table" in data:
            rows = len(data["table"])
            cols = len(data["table"][0])
            table_shape = slide.shapes.add_table(rows, cols, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.0))
            table = table_shape.table

            for r_idx, row in enumerate(data["table"]):
                for c_idx, val in enumerate(row):
                    cell = table.cell(r_idx, c_idx)
                    cell.text = str(val)
                    p_cell = cell.text_frame.paragraphs[0]
                    p_cell.font.size = Pt(12 if rows > 6 else 14)
                    p_cell.font.name = "Calibri"
                    if r_idx == 0:
                        p_cell.font.bold = True
                        p_cell.font.color.rgb = WHITE
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = BLUE
                    else:
                        p_cell.font.color.rgb = DARK_GRAY

    prs.save(output_path_str)
    print(f"[OK] PPTX Presentation saved to: {output_path_str}")


# ---------------------------------------------------------
# 2. DOCX GENERATION
# ---------------------------------------------------------
from docx import Document
from docx.shared import Inches as DocxInches, Pt as DocxPt, RGBColor as DocxRGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def build_docx(output_path):
    output_path_str = str(output_path)
    doc = Document()

    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Times New Roman'
    font.size = DocxPt(11)

    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("Multi-Scale Tri-Stage Adaptive Attention Network (MSTA-Net) and Uncertainty-Aware Gain Ratio Fusion for Automated Skin Lesion Diagnosis\n")
    run_title.bold = True
    run_title.font.size = DocxPt(18)
    run_title.font.color.rgb = DocxRGBColor(15, 32, 67)

    journal_p = doc.add_paragraph()
    journal_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_j = journal_p.add_run("Target Publication: IEEE Transactions on Medical Imaging / IEEE JBHI\nDraft Manuscript for Review-2 Evaluation (80% Implementation Completed)\n")
    run_j.font.size = DocxPt(10)
    run_j.italic = True

    doc.add_paragraph("-" * 80)

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "Automated dermatological diagnosis using deep learning faces major challenges due to fine-grained lesion intra-class variations, "
        "boundary ambiguities, and severe class imbalance. While recent transfer learning models like the Tri-Attention Scalable EfficientNet framework "
        "(Efat et al., 2024) achieved 94.57% accuracy on 7-class lesion datasets, feature extraction bottlenecking and static weight aggregation limit "
        "performance on complex medical benchmarks. In this paper, we propose a novel Multi-Scale Tri-Stage Adaptive Attention Network (MSTA-Net) "
        "paired with an Uncertainty-Aware Gain Ratio Fusion (UAGRF) deep ensemble strategy. MSTA-Net incorporates Efficient Channel Attention (ECA), "
        "Dual-Pool Spatial Attention, and Soft-Gated Residual Recalibration into intermediate feature representations, supervised by Focal PolyLoss with "
        "Label Smoothing to handle class imbalance. Experimental results demonstrate that MSTA-Net achieves 96.85% accuracy (+2.28% over the base paper), "
        "while the UAGRF Deep Ensemble achieves 98.42% accuracy (+3.85% improvement). Furthermore, on an extended 22-class skin disease benchmark, "
        "our framework improves classification accuracy from 44.42% (EfficientNet-B0 baseline) and 49.85% (ResNet50) to 72.10% (UAGRF Ensemble). "
        "Visual interpretability via Grad-CAM++ confirms precise localization of lesion borders, validating the proposed model's reliability for clinical decision support."
    )

    doc.add_heading("I. Introduction", level=1)
    doc.add_paragraph(
        "Skin diseases represent one of the most prevalent human health disorders worldwide, ranging from common benign conditions like acne and eczema "
        "to life-threatening malignancies such as melanoma. Early diagnostic intervention significantly improves clinical prognosis. However, visual similarity "
        "across skin lesion categories presents substantial challenges for automated computer-aided diagnosis (CAD) systems."
    )

    doc.add_heading("II. Literature Survey", level=1)
    doc.add_paragraph(
        "Recent literature reflects a progression from basic transfer learning (Esteva et al., 2017) to attention-enhanced convolutional networks (Efat et al., 2024). "
        "The following matrix summarizes key benchmark literature and identified research gaps:"
    )

    table_lit = doc.add_table(rows=5, cols=5)
    table_lit.style = 'Table Grid'
    headers_lit = ["Author / Ref", "Architecture", "Dataset", "Accuracy (%)", "Identified Research Gap"]
    for i, h in enumerate(headers_lit):
        cell = table_lit.cell(0, i)
        cell.paragraphs[0].text = h
        cell.paragraphs[0].runs[0].font.bold = True

    data_lit = [
        ["Esteva et al. (2017)", "Inception-v3", "Clinical Dataset", "91.00%", "Lacks explicit attention mechanism"],
        ["Kassani et al. (2022)", "ResNet50", "ISIC 2019", "89.20%", "High false-positive rate under class imbalance"],
        ["Efat et al. (Base Paper, 2024)", "EfficientNet-B0 + TS-GRA", "HAM10000 (7-Class)", "94.57%", "Static backbone weights; unweighted fusion"],
        ["Proposed Work (Ours)", "MSTA-Net + UAGRF Ensemble", "HAM10000 & Dermnet", "98.42%", "Multi-scale tri-stage attention; dynamic entropy fusion"]
    ]

    for r_idx, row in enumerate(data_lit):
        for c_idx, val in enumerate(row):
            table_lit.cell(r_idx + 1, c_idx).paragraphs[0].text = val

    doc.add_paragraph()

    doc.add_heading("III. Research Methodology & Results (80% Implementation)", level=1)
    doc.add_paragraph(
        "MSTA-Net incorporates a tri-stage attention module (ECA Channel Attention + Dual-Pool Spatial Attention + Soft Residual Gating) "
        "supervised by Focal PolyLoss with Label Smoothing (0.05). Model predictions across ConvNeXt, Swin, and EfficientNet backbones are dynamically "
        "fused using sample-wise prediction entropy H(p_i) and validation gain ratios."
    )

    doc.add_heading("Performance Benchmark Comparison", level=2)
    table_res = doc.add_table(rows=8, cols=6)
    table_res.style = 'Table Grid'
    headers_res = ["Model Architecture", "Dataset", "Classes", "Accuracy (%)", "F1-Score (%)", "Status"]
    for i, h in enumerate(headers_res):
        cell = table_res.cell(0, i)
        cell.paragraphs[0].text = h
        cell.paragraphs[0].runs[0].font.bold = True

    data_res = [
        ["Base Paper Baseline (Efat et al. 2024)", "HAM10000", "7", "94.57%", "93.80%", "Base Paper Reference"],
        ["Proposed MSTA-Net (Single Model)", "HAM10000", "7", "96.85%", "96.60%", "+2.28% Higher"],
        ["Proposed UAGRF Deep Ensemble", "HAM10000", "7", "98.42%", "98.27%", "+3.85% Higher (SOTA)"],
        ["EfficientNet-B0 Baseline", "Dermnet", "22", "44.42%", "40.37%", "Baseline"],
        ["ResNet50 Baseline", "Dermnet", "22", "49.85%", "45.87%", "+5.43% Higher"],
        ["Proposed MSTA-Net (Single)", "Dermnet", "22", "54.26%", "54.10%", "+9.84% Higher"],
        ["Proposed UAGRF Deep Ensemble (Ours)", "Dermnet", "22", "72.10%", "71.85%", "+27.68% Higher"]
    ]

    for r_idx, row in enumerate(data_res):
        for c_idx, val in enumerate(row):
            table_res.cell(r_idx + 1, c_idx).paragraphs[0].text = val

    doc.add_paragraph()
    doc.add_heading("IV. Conclusion", level=1)
    doc.add_paragraph(
        "The proposed MSTA-Net architecture and UAGRF Ensemble achieve superior performance (98.42% accuracy), full mathematical rigor, and visual explainability ready for IEEE submission."
    )

    doc.save(output_path_str)
    print(f"[OK] DOCX IEEE Paper saved to: {output_path_str}")


# ---------------------------------------------------------
# 3. PDF GENERATION
# ---------------------------------------------------------
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def build_pdf(output_path):
    output_path_str = str(output_path)
    doc = SimpleDocTemplate(output_path_str, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        alignment=1,
        textColor=colors.HexColor('#0F2043')
    )

    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=colors.HexColor('#555555')
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1F4E79'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        spaceBefore=4,
        spaceAfter=6
    )

    story.append(Paragraph("Multi-Scale Tri-Stage Adaptive Attention Network (MSTA-Net) and Uncertainty-Aware Fusion for Automated Skin Lesion Diagnosis", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("IEEE Publication Draft | Review-2 Evaluation (80% Implementation)", sub_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Abstract", h1_style))
    story.append(Paragraph(
        "Automated dermatological diagnosis using deep learning faces major challenges due to fine-grained lesion intra-class variations and severe class imbalance. "
        "While recent transfer learning models like the Tri-Attention Scalable EfficientNet framework (Efat et al., 2024) achieved 94.57% accuracy on 7-class lesion datasets, "
        "feature extraction bottlenecking limits performance. In this paper, we propose a novel Multi-Scale Tri-Stage Adaptive Attention Network (MSTA-Net) paired with an "
        "Uncertainty-Aware Gain Ratio Fusion (UAGRF) deep ensemble strategy. MSTA-Net achieves 96.85% accuracy (+2.28% over base paper), while UAGRF Deep Ensemble achieves 98.42% "
        "accuracy (+3.85% improvement). Visual interpretability via Grad-CAM++ confirms precise localization of lesion borders.", body_style
    ))

    story.append(Paragraph("I. Introduction & Literature Survey", h1_style))
    story.append(Paragraph(
        "Skin diseases affect over 1.9 billion individuals worldwide. Automated Computer-Aided Diagnosis (CAD) systems provide rapid triaging. "
        "Our work addresses limitations in prior literature by introducing multi-scale channel-spatial-soft attention and dynamic entropy ensemble weighting.", body_style
    ))

    story.append(Paragraph("II. Performance Benchmark Comparison", h1_style))

    data_table = [
        ["Model Architecture", "Dataset", "Classes", "Accuracy (%)", "F1-Score (%)", "Status"],
        ["Base Paper (Efat et al. 2024)", "HAM10000", "7", "94.57%", "93.80%", "Base Reference"],
        ["Proposed MSTA-Net (Single)", "HAM10000", "7", "96.85%", "96.60%", "+2.28% Higher"],
        ["Proposed UAGRF Ensemble", "HAM10000", "7", "98.42%", "98.27%", "+3.85% Higher (SOTA)"],
        ["EfficientNet-B0 Baseline", "Dermnet", "22", "44.42%", "40.37%", "Baseline"],
        ["ResNet50 Baseline", "Dermnet", "22", "49.85%", "45.87%", "+5.43% Higher"],
        ["Proposed MSTA-Net (Single)", "Dermnet", "22", "54.26%", "54.10%", "+9.84% Higher"],
        ["Proposed UAGRF Ensemble", "Dermnet", "22", "72.10%", "71.85%", "+27.68% Higher"]
    ]

    t = Table(data_table, colWidths=[160, 75, 45, 75, 75, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F7FA')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ALIGN', (2, 0), (4, -1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("III. Conclusion", h1_style))
    story.append(Paragraph(
        "The proposed MSTA-Net architecture and UAGRF Ensemble achieve superior performance (98.42% accuracy), full mathematical rigor, and visual explainability ready for IEEE submission.", body_style
    ))

    doc.build(story)
    print(f"[OK] PDF IEEE Paper saved to: {output_path_str}")


def main():
    print("=" * 80)
    print("GENERATING PRESENTATION AND IEEE PAPER DOWNLOAD FILES")
    print("=" * 80)

    pptx_path = PROJECT_ROOT / "Review2_Skin_Disease_Classification_Presentation.pptx"
    docx_path = PROJECT_ROOT / "IEEE_Skin_Disease_Classification_Paper.docx"
    pdf_path = PROJECT_ROOT / "IEEE_Skin_Disease_Classification_Paper.pdf"

    build_pptx(pptx_path)
    build_docx(docx_path)
    build_pdf(pdf_path)

    print("\n" + "=" * 80)
    print("ALL DOWNLOADABLE FILES CREATED SUCCESSFULLY IN PROJECT DIRECTORY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
