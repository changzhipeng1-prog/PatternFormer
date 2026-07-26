"""Assemble the a2a4 composite panel from the four standalone figures:
  row1 = examples (2x5) [A] | row2 = box (1x4) [B] | row3 = bifurcation [C] | efficiency [D]
Each A/B/C/D label sits in a white band ABOVE its row, so it never overlaps any
axis title / ylabel / ticks.  Writes a raster PNG (PIL) and a VECTOR PDF (pypdf
places the sub-PDFs; labels are a matplotlib vector-text overlay).  From paper/2D/ :
    python make_panel_2D.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

PDF = dict(ex="test/fig_a2a4_examples_2x5.pdf", box="test/fig_a2a4_box4.pdf",
           bif="data_gen/fig_a2a4_bifurcation.pdf", eff="test/timing/fig_a2a4_efficiency_main.pdf")
PNG = dict(ex="test/fig_a2a4_examples_2x5.png", box="test/fig_a2a4_box4.png",
           bif="data_gen/fig_a2a4_bifurcation.png", eff="test/timing/fig_a2a4_efficiency_main.png")
GAP = 18.0          # horizontal gap between bifurcation and efficiency in row 3


def make_pdf(out="fig_a2a4_panel.pdf"):
    from pypdf import PdfReader, PdfWriter, Transformation, PageObject
    import io, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pg = {k: PdfReader(v).pages[0] for k, v in PDF.items()}
    wh = lambda p: (float(p.mediabox.width), float(p.mediabox.height))
    w_ex, h1 = wh(pg["ex"]); W = w_ex
    w_box, h_box = wh(pg["box"]); w_bif, h_bif = wh(pg["bif"]); w_eff, h_eff = wh(pg["eff"])
    s2 = W / w_box; h2 = h_box * s2
    a_bif, a_eff = w_bif / h_bif, w_eff / h_eff
    Hc = (W - GAP) / (a_bif + a_eff); s_bif, s_eff = Hc / h_bif, Hc / h_eff; eff_w = w_eff * s_eff
    total_h = Hc + h2 + h1                                    # no bands -- rows stacked directly
    page = PageObject.create_blank_page(width=W, height=total_h)
    page.merge_transformed_page(pg["ex"],  Transformation().scale(W / w_ex).translate(0, Hc + h2))
    page.merge_transformed_page(pg["box"], Transformation().scale(s2).translate(0, Hc))
    page.merge_transformed_page(pg["bif"], Transformation().scale(s_bif).translate(0, 0))
    page.merge_transformed_page(pg["eff"], Transformation().scale(s_eff).translate(W - eff_w, 0))
    LB = W * 0.05
    pad = LB * 0.12
    # A,B overlaid at the top-left corner of rows 1,2 (compact, no separate band);
    # C,D are rendered INSIDE the row-3 sub-figures.
    labels = [("A", pad, total_h - pad), ("B", pad, Hc + h2 - pad)]
    fig = plt.figure(figsize=(W / 72.0, total_h / 72.0))
    for lab, x, y in labels:
        fig.text(x / W, y / total_h, lab, fontsize=LB * 0.85, fontweight="bold",
                 va="top", ha="left", family="sans-serif",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    buf = io.BytesIO(); fig.savefig(buf, format="pdf", transparent=True); plt.close(fig); buf.seek(0)
    page.merge_page(PdfReader(buf).pages[0])
    w = PdfWriter(); w.add_page(page)
    with open(out, "wb") as f:
        w.write(f)
    print("saved", out)


def make_png(out="fig_a2a4_panel.png", W=3000):
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    im = {k: Image.open(v).convert("RGB") for k, v in PNG.items()}
    rw = lambda img, width: img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    r1, r2 = rw(im["ex"], W), rw(im["box"], W)
    a_bif = im["bif"].width / im["bif"].height; a_eff = im["eff"].width / im["eff"].height
    Hc = round((W - GAP) / (a_bif + a_eff))
    bif = im["bif"].resize((round(Hc * a_bif), Hc), Image.LANCZOS)
    eff = im["eff"].resize((round(Hc * a_eff), Hc), Image.LANCZOS)
    r3 = Image.new("RGB", (W, Hc), (255, 255, 255)); r3.paste(bif, (0, 0)); r3.paste(eff, (W - eff.width, 0))
    total_h = r1.height + r2.height + r3.height               # no bands -- rows stacked
    canvas = Image.new("RGB", (W, total_h), (255, 255, 255))
    yr = [0, r1.height, r1.height + r2.height]                # row tops
    for r, y in zip((r1, r2, r3), yr):
        canvas.paste(r, (round((W - r.width) / 2), y))
    draw = ImageDraw.Draw(canvas)
    LB = round(W * 0.05)
    try:
        fp = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
        font = ImageFont.truetype(fp, round(LB * 0.85))
    except Exception:
        font = ImageFont.load_default()
    pad = round(LB * 0.18)
    # A,B overlaid at the top-left corner of rows 1,2 (white box behind for legibility);
    # C,D are rendered INSIDE the row-3 sub-figures.
    for lab, ytop in (("A", 0), ("B", r1.height)):
        bb = draw.textbbox((0, 0), lab, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        x, y = pad, ytop + pad
        draw.rectangle([x - 6, y - 4, x + tw + 6, y + th + 10], fill=(255, 255, 255))
        draw.text((x, y - bb[1]), lab, fill=(0, 0, 0), font=font)
    canvas.save(out, dpi=(200, 200)); print("saved", out)


if __name__ == "__main__":
    make_png()
    make_pdf()
