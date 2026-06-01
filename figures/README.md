# Figures

| File | Used in main README | Notes |
|---|---|---|
| `framework.pdf` | — (source) | Original PDF of the pipeline overview figure. |
| `framework.png` | top of README, after the author block | Rendered from `framework.pdf` at 200 DPI via `pymupdf`. Re-render after editing the PDF: `python -c "import pymupdf; d=pymupdf.open('figures/framework.pdf'); d[0].get_pixmap(dpi=200).save('figures/framework.png')"`. |
