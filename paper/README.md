# ACM Extended Abstract Draft

Files:

- `acm_extended_abstract.tex`: anonymous ACM-style 4-page extended paper draft.
- `references.bib`: BibTeX references.
- `figures/main_interpretation_figure.jpg`: lightweight figure used in the paper.

Compile with a LaTeX environment that has `acmart` installed:

```bash
pdflatex acm_extended_abstract.tex
bibtex acm_extended_abstract
pdflatex acm_extended_abstract.tex
pdflatex acm_extended_abstract.tex
```

The draft is double-blind by default via:

```latex
\documentclass[sigconf,anonymous,review]{acmart}
```

## Research Use Warning

Warning: This code is for research and educational purposes only. Any clinical deployment requires IRB approval and prospective field validation.
