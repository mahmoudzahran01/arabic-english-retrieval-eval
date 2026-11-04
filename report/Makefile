# Technical report: tables from the committed result files, then LaTeX.
#   make tables   regenerate tables/*.tex and tables/numbers.tex (every number of the text)
#   make figures  report/figures/*.pdf (the WiSE curve; skipped with a message if results/retrieval_v2/alpha_curve.json does not exist yet)
#   make pdf      tables + figures + pdflatex/bibtex
# acl.sty and acl_natbib.bst are the official *ACL style files (https://github.com/acl-org/acl-style-files); they are not covered by this repository's licence.
PY ?= python

.PHONY: tables figures pdf clean

tables:
	$(PY) make_tables.py

figures:
	$(PY) make_figures.py

pdf: tables figures
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out
