# Build the INFORMS 2026 deck with XeLaTeX. `make` builds informs2026.pdf; `make clean` removes the build files.
DECK := informs2026

$(DECK).pdf: $(DECK).tex vnslides.sty vnflow.sty $(wildcard slides/*.tex figs/*.tex data/*)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(DECK).tex

clean:
	latexmk -C $(DECK).tex

.PHONY: clean
