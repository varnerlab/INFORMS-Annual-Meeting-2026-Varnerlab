# Build the INFORMS 2026 deck with XeLaTeX. `make` builds informs2026.pdf in the theme set in informs2026.tex;
# `make both` builds informs2026-navy.pdf and informs2026-white.pdf to compare; `make clean` removes the build files.
DECK := informs2026
SOURCES := $(DECK).tex vnslides.sty vnflow.sty $(wildcard slides/*.tex figs/*.tex data/*)

$(DECK).pdf: $(SOURCES)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(DECK).tex

# one deck per theme: the theme is defined before the deck file is read, so the deck's own \providecommand line
# leaves it alone
both: $(DECK)-navy.pdf $(DECK)-white.pdf

$(DECK)-%.pdf: $(SOURCES)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -jobname=$(DECK)-$* -usepretex='\def\vntheme{$*}' $(DECK).tex

# the speaker notes alone, one page per note, in the configured theme
notes: $(DECK)-notes.pdf

$(DECK)-notes.pdf: $(SOURCES)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -jobname=$(DECK)-notes -usepretex='\def\vnshownotes{}' $(DECK).tex

clean:
	latexmk -C $(DECK).tex
	latexmk -C -jobname=$(DECK)-navy $(DECK).tex
	latexmk -C -jobname=$(DECK)-white $(DECK).tex
	latexmk -C -jobname=$(DECK)-notes $(DECK).tex

.PHONY: both notes clean
