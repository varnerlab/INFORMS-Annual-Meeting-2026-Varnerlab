PAPER := Paper_v1
BIB := References_v1
PDF := $(PAPER).pdf
ARCHIVE := INFORMS-2026-arxiv-source.tar.gz

SECTIONS := $(wildcard sections/*.tex)
TABLES := $(wildcard sections/tables/*.tex)
FIGS := $(wildcard figs/main/*.pdf) $(wildcard figs/supplement/*.pdf)
SOURCES := $(PAPER).tex $(BIB).bib neurips_2026.sty $(SECTIONS) $(TABLES) $(FIGS)

LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -bibtex -interaction=nonstopmode -halt-on-error

.PHONY: all pdf figs source watch clean distclean

all: pdf

pdf: $(PDF)

$(PDF): $(SOURCES)
	$(LATEXMK) $(LATEXMK_FLAGS) $(PAPER)

# The figures are drawn by the talk's own figure code and data (talk/figs, talk/data); see figs/src/build_figs.sh.
figs:
	sh figs/src/build_figs.sh

source: $(ARCHIVE)

# Build first so the release archive includes the current generated bibliography.
$(ARCHIVE): $(PDF) $(SOURCES) Makefile
	COPYFILE_DISABLE=1 tar -czf $(ARCHIVE) $(SOURCES) $(PAPER).bbl

watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(PAPER)

clean:
	$(LATEXMK) -c $(PAPER)
	rm -f $(PAPER).bbl $(PAPER).run.xml $(PAPER).synctex.gz $(PAPER).fff $(PAPER).ttt

distclean: clean
	rm -f $(PDF) $(ARCHIVE)
