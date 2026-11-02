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
# arXiv publishes the source, so the archive gets copies of the .tex files with their comments stripped
# (strip_comments.pl). The .bib stays out: arXiv compiles from the .bbl.
STAGE := arxiv-stage
TEX := $(PAPER).tex $(SECTIONS) $(TABLES)
SHIPPED := $(TEX) neurips_2026.sty $(FIGS) $(PAPER).bbl

$(ARCHIVE): $(PDF) $(SOURCES) Makefile strip_comments.pl
	rm -rf $(STAGE) && mkdir $(STAGE)
	COPYFILE_DISABLE=1 tar -cf - $(SHIPPED) | tar -xf - -C $(STAGE)
	cd $(STAGE) && perl ../strip_comments.pl $(TEX)
	cd $(STAGE) && COPYFILE_DISABLE=1 tar -czf ../$(ARCHIVE) $(SHIPPED)
	rm -rf $(STAGE)

watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(PAPER)

clean:
	$(LATEXMK) -c $(PAPER)
	rm -rf $(STAGE)
	rm -f $(PAPER).bbl $(PAPER).run.xml $(PAPER).synctex.gz $(PAPER).fff $(PAPER).ttt

distclean: clean
	rm -f $(PDF) $(ARCHIVE)
