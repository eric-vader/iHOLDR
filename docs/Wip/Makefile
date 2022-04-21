TUT:=$(shell basename $(CURDIR))
TARGETS = Notes 

default: $(TARGETS)

$(TARGETS):%:$(TUT)_%.md
	pandoc --highlight-style pygments  $(TUT)_$@.md -V geometry:margin=2cm,a4paper --pdf-engine=xelatex -o $(TUT)_$@.pdf --citeproc

Notes_Slides: $(TUT)_Notes.md
	pandoc --highlight-style pygments -t beamer -V theme:metropolis -V aspectratio:169 $(TUT)_Tutorial.md -o $(TUT)_$@.pdf
