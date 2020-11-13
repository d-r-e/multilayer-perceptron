NAME= multilayer-perceptron

SRC = src/*.py multilayer-perceptron.py

DOCS = docs/*.pdf

DATA = data/*.csv

test:

push:
	git add $(SRC) $(DOCS) $(DATA) makefile
	git commit -m "multilayer perceptron 42 madrid"
	git push