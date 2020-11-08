NAME= multilayer-perceptron

SRC = src/*.py

DOCS = docs/*.pdf

DATA = data/*.csv

push:
	git add $(SRC) $(DOCS) $(DATA) makefile
	git commit -m "upload from 42 Madrid"
	git push