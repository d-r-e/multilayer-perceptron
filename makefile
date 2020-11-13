NAME= multilayer-perceptron

SRC = src/*.py multilayer-perceptron.py

DOCS = docs/*.pdf

DATA = data/*.csv

test:

push:
	git add $(SRC) $(DOCS) $(DATA) makefile
	git commit -m "perceptron as in TowardsDataScience"
	git push