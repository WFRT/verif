.PHONY: test other/verif.sh dist

default: nothing

nothing:
	@ echo "This makefile does not build verif, use setup.py"

VERSION=$(shell grep __version__ verif/version.py | cut -d"=" -f2 | sed s"/ //g" | sed s"/'//g")
test:
	coverage run --source verif -m unittest discover

coverage: test
	coverage report --precision 2
	coverage html -d pages/coverage
	@echo "Coverage created in pages/coverage"

# Creating distribution for pip
dist:
	echo $(VERSION)
	rm -rf dist
	python3.8 setup.py sdist
	python3.8 setup.py bdist_wheel
	@ echo "Next, run 'twine upload dist/*'"

clean:
	python setup.py clean
	rm -rf build/
	find . -name '*.pyc' -delete
	rm -rf deb_dist
	rm -rf verif.egg-info

lint:
	# python verif/tests/pep8_test.py
	pylint -d C,R,W verif/*.py verif/tests/*.py

count:
	@wc -l verif/*.py | tail -1

other/verif.sh:
	python other/create_bash_completion.py > other/verif.sh
