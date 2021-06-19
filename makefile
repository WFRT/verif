# Use this file to build debian packages
# Create a file called stdb.cfg in this directory with the following
# contents, where # "precise" is your linux version:
# [DEFAULT]
# Suite: precise
.PHONY: other/verif.sh dist

default: nothing

nothing:
	@ echo "This makefile does not build verif, use setup.py"

VERSION=$(shell grep __version__ verif/version.py | cut -d"=" -f2 | sed s"/ //g" | sed s"/'//g")
coverage:
	#nosetests --with-coverage --cover-erase --cover-package=verif --cover-html --cover-branches
	nosetests --with-coverage --cover-erase --cover-package=verif --cover-html

test:
	nosetests

# Creating distribution for pip
dist:
	echo $(VERSION)
	rm -rf dist
	python3 setup.py sdist
	python3 setup.py bdist_wheel
	@ echo "Next, run 'twine upload dist/*'"
	@ echo "Next, run 'twine upload dist/*'"
clean:
	python setup.py clean
	rm -rf build/
	find . -name '*.pyc' -delete
	rm -rf deb_dist
	rm -rf verif.egg-info

lint:
	python verif/tests/pep8_test.py

count:
	@wc -l verif/*.py | tail -1

other/verif.sh:
	python other/create_bash_completion.py > other/verif.sh
