# Use this file to build debian packages
# Create a file called stdb.cfg in this directory with the following
# contents, where # "precise" is your linux version:
# [DEFAULT]
# Suite: precise
.PHONY: other/verif.sh

VERSION=$(shell grep __version__ verif/version.py | cut -d"=" -f2 | sed s"/ //g" | sed s"/'//g")
coverage:
	#nosetests --with-coverage --cover-erase --cover-package=verif --cover-html --cover-branches
	nosetests --with-coverage --cover-erase --cover-package=verif --cover-html

test:
	nosetests

deb_dist: makefile
	echo $(VERSION)
	rm -rf deb_dist
	python setup.py --command-packages=stdeb.command bdist_deb
	cd deb_dist/verif-$(VERSION)/ || exit; debuild -S -sa

clean:
	python setup.py clean
	rm -rf build/
	find . -name '*.pyc' -delete
	rm -rf deb_dist
	rm -rf verif.egg-info
	rm -f verif.sh verifOptions.txt

count:
	@wc -l verif/*.py | tail -1

other/verif.sh:
	python other/create_bash_completion.py > verif.sh
