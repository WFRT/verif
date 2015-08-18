# Use this file to build debian packages
# Create a file called stdb.cfg in this directory with the following
# contents, where # "precise" is your linux version:
# [DEFAULT]
# Suite: precise
VERSION=$(shell grep version= setup.py | cut -d"'" -f2)
coverage:
	#nosetests --with-coverage --cover-erase --cover-package=verif --cover-html --cover-branches
	nosetests --with-coverage --cover-erase --cover-package=verif --cover-html

test:
	nosetests

deb_dist: makefile
	rm -rf deb_dist
	python setup.py --command-packages=stdeb.command bdist_deb
	#cd deb_dist/verif-$(shell grep version= setup.py | cut -d"'" -f2)/; debuild -S -sa
	cd deb_dist/verif-$(VERSION)/; debuild -S -sa

clean:
	python setup.py clean
	rm -rf build/
	find . -name '*.pyc' -delete
	rm -rf deb_dist
	rm -rf verif.egg-info
