# Use this file to build debian packages
# Create a file called stdb.cfg in this directory with the following
# contents, where # "precise" is your linux version:
# [DEFAULT]
# Suite: precise
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

# Bash completion script
# 1) Install verif as normal (so that verif command is availble on the path)
# 2) Make this target
# 3) Move verif.sh into wherever bash completion scripts are kept (e.g. /etc/bash_completion.d or run the
#    script in your ~/.bashrc
verif.sh:
	 verif | grep "^[ ]* -" | awk '{print $$1}' > verifOptions.txt
	 python bashCompletion.py verifOptions.txt > $@
	 rm verifOptions.txt
