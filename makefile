coverage:
	#nosetests --with-coverage --cover-erase --cover-package=verif --cover-html --cover-branches
	nosetests --with-coverage --cover-erase --cover-package=verif --cover-html

test:
	nosetests
