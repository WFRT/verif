.coverage: tests/
	nosetests --with-coverage --cover-package=verif --cover-html

cover: .coverage
