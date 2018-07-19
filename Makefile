all: test

test:
	py.test --cov=./ --docstyle --codestyle --pylint --pylint-rcfile=pylintrc --pylint-error-types=WCREF --ignore=doc

black_check:
	black -S -l 100 --check *.py 

lint:
	py.test --pylint -m pylint --pylint-rcfile=pylintrc --pylint-error-types=WCREF --ignore=doc

loop:
	py.test --pylint --pylint-rcfile=pylintrc --pylint-error-types=WCREF -f --ignore=doc

debug:
	py.test --pylint --pylint-rcfile=pylintrc --pylint-error-types=WCREF -s --pdb --ignore=doc

black:
	black -l 100 -S *.py

upload:
	flit wheel --upload
