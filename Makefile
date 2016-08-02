all: test

test:
	py.test --cov=./ --pylint --pylint-rcfile=pylintrc --pylint-error-types=WCREF

loop:
	py.test --pylint --pylint-rcfile=pylintrc --pylint-error-types=WCREF -f

debug:
	py.test --pylint --pylint-rcfile=pylintrc --pylint-error-types=WCREF -s --pdb

upload:
	flit wheel --upload
