all: test

test:
	py.test --cov=./ --pylint --pylint-rcfile=pylintrc --pylint-error-types=REF

loop:
	py.test --pylint --pylint-rcfile=pylintrc --pylint-error-types=REF -f

debug:
	py.test --pylint --pylint-rcfile=pylintrc --pylint-error-types=REF -s --pdb

upload:
	flit wheel --upload
