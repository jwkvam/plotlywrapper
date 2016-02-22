all: test

test:
	py.test --cov=./ --pylint --pylint-rcfile=pylintrc --pylint-error-types=EF

loop:
	py.test --pylint --pylint-rcfile=pylintrc --pylint-error-types=EF -f

debug:
	py.test --pylint --pylint-rcfile=pylintrc --pylint-error-types=EF --pdb -s
