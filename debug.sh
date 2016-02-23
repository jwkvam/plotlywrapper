#!/bin/sh


py.test --pylint --pylint-rcfile=pylintrc --pylint-error-types=EF --pdb -s $@
