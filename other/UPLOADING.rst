Make sure
=========

- Check that we have the right version (verif/version.py)
  Should be 1.0.0-alpha.1 or 1.0.0
- make sure that travis passes

Uploading to pip
================
Run:
make dist

or

python setup.py sdist
python setup.py bdist_wheel

then

twine upload dist/* -r testpypi     # To test
twine upload dist/*                 # To upload

Add the tag
git push
git push --tags
