Make sure
=========

- Check that we have the right version (verif/version.py)
  Should be 1.0.0-alpha.1 or 1.0.0
- make sure that travis passes

Uploading to pip
================
Build package:
make dist

Upload:
twine upload dist/* -r testpypi     # To test

or

twine upload dist/*                 # To upload

Add the tag:
git tag <tag name>
git push
git push --tags
