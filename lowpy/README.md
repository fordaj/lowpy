# Developer Notes
Notes relating to package development.

## Version incrementing
1. Delete dist and build folders
2. Build distributions
```
python3 setup.py sdist bdist_wheel
```
3. Post to PyPI
```
python3 -m twine upload dist/*
```
4. To upgrade, enter the current version as an argument in pip3:
```
pip3 install lowpy==1.0.0
```
Note: May have to wait a minute or so for the new files to upload to PyPI.

## Package a non-``.py`` file
1. If not already created, add ``MANIFEST.in`` to the package root
2. In ``MANIFEST.in``, include your non-``.py`` file
```
include subfolder/data.txt
```
3. To print out the file in a packaged ``.py`` file:
```python
import pkg_resources
file = pkg_resources.resource_filename('subfolder', 'data.txt')
print(open(file).read())