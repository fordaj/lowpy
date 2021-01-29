# Developer Notes
Notes relating to package development.

# Version incrementing
1. Increment the version in setup.py
2. Delete dist and build folders
3. Build distributions
```
python3 setup.py sdist bdist_wheel
```
4. Post to PyPI
```
python3 -m twine upload dist/*
```
5. To upgrade, enter the current version as an argument in pip3:
```
pip3 install lowpy==1.0.0
```
Note: May have to wait a minute or so for the new files to upload to PyPI.



# Package a non-``.py`` file
1. Include package data in ``setup.py`` by adding the following argument:
```
include_package_data=True,
```
2. If not already created, add ``MANIFEST.in`` to the package root
3. In ``MANIFEST.in``, include your non-``.py`` file
```
include subfolder/data.txt
```
4. To print out the file in a packaged ``.py`` file:
```python
import pkg_resources
file = pkg_resources.resource_filename('subfolder', 'data.txt')
print(open(file).read())
```