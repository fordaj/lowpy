### Developer Note
For version incrementing:
1. Delete dist and build folders
2. Build distributions
```
python3 setup.py sdist bdist_wheel
```
3. Post to PyPI
```
python3 -m twine upload dist/*
```
4. To upgrade:
```
pip3 uninstall lowpy
...
pip3 install lowpy
```
Note: May have to wait a few seconds for the new files to upload to PyPI.