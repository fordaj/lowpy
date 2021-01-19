### Developer Note
For version incrementing:
1. Build distributions
```
python3 setup.py sdist bdist_wheel
```
2. Post to PyPI
```
python3 -m twine upload dist/*
```
3. To upgrade:
```
pip3 uninstall lowpy
...
pip3 install lowpy
```