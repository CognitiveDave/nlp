## readME

_"Remember, a few hours of trial and error can save you several minutes of looking at the README"_
-------------

Create a virtual environment
```shell
$ virtualenv .
```
Install requirements
```shell
$ pip install -r requirements.txt
```
Check spacy compatibility
```shell
$ python -m spacy validate
```
Download spaCy en_core_web_sm model
```shell
$ pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
```
Run app
```python
$ python cvMatcher.py
```

__Well, Buenos dias.__
