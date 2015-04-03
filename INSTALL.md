Installing Simplelearn
======================

1. Install the dependencies listed in README.md
2. Add the project directory (containing this file) to your PYTHONPATH.

### To create documentation:

    $ sphinx-apidoc -o sphinx/ -F -H Simplelearn -A "Matthew Koichi Grimes" -V 0.1 -R 0.1 -f simplelearn/
    $ cd sphinx
    $ make html
