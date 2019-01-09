=======================
Contributing to qhoptim
=======================

We want to make contributing to this project as easy and transparent as
possible.

Our Development Process
=======================

Changes and improvements will be released on an ongoing basis.

Pull Requests
=============

We actively welcome your pull requests.

1. Fork the repo and create your branch from ``master``.
2. If you've added code that should be tested, add tests_.
3. If you've changed APIs, update the documentation.
4. Ensure the `test suite`_ passes.
5. Make sure your code lints_.
6. If you haven't already, complete the `Contributor License Agreement ("CLA")`_.

.. _test suite: Tests_
.. _lints: `Coding Style`_

Contributor License Agreement ("CLA")
=====================================

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA `here`__.

__ https://code.facebook.com/cla

Issues
======

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Tests
=====

We use pytest as our unit testing framework, and all tests reside in the
``test_qhoptim/`` directory.

You may run tests by installing `pytest`__ and running the following from the
root directory of this source tree::

    $ pytest test_qhoptim/

__ https://github.com/pytest-dev/pytest


Coding Style
============

We use the `Black code style`__ with 120 character line length.

__ https://black.readthedocs.io/en/stable/the_black_code_style.html

You may autoformat your code by installing `Black`__ and running the following
from the root directory of this source tree::

    $ black .

__ https://github.com/ambv/black

In addition, you may run `flake8`__ to catch many glaring code issues.

__ https://github.com/PyCQA/flake8

License
=======

By contributing to qhoptim, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
