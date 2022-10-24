============
Installation
============

Standard installation
---------------------

The easiest way to install `kalmankit` is through pip:

.. code-block:: bash

   pip install kalmankit

Developer installation
----------------------

If you intend to use the library as a developer, you should clone the
repository and then install the library as an editable, using the requirements
file to build your environment:

.. code-block:: bash

   git clone https://github.com/Xylambda/kalmankit.git
   pip install -e kalmankit/. -r kalmankit/requirements-dev.txt


Running tests
-------------
The library makes use of `pytest` to develop the testing framework. Any
combination of commands that use the aforementioned library can be used. For
example:

.. code-block:: bash

   pytest -vv tests/. --color=yes