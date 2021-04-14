.. wikinet documentation master file, created by
   sphinx-quickstart on Wed Apr 14 10:28:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _contents:

wikinet
===================================================

``wikinet`` is a Python package. With ``wikinet``, you can

- read Wikipedia articles from zipped XML dumps,
- read through Wikipedia articles as a ``gensim`` corpus,
- create ``networkx`` networks from a list of article names, and
- run persistent homology on network.

See :ref:`usage` for details.

Data
----
Wikipedia XML dumps are available at https://dumps.wikimedia.org/enwiki. Only two files are required for reproduction: (1) ``enwiki-DATE-pages-articles-multistream.xml.bz2`` and (2) ``enwiki-DATE-pages-articles-multistream-index.txt.bz2``, where DATE is the date of the dump. Both files are multistreamed versions of the zipped files, which allow the user to access an article without unpacking the whole file. In the article, we used the archived zipped file from August 1, 2019, which is available in Dropbox_.

.. _Dropbox: https://www.dropbox.com/sh/kwsubhwf787p74k/AAA0Wf_3-SZggcvRYdrdzXBba?dl=0

Citation
--------
To cite ``wikinet`` please use the following publication: https://arxiv.org/abs/2010.08381.

Contents
--------

.. toctree::
   :maxdepth: 1

   usage
   reference
   license

Index
==================

* :ref:`genindex`
* :ref:`modindex`
