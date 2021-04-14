.. _usage:

Usage
=====

Installation
~~~~~~~~~~~~
Run ``pip install wikinet``.

Reading zipped Wikipedia XML dumps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  dump = wiki.Dump(path_xml, path_index)
  page = dump.load_page('Science')
  print(page)
  print(dump.page)
  print(dump.links) # all links
  print(dump.article_links) # just links to articles
  print(dump.years) # years in article (intro & history sections)


~~~~~~~~~~~~
