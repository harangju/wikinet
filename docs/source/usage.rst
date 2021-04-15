.. _usage:

Usage
=====

Installation
~~~~~~~~~~~~
Run ``pip install wikinet``. Then, ``import wikinet as wiki``.

Reading zipped Wikipedia XML dumps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    dump = wiki.Dump(path_xml, path_index)
    page = dump.load_page('Science')


Then, you can view the page and information about the page. ::

    print(page)
    print(dump.page)
    print(dump.links) # all links
    print(dump.article_links) # just links to articles
    print(dump.years) # years in article (intro & history sections)

Creating a network of Wikipedia articles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    network = wiki.Net.build_graph(
        name='my network', dump=dump, nodes=['Science', 'Mathematics', 'Philosophy']
    )

Optionally, for edge weights with cosine distance between ``tf-idf`` vectors of articles

::

    network = wiki.Net.build_graph(
        name='my network', dump=dump, nodes=['Science', 'Mathematics', 'Philosophy'],
        model=tfidf, # gensim.models
        dct=dct, # gensim.corpora.Dictionary
    )

Then, ``network.graph`` gives you a ``networkx.DiGraph``.
