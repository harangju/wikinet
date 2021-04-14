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

Creating a network of Wikipedia articles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    topic = 'earth science'
    dump.load_page(f"Index of {topic} articles")
    links = [str(l) for l in dump.article_links]
    network = build_graph(name=topic, dump=dump, nodes=links)
    # optionally for edge weights with cosine distance between
    # tf-idf vectors of articles
    network = build_graph(
        name=topic, dump=dump, nodes=links,
        model=tfidf, # gensim.models
        dct=dct, # gensim.corpora.Dictionary
    )
