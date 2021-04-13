import sys
import pytest

print('hello world')
print(sys.path)

import networkx as nx
import wikinet

print(wikinet.Dump(path_xml='', path_idx=''))
