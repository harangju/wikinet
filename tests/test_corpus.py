import os
import wikinet

print('hello world')

base_path = os.path.join(
    '/', 'Users','harangju','Developer','data','wiki','dumps'
)
dump = wikinet.Dump(
    path_xml=os.path.join(base_path, 'enwiki-20190801-pages-articles-multistream.xml.bz2'),
    path_idx=os.path.join(base_path, 'enwiki-20190801-pages-articles-multistream-index.txt.bz2')
)
print(wikinet.Corpus(dump, load_index=False))
