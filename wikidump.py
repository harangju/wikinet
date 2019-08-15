import bz2
import os
import xml.etree.ElementTree as ET
import mwparserfromhell as mph
import re

class WikiDump():
    def __init__(self, path_xml, path_idx):
        self._idx = {}
        self._links = []
        self._page = None
        self.path_xml = path_xml
        self.path_idx = path_idx
        
    def get_idx(self):
        if self._idx:
            return self._idx
        else:
            print('WikiDump: Loading index...')
            with bz2.BZ2File(self.path_idx, 'rb') as file:
                lines = [line for line in file]
            block_end = os.path.getsize(self.path_xml)
            offset_prev = block_end
            for line in reversed(lines):
                [offset, pid, name] = line.strip().split(b':', 2)
                offset, pid, name = (int(offset), int(pid), name.decode('utf8'))
                block_end = offset_prev if offset < offset_prev else block_end
                self._idx[name] = (offset, pid, block_end-offset)
                offset_prev = offset
            print('WikiDump: Loaded.')
            return self._idx
    idx = property(get_idx)
    
    def get_links(self):
        if self._links:
            return self._links
        elif self.page:
            self._links = [x.title for x in self.page.filter_wikilinks()]
            return self._links
    links = property(get_links)
    
    def get_page(self):
        return self._page
    
    def set_page(self, page):
        self._page = page
        self._links = []
    page = property(get_page, set_page)
    
    def load_page(self, page_name, filter_top=False):
        if page_name not in self.idx.keys():
            self.page = None
            return
        offset, pid, block_size = self.idx[page_name]
        xml = WikiDump.fetch_block(self.path_xml, offset, block_size)
        root = ET.fromstring(b'<root>' + xml + b'</root>')
        text = WikiDump.search_id(root, pid)
        text = WikiDump.filter_top_section(text) if filter_top else text
        self.page = mph.parse(text)
        return self.page
    
    @staticmethod
    def fetch_block(path, offset, block_size):
        with open(path, 'rb') as file:
            file.seek(offset)
            return bz2.decompress(file.read(block_size))
    
    @staticmethod
    def search_id(root, pid):
        for page in root.iter('page'):
            if pid == int(page.find('id').text):
                return page.find('revision').find('text').text
    
    @staticmethod
    def filter_top_section(text):
        head = re.search(r'==.*?==', text)
        idx = head.span(0)[0] if head else len(text)
        return text[:idx] #(text[:idx], text[idx:])