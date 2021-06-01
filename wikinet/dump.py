import os
import re
import bz2
import mwparserfromhell as mph
import xml.etree.ElementTree as ET

__all__ =['Dump']

class Dump:
    """``Dump`` loads and parses dumps from wikipedia from
    ``path_xml`` with index ``path_idx``.

    idx: dictionary
        ``{'page_name': (byte offset, page id, block size)}``
        Cached. Lazy.
    links: list of strings
        All links.
    article_links: list of strings
        Article links (not files, categories, etc.)
    years: list of int
        Years in the History section of a wikipedia page
        BC denoted as negative values
    page: mwparserfromhell.wikicode
        Current loaded wiki page
    path_xml: string
        Path to the zipped XML dump file.
    path_idx: string
        Path to the zipped index file.
    offset_max: int
        Maximum offset. Set as the size of the zipped dump.
    cache: xml.etree.ElementTree.Node
        Cache of the XML tree in current block
    """
    MAX_YEAR = 2020

    def __init__(self, path_xml, path_idx):
        self._idx = {}
        self._links = []
        self._article_links = []
        self._years = []
        self._page = None
        self.path_xml = path_xml
        self.path_idx = path_idx
        self.offset_max = 0
        self.cache = (0, None) # offset, cache

    @property
    def idx(self):
        if self._idx:
            return self._idx
        else:
            print('Dump: Loading index...')
            with bz2.BZ2File(self.path_idx, 'rb') as file:
                lines = [line for line in file]
            block_end = os.path.getsize(self.path_xml)
            offset_prev = block_end
            for line in reversed(lines):
                offset, pid, name = line.strip().split(b':', 2)
                offset, pid, name = (int(offset), int(pid), name.decode('utf8'))
                block_end = offset_prev if offset < offset_prev else block_end
                self._idx[name] = (offset, pid, block_end-offset)
                offset_prev = offset
            self.offset_max = max([x[0] for x in self._idx.values()])
            print('Dump: Loaded.')
            return self._idx

    @property
    def links(self):
        if self._links:
            return self._links
        elif self.page:
            self._links = [str(x.title) for x in self.page.filter_wikilinks()]
            self._links = [link.split('#')[0] for link in self._links]
            self._links = [link.split(' ') for link in self._links]
            self._links = [[words[0].capitalize()] + words[1:] for words in self._links]
            self._links = [' '.join(words) for words in self._links]
            return self._links
        else:
            return self._links

    @property
    def article_links(self):
        if self._article_links:
            return self._article_links
        elif self.links:
            self._article_links = [x for x in self.links if ':' not in x]
            return self._article_links
        else:
            return self._article_links

    @property
    def years(self):
        if self._years:
            return self._years
        elif self.page:
            history = Dump.get_history(self.page)
            top = self.page.get_sections()[0].strip_code()
            self._years = Dump.filter_years(top + history)
            return self._years
        else:
            return self._years

    @property
    def page(self):
        return self._page

    @page.setter
    def page(self, page):
        self._page = page
        self._links = []
        self._article_links = []
        self._years = []

    def load_page(self, page_name, filter_top=False):
        """Loads & returs page (``mwparserfromhell.wikicode``)
        named ``page_name`` from dump file. Returns only the
        top section if ``filter_top``.
        """
        page_name_words = page_name.split(' ')
        if len(page_name_words)>1 and page_name_words[0].islower():
            page_name = ' '.join(
                [page_name_words[0].capitalize()] +\
                [page_name_words[i] for i in range(1, len(page_name_words))]
            )
        if page_name not in self.idx.keys():
            print(f"Page '{page_name}' not in index.")
            self.page = None
            return
        offset, pid, block_size = self.idx[page_name]
        if offset == self.cache[0]:
            root = self.cache[1]
        else:
            xml = Dump.fetch_block(self.path_xml, offset, block_size)
            xml = b'<mediawiki>' + xml + b'</mediawiki>'*(offset != self.offset_max)
            root = ET.fromstring(xml)
            self.cache = (offset, root)
        text = Dump.search_id(root, pid)
        text = Dump.filter_top_section(text) if filter_top else text
        self.page = mph.parse(text, skip_style_tags = True)
        if self.page and 'REDIRECT' in self.page.strip_code():
            redirect = self.page.filter_wikilinks()[0].title
            redirect = str(redirect).split('#')[0]
            print(f"Redirect from '{page_name}' to '{redirect}'.")
            if redirect==page_name:
                print(f"Redirect page '{redirect}' same as requested page '{page_name}'.")
                return
            return self.load_page(redirect)
        else:
            return self.page

    @staticmethod
    def fetch_block(path, offset, block_size):
        """ Fetches block of ``block_size`` (``int``) bytes
        at ``offset`` (``int``) in the zipped dump at
        ``path`` (``string``) and returns the uncompressed
        text (``string``).
        """
        with open(path, 'rb') as file:
            file.seek(offset)
            return bz2.decompress(file.read(block_size))

    @staticmethod
    def search_id(root, pid):
        """Returns the text of the page with id ``pid``"""
        for page in root.iter('page'):
            if pid == int(page.find('id').text):
                return page.find('revision').find('text').text

    @staticmethod
    def filter_top_section(text):
        """Returns the top section of text,
        where the first header has the form ``==Heading==``
        """
        head = re.search(r'==.*?==', text)
        idx = head.span(0)[0] if head else len(text)
        return text[:idx] #(text[:idx], text[idx:])

    @staticmethod
    def get_history(page):
        """Returns the text of the history section.
        Returns ``""`` if not found.
        """
        headings = page.filter_headings()
        idx = [
            i for i, head in enumerate(headings)
            if 'History' in head or 'history' in head
        ]
        if not idx:
            return ""
        sections = page.get_sections(include_headings=True)
        history = str(sections[idx[0]+1].strip_code())
        return history

    @staticmethod
    def filter_years(text, get_matches=False):
        """Filters the years from text."""
        months = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ]
        prepositions = [
            'around', 'after', 'at', 'as', 'approximately',
            'before', 'between', 'by', 'during', 'from',
            'in', 'near', 'past', 'since', 'until', 'within'
        ] # removed: about, on
        conjugations = ['and']
        articles = ['the']
        times = ['early', 'mid', 'late']
        patterns = months + prepositions + conjugations + articles + times
        re_string = r'\b(' + '|'.join(patterns) + r')\b(\s|-)\b([0-9]{3,4})s?\b(?i)(?!\sMYA)\s?(BCE|BC)?'
        year_matches = list(re.finditer(re_string, text, re.IGNORECASE))
        years = [
            int(match.group(3)) * (-2*bool(match.group(4))+1)
            for match in year_matches
        ]
        re_string = r'([0-9]{1,2})(st|nd|rd|th) century\s?(BCE|BC)?'
        century_matches = list(re.finditer(re_string, text, re.IGNORECASE))
        centuries = [
            (int(match.group(1)) * 100 - 100) * (-2*bool(match.group(2))+1)
            for match in century_matches
        ]
        years += centuries
        years = [y for y in years if y<Dump.MAX_YEAR]
        if get_matches:
            return sorted(years + centuries), year_matches + century_matches
        else:
            return sorted(years + centuries)
