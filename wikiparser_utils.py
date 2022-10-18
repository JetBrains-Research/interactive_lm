import mwxml
import wikitextparser as wtp
from tqdm import tqdm
import pandas as pd
import re

class WikiRevision:
    def __init__(self, page_id, revision_id, parent_revision_id, text, comment, title, timestamp):
        self.text = text
        self.title = title
        self.comment = comment
        self.timestamp = timestamp
        self.page_id = page_id
        self.parent_revision_id = parent_revision_id
        self.revision_id = revision_id
        
        try:
            self.parsed_text = wtp.parse(text)
        except:
            pass
        
    def get_plain_text(self):
        return wtp.remove_markup(self.text)
    
    def get_sections(self):
        sections = [(sec.title, sec.string) for sec in self.parsed_text.sections]
        return sections
    
    def get_clean_sections(self):
        sections = [(sec.title, wtp.remove_markup(sec.string)) for sec in self.parsed_text.sections]
        return sections
        
    def get_links(self):
        links = []
        tags = self.parsed_text.get_tags()
        for tag in tags:
            title = re.findall(r'title=\s*([^|]+)', tag.string)
            urls = re.findall(r'url=\s*([^|]+)', tag.string)
            
            title_str = title[0] if title else ''
            if urls:
                for link in urls:
                    links.append((title_str, link.strip()))
        return links        
        

class WikiPage:
    def __init__(self, page_id, title, revisions):
        self.page_id = page_id
        self.title = title
        self.revisions = revisions
        
    def get_revision(self):
        revisions = {}
        for revision in self.revisions:
            p_id = -1 if revision.parent_id is None else revision.parent_id
            r_id = revision.id
            new_revision = WikiRevision(self.page_id, r_id, p_id, revision.text, revision.comment, self.title, revision.timestamp)
            revisions[r_id] = new_revision
        return revisions


class WikiXMLDump:
    def __init__(self, dump_path):
        self.dump_path = dump_path
        
    def get_pages(self):
        id2page = {}
        dump = mwxml.Dump.from_file(open(self.dump_path, encoding="utf-8"))
        for p in tqdm(dump, position=0, leave=True):
            revisions = []
            for rev in p:
                revisions.append(rev)

            id2page[p.id] = WikiPage(p.id, p.title, revisions)    
        return id2page
    
    def get_exact_page(self, num):
        id2page = {}
        dump = mwxml.Dump.from_file(open(self.dump_path, encoding="utf-8"))
        for p in tqdm(dump, position=0, leave=True):
            if p.id == num:
                revisions = []
                for rev in p:
                    revisions.append(rev)   
                return WikiPage(p.id, p.title, revisions) 
        return None
    
    def get_revision(self):
        id2page2revision = {}
        id2page = self.get_pages()
        for page_id, page in tqdm(id2page.items(), position=0, leave=True):
            id2rev_page = page.get_revision()
            if id2rev_page:
                for rev_id, rev in id2rev_page.items():
                    if page_id not in id2page2revision:
                        id2page2revision[page_id] = {}
                    id2page2revision[page_id][rev_id] = rev
        return id2page, id2page2revision
            
