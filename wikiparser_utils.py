import mwxml
import wikitextparser as wtp
from tqdm import tqdm
import pandas as pd
import re
import json
import os
import string

def del_punkt(str_):
    return str_.translate(str.maketrans("","", string.punctuation))


class WikiRevision:
    def __init__(self, page_id=None, revision_id=None, parent_revision_id=None, 
                 text=None, comment=None, title=None, timestamp=None):
        self.text = text
        self.title = title
        self.comment = comment
        try:
            self.timestamp = timestamp.strftime("%Y/%m/%d %H:%M:%S")
        except:
            self.timestamp = ""
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
            urls = re.findall(r'url=\s*([^}<|]+)', tag.string)
            
            title_str = title[0] if title else ''
            if urls:
                for link in urls:
                    links.append((title_str, link.strip()))
        return links 
    
    def save(self, filepath):
        final_path = f"{filepath}/{self.revision_id}.json"
        json_obj = {
            "text": self.text,
            "title": self.title,
            "comment": self.comment,
            "timestamp": self.timestamp,
            "page_id": self.page_id,
            "parent_revision_id": self.parent_revision_id,
            "revision_id": self.revision_id
        }
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(json_obj, f)
        
    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            json_obj = json.load(f)
        self.text = json_obj["text"]
        self.title = json_obj["title"]
        self.comment = json_obj["comment"]
        self.timestamp = json_obj["timestamp"]
        self.page_id = json_obj["page_id"]
        self.parent_revision_id = json_obj["parent_revision_id"]
        self.revision_id = json_obj["revision_id"]
        
        try:
            self.parsed_text = wtp.parse(self.text)
        except:
            pass
        
        

class WikiPage:
    def __init__(self, page_id=None, title=None, revisions=None):
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
    
    def load_revisions(self, filepath):
        revisions = {}
        revisions_nums = os.listdir(filepath)
        revs = sorted([int(part.split('.json')[0]) for part in revisions_nums])
        for revision in revs:
            final_path = f"{filepath}/{revision}.json"
            new_revision = WikiRevision()
            new_revision.load(final_path)
            revisions[revision] = new_revision
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
    
    def get_save_on_disk(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        dump = mwxml.Dump.from_file(open(self.dump_path, encoding="utf-8"))
        for p in tqdm(dump, position=0, leave=True):
            revisions = []
            for rev in p:
                revisions.append(rev)

            new_page = WikiPage(p.id, p.title, revisions)
            for revision in revisions:
                p_id = -1 if revision.parent_id is None else revision.parent_id
                r_id = revision.id
                new_revision = WikiRevision(new_page.page_id, r_id, p_id, revision.text, revision.comment, new_page.title, revision.timestamp)
                final_path = f"{filepath}/{del_punkt(new_page.title)}_{new_page.page_id}"
                if not os.path.exists(final_path):
                    os.makedirs(final_path)
                new_revision.save(f"{final_path}")
            
