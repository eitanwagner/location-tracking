

import requests
import json
from bs4 import BeautifulSoup
from time import sleep

def read_nums(filename):
    # Read in the lines from the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [int(l) for l in lines]

def combine_lines_by_length(filename, lengths):
    # Read in the lines from the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove newline characters from the end of each line
    lines = [line.strip() for line in lines]

    # Initialize the list of combined lines
    combined_lines = []
    i = 0

    # Loop over the desired lengths
    for length in lengths:
        # Combine lines until the desired length is reached
        combined_line = " ".join(lines[i: i+length])
        # Add the combined line to the list of combined lines
        combined_lines.append(combined_line)
        i += length

    # Return the list of combined lines
    return combined_lines

def is_nba(combined_lines, ids):
    return [(id, l) for id, l in zip(ids, combined_lines) if (l.find(" nba") > -1 and l.find("basketball player") > -1
                                                              and l.find("europe") == -1)]

def get_wikipedia_pages(page_ids, filename=""):
    # Build the URL for the API request
    articles = {}
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        # "rvprop": "content",
        "pageids": "|".join(str(id) for id in page_ids),
    }
    url = f"{base_url}?{requests.compat.urlencode(params)}"

    # Send the API request and save the response to a file
    response = requests.get(url)
    import json
    d = json.loads(response.text)['query']

    for v in d['pages'].values():
        if 'extract' in v:
            articles[v['title']] = extract_sections_with_paragraphs(v['extract'])
    return articles


import requests
import mwparserfromhell

from bs4 import BeautifulSoup
from bs4.element import Tag

def extract_sections_with_paragraphs(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')

    sections = []
    # headers = ['h1', 'h2', 'h3', 'h4']
    headers = ['h1', 'h2', 'h3']
    careers = ['High school career', 'College career', 'Professional career', 'Retirement', 'Coaching career', 'Executive career']
    to_remove = ['Career statistics', 'External links', 'References', 'Awards and honors', 'Regular season', 'Playoffs',
                 'Political views', 'Personal life', 'Awards and accomplishments']
    # first part
    section_paragraphs = []
    for c in soup.children:
        if isinstance(c, Tag):
            if c.name in headers:
                sections.append({'title': "", 'h': "", 'paragraphs': section_paragraphs})
                break
            txt = c.get_text()
            if len(txt) > 1:
                section_paragraphs.append(txt)
        elif len(c) > 1:
            section_paragraphs.append(c)

    # s = {"Summary": sections[-1]}
    add_h = False
    for header in soup.find_all(headers):
        section_title = header.get_text()
        section_paragraphs = []
        for sibling in header.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
            # if sibling.name and sibling.name.startswith('h'):
                # if sibling.name == "h4":
                #     continue
                add_h = sibling.name > header.name and len(section_paragraphs) == 0 and section_title not in to_remove and sibling.get_text() not in to_remove
                break
            if sibling.name == 'p':
                section_paragraphs.append(sibling.get_text())
        if (len(section_paragraphs) > 0 or add_h) and section_title not in to_remove:
            sections.append({'title': section_title, 'h': header.name, 'paragraphs': section_paragraphs})

    return sections

def extract_plain_text_from_wikipedia_pages(page_ids, filename):
    # Build the URL for the API request
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
    "action": "query",
    "format": "json",
    "prop": "revisions",
    "rvprop": "content",
    "pageids": "|".join(str(id) for id in page_ids),
    }
    url = f"{base_url}?{requests.compat.urlencode(params)}"

    # Send the API request and extract plain text from the response
    response = requests.get(url)
    import json
    d = json.loads(response.text)['query']
    with open(filename, "w", encoding="utf-8") as file:
        for page in mwparserfromhell.parse(response.text).filter_text():
            file.write(str(page) + "\n")

def get_nba_bios():
    """
    Get the NBA bios from the wikipedia dataset
    :return:
    """
    path = "/cs/labs/oabend/eitan.wagner/downloads/rlebret-wikipedia-biography-dataset-d0d6c78/wikipedia-biography-dataset/train/"
    filename = path + "train.sent"
    lengths = read_nums(path + "train.nb")
    ids = read_nums(path + "train.id")
    combined = combine_lines_by_length(filename, lengths=lengths)
    i_n = is_nba(combined, ids)

    articles = {}
    page_ids = list(zip(*i_n))[0]
    for pid in page_ids:
        articles |= get_wikipedia_pages([pid])
        sleep(1)
    filename = path + "nba_players.json"
    with open(filename, 'w', encoding="utf-8") as outfile:
        json.dump(articles, outfile)

def make_nba_data(data_path):
    """
    Make the NBA data from the wikipedia dataset
    :param data_path:
    :return:
    """
    with open(data_path + "/nba_players.json", 'r', encoding="utf-8") as infile:
        articles = json.load(infile)
    with open(data_path + "/nba_teams.json", 'r', encoding="utf-8") as infile:
        teams = json.load(infile)
    with open(data_path + "/old_nba_teams.json", 'r', encoding="utf-8") as infile:
        old_teams = json.load(infile)

    early_careers = ['High school career', 'High school and college career', 'College career', 'Early life and college']
    late_careers = ['Retirement', 'Injury and retirement', 'Coaching career', 'Executive career', "Coaching"]
    conversion = {"New Orleans Hornets": "New Orleans Pelicans", "Charlotte Bobcats": "Charlotte Hornets", "Vancouver Grizzlies": "Memphis Grizzlies",
                  "Seattle SuperSonics": "Oklahoma City Thunder", "New Jersey Nets": "Brooklyn Nets", "Denver Rockets": "Denver Nuggets",}
    ignore = ["Coaching", "Career highlights", "Corporate", "Enemies", "Spurs, Bucks, Blazers, Rockets, Raptors, and Bullets",
              "Nuggets, Raptors and Lakers", "Legal issues", "Career notes", "After Retirement", "College",
              "Expulsion and reinstatement", "Role in Stephen Curry endorsement contract", "NBA suspension", "Philanthropy", "Spirituality"]

    data = {}
    for n, a in articles.items():
        has_teams = []
        professional = False
        late = False
        # single header means no teams names
        if len(a) < 2:
            continue
        paras = []
        for h in a[1:]:
            title = h['title'].split('(')[0].split(':')[0].split('/')[0].strip()
            if title in ignore:
                continue
            if h['h'] == 'h2':
                late = title in late_careers
                _title = title
                professional = title == 'Professional career'
            if not late:
                _title = ""
            if title in conversion:
                title = conversion[title]
            if title in teams and not late:
                _title = title
                has_teams.append(_title)
            if title in old_teams and not late:
                _title = 'Old team'
                has_teams.append(_title)
            elif title in early_careers:
                _title = 'Early careers'
            elif professional and h['h'] != 'h2':
                if title.startswith("Second stint with"):
                    title = "Return to " + title.split("Second stint with")[1]
                if title.startswith("Return to"):
                    # print("return to")
                    return_to = [t for t in has_teams if (t.startswith(title.split("Return to ")[1]) or
                                                          t.endswith(title.split()[-1]))]
                    if len(return_to) == 0:
                        if len([t for t in old_teams if (t.startswith(title.split("Return to ")[1]) or
                                                         t.endswith(title.split()[-1]))]) > 0:
                            _title = "Old team"
                    else:
                        _title = return_to[0]
                if _title == "":
                    if title.find("draft") == -1:
                        _title = 'Non-NBA team'
                    if title.find("retirement") > -1:
                        _title = 'Retirement'
            if _title != "":
                for _h in h['paragraphs']:
                    paras.append([_h, _title])
        if len(has_teams) > 0:
            data[n] = paras


    with open(data_path + "/nba_data.json", 'w', encoding="utf-8") as outfile:
        json.dump(data, outfile)
    return


if __name__ == "__main__":
    from utils_ import parse_args
    args = parse_args()
    data_path = args.base_path + "/data/"
    make_nba_data(data_path=data_path)
    print("Done")

