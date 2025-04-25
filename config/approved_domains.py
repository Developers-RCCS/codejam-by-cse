# config/approved_domains.py
"""
Configuration file containing the list of competition-approved domains
for web searches, organized by topic.
"""

# Dictionary of approved domains organized by topic category
APPROVED_DOMAINS = {
    "wright_brothers": [
        "kids.nationalgeographic.com/history/article/wright-brothers",
        "en.wikipedia.org/wiki/Wright_Flyer",
        "airandspace.si.edu/collection-objects/1903-wright-flyer/nasm_A19610048000",
        "en.wikipedia.org/wiki/Wright_brothers",
        "spacecenter.org/a-look-back-at-the-wright-brothers-first-flight/"
    ],
    "education_sri_lanka": [
        "udithadevapriya.medium.com/a-history-of-education-in-sri-lanka-bf2d6de2882c",
        "en.wikipedia.org/wiki/Education_in_Sri_Lanka",
        "thuppahis.com/2018/05/16/the-earliest-missionary-english-schools-challenging-shirley-somanader/",
        "www.elivabooks.com/pl/book/book-6322337660",
        "quizgecko.com/learn/christian-missionary-organizations-in-sri-lanka-bki3tu"
    ],
    "mahaweli_development": [
        "en.wikipedia.org/wiki/Mahaweli_Development_programme",
        "www.cmg.lk/largest-irrigation-project",
        "mahaweli.gov.lk/Corporate%20Plan%202019%20-%202023.pdf",
        "www.sciencedirect.com/science/article/pii/S0016718524002082",
        "www.sciencedirect.com/science/article/pii/S2405844018381635"
    ],
    "marie_antoinette": [
        "www.britannica.com/story/did-marie-antoinette-really-say-let-them-eat-cake",
        "genikuckhahn.blog/2023/06/10/marie-antoinette-and-the-infamous-phrase-did-she-really-say-let-them-eat-cake/",
        "www.instagram.com/mottahedehchina/p/Cx07O8XMR8U/",
        "www.reddit.com/r/HistoryMemes/comments/rqgcjs/let_them_eat_cake_is_the_most_famous_quote/",
        "www.history.com/news/did-marie-antoinette-really-say-let-them-eat-cake"
    ],
    "adolf_hitler": [
        "encyclopedia.ushmm.org/content/en/article/adolf-hitler-early-years-1889-1921",
        "en.wikipedia.org/wiki/Adolf_Hitler",
        "encyclopedia.ushmm.org/content/en/article/adolf-hitler-early-years-1889-1913",
        "www.history.com/articles/adolf-hitler",
        "www.bbc.co.uk/teach/articles/zbrx8xs"
    ]
}

# Flattened list for simple validation
ALL_APPROVED_DOMAINS = [domain for category in APPROVED_DOMAINS.values() for domain in category]