# Early crawler skeleton - fetch single page only
import requests
from bs4 import BeautifulSoup

def fetch(url):
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text()
