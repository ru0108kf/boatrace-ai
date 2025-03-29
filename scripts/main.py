import os
from scraper import BoatraceScraper
from analyzer import BoatraceAnalyzer

# 共通設定
folder = "C:\\Users\\msy-t\\boatrace-ai\\data"
B_url = "https://www1.mbrace.or.jp/od2/B/dindex.html"
K_url = "https://www1.mbrace.or.jp/od2/K/dindex.html"

scraper = BoatraceScraper(folder, BorK="K")
scraper.scrape_and_process_data_for_date_range(start_date="2025-02-18",end_date="2025-03-27")