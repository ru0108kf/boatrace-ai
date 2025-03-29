import os
from scraper import BoatraceScraper
from analyzer import BoatraceAnalyzer

# 共通設定
folder = "C:\\Users\\msy-t\\boatrace-ai\\data"
B_url = "https://www1.mbrace.or.jp/od2/B/dindex.html"
K_url = "https://www1.mbrace.or.jp/od2/K/dindex.html"

#scraper = BoatraceScraper(folder, BorK="K")
#scraper.scrape_and_process_data_for_date_range(start_date="2025-02-18",end_date="2025-03-27")

analyzer = BoatraceAnalyzer(folder = "C:\\Users\\msy-t\\boatrace-ai\\data")
result = analyzer.aggregate_data(start_date="2024-11-01", end_date="2025-03-28", venue="全国")
result.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\全国_result.csv", index=False, encoding="shift_jis")
venue_name="住之江"
results_by_venue = analyzer.aggregate_data(start_date="2024-11-01", end_date="2025-03-28", venue=venue_name)
results_by_venue.to_csv(f"C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\{venue_name}_result.csv", index=False, encoding="shift_jis")