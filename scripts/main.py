import os
from datetime import datetime, timedelta
from base import BoatraceBase
from scraper import BoatraceScraper
from scraper import BoatraceLatestDataScraper
from analyzer import BoatraceAnalyzer

# ===============変更しないと駄目なところ===============
folder = "C:\\Users\\msy-t\\boatrace-ai\\data"
# =====================変更してOK======================
today_date = "2025-04-21"#datetime.now().strftime('%Y-%m-%d')
race_no = 1
venue = "戸田"
# ====================================================

# ===================以下は変更不要====================
today_datetime = datetime.strptime(today_date, '%Y-%m-%d')
yesterday = (today_datetime - timedelta(days=1)).strftime('%Y-%m-%d')


scraperK = BoatraceScraper(folder, BorK="K")
current_date = scraperK.find_current_dates(date=yesterday,BorK="K")
if current_date!=None:
    if current_date==today_date:
        scraperK.scrape_and_process_data(start_date=yesterday,end_date=yesterday)
        scraperK.scrape_and_process_data(start_date=current_date,end_date=yesterday)

scraperB = BoatraceScraper(folder, BorK="B")
current_date = scraperB.find_current_dates(date=today_date,BorK="B")
if current_date!=None:
    if current_date==today_date:
        scraperB.scrape_and_process_data(start_date=yesterday,end_date=yesterday)
    else:
        scraperB.scrape_and_process_data(start_date=current_date,end_date=today_date)

boat_number_1, boat_number_2, boat_number_3, boat_number_4, boat_number_5, boat_number_6 = scraperB.get_player_numbers_by_race(date=today_date,venue=venue,race_number=race_no)

analyzer = BoatraceAnalyzer(folder)
"""
outputs = analyzer.get_boatrace_data(start_date="2024-11-01", end_date=yesterday, venue="全国",
                            boat_number_1=boat_number_1, boat_number_2=boat_number_2,
                            boat_number_3=boat_number_3, boat_number_4=boat_number_4,
                            boat_number_5=boat_number_5, boat_number_6=boat_number_6)

print(outputs)

latestdata = BoatraceLatestDataScraper()
latestoutputs = latestdata.get_latest_boatrace_data(place_name=venue,race_no=race_no,date=today_date)

print(latestoutputs)"""