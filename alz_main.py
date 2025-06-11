import os
from datetime import datetime, timedelta
import boatrace.scraper
from boatrace.latestscraper import BoatraceLatestDataScraper
from boatrace.analyzer import BoatraceAnalyzer

# =====================変更してOK======================
today_date = "2025-04-21"#datetime.now().strftime('%Y-%m-%d')
race_no = 1
venue = "戸田"
# ====================================================

# ===================以下は変更不要====================
folder = os.path.dirname(os.path.abspath(__file__)) + "/data"
boatrace.scraper.run(folder,today_date)

# メイン処理
analyzer = BoatraceAnalyzer(folder)
outputs = analyzer.get_boatrace_data(date=today_date,venue=venue,race_number=race_no,target_venue="全国",lookback_days=365)

print(outputs)

latestdata = BoatraceLatestDataScraper()
latestoutputs = latestdata.get_latest_boatrace_data(place_name=venue,race_no=race_no,date=today_date)

print(latestoutputs)