import os
from datetime import datetime, timedelta
import joblib
from scraper import BoatraceScraper
from machinelarning import BoatraceML

# ===============変更しないと駄目なところ===============
folder = "C:\\Users\\msy-t\\boatrace-ai\\data"
# =====================変更してOK======================
today_date = datetime.now().strftime('%Y-%m-%d')#"2025-05-21"
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
        scraperB.scrape_and_process_data(start_date=today_date,end_date=today_date)
    else:
        scraperB.scrape_and_process_data(start_date=current_date,end_date=today_date)
boatrace_ml = BoatraceML(folder)

# モデルの読み込み
model_one = joblib.load('saved_models/model_one.joblib')
model_defeat_one = joblib.load('saved_models/model_defeat_one.joblib')
model_twos = joblib.load('saved_models/model_twos_dict.pkl')
scaling_params = joblib.load('saved_models/scaling_params.pkl')

# 入力を受け取る
input_str = input("展示タイムを入力(スペースで区切る): ")
numbers = list(map(float, input_str.split()))
Exhibition_time = {1: numbers[0],2: numbers[1],3: numbers[2],4: numbers[3],5: numbers[4],6: numbers[5]}
# ====================================================
race_no = 11
venue = "桐生"
weather="曇り"
wind_dir="北東"
wind_spd=0
wave_hgt=0
# ====================================================
boatrace_ml.run_pipeline_jissen(model_one,model_defeat_one,model_twos,target_date=today_date, place=venue, race_no=race_no,weather=weather,wind_dir=wind_dir,wind_spd=wind_spd,wave_hgt=wave_hgt, Exhibition_time=Exhibition_time)

