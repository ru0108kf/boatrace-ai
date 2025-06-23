import os
from datetime import datetime, timedelta
import joblib
from boatrace.boatraceml import BoatraceML
import  boatrace.scraper
import boatrace.analyzer
from boatrace.crawler import BoatRaceOddsScraper
import pandas as pd

# =====================変更してOK======================
today_date = datetime.now().strftime('%Y-%m-%d')
main = False
practice = False
# ====================================================

# ===================以下は変更不要====================
boatrace_ml = BoatraceML()
folder = boatrace_ml.data_folder
boatrace.scraper.run(today_date)
boatrace.analyzer.run_agg(today_date)

# メイン処理
if main:
    result_df,model_one,model_defeat_one,model_twos,model_threes = boatrace_ml.run_pipeline(compile=True,
        train_start_date = "2021-01-01",train_end_date = "2025-05-31",test_start_date = "2025-06-01",test_end_date = "2025-06-18")
        
    # モデルを保存する
    result_df.to_csv(folder+"/agg_results/result_df.csv", index=False, encoding="shift_jis")
    joblib.dump(model_one, folder+'/saved_models/model_one.joblib')
    joblib.dump(model_defeat_one,folder+'/saved_models/model_defeat_one.joblib')
    joblib.dump(model_twos, folder+'/saved_models/model_twos_dict.pkl')
    joblib.dump(model_threes, folder+'/saved_models/model_threes_dict.pkl')
    
# テスト
#result_df = pd.read_csv(folder+"/agg_results/result_df.csv", encoding="shift_jis")
#boatrace_ml.run_pipeline_validation(compile=True,result_df=result_df)

if practice:
    # モデルの読み込み
    model_one = joblib.load(folder+'/saved_models/model_one.joblib')
    model_defeat_one = joblib.load(folder+'/saved_models/model_defeat_one.joblib')
    model_twos = joblib.load(folder+'/saved_models/model_twos_dict.pkl')
    model_threes = joblib.load(folder+'/saved_models/model_threes_dict.pkl')
    scaling_params = joblib.load(folder+'/saved_models/scaling_params.pkl')

    # 入力を受け取る
    input_str = input("展示タイムを入力(スペースで区切る): ")
    numbers = list(map(float, input_str.split()))
    Exhibition_time = {1: numbers[0],2: numbers[1],3: numbers[2],4: numbers[3],5: numbers[4],6: numbers[5]}
    # ====================================================
    race_no = 7
    venue = "蒲郡"
    weather="晴れ"
    wind_dir="北東"#無風
    wind_spd=2
    wave_hgt=1
    # ====================================================
    date_obj = datetime.strptime(today_date, '%Y-%m-%d')
    hd = date_obj.strftime('%Y/%m/%d')
    rno = f"{int(race_no)}R"
    odds = BoatRaceOddsScraper()
    all_odds = odds.get_odds(rno=rno,jcd=venue,hd=hd)
    boatrace_ml.run_pipeline_practice(model_one,model_defeat_one,model_twos,model_threes,all_odds,target_date=today_date, place=venue, race_no=race_no,weather=weather,wind_dir=wind_dir,wind_spd=wind_spd,wave_hgt=wave_hgt, Exhibition_time=Exhibition_time)