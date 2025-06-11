import os
from datetime import datetime, timedelta
import joblib
from boatrace.boatraceml import BoatraceML
import  boatrace.scraper

# =====================変更してOK======================
today_date = "2025-05-21"#datetime.now().strftime('%Y-%m-%d')
# ====================================================

# ===================以下は変更不要====================
folder = os.path.dirname(os.path.abspath(__file__)) + "/data"
boatrace.scraper.run(folder,today_date)

# メイン処理
boatrace_ml = BoatraceML(folder)

#result_df,model_one,model_defeat_one,model_twos,model_threes = boatrace_ml.run_pipeline(compile=False)
    
# モデルを保存する
#result_df.to_csv(folder+"/agg_results/result_df.csv", index=False, encoding="shift_jis")
#joblib.dump(model_one, folder+'/saved_models/model_one.joblib')
#joblib.dump(model_defeat_one,folder+'/saved_models/model_defeat_one.joblib')
#joblib.dump(model_twos, folder+'/saved_models/model_twos_dict.pkl')
#joblib.dump(model_threes, folder+'/saved_models/model_threes_dict.pkl')
# テスト
boatrace_ml.run_pipeline_ellipsis()

# モデルの読み込み
model_one = joblib.load(folder+'/saved_models/model_one.joblib')
model_defeat_one = joblib.load(folder+'/saved_models/model_defeat_one.joblib')
model_twos = joblib.load(folder+'/saved_models/model_twos_dict.pkl')
scaling_params = joblib.load(folder+'/saved_models/scaling_params.pkl')
"""
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
"""