import os
from boatrace.analyzer import BoatraceAnalyzer
import boatrace.scraper

# ==================変更してもOK==================
# 集計期間
start_date = "2023-04-01"
end_date = "2025-05-31"
# ===============================================
analyzer = BoatraceAnalyzer()
folder = analyzer.data_folder
boatrace.scraper.run(end_date)

#result = analyzer.get_base_data(start_date=start_date, end_date=end_date, venue="全国")
#result.to_csv(f"{folder}/agg_results/national_agg.csv", index=False, encoding="shift_jis")

# 特定のレース場のデータを集計
#venue_name="住之江"
#results_by_venue = analyzer.get_base_data(start_date=start_date, end_date=end_date, venue=venue_name)
#results_by_venue.to_csv(f"{folder}/agg_results/{venue_name}_agg.csv", index=False, encoding="shift_jis")

# 逃げデータのみの集計
#result_escape_only = analyzer.get_escape_only_data(start_date=start_date, end_date=end_date, venue="全国")
#result_escape_only.to_csv(f"{folder}/agg_results/escape_national_agg.csv", index=False, encoding="shift_jis")

# 差しまくりデータの集計
#sasi_makuri_results = analyzer.get_sasi_makuri_data(start_date=start_date, end_date=end_date, venue="全国")
#sasi_makuri_results.to_csv(f"{folder}/agg_results/sasi_makuri_national_agg.csv", index=False, encoding="shift_jis")

# 1年分の統計データを保存
#analyzer.save_agg_data(start_date,end_date)

# BとKのデータを結合
analyzer.get_merge_data(start_date,end_date)