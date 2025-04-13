import pandas as pd
from datetime import datetime, timedelta
import os

from base import BoatraceBase

class BoatraceAnalyzer(BoatraceBase):
    def __init__(self, folder):
        super().__init__(folder)
        self.B_csv_folder = os.path.join(self.folder, "B_csv")
        self.K_csv_folder = os.path.join(self.folder, "K_csv")
        
    def load_and_process_file(self, file_path):
        """必要なパラメータを抽出、追加してデータフレームを作成する
        Args:
            file_path: ファイルのパス
        Returns:
            dataframe: データフレーム
        """
        try:
            df = pd.read_csv(file_path, encoding="shift_jis")
        except FileNotFoundError:
            return None  # ファイルが存在しない場合は None を返す
        
        # ファイル名から日付を取得
        file_name = os.path.basename(file_path)
        date = file_name[1:7]
        
        # レースIDを作成する（各レースを一意に識別するためのID）
        # 日付, レース場, レース番号を組み合わせて作成
        df["レースID"] = date + "_" + df["レース場"] + "_" + df["レース番号"].astype(str)
        
        # 優勝した艇番を取得 (レースIDごとに1着の艇番を取得する)
        df["勝者"] = df.groupby("レースID")["艇番"].transform(lambda x: x.iloc[0])
        
        # フラグを作成し、整数に変換
        df["1着"] = (df["着順"] == 1).astype(int)
        df["2着"] = (df["着順"] == 2).astype(int)
        df["3着"] = (df["着順"] == 3).astype(int)
        df["4着"] = (df["着順"] == 4).astype(int)
        df["5着"] = (df["着順"] == 5).astype(int)
        df["6着"] = (df["着順"] == 6).astype(int)
        
        # 決まり手のフラグ（勝者のみ）
        decision_types = ["逃げ", "差し", "まくり", "まくり差し", "抜き", "恵まれ"]
        for k in decision_types:
            df[k] = 0  # 初期化

        # 決まり手が該当する行（かつ着順が1）だけ1にする
        for k in decision_types:
            df.loc[(df["決まり手"] == k) & (df["着順"] == 1), k] = 1
            
        # 敗因マッピング
        b1_defeat_map = {"差し": "差され", "まくり": "まくられ", "まくり差し": "まくり差され"}
        b2_defeat_map = {"逃げ": "逃し"}

        # 敗因カラム追加
        for col in list(b1_defeat_map.values()) + list(b2_defeat_map.values()):
            df[col] = 0

        # 各レースごとに処理
        for race_id, group in df.groupby("レースID"):
            if group.empty:
                continue

            decision = group["決まり手"].iloc[0]
            winner = group.loc[group["着順"] == 1, "艇番"].values
            if len(winner) == 0:
                continue
            winner = winner[0]

            # 1号艇処理
            b1 = group[group["艇番"] == 1]
            if not b1.empty and winner != 1 and decision in b1_defeat_map:
                df.loc[b1.index, b1_defeat_map[decision]] = 1

            # 2号艇処理
            b2 = group[group["艇番"] == 2]
            if not b2.empty and winner != 2 and decision in b2_defeat_map:
                df.loc[b2.index, b2_defeat_map[decision]] = 1
        
        return df
    
    def make_raw_data(self, start_date, end_date):
        """指定した日付範囲内のCSVファイルを読み込み、必要なパラメータを抽出してデータフレームを作成する
        Args:
            start_date: 開始日（YYYY-MM-DD形式）
            end_date: 終了日（YYYY-MM-DD形式）
        Returns:
            dataframe: データフレーム
        """
        file_names = self.generate_date_list(start_date, end_date)
        file_paths = [os.path.join(self.K_csv_folder, f"K{file_name}.csv") for file_name in file_names]

        df_list = []
        for file_path in file_paths:
            df = self.load_and_process_file(file_path)
            if df is not None:
                df_list.append(df)

        if len(df_list) == 0:
            raise FileNotFoundError("指定した日付範囲内に有効なCSVファイルが存在しません。")
        
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    
    def base_data(self, start_date, end_date, venue="全国"):
        df = self.make_raw_data(start_date, end_date)

        # 全出走データ + 決まり手フラグ + 敗因フラグを集計
        agg_columns = {
            "着順": "count",          # 出走数
            "1着": "sum",             # 勝利回数
            "2着": "sum",             # 2着回数
            "3着": "sum",             # 3着回数
            "スタートタイミング": "mean",  # 平均ST
            "逃げ": "sum",
            "差し": "sum",
            "まくり": "sum",
            "まくり差し": "sum",
            "抜き": "sum",
            "恵まれ": "sum",
            "差され": "sum", 
            "まくられ": "sum", 
            "まくり差され": "sum",
            "逃し": "sum"
        }
        
        # スタートタイミングの平均値を計算
        avg_st_time = df.groupby(["登録番号", "選手名"])["スタートタイミング"].mean().reset_index()
        avg_st_time.columns = ["登録番号", "選手名", "平均ST"]

        if venue == "全国":
            # 統合して1回の groupby + agg にまとめる
            result = (df.groupby(["登録番号", "選手名", "艇番"]).agg(agg_columns).reset_index().fillna(0))
            # 列名をわかりやすく変更
            result = result.rename(columns={"着順": "出走数","1着": "勝利回数","2着": "2着回数","3着": "3着回数","スタートタイミング": "平均ST"})
            # 全体平均STを別で計算
            avg_st_time = df.groupby(["登録番号", "選手名"])["スタートタイミング"].mean().reset_index()
            avg_st_time.columns = ["登録番号", "選手名", "全体平均ST"]
            # マージして「全体平均ST」列を追加
            result = pd.merge(result, avg_st_time, on=["登録番号", "選手名"], how="left")
            
        else:
            # 指定されたレース場のみフィルタリング
            venue_data = df[df["レース場"] == venue]
            if venue_data.empty:
                raise ValueError(f"指定されたレース場「{venue}」のデータが存在しません。")
            result = (venue_data.groupby(["登録番号", "選手名", "艇番"]).agg(agg_columns).reset_index().fillna(0))
            result = result.rename(columns={"着順": "出走数","1着": "勝利回数","2着": "2着回数","3着": "3着回数","スタートタイミング": "平均ST"})
            avg_st_time = venue_data.groupby(["登録番号", "選手名"])["スタートタイミング"].mean().reset_index()
            avg_st_time.columns = ["登録番号", "選手名", "全体平均ST"]
            result = pd.merge(result, avg_st_time, on=["登録番号", "選手名"], how="left")
            
        return result
           
    def escape_only_data(self, start_date, end_date, venue="全国"):
        df = self.make_raw_data(start_date, end_date)

        # 決まり手が"逃げ"のデータを抽出
        escaped_df = df[df["決まり手"] == "逃げ"]
        agg_columns = {
            "着順": "count",
            "1着": "sum",
            "2着": "sum",
            "3着": "sum"
        }

        if venue=="全国":
            result = escaped_df.groupby(["登録番号", "選手名", "艇番"]).agg(agg_columns).reset_index()
            result.columns = ["登録番号", "選手名", "艇番", "出走数", "勝利回数", "2着回数", "3着回数"]
            return result

        else:
            # レース場が指定された場合、そのレース場のみフィルタリング
            venue_data = escaped_df[escaped_df["レース場"] == venue]
            
            if venue_data.empty:
                raise ValueError(f"指定されたレース場「{venue}」のデータが存在しません。")
            
            # 指定されたレース場のデータを集計
            result = venue_data.groupby(["登録番号", "選手名", "艇番"]).agg(agg_columns).reset_index()
            result.columns = ["登録番号", "選手名", "艇番", "出走数", "勝利回数", "2着回数", "3着回数"]
            return result
    
    def sasi_makuri_data(self, start_date, end_date, venue="全国"):
        df = self.make_raw_data(start_date, end_date)

        # 条件に合致する場合にフラグを立てる（事前にすべて0で初期化）
        df = df.assign(
            no1_sasare_by_no2=((df["艇番"] == 1) & (df["決まり手"] == "差し") & (df["勝者"] == 2)).astype(int),
            no1_makurare_by_no2=((df["艇番"] == 1) & (df["決まり手"] == "まくり") & (df["勝者"] == 2)).astype(int),
            no1_makurare_by_no3=((df["艇番"] == 1) & (df["決まり手"] == "まくり") & (df["勝者"] == 3)).astype(int),
            no1_makurisasare_by_no3=((df["艇番"] == 1) & (df["決まり手"] == "まくり差し") & (df["勝者"] == 3)).astype(int),

            no2_sasi=((df["艇番"] == 2) & (df["決まり手"] == "差し") & (df["勝者"] == 2)).astype(int),
            no2_makuri=((df["艇番"] == 2) & (df["決まり手"] == "まくり") & (df["勝者"] == 2)).astype(int),

            no3_makuri=((df["艇番"] == 3) & (df["決まり手"] == "まくり") & (df["勝者"] == 3)).astype(int),
            no3_makurisasi=((df["艇番"] == 3) & (df["決まり手"] == "まくり差し") & (df["勝者"] == 3)).astype(int),

            no4_sasi=((df["艇番"] == 4) & (df["決まり手"] == "差し") & (df["勝者"] == 4)).astype(int),
            no4_makuri=((df["艇番"] == 4) & (df["決まり手"] == "まくり") & (df["勝者"] == 4)).astype(int),
            no4_makurisasi=((df["艇番"] == 4) & (df["決まり手"] == "まくり差し") & (df["勝者"] == 4)).astype(int),

            no5_sasi=((df["艇番"] == 5) & (df["決まり手"] == "差し") & (df["勝者"] == 5)).astype(int),
            no5_makuri=((df["艇番"] == 5) & (df["決まり手"] == "まくり") & (df["勝者"] == 5)).astype(int),
            no5_makurisasi=((df["艇番"] == 5) & (df["決まり手"] == "まくり差し") & (df["勝者"] == 5)).astype(int),

            no6_sasi=((df["艇番"] == 6) & (df["決まり手"] == "差し") & (df["勝者"] == 6)).astype(int),
            no6_makuri=((df["艇番"] == 6) & (df["決まり手"] == "まくり") & (df["勝者"] == 6)).astype(int),
            no6_makurisasi=((df["艇番"] == 6) & (df["決まり手"] == "まくり差し") & (df["勝者"] == 6)).astype(int)
        )

        # 使用する列だけフィルタリング
        flag_cols = [col for col in df.columns if col.startswith("no")]
        df_filtered = df[["登録番号", "選手名", "レース場"] + flag_cols]

        # 会場でフィルター（全国 or 特定会場）
        if venue != "全国":
            df_filtered = df_filtered[df_filtered["レース場"] == venue]
            if df_filtered.empty:
                raise ValueError(f"指定されたレース場「{venue}」のデータが存在しません。")

        # 集計
        result = df_filtered.groupby(["登録番号", "選手名"])[flag_cols].sum().reset_index()

        return result

    def get_boatrace_data(self, start_date, end_date, venue, boat_number_1=False, boat_number_2=False, 
                        boat_number_3=False, boat_number_4=False, boat_number_5=False, boat_number_6=False):
        """
        選手登録番号を入力し、各艇のデータを取得する
        """
        # 基本データを一度だけ取得
        result = self.base_data(start_date=start_date, end_date=end_date, venue=venue)
        result_sasi_makuri = self.sasi_makuri_data(start_date=start_date, end_date=end_date, venue=venue)
        result_escape_only = self.escape_only_data(start_date=start_date, end_date=end_date, venue=venue)
        
        outputs = {}
        
        # 各艇番号ごとの処理を共通化
        boat_configs = [
            (1, boat_number_1, ["1号艇"], ["艇番"]),
            (2, boat_number_2, ["2号艇"], ["艇番"]),
            (3, boat_number_3, ["3号艇"], ["艇番"]),
            (4, boat_number_4, ["4号艇"], ["艇番"]),
            (5, boat_number_5, ["5号艇"], ["艇番"]),
            (6, boat_number_6, ["6号艇"], ["艇番"])
        ]
        
        for boat_num, boat_number, output_keys, filter_cols in boat_configs:
            if not boat_number:
                continue
                
            # 共通フィルタ条件
            filters = {"登録番号": boat_number}
            if filter_cols:
                filters.update({"艇番": boat_num})
            
            # データフィルタリング（一度だけ行う）
            base_df = result[(result["登録番号"] == boat_number) & (result["艇番"] == boat_num)] if filter_cols else result[result["登録番号"] == boat_number]
            sasi_makuri_df = result_sasi_makuri[result_sasi_makuri["登録番号"] == boat_number]
            escape_df = result_escape_only[(result_escape_only["登録番号"] == boat_number) & (result_escape_only["艇番"] == boat_num)] if boat_num in [2,3,4,5,6] else None
            
            # 出走数（分母）を計算
            race_count = base_df["出走数"].iloc[0] if not base_df.empty and base_df["出走数"].iloc[0] > 0 else 0
            
            # 艇番ごとの特別な計算
            if boat_num == 1:
                output_data = {
                    "選手名": base_df["選手名"].iloc[0] if not base_df.empty else "Unknown",
                    "1着率": (base_df["勝利回数"].iloc[0] / race_count) if not base_df.empty and race_count > 0 else 0,
                    "2号艇にまくられさされ率": (
                        (sasi_makuri_df["no1_sasare_by_no2"].iloc[0] + sasi_makuri_df["no1_makurare_by_no2"].iloc[0]) / race_count 
                        if not sasi_makuri_df.empty and race_count > 0 else 0
                    ),
                    "3号艇にまくられさされ率": (
                        (sasi_makuri_df["no1_makurare_by_no3"].iloc[0] + sasi_makuri_df["no1_makurisasare_by_no3"].iloc[0]) / race_count 
                        if not sasi_makuri_df.empty and race_count > 0 else 0
                    ),
                    "平均ST": base_df["全体平均ST"].iloc[0] if not base_df.empty else 0
                }
            elif boat_num == 2:
                output_data = {
                    "選手名": base_df["選手名"].iloc[0] if not base_df.empty else "Unknown",
                    "差し率": (sasi_makuri_df["no2_sasi"].iloc[0] / race_count) if not sasi_makuri_df.empty and race_count > 0 else 0,
                    "まくり率": (sasi_makuri_df["no2_makuri"].iloc[0] / race_count) if not sasi_makuri_df.empty and race_count > 0 else 0,
                    "逃し率": (escape_df["出走数"].iloc[0] / race_count) if escape_df is not None and not escape_df.empty and race_count > 0 else 0,
                    "1号艇が逃げた時の2-3着率": (
                        (escape_df["2着回数"].iloc[0] + escape_df["3着回数"].iloc[0]) / race_count 
                        if escape_df is not None and not escape_df.empty and race_count > 0 else 0
                    ),
                    "平均ST": base_df["全体平均ST"].iloc[0] if not base_df.empty else 0
                }
            elif boat_num == 3:
                output_data = {
                    "選手名": base_df["選手名"].iloc[0] if not base_df.empty else "Unknown",
                    "まくり率": (sasi_makuri_df["no3_makuri"].iloc[0] / race_count) if not sasi_makuri_df.empty and race_count > 0 else 0,
                    "まくり差し率": (sasi_makuri_df["no3_makurisasi"].iloc[0] / race_count) if not sasi_makuri_df.empty and race_count > 0 else 0,
                    "1号艇が逃げた時の2-3着率": (
                        (escape_df["2着回数"].iloc[0] + escape_df["3着回数"].iloc[0]) / race_count 
                        if escape_df is not None and not escape_df.empty and race_count > 0 else 0
                    ),
                    "平均ST": base_df["全体平均ST"].iloc[0] if not base_df.empty else 0
                }
            elif boat_num in [4,5,6]:
                output_data = {
                    "選手名": base_df["選手名"].iloc[0] if not base_df.empty else "Unknown",
                    "差し1着率": (sasi_makuri_df[f"no{boat_num}_sasi"].iloc[0] / race_count) if not sasi_makuri_df.empty and race_count > 0 else 0,
                    "まくり1着率": (sasi_makuri_df[f"no{boat_num}_makuri"].iloc[0] / race_count) if not sasi_makuri_df.empty and race_count > 0 else 0,
                    "まくり差し1着率": (sasi_makuri_df[f"no{boat_num}_makurisasi"].iloc[0] / race_count) if not sasi_makuri_df.empty and race_count > 0 else 0,
                    "1号艇が逃げた時の2-3着率": (
                        (escape_df["2着回数"].iloc[0] + escape_df["3着回数"].iloc[0]) / race_count 
                        if escape_df is not None and not escape_df.empty and race_count > 0 else 0
                    ),
                    "平均ST": base_df["全体平均ST"].iloc[0] if not base_df.empty else 0
                }
            
            outputs[output_keys[0]] = output_data
        
        return outputs
    
    def merge_data(self,start_date="2024-04-01", end_date="2025-03-31"):
        list = self.generate_date_list(start_date, end_date)
        for name in list:
            df_K = pd.read_csv(self.folder+f"\\K_csv\\K{name}.csv", encoding='shift-jis')
            df_B = pd.read_csv(self.folder+f"\\B_csv\\B{name}.csv", encoding='shift-jis')

            # 結合と重複カラムを削除
            common_columns = ['レース場', 'レース番号', '登録番号']
            overlapping_columns = ['レース種別','艇番','選手名','モーター番号','ボート番号']
            
            df_K = df_K.drop(columns=overlapping_columns)
            merged_df = pd.merge(df_B, df_K, on=common_columns, how='inner')
            
            # 3カ月前の日付を取得
            three_months_ago = (datetime.strptime(name, "%y%m%d") - timedelta(days=366)).strftime("%Y-%m-%d")
            # 1日前の日付を取得
            one_day_ago = (datetime.strptime(name, "%y%m%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            
            # 基本データの取得
            base_df = self.base_data(start_date=three_months_ago, end_date=one_day_ago, venue="全国") 
            
            # データを結合する
            merged_df = pd.merge(merged_df, base_df[['登録番号', '艇番', '平均ST','全体平均ST']], on=['登録番号', '艇番'], how='left')
            
            # 1着率を計算
            merged_df['1着率'] = base_df['勝利回数'] / base_df['出走数']
            # 2-3着率を計算
            merged_df['2-3着率'] = (base_df['2着回数']+base_df['3着回数']) / base_df['出走数']
            # 逃し率
            merged_df['逃し率'] = base_df['逃し'] / base_df['出走数']
            
            merged_df.to_csv(self.folder+f"\\merged_csv\\{name}.csv", index=False, encoding='shift-jis')      
 

if __name__ == "__main__":
    # ==================変更すべき欄==================
    folder = "C:\\Users\\msy-t\\boatrace-ai\\data"
    # ==================変更してもOK==================
    # 集計期間
    start_date = "2024-04-01"
    end_date = "2025-03-31"
    
    # ===============================================
    analyzer = BoatraceAnalyzer(folder)
    result = analyzer.base_data(start_date=start_date, end_date=end_date, venue="全国")
    result.to_csv(f"{folder}\\agg_results\\national_agg.csv", index=False, encoding="shift_jis")
    
    # 特定のレース場のデータを集計
    venue_name="住之江"
    results_by_venue = analyzer.base_data(start_date=start_date, end_date=end_date, venue=venue_name)
    results_by_venue.to_csv(f"{folder}\\agg_results\\{venue_name}_agg.csv", index=False, encoding="shift_jis")
    
    # 逃げデータのみの集計
    result_escape_only = analyzer.escape_only_data(start_date=start_date, end_date=end_date, venue="全国")
    result_escape_only.to_csv(f"{folder}\\agg_results\\escape_national_agg.csv", index=False, encoding="shift_jis")
    
    # 差しまくりデータの集計
    sasi_makuri_results = analyzer.sasi_makuri_data(start_date=start_date, end_date=end_date, venue="全国")
    sasi_makuri_results.to_csv(f"{folder}\\agg_results\\sasi_makuri_national_agg.csv", index=False, encoding="shift_jis")
    
    # BとKのデータを結合
    analyzer.merge_data(start_date,end_date)


