import pandas as pd
import os

from base import BoatraceBase  # ← 親クラスのインポート

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
        
        # ファイル名から日付を取得 (例: K240301.csv -> 240301)
        file_name = os.path.basename(file_path)
        date = file_name[1:7]  # 「K240301.csv」から「240301」を取り出す
        
        # レースIDを作成する（各レースを一意に識別するためのID）
        # 日付, レース場, レース番号を組み合わせて作成
        df["レースID"] = date + "_" + df["レース場"] + "_" + df["レース番号"].astype(str)
        
        # 優勝した艇番を取得 (レースIDごとに1着の艇番を取得する)
        df["勝者"] = df.groupby("レースID")["艇番"].transform(lambda x: x.iloc[0])
        
        # フラグを作成し、整数に変換
        df["1着"] = (df["着順"] == 1).astype(int)
        df["2着"] = (df["着順"] == 2).astype(int)
        df["3着"] = (df["着順"] == 3).astype(int)
        
        decision_types = ["逃げ", "差し", "まくり", "まくり差し", "抜き", "恵まれ"]
        for k in decision_types:
            df[k] = (df["決まり手"] == k).astype(int)
        
        return df
    
    def make_raw_data(self, start_date, end_date):
        """指定した日付範囲内のCSVファイルを読み込み、必要なパラメータを抽出してデータフレームを作成する
        Args:
            start_date (_type_): 開始日（YYYY-MM-DD形式）
            end_date (_type_): 終了日（YYYY-MM-DD形式）
        Raises:
            FileNotFoundError: _description_
        Returns:
            type: データフレーム
        """
        file_names = self.generate_date_list(start_date, end_date, "K")
        file_paths = [os.path.join(self.K_csv_folder, f"{file_name}.csv") for file_name in file_names]

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
        
        agg_columns = {
            "着順": "count",
            "1着": "sum",
            "2着": "sum",
            "3着": "sum",
            "スタートタイミング": "mean",
            "逃げ": "sum",
            "差し": "sum",
            "まくり": "sum",
            "まくり差し": "sum",
            "抜き": "sum",
            "恵まれ": "sum"
        }

        if venue=="全国":
            result = df.groupby(["登録番号", "選手名", "艇番"]).agg(agg_columns).reset_index()
            result.columns = ["登録番号", "選手名", "艇番", "出走数", "勝利回数", "2着回数", "3着回数", "平均ST", 
                            "逃げ", "差し", "まくり", "まくり差し", "抜き", "恵まれ"]
            return result
        
        else:
            # レース場が指定された場合、そのレース場のみフィルタリング
            venue_data = df[df["レース場"] == venue]
            
            if venue_data.empty:
                raise ValueError(f"指定されたレース場「{venue}」のデータが存在しません。")
            
            # 指定されたレース場のデータを集計
            result = venue_data.groupby(["登録番号", "選手名", "艇番"]).agg(agg_columns).reset_index()
            result.columns = ["登録番号", "選手名", "艇番", "出走数", "勝利回数", "2着回数", "3着回数", "平均ST", 
                            "逃げ", "差し", "まくり", "まくり差し", "抜き", "恵まれ"]
            return result

    def st_time_data(self, start_date, end_date, venue="全国"):
        df = self.make_raw_data(start_date, end_date)

        # スタートタイミングが0.0未満のデータを除外
        df = df[df["スタートタイミング"] >= 0.0]

        # スタートタイミングの平均値を計算
        avg_st_time = df.groupby(["登録番号", "選手名"])["スタートタイミング"].mean().reset_index()
        avg_st_time.columns = ["登録番号", "選手名", "平均ST"]
        
        if venue=="全国":
            return avg_st_time
        
        else:
            # レース場が指定された場合、そのレース場のみフィルタリング
            venue_data = df[df["レース場"] == venue]
            
            if venue_data.empty:
                raise ValueError(f"指定されたレース場「{venue}」のデータが存在しません。")
            
            # 指定されたレース場のデータを集計
            result = venue_data.groupby(["登録番号", "選手名"])["スタートタイミング"].mean().reset_index()
            result.columns = ["登録番号", "選手名", "平均ST"]
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

        # 1号艇で2号艇に差されたデータを抽出
        no1_sasare_by_no2_df = df[(df["艇番"] == 1) & (df["決まり手"] == "差し") & (df["勝者"] == 2)].copy()
        no1_sasare_by_no2_df.loc[:, "no1_sasare_by_no2"] = 1

        # 1号艇で2号艇にまくられたデータを抽出
        no1_makurare_by_no2_df = df[(df["艇番"] == 1) & (df["決まり手"] == "まくり") & (df["勝者"] == 2)].copy()
        no1_makurare_by_no2_df.loc[:, "no1_makurare_by_no2"] = 1

        # 1号艇で3号艇にまくられたデータを抽出
        no1_makurare_by_no3_df = df[(df["艇番"] == 1) & (df["決まり手"] == "まくり") & (df["勝者"] == 3)].copy()
        no1_makurare_by_no3_df.loc[:, "no1_makurare_by_no3"] = 1

        # 1号艇で3号艇にまくられたデータを抽出
        no1_makurisasare_by_no3_df = df[(df["艇番"] == 1) & (df["決まり手"] == "まくり") & (df["勝者"] == 3)].copy()
        no1_makurisasare_by_no3_df.loc[:, "no1_makurisasare_by_no3"] = 1
        
        # 2号艇でさしたデータを抽出
        no2_sasi_df = df[(df["艇番"] == 2) & (df["決まり手"] == "差し") & (df["勝者"] == 2)].copy()
        no2_sasi_df.loc[:, "no2_sasi"] = 1
        
        # 2号艇でまくったデータを抽出
        no2_makuri_df = df[(df["艇番"] == 2) & (df["決まり手"] == "まくり") & (df["勝者"] == 2)].copy()
        no2_makuri_df.loc[:, "no2_makuri"] = 1
        
        # 3号艇でまくったデータを抽出
        no3_makuri_df = df[(df["艇番"] == 3) & (df["決まり手"] == "まくり") & (df["勝者"] == 3)].copy()
        no3_makuri_df.loc[:, "no3_makuri"] = 1
        
        # 3号艇でまくり差ししたデータを抽出
        no3_makurizasi_df = df[(df["艇番"] == 3) & (df["決まり手"] == "まくり差し") & (df["勝者"] == 3)].copy()
        no3_makurizasi_df.loc[:, "no3_makurizasi"] = 1
        
        # 4号艇でさしたデータを抽出
        no4_sasi_df = df[(df["艇番"] == 4) & (df["決まり手"] == "差し") & (df["勝者"] == 4)].copy()
        no4_sasi_df.loc[:, "no4_sasi"] = 1
        
        # 4号艇でまくったデータを抽出
        no4_makuri_df = df[(df["艇番"] == 4) & (df["決まり手"] == "まくり") & (df["勝者"] == 4)].copy()
        no4_makuri_df.loc[:, "no4_makuri"] = 1
        
        # 4号艇でまくり差ししたデータを抽出
        no4_makurizasi_df = df[(df["艇番"] == 4) & (df["決まり手"] == "まくり差し") & (df["勝者"] == 4)].copy()
        no4_makurizasi_df.loc[:, "no4_makurizasi"] = 1
        
        # 5号艇でさしたデータを抽出
        no5_sasi_df = df[(df["艇番"] == 5) & (df["決まり手"] == "差し") & (df["勝者"] == 5)].copy()
        no5_sasi_df.loc[:, "no5_sasi"] = 1
        
        # 5号艇でまくったデータを抽出
        no5_makuri_df = df[(df["艇番"] == 5) & (df["決まり手"] == "まくり") & (df["勝者"] == 5)].copy()
        no5_makuri_df.loc[:, "no5_makuri"] = 1
        
        # 5号艇でまくり差ししたデータを抽出
        no5_makurizasi_df = df[(df["艇番"] == 5) & (df["決まり手"] == "まくり差し") & (df["勝者"] == 5)].copy()
        no5_makurizasi_df.loc[:, "no5_makurizasi"] = 1
        
        # 6号艇でさしたデータを抽出
        no6_sasi_df = df[(df["艇番"] == 6) & (df["決まり手"] == "差し") & (df["勝者"] == 6)].copy()
        no6_sasi_df.loc[:, "no6_sasi"] = 1
        
        # 6号艇でまくったデータを抽出
        no6_makuri_df = df[(df["艇番"] == 6) & (df["決まり手"] == "まくり") & (df["勝者"] == 6)].copy()
        no6_makuri_df.loc[:, "no6_makuri"] = 1
        
        # 6号艇でまくり差ししたデータを抽出
        no6_makurizasi_df = df[(df["艇番"] == 6) & (df["決まり手"] == "まくり差し") & (df["勝者"] == 6)].copy()
        no6_makurizasi_df.loc[:, "no6_makurizasi"] = 1

        # 全データをまとめる
        sasi_makuri_df = pd.concat([
            no1_sasare_by_no2_df, no1_makurare_by_no2_df, no1_makurare_by_no3_df, no1_makurisasare_by_no3_df,
            no2_sasi_df, no2_makuri_df, no3_makuri_df, no3_makurizasi_df,
            no4_sasi_df, no4_makuri_df, no4_makurizasi_df,
            no5_sasi_df, no5_makuri_df, no5_makurizasi_df,
            no6_sasi_df, no6_makuri_df, no6_makurizasi_df
        ], ignore_index=True)

        # 存在しないカラムを補完（ゼロ埋め）
        for col in ["no1_sasare_by_no2", "no1_makurare_by_no2", "no1_makurare_by_no3", "no1_makurisasare_by_no3",
                    "no2_sasi", "no2_makuri", "no3_makuri", "no3_makurizasi",
                    "no4_sasi", "no4_makuri", "no4_makurizasi",
                    "no5_sasi", "no5_makuri", "no5_makurizasi",
                    "no6_sasi", "no6_makuri", "no6_makurizasi"]:
            if col not in sasi_makuri_df.columns:
                sasi_makuri_df[col] = 0

        # 集計用の定義
        agg_columns = {
            "no1_sasare_by_no2": "sum",
            "no1_makurare_by_no2": "sum",
            "no1_makurare_by_no3": "sum",
            "no1_makurisasare_by_no3": "sum",
            "no2_sasi": "sum",
            "no2_makuri": "sum",
            "no3_makuri": "sum",
            "no3_makurizasi": "sum",
            "no4_sasi": "sum",
            "no4_makuri": "sum",
            "no4_makurizasi": "sum",
            "no5_sasi": "sum",
            "no5_makuri": "sum",
            "no5_makurizasi": "sum",
            "no6_sasi": "sum",
            "no6_makuri": "sum",
            "no6_makurizasi": "sum"
        }

        if venue == "全国":
            result = sasi_makuri_df.groupby(["登録番号", "選手名"]).agg(agg_columns).reset_index()
            return result

        else:
            # レース場が指定された場合、そのレース場のみフィルタリング
            venue_data = sasi_makuri_df[sasi_makuri_df["レース場"] == venue]
            
            if venue_data.empty:
                raise ValueError(f"指定されたレース場「{venue}」のデータが存在しません。")
            
            # 指定されたレース場のデータを集計
            result = venue_data.groupby(["登録番号", "選手名"]).agg(agg_columns).reset_index()
            return result

    def get_boatrace_data(self,start_date, end_date, venue, boat_number_1, boat_number_2, boat_number_3, boat_number_4, boat_number_5, boat_number_6):
        """
        選手登録番号を入力し、各艇のデータを取得する
        """
        result = self.base_data(start_date=start_date, end_date=end_date, venue=venue)
        result_sasi_makuri = self.sasi_makuri_data(start_date=start_date, end_date=end_date, venue=venue)
        result_escape_only = self.escape_only_data(start_date=start_date, end_date=end_date, venue=venue)
        result_st_time = self.st_time_data(start_date=start_date, end_date=end_date, venue=venue)
        
        # 1号艇のデータ
        one_df = result[(result["登録番号"] == boat_number_1) & (result["艇番"] == 1)]
        one_st_df = result_st_time[(result_st_time["登録番号"] == boat_number_1)]
        one_sasi_makuri_df = result_sasi_makuri[(result_sasi_makuri["登録番号"] == boat_number_1)]
        # 選手名
        one_name = one_df["選手名"].iloc[0] if not one_df.empty else "Unknown"
        # 1号艇の1着率,1号艇2号艇にまくられさされ率,1号艇3号艇にまくられさされ率,平均ST
        one_rate = one_df["勝利回数"].iloc[0] / one_df["出走数"].iloc[0] if not one_df.empty and one_df["出走数"].iloc[0] > 0 else 0
        one_overtaken_no2_rate = (
            (one_sasi_makuri_df["no1_sasare_by_no2"].iloc[0] + one_sasi_makuri_df["no1_makurare_by_no2"].iloc[0]) 
            / one_df["出走数"].iloc[0] if not one_sasi_makuri_df.empty and not one_df.empty and one_df["出走数"].iloc[0] > 0 else 0
        )
        one_overtaken_no3_rate = (
            (one_sasi_makuri_df["no1_makurare_by_no3"].iloc[0] + one_sasi_makuri_df["no1_makurisasare_by_no3"].iloc[0]) 
            / one_df["出走数"].iloc[0] if not one_sasi_makuri_df.empty and not one_df.empty and one_df["出走数"].iloc[0] > 0 else 0
        )
        one_st_time = one_st_df["平均ST"].iloc[0] if not one_st_df.empty else 0

        # 2号艇のデータ
        two_df = result[(result["登録番号"] == boat_number_2) & (result["艇番"] == 2)]
        two_st_df = result_st_time[(result_st_time["登録番号"] == boat_number_2)]
        two_escape_only_df = result_escape_only[(result_escape_only["登録番号"] == boat_number_2) & (result_escape_only["艇番"] == 2)]
        two_sasi_makuri_df = result_sasi_makuri[(result_sasi_makuri["登録番号"] == boat_number_2)]
        # 選手名
        two_name = two_df["選手名"].iloc[0] if not two_df.empty else "Unknown"
        # 2号艇の差し率、まくり率、逃し率、1号艇が逃げた時の2－3着率,平均ST
        two_sasi_rate = (
            two_sasi_makuri_df["no2_sasi"].iloc[0] / two_df["出走数"].iloc[0]
            if not two_sasi_makuri_df.empty and not two_df.empty and two_df["出走数"].iloc[0] > 0
            else 0
        )
        two_makuri_rate = (
            two_sasi_makuri_df["no2_makuri"].iloc[0] / two_df["出走数"].iloc[0]
            if not two_sasi_makuri_df.empty and not two_df.empty and two_df["出走数"].iloc[0] > 0
            else 0
        )
        two_nogashi_rate = (
            two_escape_only_df["出走数"].iloc[0] / two_df["出走数"].iloc[0]
            if not two_escape_only_df.empty and not two_df.empty and two_df["出走数"].iloc[0] > 0
            else 0
        )
        two_no1_nogashi_23_rate = (
            (two_escape_only_df["2着回数"].iloc[0] + two_escape_only_df["3着回数"].iloc[0])
            / two_df["出走数"].iloc[0] if not two_escape_only_df.empty and not two_df.empty and two_df["出走数"].iloc[0] > 0 else 0
        )
        two_st_time = two_st_df["平均ST"].iloc[0] if not two_st_df.empty else 0

        # 3号艇のデータ
        three_df = result[(result["登録番号"] == boat_number_3) & (result["艇番"] == 3)]
        three_st_df = result_st_time[(result_st_time["登録番号"] == boat_number_3)]
        three_sasi_makuri_df = result_sasi_makuri[(result_sasi_makuri["登録番号"] == boat_number_3)]
        three_escape_only_df = result_escape_only[(result_escape_only["登録番号"] == boat_number_3) & (result_escape_only["艇番"] == 3)]
        # 選手名
        three_name = three_df["選手名"].iloc[0] if not three_df.empty else "Unknown"
        # 3号艇のまくり1着率、まくり差し1着率、1号艇が逃げた時の2－3着率,平均ST
        three_makuri_rate = (
            three_sasi_makuri_df["no3_makuri"].iloc[0] / three_df["出走数"].iloc[0]
            if not three_sasi_makuri_df.empty and not three_df.empty and three_df["出走数"].iloc[0] > 0
            else 0
        )
        three_makurizasi_rate = (
            three_sasi_makuri_df["no3_makurizasi"].iloc[0] / three_df["出走数"].iloc[0]
            if not three_sasi_makuri_df.empty and not three_df.empty and three_df["出走数"].iloc[0] > 0
            else 0
        )
        three_no1_nogashi_23_rate = (
            (three_escape_only_df["2着回数"].iloc[0] + three_escape_only_df["3着回数"].iloc[0])
            / three_df["出走数"].iloc[0] if not three_escape_only_df.empty and not three_df.empty and three_df["出走数"].iloc[0] > 0 else 0
        )
        three_st_time = three_st_df["平均ST"].iloc[0] if not three_st_df.empty else 0

        # 4号艇のデータ
        four_df = result[(result["登録番号"] == boat_number_4) & (result["艇番"] == 4)]
        four_st_df = result_st_time[(result_st_time["登録番号"] == boat_number_4)]
        four_sasi_makuri_df = result_sasi_makuri[(result_sasi_makuri["登録番号"] == boat_number_4)]
        four_escape_only_df = result_escape_only[(result_escape_only["登録番号"] == boat_number_4) & (result_escape_only["艇番"] == 4)]
        # 選手名
        four_name = four_df["選手名"].iloc[0] if not four_df.empty else "Unknown"
        # 4号艇の差し1着率,まくり1着率,まくり差し1着率,1号艇が逃げた時の2－3着率,平均ST
        four_sasi_rate = (
            four_sasi_makuri_df["no4_sasi"].iloc[0] / four_df["出走数"].iloc[0]
            if not four_sasi_makuri_df.empty and not four_df.empty and four_df["出走数"].iloc[0] > 0
            else 0
        )
        four_makuri_rate = (
            four_sasi_makuri_df["no4_makuri"].iloc[0] / four_df["出走数"].iloc[0]
            if not four_sasi_makuri_df.empty and not four_df.empty and four_df["出走数"].iloc[0] > 0
            else 0
        )
        four_makurizasi_rate = (
            four_sasi_makuri_df["no4_makurizasi"].iloc[0] / four_df["出走数"].iloc[0]
            if not four_sasi_makuri_df.empty and not four_df.empty and four_df["出走数"].iloc[0] > 0
            else 0
        )
        four_no1_nogashi_23_rate = (
            (four_escape_only_df["2着回数"].iloc[0] + four_escape_only_df["3着回数"].iloc[0])
            / four_df["出走数"].iloc[0] if not four_escape_only_df.empty and not four_df.empty and four_df["出走数"].iloc[0] > 0 else 0
        )
        four_st_time = four_st_df["平均ST"].iloc[0] if not four_st_df.empty else 0

        # 5号艇のデータ
        five_df = result[(result["登録番号"] == boat_number_5) & (result["艇番"] == 5)]
        five_st_df = result_st_time[(result_st_time["登録番号"] == boat_number_5)]
        five_sasi_makuri_df = result_sasi_makuri[(result_sasi_makuri["登録番号"] == boat_number_5)]
        five_escape_only_df = result_escape_only[(result_escape_only["登録番号"] == boat_number_5) & (result_escape_only["艇番"] == 5)]
        # 選手名
        five_name = five_df["選手名"].iloc[0] if not five_df.empty else "Unknown"
        # 5号艇の差し1着率,まくり1着率,まくり差し1着率,1号艇が逃げた時の2－3着率,平均ST
        five_sasi_rate = (
            five_sasi_makuri_df["no5_sasi"].iloc[0] / five_df["出走数"].iloc[0]
            if not five_sasi_makuri_df.empty and not five_df.empty and five_df["出走数"].iloc[0] > 0
            else 0
        )
        five_makuri_rate = (
            five_sasi_makuri_df["no5_makuri"].iloc[0] / five_df["出走数"].iloc[0]
            if not five_sasi_makuri_df.empty and not five_df.empty and five_df["出走数"].iloc[0] > 0
            else 0
        )
        five_makurizasi_rate = (
            five_sasi_makuri_df["no5_makurizasi"].iloc[0] / five_df["出走数"].iloc[0]
            if not five_sasi_makuri_df.empty and not five_df.empty and five_df["出走数"].iloc[0] > 0
            else 0
        )
        five_no1_nogashi_23_rate = (
            (five_escape_only_df["2着回数"].iloc[0] + five_escape_only_df["3着回数"].iloc[0])
            / five_df["出走数"].iloc[0] if not five_escape_only_df.empty and not five_df.empty and five_df["出走数"].iloc[0] > 0 else 0
        )
        five_st_time = five_st_df["平均ST"].iloc[0] if not five_st_df.empty else 0

        # 6号艇のデータ
        six_df = result[(result["登録番号"] == boat_number_6) & (result["艇番"] == 6)]
        six_st_df = result_st_time[(result_st_time["登録番号"] == boat_number_6)]
        six_sasi_makuri_df = result_sasi_makuri[(result_sasi_makuri["登録番号"] == boat_number_6)]
        six_escape_only_df = result_escape_only[(result_escape_only["登録番号"] == boat_number_6) & (result_escape_only["艇番"] == 6)]
        # 選手名
        six_name = six_df["選手名"].iloc[0] if not six_df.empty else "Unknown"
        # 6号艇の差し1着率,まくり1着率,まくり差し1着率,1号艇が逃げた時の2－3着率,平均ST
        six_sasi_rate = (
            six_sasi_makuri_df["no6_sasi"].iloc[0] / six_df["出走数"].iloc[0]
            if not six_sasi_makuri_df.empty and not six_df.empty and six_df["出走数"].iloc[0] > 0
            else 0
        )
        six_makuri_rate = (
            six_sasi_makuri_df["no6_makuri"].iloc[0] / six_df["出走数"].iloc[0]
            if not six_sasi_makuri_df.empty and not six_df.empty and six_df["出走数"].iloc[0] > 0
            else 0
        )
        six_makurizasi_rate = (
            six_sasi_makuri_df["no6_makurizasi"].iloc[0] / six_df["出走数"].iloc[0]
            if not six_sasi_makuri_df.empty and not six_df.empty and six_df["出走数"].iloc[0] > 0
            else 0
        )
        six_no1_nogashi_23_rate = (
            (six_escape_only_df["2着回数"].iloc[0] + six_escape_only_df["3着回数"].iloc[0])
            / six_df["出走数"].iloc[0] if not six_escape_only_df.empty and not six_df.empty and six_df["出走数"].iloc[0] > 0 else 0
        )
        six_st_time = six_st_df["平均ST"].iloc[0] if not six_st_df.empty else 0

        # 結果を辞書形式で返す
        outputs = {
            "1号艇": {
                "選手名": one_name,
                "1着率": one_rate,
                "2号艇にまくられさされ率": one_overtaken_no2_rate,
                "3号艇にまくられさされ率": one_overtaken_no3_rate,
                "平均ST": one_st_time
                },
            "2号艇": {
                "選手名": two_name,
                "差し率": two_sasi_rate,
                "まくり率": two_makuri_rate,
                "逃し率": two_nogashi_rate,
                "1号艇が逃げた時の2-3着率": two_no1_nogashi_23_rate,
                "平均ST": two_st_time
                },
            "3号艇": {
                "選手名": three_name,
                "まくり1着率": three_makuri_rate,
                "まくり差し1着率": three_makurizasi_rate,
                "1号艇が逃げた時の2-3着率": three_no1_nogashi_23_rate,
                "平均ST": three_st_time
                },
            "4号艇": {
                "選手名": four_name,
                "差し1着率": four_sasi_rate,
                "まくり1着率": four_makuri_rate,
                "まくり差し1着率": four_makurizasi_rate,
                "1号艇が逃げた時の2-3着率": four_no1_nogashi_23_rate,
                "平均ST": four_st_time
                },
            "5号艇": {
                "選手名": five_name,
                "差し1着率": five_sasi_rate,
                "まくり1着率": five_makuri_rate,
                "まくり差し1着率": five_makurizasi_rate,
                "1号艇が逃げた時の2-3着率": five_no1_nogashi_23_rate,
                "平均ST": five_st_time
                },
            "6号艇": {
                "選手名": six_name,
                "差し1着率": six_sasi_rate,
                "まくり1着率": six_makuri_rate,
                "まくり差し1着率": six_makurizasi_rate,
                "1号艇が逃げた時の2-3着率": six_no1_nogashi_23_rate,
                "平均ST": six_st_time
                }
            }
        return outputs
        
if __name__ == "__main__":
    analyzer = BoatraceAnalyzer(folder = "C:\\Users\\msy-t\\boatrace-ai\\data")
    start_date = "2024-11-01"
    end_date = "2025-03-30"
    result = analyzer.base_data(start_date=start_date, end_date=end_date, venue="全国")
    result.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\全国_result.csv", index=False, encoding="shift_jis")
    # 特定のレース場のデータを集計
    venue_name="住之江"
    results_by_venue = analyzer.base_data(start_date=start_date, end_date=end_date, venue=venue_name)
    results_by_venue.to_csv(f"C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\{venue_name}_result.csv", index=False, encoding="shift_jis")
    # スタートタイミングの集計
    st_time_results = analyzer.st_time_data(start_date=start_date, end_date=end_date, venue="全国")
    st_time_results.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\st_time_全国_result.csv", index=False, encoding="shift_jis")
    # 逃げデータのみの集計
    result_escape_only = analyzer.escape_only_data(start_date=start_date, end_date=end_date, venue="全国")
    result_escape_only.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\escape_全国_result.csv", index=False, encoding="shift_jis")
    # 差しまくりデータの集計
    sasi_makuri_results = analyzer.sasi_makuri_data(start_date=start_date, end_date=end_date, venue="全国")
    sasi_makuri_results.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\sasi_makuri_全国_result.csv", index=False, encoding="shift_jis")
