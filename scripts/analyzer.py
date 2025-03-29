import pandas as pd
import os

from base import BoatraceBase  # ← 親クラスのインポート

class BoatraceAnalyzer(BoatraceBase):
    def __init__(self, folder):
        super().__init__(folder)
        self.B_csv_folder = os.path.join(self.folder, "B_csv")
        self.K_csv_folder = os.path.join(self.folder, "K_csv")
        

    def load_and_process_file(self, file_path):
        try:
            df = pd.read_csv(file_path, encoding="shift_jis")
        except FileNotFoundError:
            return None  # ファイルが存在しない場合は None を返す
        
        # フラグを作成し、整数に変換
        df["1着"] = (df["着順"] == 1).astype(int)
        df["2着"] = (df["着順"] == 2).astype(int)
        df["3着"] = (df["着順"] == 3).astype(int)
        
        decision_types = ["逃げ", "差し", "まくり", "まくり差し", "抜き", "恵まれ"]
        for k in decision_types:
            df[k] = (df["決まり手"] == k).astype(int)
        
        return df
    
    def aggregate_data(self, start_date, end_date, all_venues=False):
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

        if all_venues:
            results = {}
            for venue_name, group in combined_df.groupby("レース場"):
                result = group.groupby(["登録番号", "選手名", "艇番"]).agg(agg_columns).reset_index()
                result.columns = ["登録番号", "選手名", "艇番", "出走数", "勝利回数", "2着回数", "3着回数", "平均ST", 
                                "逃げ", "差し", "まくり", "まくり差し", "抜き", "恵まれ"]
                results[venue_name] = result
            return results
        else:
            result = combined_df.groupby(["登録番号", "選手名", "艇番"]).agg(agg_columns).reset_index()
            result.columns = ["登録番号", "選手名", "艇番", "出走数", "勝利回数", "2着回数", "3着回数", "平均ST", 
                            "逃げ", "差し", "まくり", "まくり差し", "抜き", "恵まれ"]
            return result


#if __name__ == "__main__":
analyzer = BoatraceAnalyzer(folder = "C:\\Users\\msy-t\\boatrace-ai\\data")
#result = analyzer.aggregate_data(start_date="2024-11-01", end_date="2025-03-28")
#result.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\aggregated_result.csv", index=False, encoding="shift_jis")

results_by_venue = analyzer.aggregate_data(start_date="2024-11-01", end_date="2025-03-28", all_venues=True)
for venue_name, result_df in results_by_venue.items():
    file_path = f"C:\\Users\\msy-t\\boatrace-ai\\data\\{venue_name}_result.csv"
    result_df.to_csv(file_path, index=False, encoding="shift_jis")
    print(f"{venue_name} の成績を保存しました: {file_path}")