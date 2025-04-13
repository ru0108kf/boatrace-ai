import os
from datetime import datetime, timedelta
import re
import pandas as pd

class BoatraceBase:
    def __init__(self, folder):
        self.folder = folder

    def generate_date_list(self, start_date, end_date):
        """
        start_date から end_date までの日付を基に、指定の形式で日付リストを生成する。
        
        :param start_date: 開始日（YYYY-MM-DD形式）
        :param end_date: 終了日（YYYY-MM-DD形式）
        :return: 日付リスト（例: ["250221", "250222", ...]）
        """
        # 日付を datetime 型に変換
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        date_list = []
        
        # 開始日から終了日までの日付を生成
        current_date = start_dt
        while current_date <= end_dt:
            formatted_date = current_date.strftime(f"%y%m%d")
            date_list.append(formatted_date)
            current_date += timedelta(days=1)
        
        return date_list
    
    def find_current_dates(self, date, BorK):
        """
        指定されたフォルダ内のファイル名を調べて、欠けている日付をリストとして返す
        """
        path = self.folder + "\\" + BorK + "_lzh"
        
        # 指定されたフォルダ内の .lzh ファイル名を取得
        file_list = [f for f in os.listdir(path) if f.endswith(".lzh")]
        

        # 今日の日付を取得
        today = datetime.strptime(date, '%Y-%m-%d').date()
        
        # ファイル名から日付を抽出して、datetime オブジェクトに変換する
        dates = []
        for file_name in file_list:
            if BorK == "B":
                match = re.search(r'b(\d{6})\.lzh', file_name)
            else:
                match = re.search(r'k(\d{6})\.lzh', file_name)
            if match:
                date_str = match.group(1)
                date = datetime.strptime("20" + date_str, '%Y%m%d').date()
                dates.append(date)
        
        # 日付を昇順にソート
        dates.sort()
        
        # ファイル内の最も新しい日付を取得
        latest_date = dates[-1]
        
        # 今日までの間でファイルに存在しない日付を見つける
        current_date = latest_date + timedelta(days=1)
        while current_date <= today:
            if current_date not in dates:
                return current_date.strftime('%Y-%m-%d')  # 欠けている日付を返す
            current_date += timedelta(days=1)
        
        return None  # すべて連続している場合は None を返す
    
    def get_player_numbers_by_race(self, date, venue, race_number):
        """日付とレース場とレース番号を入力として、対応する選手情報を出力する関数"""
        
        file_name = self.generate_date_list(date, date)[0]
        path = self.folder + "\\B_csv\\B" + file_name + ".csv"
        
        try:
            df = pd.read_csv(path, encoding='shift-jis')
        except FileNotFoundError:
            raise FileNotFoundError(f"指定されたCSVファイルが見つかりません: {path}")
        except Exception as e:
            raise Exception(f"CSVファイルの読み込み中にエラーが発生しました: {e}")
        
        # レース場とレース番号でフィルタリング
        filtered_data = df[(df['レース場'] == venue) & (df['レース番号'] == race_number)]
        
        if filtered_data.empty:
            raise ValueError(f"フィルタリング結果が空です。該当するデータが見つかりません。venue: {venue}, race_number: {race_number}")
        
        if len(filtered_data) < 6:
            raise IndexError(f"フィルタリング後のデータが不足しています。必要なデータ数: 6、取得できたデータ数: {len(filtered_data)}")
        
        try:
            boat_number_1 = filtered_data['登録番号'].iloc[0]
            boat_number_2 = filtered_data['登録番号'].iloc[1]
            boat_number_3 = filtered_data['登録番号'].iloc[2]
            boat_number_4 = filtered_data['登録番号'].iloc[3]
            boat_number_5 = filtered_data['登録番号'].iloc[4]
            boat_number_6 = filtered_data['登録番号'].iloc[5]
        except IndexError as e:
            raise IndexError("必要なボート番号が不足しています。データが完全ではない可能性があります。") from e

        return boat_number_1, boat_number_2, boat_number_3, boat_number_4, boat_number_5, boat_number_6
