import os
from datetime import datetime, timedelta
import re

class BoatraceBase:
    def __init__(self, folder):
        self.folder = folder

    def generate_date_list(self, start_date, end_date, BorK):
        """
        start_date から end_date までの日付を基に、指定の形式で日付リストを生成する。
        
        :param start_date: 開始日（YYYY-MM-DD形式）
        :param end_date: 終了日（YYYY-MM-DD形式）
        :param BorK: ボートレースの種類（"B" または "K"）
        :return: 日付リスト（例: ["B250221", "B250222", ...]）
        """
        # 日付を datetime 型に変換
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        date_list = []
        
        # 開始日から終了日までの日付を生成
        current_date = start_dt
        while current_date <= end_dt:
            formatted_date = current_date.strftime(f"{BorK}%y%m%d")
            date_list.append(formatted_date)
            current_date += timedelta(days=1)
        
        return date_list
    
    def find_current_dates(self,today_date, BorK="K"):
        """
        指定されたフォルダ内のファイル名を調べて、欠けている日付をリストとして返す
        """
        path = self.folder + "\\" + BorK + "_lzh"
        
        # 指定されたフォルダ内の .lzh ファイル名を取得
        file_list = [f for f in os.listdir(path) if f.endswith(".lzh")]

        # 今日の日付を取得
        today = datetime.strptime(today_date, '%Y-%m-%d').date()
        
        # ファイル名から日付を抽出して、datetime オブジェクトに変換する
        dates = []
        for file_name in file_list:
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
    

