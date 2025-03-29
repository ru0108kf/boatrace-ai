import os
from datetime import datetime, timedelta

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
    

