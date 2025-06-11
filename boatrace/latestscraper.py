import os
import time
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

import re
import pandas as pd

class BoatraceLatestDataScraper:
    def scrape_the_latest_info(self,place_name, race_no, date):
        """直前情報をボートレース日和からスクレイピング"""
        place_no = {
            "桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6,
            "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12,
            "尼崎": 13, "鳴門": 14, "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18,
            "下関": 19, "若松": 20, "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24
        }
        place_number = place_no.get(place_name)
        if place_number is None:
            return "The specified venue cannot be found."

        try:
            # 日付の形式を確認（例: 2025-03-31）
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%Y%m%d")
        except ValueError:
            return "Date format is incorrect. Please enter the date in the format 'YYYYY-MM-DD'."
        
        # レース番号の確認
        if not (1 <= race_no <= 12):
            return "Race numbers must be between 1 and 12."
        
        # URLを作成
        url = f"https://kyoteibiyori.com/race_shusso.php?place_no={place_number}&race_no={race_no}&hiduke={formatted_date}&slider=4"
        print(url)
        
        # Seleniumの設定
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # ヘッドレスモードでブラウザを表示せずに実行
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # ChromeDriverを自動でインストールして取得
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        # URLを指定
        driver.get(url)

        # ページを完全にロードするまで待機
        time.sleep(3)

        # 指定されたセクションの HTML を取得
        try:
            section = driver.find_element(By.XPATH, '/html/body/div[8]/div[1]/section')
            table_html = section.get_attribute('outerHTML')
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            driver.quit()
            exit()

        # BeautifulSoupを使ってテーブルをパース
        soup = BeautifulSoup(table_html, 'html.parser')

        # 表の中で必要な部分のみ抽出
        target_table = soup.find_all('tr')

        # データを保存するリスト
        data = []
        for row in target_table:
            cols = [col.text.strip() for col in row.find_all(['th', 'td'])]
            if cols:  # 空でない行を追加
                data.append(cols)

        # DataFrame に変換
        df = pd.DataFrame(data)

        # ブラウザを終了
        driver.quit()
        
        # 抽出対象の項目名
        target_rows = ['展示', '周回', '周り足', "直線", "ST"]

        # データフレーム内の各行の最初の要素を確認し、ターゲットの行を抽出
        filtered_df = df[df.iloc[:, 0].astype(str).isin(target_rows)].copy()

        # 必要に応じて列名を設定
        filtered_df.columns = ['項目', '1号艇', '2号艇', '3号艇', '4号艇', '5号艇', '6号艇']
        
        return filtered_df

    def scrape_wether(self, place_name, race_no, date):
        """
        直前の天気情報をボーダースからスクレイピング
        """
        place_en_name = {
            "桐生": "kiryu", "戸田": "toda", "江戸川": "edogawa", "平和島": "heiwajima", "多摩川": "tamagawa", "浜名湖": "hamanako",
            "蒲郡": "gamagori", "常滑": "tokoname", "津": "tsu", "三国": "mikuni", "びわこ": "biwako", "住之江": "suminoe",
            "尼崎": "amagasaki", "鳴門": "naruto", "丸亀": "marugame", "児島": "kojima", "宮島": "miyajima", "徳山": "tokuyama",
            "下関": "simonoseki", "若松": "wakamatsu", "芦屋": "asiya", "福岡": "fukuoka", "唐津": "karatsu", "大村": "oomura"
        }    
        place_name = place_en_name.get(place_name)
        if place_name is None:
            return "The specified venue cannot be found."
        
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date format is incorrect. Please enter the date in the format 'YYYYY-MM-DD'.")
        
        if not (1 <= race_no <= 12):
            return "Race numbers must be between 1 and 12."
        
        url = f"https://boaters-boatrace.com/race/{place_name}/{date}/{race_no}R?content=last-minute&last-minute-content=last-minute"
        print(url)
        
        # Seleniumの設定
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # ChromeDriverを自動でインストールして取得
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        # URLを指定
        driver.get(url)

        # ページを完全にロードするまで待機
        time.sleep(3)

        # ページ全体の HTML を取得
        page_html = driver.page_source

        # BeautifulSoupでパース
        soup = BeautifulSoup(page_html, 'html.parser')

        # <p> タグの中からクラス名 'chakra-text' を持つ要素を全て抽出
        data_elements = soup.find_all('p', class_="chakra-text css-86z7wg")

        # 抽出したデータを保存するリスト
        data = [element.get_text(strip=True) for element in data_elements]

        # データフレームに変換
        df = pd.DataFrame(data, columns=["気象情報"])

        # ブラウザを終了
        driver.quit()

        return df
   
    def get_latest_boatrace_data(self, place_name, race_no, date):
        """
        最新のレースデータを取得する
        """
        df = self.scrape_the_latest_info(place_name, race_no, date)
        wether_df = self.scrape_wether(place_name, race_no, date)
        
        weather_info = {}

        # インデックス数による場合分け
        if len(wether_df) == 5:
            weather_info["風速"] = "N/A"
            weather_info["風向き"] = "N/A"
            weather_info["天候"] = wether_df.iloc[1, 0]
            weather_info["気温"] = wether_df.iloc[2, 0]
            weather_info["波高"] = wether_df.iloc[3, 0]
            weather_info["水温"] = wether_df.iloc[4, 0]
        elif len(wether_df) == 6:
            weather_info["風速"] = wether_df.iloc[0, 0]
            weather_info["風向き"] = wether_df.iloc[1, 0]
            weather_info["天候"] = wether_df.iloc[2, 0]
            weather_info["気温"] = wether_df.iloc[3, 0]
            weather_info["波高"] = wether_df.iloc[4, 0]
            weather_info["水温"] = wether_df.iloc[5, 0]
        
        outputs = {
            "1号艇": {
                "展示": df["1号艇"][df["項目"]=="展示"].values[0],
                "周回": df["1号艇"][df["項目"]=="周回"].values[0],
                "周り足": df["1号艇"][df["項目"]=="周り足"].values[0],
                "直線": df["1号艇"][df["項目"]=="直線"].values[0],
                "ST": df["1号艇"][df["項目"]=="ST"].values[0]
            },
            "2号艇": {
                "展示": df["2号艇"][df["項目"]=="展示"].values[0],
                "周回": df["2号艇"][df["項目"]=="周回"].values[0],
                "周り足": df["2号艇"][df["項目"]=="周り足"].values[0],
                "直線": df["2号艇"][df["項目"]=="直線"].values[0],
                "ST": df["2号艇"][df["項目"]=="ST"].values[0]
            },
            "3号艇": {
                "展示": df["3号艇"][df["項目"]=="展示"].values[0],
                "周回": df["3号艇"][df["項目"]=="周回"].values[0],
                "周り足": df["3号艇"][df["項目"]=="周り足"].values[0],
                "直線": df["3号艇"][df["項目"]=="直線"].values[0],
                "ST": df["3号艇"][df["項目"]=="ST"].values[0]
            },
            "4号艇": {
                "展示": df["4号艇"][df["項目"]=="展示"].values[0],
                "周回": df["4号艇"][df["項目"]=="周回"].values[0],
                "周り足": df["4号艇"][df["項目"]=="周り足"].values[0],
                "直線": df["4号艇"][df["項目"]=="直線"].values[0],
                "ST": df["4号艇"][df["項目"]=="ST"].values[0]
            },
            "5号艇": {
                "展示": df["5号艇"][df["項目"]=="展示"].values[0],
                "周回": df["5号艇"][df["項目"]=="周回"].values[0],
                "周り足": df["5号艇"][df["項目"]=="周り足"].values[0],
                "直線": df["5号艇"][df["項目"]=="直線"].values[0],
                "ST": df["5号艇"][df["項目"]=="ST"].values[0]
            },
            "6号艇": {
                "展示": df["6号艇"][df["項目"]=="展示"].values[0],
                "周回": df["6号艇"][df["項目"]=="周回"].values[0],
                "周り足": df["6号艇"][df["項目"]=="周り足"].values[0],
                "直線": df["6号艇"][df["項目"]=="直線"].values[0],
                "ST": df["6号艇"][df["項目"]=="ST"].values[0]
            },
            "気象情報": weather_info
        }
        return outputs