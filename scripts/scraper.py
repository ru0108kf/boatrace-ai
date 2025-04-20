import os
import time
from datetime import datetime, timedelta
import lhafile
import traceback
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

import re
import pandas as pd

from base import BoatraceBase

class BoatraceScraper(BoatraceBase):
    def __init__(self,folder,BorK="B"):
        super().__init__(folder)
        self.BorK = BorK
        self.download_folder = os.path.join(self.folder, f"{BorK}_lzh")
        self.kaitou_folder = os.path.join(self.folder, f"{BorK}_txt")
        self.csv_folder = os.path.join(self.folder,f"{BorK}_csv")
        self.odds_folder = os.path.join(self.folder,"o_csv")
        os.makedirs(self.download_folder, exist_ok=True)
        os.makedirs(self.kaitou_folder, exist_ok=True)
        self.url = "https://www1.mbrace.or.jp/od2/B/dindex.html" if BorK == "B" else "https://www1.mbrace.or.jp/od2/K/dindex.html"
        
        # Chrome options for downloading
        self.options = webdriver.ChromeOptions()
        self.options.add_experimental_option("prefs", {
            "download.default_directory": self.download_folder,
            "download.prompt_for_download": False,
            "directory_upgrade": True,
            "safebrowsing.enabled": True
        })
        
        # Data placeholders
        self.program_data = None
        self.result_data = None
        self.pre_race_data = None

    def extract_lzh_file(self, file_name):
        """指定されたLZHファイルを解凍"""
        os.makedirs(self.kaitou_folder, exist_ok=True)  # 解凍先フォルダがなければ作成
        
        # LZHファイルのパスを作成
        lzh_file_path = os.path.join(self.download_folder, f"{file_name}.lzh")

        # LZHファイルを開く
        file = lhafile.Lhafile(lzh_file_path)
        
        # LZHファイル内の実際のファイル名を取得
        info = file.infolist()
        name = info[0].filename

        # 解凍したファイルを保存
        with open(os.path.join(self.kaitou_folder, name), "wb") as f:
            f.write(file.read(name))

    def scrape_data(self, start_date, end_date):
        """指定された日付範囲のデータをまとめてダウンロード"""
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        browser.implicitly_wait(10)

        try:
            browser.get(self.url)
            current_month = None

            while current_date <= end_date_dt:
                year, month, day = current_date.year, current_date.month, current_date.day

                if current_month != (year, month):
                    current_month = (year, month)

                    # menuフレームに切り替え
                    browser.switch_to.default_content()
                    browser.switch_to.frame('menu')

                    now = datetime.now()
                    months_ago = (now.year - year) * 12 + (now.month - month)

                    # 月のドロップダウンを選択
                    dropdown = browser.find_element(By.NAME, 'MONTH')
                    select = Select(dropdown)
                    select.select_by_index(months_ago + 1)

                    # フレームをリセットしてJYOUに切り替え
                    browser.switch_to.default_content()
                    time.sleep(1)
                    browser.switch_to.frame("JYOU")
                else:
                    # 月が変わっていない場合はフレームだけ切り替える
                    browser.switch_to.default_content()
                    browser.switch_to.frame("JYOU")

                try:
                    radio_buttons = browser.find_elements(By.XPATH, "//*[@type='radio']")
                    radio_buttons[day - 1].click()
                    time.sleep(1)

                    download_button = browser.find_element(By.XPATH, "//*[@value='ダウンロード開始']")
                    download_button.click()
                    time.sleep(2)

                except Exception as e:
                    print(f"{current_date.strftime('%Y-%m-%d')} の処理中にエラーが発生しました: {e}")
                    traceback.print_exc()

                current_date += timedelta(days=1)

        except Exception as e:
            print(f"全体処理中にエラーが発生しました: {e}")
            traceback.print_exc()
        finally:
            browser.quit()

    def parse_B_txt(self, file_name):
        """
        番組表のtxtを解析し、 DataFrame に変換する

        :param file_name: 解析するテキストファイルの名前
        :return: 解析したDataFrame
        """
        os.makedirs(self.csv_folder, exist_ok=True)  # 保存先フォルダがなければ作成
        
        # txtファイルのパスを作成
        txt_file_path = os.path.join(self.kaitou_folder, f"{file_name}.txt")
        
        # ファイルの読み込み
        with open(txt_file_path, "r", encoding="shift_jis") as f:
            lines = f.readlines()

        race_data = []
        race_info = None
        race_place = None
        
        # 行ごとに解析
        for line in lines:
            line = line.strip()
            if not line:  # 空行はスキップ
                continue

            # レース場の取得
            match_place = re.search(r"ボートレース(.{3})", line)
            if match_place:
                race_place = match_place.group(1).replace("　", "").strip()

            # レース情報の取得
            match_race = re.match(r"([０-９])Ｒ\s+(\S+)", line)
            if match_race:
                race_number_zen = match_race.group(1)
                zen_to_han = str.maketrans("０１２３４５６７８９", "0123456789")
                race_number_han = race_number_zen.translate(zen_to_han)  # 半角に変換
                race_info = {
                    "レース場": race_place,
                    "レース番号": race_number_han,
                    "レース種別": match_race.group(2)
                }

            # 選手情報の取得
            match_player = re.match(r"^\d\s\d{4}", line)
            if match_player and race_info:
                race_player = {
                    "艇番": line[:1].strip(),
                    "登録番号": line[1:6].strip(),
                    "選手名": line[6:10].strip(),
                    "年齢": line[10:12].strip(),
                    "支部": line[12:14].strip(),
                    "体重": line[14:16].strip(),
                    "級別": line[16:18].strip(),
                    "全国勝率": line[18:23].strip(),
                    "全国2連対率": line[23:29].strip(),
                    "当地勝率": line[29:34].strip(),
                    "当地2連対率": line[34:40].strip(),
                    "モーター番号": line[40:43].strip(),
                    "モーター2連対率": line[43:49].strip(),
                    "ボート番号": line[49:52].strip(),
                    "ボート2連対率": line[52:58].strip()
                }
                
                # レース情報と結合して1行分のレースデータを作成
                full_data = {**race_info, **race_player}
                race_data.append(full_data)

        # DataFrame に変換
        columns = [
            "レース場", "レース番号", "レース種別",
            "艇番", "登録番号", "選手名", "年齢", "支部", "体重", "級別",
            "全国勝率", "全国2連対率", "当地勝率", "当地2連対率",
            "モーター番号", "モーター2連対率", "ボート番号", "ボート2連対率"
        ]

        df = pd.DataFrame(race_data, columns=columns)
        
        csv_path = os.path.join(self.csv_folder, f"{file_name}.csv")
        df.to_csv(csv_path, index=False, encoding="shift_jis")
        
        return df

    def parse_K_txt(self, file_name):
        """
        レース成績のtxtを解析し、DataFrame に変換する
        
        :param file_name: 解析するテキストファイルの名前
        :return: 解析したDataFrame
        """
        os.makedirs(self.csv_folder, exist_ok=True)  # 保存先フォルダがなければ作成
        
        # txtファイルのパスを作成
        txt_file_path = os.path.join(self.kaitou_folder, f"{file_name}.txt")
        
        # ファイルの読み込み
        with open(txt_file_path, "r", encoding="shift_jis") as f:
            lines = f.readlines()
        
        race_data = []
        race_info = None
        race_place = None
        
        for line in lines:
            line = line.strip()
            if not line:  # 空行はスキップ
                continue
        
            # レース場の取得
            match_place = re.search(r"(^.+?)［成績］", line)
            if match_place:
                race_place = match_place.group(1).replace("　", "").strip()
        
            # レース情報の取得
            match_race = re.match(r"([0-9]+)R\s+(\S+).*?H\d+m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm", line)
            if match_race:
                race_info = {
                    "レース場": race_place,
                    "レース番号": match_race.group(1),
                    "レース種別": match_race.group(2),
                    "天候": match_race.group(3),
                    "風向": match_race.group(4),
                    "風速(m)": match_race.group(5),
                    "波高(cm)": match_race.group(6),
                    "決まり手": "",
                }
        
            # 決まり手の取得
            match_kimarite = re.search(r"ﾚｰｽﾀｲﾑ\s+(.+)", line)
            if match_kimarite and race_info:
                race_info["決まり手"] = match_kimarite.group(1).replace("　", "").strip()
        
            # 選手データの取得
            match_player = re.match(r"^\d{2}\s+\d\s+\d{4}", line)
            if match_player and race_info:
                race_player = {
                    "着順": line[0:4].strip(),
                    "艇番": line[4:6].strip(),
                    "登録番号": line[6:11].strip(),
                    "選手名": line[11:19].replace("　", "").strip(),
                    "モーター番号": line[19:22].strip(),
                    "ボート番号": line[22:27].strip(),
                    "展示タイム": line[27:33].strip(),
                    "進入": line[33:37].strip(),
                    "スタートタイミング": line[37:45].strip(),
                    "レースタイム": line[45:56].strip(),
                }
        
                # レース情報と結合して1行分のレースデータを作成
                full_data = {**race_info, **race_player}
                race_data.append(full_data)
        
        # DataFrame 変換
        columns = [
            "レース場", "レース番号", "レース種別", "艇番", "登録番号", "選手名", "モーター番号", "ボート番号",
            "天候", "風向", "風速(m)", "波高(cm)", "展示タイム",
            "着順", "決まり手", "進入", "スタートタイミング", "レースタイム"
        ]
        
        df = pd.DataFrame(race_data, columns=columns)
        
        csv_path = os.path.join(self.csv_folder, f"{file_name}.csv")
        df.to_csv(csv_path, index=False, encoding="shift_jis")
        
        return df
    
    def parse_all_odds_from_K(self,file_name):
        os.makedirs(self.odds_folder, exist_ok=True)  # 保存先フォルダがなければ作成
        
        # txtファイルのパスを作成
        txt_file_path = os.path.join(self.folder+"\\K_txt", f"{file_name}.txt")
        
        file_date = "20" + file_name[1:3] + "-" + file_name[3:5] + "-" + file_name[5:7]
        
        # ファイルの読み込み
        with open(txt_file_path, "r", encoding="shift_jis") as f:
            lines = f.readlines()

        odds_data = []
        current_place = None
        race_number = None

        for line in lines:
            line = line.strip()

            # 場所取得（例: ボートレース大村）
            if "ボートレース" in line:
                match = re.search(r"ボートレース(.+)", line)
                if match:
                    current_place = match.group(1).replace("　", "").strip()

            # レース番号取得（例: " 1R"）
            match_race = re.match(r"^\s*(\d{1,2})R", line)
            if match_race:
                race_number = int(match_race.group(1))

            # 舟券種ごとのパターンマッチング
            for bet_type in ["単勝", "複勝", "２連単", "２連複", "拡連複", "３連単", "３連複"]:
                if line.startswith(bet_type):
                    # 組合せと払戻金をすべて抽出（同じ行に複数あるため）
                    matches = re.findall(r"(\d(?:-\d)?(?:-\d)?)\s+([\d,]+)", line)
                    for combo, payout in matches:
                        odds_data.append({
                            "日付": file_date,
                            "レース場": current_place,
                            "レース番号": race_number,
                            "舟券種": bet_type,
                            "組合せ": combo,
                            "払戻金": int(payout.replace(",", ""))
                        })

        df = pd.DataFrame(odds_data)
        csv_path = os.path.join(self.odds_folder, f"o{file_name}.csv")
        df.to_csv(csv_path, index=False, encoding="shift_jis")
        return df

    def scrape_and_process_data(self, start_date, end_date):
        """スクレイピングと解凍, CSVに変換までを行う一連の流れ"""
        
        print(f"スクレイピング開始: {start_date} ～ {end_date}")
        self.scrape_data(start_date, end_date)
        print("スクレイピング完了")
        
        file_lists = []
        for file in self.generate_date_list(start_date, end_date):
            file_lists.append(self.BorK + file)
        
        print("解凍処理開始")
        for file_name in file_lists:
            self.extract_lzh_file(file_name)
        print("解凍処理完了")

        if self.BorK == "B":
            print("番組表テキストの解析（B方式）開始")
            for file_name in file_lists:
                self.parse_B_txt(file_name)
            print("B番組表CSV変換完了")
        elif self.BorK == "K":
            print("番組表テキストの解析（K方式）開始")
            for file_name in file_lists:
                self.parse_K_txt(file_name)
                self.parse_all_odds_from_K(file_name)
            print("K番組表CSV変換完了")

        print("全処理完了")

class BoatraceLatestDataScraper():
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
    

if __name__ == "__main__":
    # ==================変更すべき欄==================
    folder = "C:\\Users\\msy-t\\boatrace-ai\\data"
    # ==================変更してもOK==================
    scraper = BoatraceScraper(folder, BorK="K")
            
    start_date = "2022-04-01"
    end_date = "2022-12-31"
    # ===============成績表スクレイピング===============
    scraperK = BoatraceScraper(folder, BorK="K")
    scraperK.scrape_and_process_data(start_date,end_date)
    # ===============番組表スクレイピング===============
    scraperB = BoatraceScraper(folder, BorK="B")
    scraperB.scrape_and_process_data(start_date,end_date)
