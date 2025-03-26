import os
import time
import lhafile
import pandas as pd
import requests
import re
import traceback
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class BoatraceManager:
    def __init__(self, folder='../data', B_url="https://www1.mbrace.or.jp/od2/B/dindex.html", K_url="https://www1.mbrace.or.jp/od2/K/dindex.html"):
        self.folder = folder
        self.B_url = B_url
        self.K_url = K_url

        # Chrome options for downloading
        self.options = webdriver.ChromeOptions()

        # Data placeholders
        self.program_data = None
        self.result_data = None
        self.pre_race_data = None

    def _set_folders(self, BorK="B"):
        """BorKに基づいてフォルダのパスとurlを設定する"""
        self.download_folder = os.path.join(self.folder, f"{BorK}_lzh")
        self.kaitou_folder = os.path.join(self.folder, f"{BorK}_txt")
        
        if BorK == "B":
            self.url = self.B_url
        elif BorK == "K":
            self.url = self.K_url
        else:
            raise ValueError("BorK must be either 'B' or 'K'")  # BorK が "B" または "K" でない場合にエラーを発生させる

        print(f"ダウンロードフォルダ: {self.download_folder}")
        print(f"解凍フォルダ: {self.kaitou_folder}")
        print(f"使用するURL: {self.url}")

    def scrape_data_by_month(self, months=12, BorK="B"):
        """今日からさかのぼって、スクレイピングを行い、データをダウンロード"""
        self._set_folders(BorK)  # BorKに基づいてフォルダを設定
        
        # Chrome options for downloading
        self.options.add_experimental_option("prefs", {"download.default_directory": self.download_folder})
        
        browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        browser.implicitly_wait(10)  # 最大10秒待機

        try:
            # URLへ移動
            browser.get(self.url)
    
            for i in range(months):
                # フレームを選択
                browser.switch_to.frame('menu')
    
                # 月のドロップダウンを選択
                dropdown = browser.find_element(By.NAME, 'MONTH')
                select = Select(dropdown)
                select.select_by_index(i + 1 + 2)#####
    
                # フレームをリセット
                browser.switch_to.default_content()
                time.sleep(1)
    
                # "JYOU" フレームに切り替え
                browser.switch_to.frame("JYOU")
    
                # ラジオボタンを取得しクリックしてダウンロード
                radio_buttons = browser.find_elements(By.XPATH, "//*[@type='radio']")
                for j, radio in enumerate(radio_buttons):
                    radio.click()
                    time.sleep(1)  # 適度な間隔
                    download_button = browser.find_element(By.XPATH, "//*[@value='ダウンロード開始']")
                    download_button.click()
                    time.sleep(2)
    
                # フレームをリセット
                browser.switch_to.default_content()
                time.sleep(2)

        except Exception as e:
            print(f"エラーが発生しました: {e}")
            traceback.print_exc()  # エラーの詳細を表示
        finally:
            browser.quit()
        
    def scrape_data_for_single_day(self, target_date, BorK="B"):
        """指定された1日の日付のデータをダウンロード"""
        self._set_folders(BorK)  # BorKに基づいてフォルダを設定
        
        browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        browser.implicitly_wait(10)  # 最大10秒待機
    
        try:
            browser.get(self.url)
    
            # フレームの切り替え (もしフレーム内にある場合)
            browser.switch_to.frame('menu')
    
            # 日付フィールドが見つかるまで待機
            date_field = WebDriverWait(browser, 20).until(
                EC.presence_of_element_located((By.NAME, 'date'))
            )
            date_field.clear()  # 入力欄をクリア
            date_field.send_keys(target_date)
    
            # フレームをリセット
            browser.switch_to.default_content()
    
            # "JYOU" フレームに切り替え
            browser.switch_to.frame("JYOU")
    
            # ラジオボタンを取得しクリックしてダウンロード
            radio_buttons = browser.find_elements(By.XPATH, "//*[@type='radio']")
            for radio in radio_buttons:
                radio.click()
                time.sleep(1)  # 適度な間隔
                download_button = WebDriverWait(browser, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//*[@value='ダウンロード開始']"))
                )
                download_button.click()
                time.sleep(2)
    
            # フレームをリセット
            browser.switch_to.default_content()
            time.sleep(2)
    
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            traceback.print_exc()  # エラーの詳細を表示
        finally:
            browser.quit()

    def extract_lzh_files(self):
        """ダウンロードしたLZHファイルを解凍"""
        os.makedirs(self.kaitou_folder, exist_ok=True)
        lzh_file_list = os.listdir(self.download_folder)
        for lzh_file_name in lzh_file_list:
            file = lhafile.Lhafile(os.path.join(self.download_folder, lzh_file_name))

            # 解凍したファイルの名前を取得
            info = file.infolist()
            name = info[0].filename

            # 解凍したファイルを保存
            with open(os.path.join(self.kaitou_folder, name), "wb") as f:
                f.write(file.read(name))

    def download_and_extract_by_month(self, months=12, BorK="B"):
        """スクレイピングと解凍を行う一連の流れ"""
        self.scrape_data_by_month(months, BorK)
        self.extract_lzh_files()

    def download_and_extract_for_single_day(self, target_date, BorK="B"):
        """指定された1日分の日付でスクレイピングと解凍を行う一連の流れ"""
        self.scrape_data_for_single_day(target_date, BorK)
        self.extract_lzh_files()

    def parse_program_txt(self, file_path):
        """
        番組表のテキストファイルを読み込み、レースデータを解析してDataFrameを返す。

        :param file_path: 解析するテキストファイルのパス
        :return: 解析したDataFrame
        """
        # ファイルの読み込み
        with open(file_path, "r", encoding="shift_jis") as f:
            lines = f.readlines()

        race_data = []
        race_info = None
        race_place = None
        current_race = []

        # 行ごとに解析
        for line in lines:
            line = line.strip()
            if not line:  # 空行はスキップ
                continue

            # レース場情報の取得
            match_place = re.search(r"ボートレース(.{3})", line)
            if match_place:
                race_place = match_place.group(1)

            # レース情報の取得
            match_race = re.match(r"([０-９])Ｒ\s+(\S+)", line)
            if match_race:
                if current_race:
                    race_data.extend(current_race)
                    current_race = []

                race_number = match_race.group(1)
                race_type = match_race.group(2)
                race_info = (race_number, race_type, race_place)

            # 選手情報の取得
            match_player = re.match(r"^\d\s\d{4}", line)
            if match_player:
                parts = line.split(maxsplit=2)  # 先頭2つ（艇番, 登録番号）は固定なので分割

                boat_number = parts[0]  # 艇番
                raw_data = parts[1] + " " + parts[2]  # 登録番号以降のデータを結合して処理

                # 登録番号 (4桁固定)
                register_number = raw_data[:4].strip()
                # 選手名 (4文字固定)
                name = raw_data[4:8].strip()
                # 年齢 (2桁固定)
                age = raw_data[8:10].strip()
                # 支部 (2文字固定)
                region = raw_data[10:12].strip()
                # 体重 (2桁固定)
                weight = raw_data[12:14].strip()
                # 級別 (A1, A2, B1, B2)
                rank = raw_data[14:16].strip()

                # 残りのデータを適切に分割
                remaining_data = raw_data[16:].split()

                # None を空文字に変換
                remaining_data = [str(x) if x is not None else "" for x in remaining_data]

                win_rate_national = remaining_data[0]
                win_rate_national_2 = remaining_data[1]
                win_rate_local = remaining_data[2]
                win_rate_local_2 = remaining_data[3]
                motor_no = remaining_data[4]
                motor_rate = remaining_data[5]
                boat_no = remaining_data[6]
                boat_rate = remaining_data[7]

                current_race.append([
                    race_info[0], race_info[1], race_info[2],
                    boat_number, register_number, name, age, region, weight, rank,
                    win_rate_national, win_rate_national_2, win_rate_local, win_rate_local_2,
                    motor_no, motor_rate, boat_no, boat_rate
                ])

        # 最後のレースデータを追加
        if current_race:
            race_data.extend(current_race)

        # DataFrame に変換
        columns = [
            "レース番号", "レース種別", "レース場", 
            "艇番", "登録番号", "選手名", "年齢", "所属", "体重", "級別",
            "全国勝率", "全国2連対率", "当地勝率", "当地2連対率",
            "モーター番号", "モーター2連対率", "ボート番号", "ボート2連対率"
        ]

        df = pd.DataFrame(race_data, columns=columns)
        return df

    def program_save_csv(self,file_path,output_folder):
        df = self.parse_program_txt(file_path)
        # CSVとして保存
        csv_path = f"{output_folder}/boatrace_data.csv"
        df.to_csv(csv_path, index=False, encoding="shift_jis")
        print(f"CSVファイルが保存されました: {csv_path}")

    def parse_result(self, file_path):
        """txtからレース結果を解析し、DataFrame に変換する"""
        with open(file_path, "r", encoding="shift_jis") as f:
            lines = f.readlines()
        
        race_data = []
        race_info = None
        race_place = None
        race_player = None
        
        for line in lines:
            line = line.strip()
        
            # レース場の取得
            match_place = re.search(r"(.{1}　.{1})［", line)
            if match_place:
                race_place = match_place.group(1)
        
            # レース情報の取得
            match_race = re.match(r"([0-9]+R)\s+(\S+).*?H\d+m\s+(\S+)\s+風\s+(\S+)\s+(\d+m)\s+波\s+(\d+cm)", line)
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
                    "モーター": line[19:22].strip(),
                    "ボート": line[22:27].strip(),
                    "展示タイム": line[27:33].strip(),
                    "進入": line[33:37].strip(),
                    "スタートタイミング": line[37:45].strip(),
                    "レースタイム": line[45:56].strip(),
                }
        
                # レース情報と結合して1行分のレースデータを作成
                full_data = {**race_info, **race_player}
                race_data.append(full_data)
        
        # DataFrame 変換
        columns = ["レース場", "レース番号", "レース種別", "天候", "風向", "風速(m)", "波高(cm)", "決まり手",
                   "着順", "艇番", "登録番号", "選手名", "モーター", "ボート", "展示タイム", 
                   "進入", "スタートタイミング", "レースタイム"]
        
        df = pd.DataFrame(race_data, columns=columns)
        return df

    def parse_pre_race_info(self, soup):
        """直前情報を解析して、DataFrame に変換する"""
        pre_race_data = []
        return pd.DataFrame(pre_race_data)

    def export_csv(self, data, path):
        """指定したデータをCSVとして保存"""
        data.to_csv(path, index=False, encoding="shift_jis")
        print(f"CSVファイルが保存されました: {path}")