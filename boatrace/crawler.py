import time
import pandas as pd
import re
from datetime import datetime, timedelta
from http.client import RemoteDisconnected
from bs4 import BeautifulSoup
import urllib.request
import os
from .base.base import BoatraceBase


class BoatRaceOddsScraper:
    def __init__(self):
        self.jcd_dict = self._make_jcd_dict()

    def _make_jcd_dict(self):
        return {
            "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05", "浜名湖": "06", "蒲郡": "07", "常滑": "08",
            "津": "09", "三国": "10", "びわこ": "11", "住之江": "12", "尼崎": "13", "鳴門": "14", "丸亀": "15", "児島": "16",
            "宮島": "17", "徳山": "18", "下関": "19", "若松": "20", "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
        }

    def _daterange(self, start_date, end_date):
        for n in range((end_date - start_date).days):
            yield start_date + timedelta(n)

    def make_hd_list(self, date_from, date_to):
        start_date = datetime.strptime(date_from, '%Y%m%d').date()
        end_date = datetime.strptime(date_to, '%Y%m%d').date()
        return [i.strftime('%Y/%m/%d') for i in self._daterange(start_date, end_date)]

    def _make_url(self, what, rno, jcd, hd):
        rno_num = rno[:-1]
        hd_formatted = hd[0:4] + hd[5:7] + hd[8:10]
        
        jcd_code = self.jcd_dict.get(jcd)
        if not jcd_code:
            raise ValueError(f"不明な会場名: {jcd}")

        return f"http://boatrace.jp/owpc/pc/race/{what}?rno={rno_num}&jcd={jcd_code}&hd={hd_formatted}"

    def _html_parser(self, site_url):
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0",
        }
        try:
            request = urllib.request.Request(url=site_url, headers=headers)
            response = urllib.request.urlopen(request)
            html = response.read().decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser') 
            return soup
        except RemoteDisconnected:
            print(f"remote disconnected error for {site_url}!")
            return None
        except ConnectionResetError:
            print(f"Connection Reset error for {site_url}!")
            return None
        except Exception as e:
            print(f"HTMLパース中に予期せぬエラーが発生しました: {e} for {site_url}")
            return None

    def _scrape_odds_trifecta(self, soup, rno, jcd, hd):
        odds_dict = {
            "日付": "-".join([hd[0:4], hd[5:7], hd[8:10]]),
            "レース場": jcd,
            "レース番号": rno[:-1],
            "券種": "3連単"
        }

        set_number_list = [
            " 1-2-3", " 2-1-3", " 3-1-2", " 4-1-2", " 5-1-2", " 6-1-2",
            " 1-2-4", " 2-1-4", " 3-1-4", " 4-1-3", " 5-1-3", " 6-1-3",
            " 1-2-5", " 2-1-5", " 3-1-5", " 4-1-5", " 5-1-4", " 6-1-4",
            " 1-2-6", " 2-1-6", " 3-1-6", " 4-1-6", " 5-1-6", " 6-1-5",
            " 1-3-2", " 2-3-1", " 3-2-1", " 4-2-1", " 5-2-1", " 6-2-1",
            " 1-3-4", " 2-3-4", " 3-2-4", " 4-2-3", " 5-2-3", " 6-2-3",
            " 1-3-5", " 2-3-5", " 3-2-5", " 4-2-5", " 5-2-4", " 6-2-4",
            " 1-3-6", " 2-3-6", " 3-2-6", " 4-2-6", " 5-2-6", " 6-2-5",
            " 1-4-2", " 2-4-1", " 3-4-1", " 4-3-1", " 5-3-1", " 6-3-1",
            " 1-4-3", " 2-4-3", " 3-4-2", " 4-3-2", " 5-3-2", " 6-3-2",
            " 1-4-5", " 2-4-5", " 3-4-5", " 4-3-5", " 5-3-4", " 6-3-4",
            " 1-4-6", " 2-4-6", " 3-4-6", " 4-3-6", " 5-3-6", " 6-3-5",
            " 1-5-2", " 2-5-1", " 3-5-1", " 4-5-1", " 5-4-1", " 6-4-1",
            " 1-5-3", " 2-5-3", " 3-5-2", " 4-5-2", " 5-4-2", " 6-4-2",
            " 1-5-4", " 2-5-4", " 3-5-4", " 4-5-3", " 5-4-3", " 6-4-3",
            " 1-5-6", " 2-5-6", " 3-5-6", " 4-5-6", " 5-4-6", " 6-4-5",
            " 1-6-2", " 2-6-1", " 3-6-1", " 4-6-1", " 5-6-1", " 6-5-1",
            " 1-6-3", " 2-6-3", " 3-6-2", " 4-6-2", " 5-6-2", " 6-5-2",
            " 1-6-4", " 2-6-4", " 3-6-4", " 4-6-3", " 5-6-3", " 6-5-3",
            " 1-6-5", " 2-6-5", " 3-6-5", " 4-6-5", " 5-6-4", " 6-5-4",
        ]
        
        all_table1_divs = soup.find_all(class_="table1")

        if len(all_table1_divs) < 2: 
            return None
            
        odds_tbody = all_table1_divs[-1].find('table').find(class_="is-p3-0")
        
        if not odds_tbody:
            return None

        odds_list_soup = odds_tbody.find_all(class_="oddsPoint")
        
        if not odds_list_soup:
            return None

        for i, odds_tag in enumerate(odds_list_soup):
            if i < len(set_number_list): 
                odds_dict[set_number_list[i]] = odds_tag.text
            else:
                break 

        return pd.json_normalize([odds_dict])

    def _scrape_odds_exacta(self, soup, rno, jcd, hd):
        odds_dict = {
            "日付": "-".join([hd[0:4], hd[5:7], hd[8:10]]),
            "レース場": jcd,
            "レース番号": rno[:-1],
            "券種": "2連単"
        }

        set_number_list = [
            " 1-2", " 2-1", " 3-1", " 4-1", " 5-1", " 6-1",
            " 1-3", " 2-3", " 3-2", " 4-2", " 5-2", " 6-2",
            " 1-4", " 2-4", " 3-4", " 4-3", " 5-3", " 6-3",
            " 1-5", " 2-5", " 3-5", " 4-5", " 5-4", " 6-4",
            " 1-6", " 2-6", " 3-6", " 4-6", " 5-6", " 6-5",
        ]
        
        all_table1_divs = soup.find_all(class_="table1")

        if len(all_table1_divs) < 2: 
            return None
        
        odds_tbody = all_table1_divs[-1].find('table').find(class_="is-p3-0")

        if not odds_tbody:
            return None

        odds_list_soup = odds_tbody.find_all(class_="oddsPoint")

        if not odds_list_soup:
            return None
            
        for i, odds_tag in enumerate(odds_list_soup):
            if i < len(set_number_list): 
                odds_dict[set_number_list[i]] = odds_tag.text
            else:
                break

        return pd.json_normalize([odds_dict])

    def get_odds(self, rno, jcd, hd, odds_type="3t"):
        if odds_type == "3t":
            crawl_key = "odds3t"
            extractor = self._scrape_odds_trifecta
        elif odds_type == "2tf":
            crawl_key = "odds2tf"
            extractor = self._scrape_odds_exacta
        else:
            raise ValueError("odds_typeは '3t' (3連単) または '2tf' (2連単) のいずれかを指定してください。")

        race_odds_url = self._make_url(crawl_key, rno, jcd, hd)
        print(f"URL: {race_odds_url}") 

        soup = self._html_parser(race_odds_url)
        if soup is None:
            print("HTMLの取得またはパースに失敗しました。")
            return None

        try:
            odds_df = extractor(soup, rno, jcd, hd)
            if odds_df is not None:
                odds_df = odds_df.set_index(["日付", "レース場", "レース番号"])
            return odds_df
        except IndexError:
            print("指定されたレースのオッズデータが見つかりませんでした。HTML構造を確認してください。")
            return None
        except AttributeError as e:
            print(f"HTML要素の取得に失敗しました。サイトのHTML構造が変更された可能性があります。エラー: {e}")
            return None
        except Exception as e:
            print(f"オッズ情報の抽出中に予期せぬエラーが発生しました: {e}")
            return None

# --- CSVファイル処理のメインスクリプト ---
def main_process_csv_and_get_odds(folder, file_name):
    scraper = BoatRaceOddsScraper()
    input_csv_path = os.path.join(folder, "B_csv", f"{file_name}.csv") # パス結合をより堅牢に
    output_csv_path = os.path.join(folder, "Odds_csv", f"Odds{file_name[1:]}.csv") # パス結合をより堅牢に
    
    # 出力フォルダが存在しない場合は作成
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    try:
        # encoding="shift-jis" を指定
        df_input = pd.read_csv(input_csv_path, encoding="shift-jis")
        print(f"'{input_csv_path}' を読み込みました。")
        print(f"データ数: {len(df_input)}件")
    except FileNotFoundError:
        print(f"エラー: ファイル '{input_csv_path}' が見つかりません。ファイル名とパスを確認してください。")
        return
    except UnicodeDecodeError:
        print(f"エラー: '{input_csv_path}' のエンコーディングが Shift-JIS ではありません。")
        print("別のエンコーディング (例: 'utf-8') を試すか、ファイルを確認してください。")
        return

    all_odds_dfs = []

    # 重複するレース情報をスキップするためのセット
    processed_races = set()

    # CSVの各行を処理
    for index, row in df_input.iterrows():
        try:
            # CSVからのデータ抽出
            # カラム名は提供されたCSVのヘッダーに合わせる
            the_jcd_raw = str(row['レース場']) 
            # レース場名から全角スペースを削除
            the_jcd = the_jcd_raw.replace('　', '') 
            
            the_rno_str = str(row['レース番号']) 

            # 日付フォーマットの変換 (ファイル名 B220401 -> 2022/04/01)
            date_part = file_name[1:] # "220401"
            date_obj = datetime.strptime(date_part, '%y%m%d')
            the_hd = date_obj.strftime('%Y/%m/%d')
            
            # レース番号の変換 (1 -> 1R)
            the_rno = f"{int(the_rno_str)}R"

            # 重複チェック: 既に処理したレースはスキップ
            race_identifier = (the_hd, the_jcd, the_rno)
            if race_identifier in processed_races:
                # print(f"  -> {the_hd} {the_jcd} {the_rno} は既に処理済みです。スキップします。") # デバッグ時はコメントアウト
                continue
            
            processed_races.add(race_identifier) # 処理済みとして追加

            print(f"\n--- 処理中: {the_hd} {the_jcd} {the_rno} の3連単オッズ ---")
            odds_df = scraper.get_odds(the_rno, the_jcd, the_hd, odds_type="3t")
            
            if odds_df is not None:
                all_odds_dfs.append(odds_df)
            else:
                print(f"  -> {the_hd} {the_jcd} {the_rno} のオッズ取得に失敗したためスキップします。")
            
            time.sleep(1) # サーバーへの負荷軽減のため、1秒待機

        except KeyError as e:
            print(f"エラー: CSVファイルの必要なカラムが見つかりません。'{e}'。CSVのヘッダーを確認してください。")
            print("期待されるカラム: 'レース場', 'レース番号'")
            return
        except Exception as e:
            print(f"処理中に予期せぬエラーが発生しました (行 {index}, データ: {row.to_dict()}): {e}")
            continue

    if all_odds_dfs:
        final_odds_df = pd.concat(all_odds_dfs, ignore_index=False)
        
        # CSVファイルを保存 (encoding='shift-jis' で Excel 互換性を高める)
        final_odds_df.to_csv(output_csv_path, encoding='shift-jis') 
        print(f"\n全てのオッズデータを '{output_csv_path}' に保存しました。")
    else:
        print("\n取得できたオッズデータがありませんでした。")

# --- メイン実行部分 ---
if __name__ == "__main__":
    print(f"現在の作業ディレクトリ: {os.getcwd()}")
    # pandasの表示設定
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # 処理する日付範囲
    start_date_test = "2025-06-01" 
    end_date_test = "2025-06-18" 
    print(f"スクレイピング開始: {start_date_test} ～ {end_date_test}")
    
    base_handler = BoatraceBase()
    file_lists_to_process = []
    for date_str in base_handler.generate_date_list(start_date_test, end_date_test):
        file_lists_to_process.append(f"B{date_str}")
    for file_name_b_prefix in file_lists_to_process:
        main_process_csv_and_get_odds(base_handler.data_folder, file_name_b_prefix)