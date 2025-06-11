import time
import pandas as pd
import re
from datetime import datetime, timedelta
from http.client import RemoteDisconnected
from bs4 import BeautifulSoup
import urllib.request

class BoatRaceOddsScraper:
    def __init__(self):
        """
        BoatRaceOddsScraperクラスのコンストラクタ。
        会場コードの辞書を初期化します。
        """
        self.jcd_dict = self._make_jcd_dict()

    def _make_jcd_dict(self):
        """
        会場コードの辞書を作成します。
        """
        return {
            "桐　生": "01", "戸　田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05", "浜名湖": "06", "蒲　郡": "07", "常　滑": "08",
            "　津　": "09", "三　国": "10", "びわこ": "11", "住之江": "12", "尼　崎": "13", "鳴　門": "14", "丸　亀": "15", "児　島": "16",
            "宮　島": "17", "徳　山": "18", "下　関": "19", "若　松": "20", "芦　屋": "21", "福　岡": "22", "唐　津": "23", "大　村": "24"
        }

    def _daterange(self, start_date, end_date):
        """
        指定された期間の日付を生成するジェネレータ。
        """
        for n in range((end_date - start_date).days):
            yield start_date + timedelta(n)

    def make_hd_list(self, date_from, date_to):
        """
        期間の始まりと終わりをインプットすることで、
        クロール対象日付のリストを作成します。

        :param date_from: クロールの開始日。"yyyymmdd" の形で入力
        :param date_to: この日の前日のレースまでクロールを行う。"yyyymmdd" の形で入力
        :return: クロール対象日付のリスト。例えば、['2019/04/09', '2019/04/10']のような形。
        """
        start_date = datetime.strptime(date_from, '%Y%m%d').date()
        end_date = datetime.strptime(date_to, '%Y%m%d').date()
        return [i.strftime('%Y/%m/%d') for i in self._daterange(start_date, end_date)]

    def _make_url(self, what, rno, jcd, hd):
        """
        指定された情報（オッズの種類）、レース番号、会場、開催日を基に、
        ボートレース公式サイトのURLを構築します。

        :param what: 何をクロールするか。"odds3t"（3連単オッズ）または "odds2tf"（2連単オッズ）
        :param rno: レース番号。例: "8R"
        :param jcd: 会場名。例: "住之江"
        :param hd: 開催日。例: "2019/03/28" (yyyy/mm/dd形式の文字列)
        :return: 公式サイトのページのURL
        """
        rno_num = rno[:-1] # "11R" -> "11"
        hd_formatted = hd[0:4] + hd[5:7] + hd[8:10] # "2019/10/06" -> "20191006"
        
        jcd_code = self.jcd_dict.get(jcd)
        if not jcd_code:
            raise ValueError(f"不明な会場名: {jcd}")

        return f"http://boatrace.jp/owpc/pc/race/{what}?rno={rno_num}&jcd={jcd_code}&hd={hd_formatted}"

    def _html_parser(self, site_url):
        """
        指定されたURLからHTMLコンテンツを取得し、BeautifulSoupオブジェクトとしてパースします。
        ネットワークエラーをハンドリングします。取得したHTMLも出力します。

        :param site_url: パースするウェブサイトのURL
        :return: BeautifulSoupオブジェクト、またはエラー発生時にNone
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0",
        }
        try:
            request = urllib.request.Request(url=site_url, headers=headers)
            response = urllib.request.urlopen(request)
            html = response.read().decode('utf-8')
            
            # --- デバッグ出力 ---
            print("\n--- 取得したHTMLコンテンツのプレビュー (先頭1000文字) ---")
            print(html[:1000])
            print("\n--- 取得したHTMLコンテンツのプレビュー (末尾1000文字) ---")
            print(html[-1000:])
            # --- デバッグ出力ここまで ---

            # パーサーを 'html.parser' に戻す
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
        """
        3連単のオッズをクロールします。
        :param soup: BeautifulSoupオブジェクト
        :return: pandas DataFrame形式のオッズデータ
        """
        odds_dict = {
            "date": "-".join([hd[0:4], hd[5:7], hd[8:10]]),
            "venue": jcd,
            "raceNumber": rno[:-1],
            "placeBed": "trifecta"
        }

        # 3連単の組み合わせリスト (6艇の場合、6*5*4 = 120通り)
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
        print(f"\n--- debug: soup.find_all(class_='table1') の要素数: {len(all_table1_divs)} ---")

        if len(all_table1_divs) < 2: 
            print("オッズのテーブル構造が見つかりませんでした。(table1の数が不足)")
            return None
            
        odds_tbody = all_table1_divs[-1].find('table').find(class_="is-p3-0")
        
        if not odds_tbody:
            print("3連単オッズのtbody (.is-p3-0) が見つかりませんでした。")
            return None

        odds_list_soup = odds_tbody.find_all(class_="oddsPoint")
        
        if not odds_list_soup:
            print("3連単オッズの値 (.oddsPoint) が見つかりませんでした。")
            return None

        for i, odds_tag in enumerate(odds_list_soup):
            if i < len(set_number_list): 
                odds_dict[set_number_list[i]] = odds_tag.text
            else:
                print(f"警告: 組み合わせリストの数 ({len(set_number_list)}) を超えるオッズタグが見つかりました ({i+1}番目)。")
                break 

        # --- 修正箇所 ---
        return pd.json_normalize([odds_dict])

    def _scrape_odds_exacta(self, soup, rno, jcd, hd):
        """
        2連単のオッズをクロールします。
        :param soup: BeautifulSoupオブジェクト
        :return: pandas DataFrame形式のオッズデータ
        """
        odds_dict = {
            "date": "-".join([hd[0:4], hd[5:7], hd[8:10]]),
            "venue": jcd,
            "raceNumber": rno[:-1],
            "placeBed": "exacta"
        }

        # 2連単の組み合わせリスト (6*5 = 30通り)
        set_number_list = [
            " 1-2", " 2-1", " 3-1", " 4-1", " 5-1", " 6-1",
            " 1-3", " 2-3", " 3-2", " 4-2", " 5-2", " 6-2",
            " 1-4", " 2-4", " 3-4", " 4-3", " 5-3", " 6-3",
            " 1-5", " 2-5", " 3-5", " 4-5", " 5-4", " 6-4",
            " 1-6", " 2-6", " 3-6", " 4-6", " 5-6", " 6-5",
        ]
        
        all_table1_divs = soup.find_all(class_="table1")
        print(f"\n--- debug: soup.find_all(class_='table1') の要素数 (2連単): {len(all_table1_divs)} ---")

        if len(all_table1_divs) < 2: 
            print("オッズのテーブル構造が見つかりませんでした。(table1の数が不足)")
            return None
        
        odds_tbody = all_table1_divs[-1].find('table').find(class_="is-p3-0")

        if not odds_tbody:
            print("2連単オッズのtbody (.is-p3-0) が見つかりませんでした。")
            return None

        odds_list_soup = odds_tbody.find_all(class_="oddsPoint")

        if not odds_list_soup:
            print("2連単オッズの値 (.oddsPoint) が見つかりませんでした。")
            return None
            
        for i, odds_tag in enumerate(odds_list_soup):
            if i < len(set_number_list): 
                odds_dict[set_number_list[i]] = odds_tag.text
            else:
                print(f"警告: 組み合わせリストの数 ({len(set_number_list)}) を超えるオッズタグが見つかりました ({i+1}番目)。")
                break

        # --- 修正箇所 ---
        return pd.json_normalize([odds_dict])

    def get_odds(self, rno, jcd, hd, odds_type="3t"):
        """
        指定されたレースのオッズ情報を取得するメインメソッド。
        :param rno: レース番号 (例: "11R")
        :param jcd: 開催場 (例: "住之江")
        :param hd: 日付 (例: "2019/10/06")
        :param odds_type: 取得するオッズの種類 ("3t" for 3連単, "2tf" for 2連単)
        :return: pandas DataFrame形式のオッズデータ、またはNone
        """
        if odds_type == "3t":
            crawl_key = "odds3t"
            extractor = self._scrape_odds_trifecta
        elif odds_type == "2tf":
            crawl_key = "odds2tf"
            extractor = self._scrape_odds_exacta
        else:
            raise ValueError("odds_typeは '3t' (3連単) または '2tf' (2連単) のいずれかを指定してください。")

        # クロール対象サイトのURL作成
        race_odds_url = self._make_url(crawl_key, rno, jcd, hd)
        print(f"URL: {race_odds_url}")

        # HTMLをパース
        soup = self._html_parser(race_odds_url)
        if soup is None:
            print("HTMLの取得またはパースに失敗しました。")
            return None

        try:
            # 対象サイトからオッズ情報をクロール
            odds_df = extractor(soup, rno, jcd, hd)
            if odds_df is not None:
                odds_df = odds_df.set_index(["date", "venue", "raceNumber"])
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

# --- 使用例 ---
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    scraper = BoatRaceOddsScraper()

    current_datetime = datetime.now()
    # 開催日を昨日以前の日付にする (本日開催されていない場合に備え)
    target_date = current_datetime - timedelta(days=1)
    the_hd = target_date.strftime('%Y/%m/%d')
    print(f"対象日付: {the_hd}\n") 

    the_rno = "11R"
    the_jcd = "三　国" 

    print(f"--- {the_hd} {the_jcd} {the_rno} の3連単オッズ ---")
    trifecta_odds_df = scraper.get_odds(the_rno, the_jcd, the_hd, odds_type="3t")
    if trifecta_odds_df is not None:
        print(trifecta_odds_df)
    else:
        print("3連単オッズを取得できませんでした。")

    print(f"\n--- {the_hd} {the_jcd} {the_rno} の2連単オッズ ---")
    exacta_odds_df = scraper.get_odds(the_rno, the_jcd, the_hd, odds_type="2tf")
    if exacta_odds_df is not None:
        print(exacta_odds_df)
    else:
        print("2連単オッズを取得できませんでした。")

    print("\n--- 特定期間の日付リスト生成の例 ---")
    date_from_str = current_datetime.strftime('%Y%m%d')
    date_to_str = (current_datetime + timedelta(days=3)).strftime('%Y%m%d')
    date_list = scraper.make_hd_list(date_from_str, date_to_str)
    print(f"{date_from_str} から {date_to_str} までの日付: {date_list}")