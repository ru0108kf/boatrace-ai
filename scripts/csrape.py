from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

class BoatraceLatestDataScraper():
    def scrape_the_latest_info(self,url):
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
    


