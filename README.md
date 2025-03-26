# boarrace-ai
競艇アルゴリズムの構築を目指すプロジェクトです。

## 機能概要
- 番組表/結果のスクレイピング機能&データ分析
- 直前情報のスクレイピング機能（未実装）
- アルゴリズム（未実装）
- 機械学習（未実装）
- Web/スマホアプリ化（未実装）

---

## ディレクトリ構成
boatrace-analyzer/ ├─ data/ # テキストデータ（番組表・結果）保存用 ├─ outputs/ # 解析結果CSVなど ├─ scripts/
│ ├─ boatrace_manager.py # データ処理クラス │ └─ scraper.py # スクレイピング処理 ├─ main.py # 実行スクリプト（開発用） ├─ requirements.txt # 必要なライブラリ一覧 └─ README.md
<pre> ## 🗂 ディレクトリ構成 ``` boatrace-analyzer/ ├── data/ ├── outputs/ └── main.py ``` </pre>
---

## 使用方法

1. テキストファイルからCSVを生成：

~~~bash
python main.py --mode parse --file data/B250217.TXT
~~~

## 開発環境
- Python 3.9+
- pandas
- requests
- beautifulsoup4

## Author
- 管理者：@ru0108kf
- リーダー：usami
  
