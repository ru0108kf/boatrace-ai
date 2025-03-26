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
```text
boatrace-analyzer/
├── data/                  # 生データ（TXTなど）
│   ├── K_lzh  # 競走成績_lxh
│   ├── B_lzh  # 番組表_lzh
│   ├── K_txt  # 競走成績_txt
│   ├── B_txt  # 番組表_txt
├── outputs/               # 出力CSV
│   ├── K_txt  # 競走成績_csv
│   ├── B_csv  # 番組表_csv
├── scripts/
│   ├── boatrace_manager.py  # 処理をまとめたクラス
│   ├── parser_program.py    # 番組表パーサー
│   ├── parser_result.py     # 結果パーサー
│   └── scraper.py           # スクレイピング処理
├── main.py                # 実行用スクリプト
├── requirements.txt       # 必要ライブラリ
└── README.md
```
## 開発環境
- Python 3.9+
- pandas
- requests
- beautifulsoup4

## Author
- 管理者：@ru0108kf
- リーダー：usami
  
