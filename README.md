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
├── outputs/               # 出力CSV
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
  
