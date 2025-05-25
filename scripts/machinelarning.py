import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import japanize_matplotlib
from itertools import permutations
import joblib
import pickle

from analyzer import BoatraceAnalyzer

class DataCompiler(BoatraceAnalyzer):
    """データ収集と前処理を担当するクラス"""
    def __init__(self, folder):
        super().__init__(folder)
        self.folder = folder
        self.scaler = None
        self.categorical_cols = ['レース場','天候', '風向',
                                '1号艇_登録番号', '1号艇_支部', '1号艇_級別', '1号艇_モーター番号', '1号艇_ボート番号',
                                '2号艇_登録番号', '2号艇_支部', '2号艇_級別', '2号艇_モーター番号', '2号艇_ボート番号',
                                '3号艇_登録番号', '3号艇_支部', '3号艇_級別', '3号艇_モーター番号', '3号艇_ボート番号',
                                '4号艇_登録番号', '4号艇_支部', '4号艇_級別', '4号艇_モーター番号', '4号艇_ボート番号',
                                '5号艇_登録番号', '5号艇_支部', '5号艇_級別', '5号艇_モーター番号', '5号艇_ボート番号',
                                '6号艇_登録番号', '6号艇_支部', '6号艇_級別', '6号艇_モーター番号', '6号艇_ボート番号']
        self.numeric_cols = None
        self.categories = None
        self.feature_columns = None
        self.categorical_indices = None

    def compile_race_data(self):
        """レースデータをコンパイル"""
        merged_csv_folder = os.path.join(self.folder, "merged_csv")
        all_files = [os.path.join(merged_csv_folder, f) for f in os.listdir(merged_csv_folder) if f.endswith('.csv')]

        all_dataframes = []
        for filepath in all_files:
            # CSV読み込み
            df = pd.read_csv(filepath,encoding="shift-jis")

            # 出力用リスト
            records = []

            # レースごとに処理
            for (date, place, race_no), group in df.groupby(['日付','レース場', 'レース番号']):
                if len(group) != 6:
                    continue  # 6艇そろってないレースは除外

                # 1-3着をとった艇番
                winner_row = group[group['着順'] == 1].iloc[0] if not group[group['着順'] == 1].empty else None
                second_row = group[group['着順'] == 2].iloc[0] if not group[group['着順'] == 2].empty else None
                third_row = group[group['着順'] == 3].iloc[0] if not group[group['着順'] == 3].empty else None
                
                if winner_row is None or second_row is None or third_row is None:
                    print(f"欠損データをスキップ: {group[['日付', 'レース場', 'レース番号']].iloc[0]}")
                    continue

                # 舟券学習用データ構造
                record = {
                    '日付': date,
                    'レース場': place,
                    'レース番号': race_no,
                    '1着艇': int(winner_row['艇番']),
                    '2着艇': int(second_row['艇番']),
                    '3着艇': int(third_row['艇番']),
                    '天候' : winner_row['天候'],
                    '風向' : winner_row['風向'],
                    '風速(m)' : int(winner_row['風速(m)']),
                    '波高(cm)' : int(winner_row['波高(cm)'])
                }

                # 各艇のデータを展開
                for i in range(1, 7):
                    boat = group[group['艇番'] == i]
                    if boat.empty:
                        continue

                    b = boat.iloc[0]
                    prefix = f"{i}号艇"
                    record.update({
                        f'{prefix}_登録番号': b['登録番号'],
                        f'{prefix}_年齢': b['年齢'],
                        f'{prefix}_支部': b['支部'],
                        f'{prefix}_体重': b['体重'],
                        f'{prefix}_級別': b['級別'],
                        f'{prefix}_全国勝率': b['全国勝率'],
                        f'{prefix}_全国2連対率': b['全国2連対率'],
                        f'{prefix}_当地勝率': b['当地勝率'],
                        f'{prefix}_当地2連対率': b['当地2連対率'],
                        f'{prefix}_モーター番号': b['モーター番号'],
                        f'{prefix}_モーター2連対率': b['モーター2連対率'],
                        f'{prefix}_ボート番号': b['ボート番号'],
                        f'{prefix}_ボート2連対率': b['ボート2連対率'],
                        f'{prefix}_展示タイム': b['展示タイム'],
                        f'{prefix}_平均ST': b['平均ST'],
                        f'{prefix}_全体平均ST': b['全体平均ST'],
                        f'{prefix}_1着率': b['1着率'],
                        f'{prefix}_2着率': b['2着率'],
                        f'{prefix}_3着率': b['3着率'],
                    })
                    
                    if i == 1:
                        record.update({
                            f'{prefix}_逃げ率': b['逃げ率'],
                            f'{prefix}_差され率': b['差され率'],
                            f'{prefix}_まくられ率': b['まくられ率'],
                            f'{prefix}_まくり差され率': b['まくり差され率'],
                        })
                    elif i == 2:
                        record.update({
                            f'{prefix}_逃し率': b['逃し率'],
                            f'{prefix}_差し率': b['差し率'],
                            f'{prefix}_まくり率': b['まくり率'],
                        })
                    else:
                        record.update({
                            f'{prefix}_差し率': b['差し率'],
                            f'{prefix}_まくり率': b['まくり率'],
                            f'{prefix}_まくり差し率': b['まくり差し率'],
                        })
                        
                records.append(record)
            
            all_dataframes.append(pd.DataFrame(records))

        return pd.concat(all_dataframes, ignore_index=True)
    
    def compile_odds_data(self):
        """オッズデータをコンパイル"""
        o_csv_folder = os.path.join(self.folder, "o_csv")
        all_files = [os.path.join(o_csv_folder, f) for f in os.listdir(o_csv_folder) if f.endswith('.csv')]

        all_dataframes = []
        for filepath in all_files:
            try:
                df = pd.read_csv(filepath, encoding='shift-jis')
            except Exception as e:
                print(f"読み込みエラー: {filepath} → {e}")
                continue

            all_dataframes.append(df)

        # すべてのDataFrameをまとめて返す
        if all_dataframes:
            return pd.concat(all_dataframes, ignore_index=True)
        else:
            return pd.DataFrame()
 
    def preprocess_features(self, df):
        """特徴量の前処理（全モデルで共通）"""
        # 基本となる特徴量のみ保持
        df = df.drop(columns=['日付', 'レース番号', '1着艇', '2着艇', '3着艇'], errors='ignore')
        
        # 数値列の特定（実行時1回だけ）
        if self.numeric_cols is None:
            self.numeric_cols = [col for col in df.columns if col not in self.categorical_cols]
            self.feature_columns = self.categorical_cols + self.numeric_cols
        
        # カテゴリカル変数処理
        for col in self.categorical_cols:
            if col in df.columns:
                # 未知のカテゴリを'missing'に変換
                df[col] = df[col].fillna('missing')
                df[col] = pd.Categorical(
                    df[col], 
                    categories=list(df[col].unique()))  # 自動的にカテゴリを設定
            else:
                print(f"Warning: Column {col} not found in dataframe")
        
        # 数値変数の標準化
        if self.scaler is None:
            self.scaler = StandardScaler()
            df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols].fillna(0).astype('float32'))
        else:
            df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols].fillna(0).astype('float32'))
        
        # カテゴリカル特徴量のインデックス取得
        self.categorical_indices = [
            i for i, col in enumerate(self.feature_columns)
            if col in self.categorical_cols
        ]
        
        # 特徴量の順序を統一
        return df[self.feature_columns]
    
    def get_binary_target(self, df, boat_num=1, top_num=1):
        """バイナリ分類用ターゲット生成"""
        if top_num == 1:
            return (df['1着艇'] == boat_num).astype(int).values
        elif top_num == 2:
            return (df['2着艇'] == boat_num).astype(int).values
        elif top_num == 3:
            return (df['3着艇'] == boat_num).astype(int).values
        else:
            raise ValueError("top_num must be 1, 2, or 3")
    
    def get_multiclass_target(self, df, top_num=1):
        """多クラス分類用ターゲット生成"""
        if top_num == 1:
            return df['1着艇'].values
        elif top_num == 2:
            return df['2着艇'].values
        elif top_num == 3:
            return df['3着艇'].values
        else:
            raise ValueError("top_num must be 1, 2, or 3")
    
    def save_preprocessor(self, filepath):
        """前処理情報を保存"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'categorical_cols': self.categorical_cols,
                'numeric_cols': self.numeric_cols,
                'categories': self.categories,
                'scaler_mean': self.scaler.mean_,
                'scaler_scale': self.scaler.scale_,
                'feature_columns': self.feature_columns
            }, f)
    
    @classmethod
    def load_preprocessor(cls, filepath,folder):
        """保存した前処理情報を読み込み"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(folder)
        preprocessor.categorical_cols = data['categorical_cols']
        preprocessor.numeric_cols = data['numeric_cols']
        preprocessor.categories = data['categories']
        preprocessor.feature_columns = data['feature_columns']
        
        preprocessor.scaler = StandardScaler()
        preprocessor.scaler.mean_ = data['scaler_mean']
        preprocessor.scaler.scale_ = data['scaler_scale']
        
        return preprocessor
    
    def compile_race_data_B(self, target_date="2024-04-01", place='大村', race_no=1,weather='晴',wind_dir='東',wind_spd=0,wave_hgt=1, Exhibition_time={1:7.00, 2:7.00, 3:7.00, 4:7.00, 5:7.00, 6:7.00}):
        """指定したレースのデータをコンパイルして1行にまとめる"""
        # CSV読み込み
        df = self._format_race_data_(target_date=target_date)
        
        # 指定したレースのデータのみ抽出
        race_data = df[(df['日付'] == target_date) & 
                    (df['レース場'] == place) & 
                    (df['レース番号'] == race_no)]
                
        # 6艇そろってない場合は空のDataFrameを返す
        if len(race_data) != 6:
            return pd.DataFrame()
        
        # 出力用レコード
        record = {
            '日付': target_date,
            'レース場': place,
            'レース番号': race_no,
            '天候' : weather,
            '風向' : wind_dir,
            '風速(m)' : wind_spd,
            '波高(cm)' : wave_hgt
        }
        
        # 各艇のデータを展開
        for i in range(1, 7):
            boat = race_data[race_data['艇番'] == i]
            if boat.empty:
                continue
                
            b = boat.iloc[0]
            prefix = f"{i}号艇"
            record.update({
                f'{prefix}_登録番号': b['登録番号'],
                f'{prefix}_年齢': b['年齢'],
                f'{prefix}_支部': b['支部'],
                f'{prefix}_体重': b['体重'],
                f'{prefix}_級別': b['級別'],
                f'{prefix}_全国勝率': b['全国勝率'],
                f'{prefix}_全国2連対率': b['全国2連対率'],
                f'{prefix}_当地勝率': b['当地勝率'],
                f'{prefix}_当地2連対率': b['当地2連対率'],
                f'{prefix}_モーター番号': b['モーター番号'],
                f'{prefix}_モーター2連対率': b['モーター2連対率'],
                f'{prefix}_ボート番号': b['ボート番号'],
                f'{prefix}_ボート2連対率': b['ボート2連対率'],
                f'{prefix}_展示タイム': Exhibition_time.get(i, 7.00),  # デフォルト7.00
                f'{prefix}_平均ST': b['平均ST'],
                f'{prefix}_全体平均ST': b['全体平均ST'],
                f'{prefix}_1着率': b['1着率'],
                f'{prefix}_2着率': b['2着率'],
                f'{prefix}_3着率': b['3着率'],
                f'{prefix}_逃し率': b['逃し率'],
            })
            
            if i == 1:
                record.update({
                    f'{prefix}_逃げ率': b['逃げ率'],
                    f'{prefix}_差され率': b['差され率'],
                    f'{prefix}_まくられ率': b['まくられ率'],
                    f'{prefix}_まくり差され率': b['まくり差され率'],
                })
            elif i == 2:
                record.update({
                    f'{prefix}_逃し率': b['逃し率'],
                    f'{prefix}_差し率': b['差し率'],
                    f'{prefix}_まくり率': b['まくり率'],
                })
            else:
                record.update({
                    f'{prefix}_差し率': b['差し率'],
                    f'{prefix}_まくり率': b['まくり率'],
                    f'{prefix}_まくり差し率': b['まくり差し率'],
                })
        
        # 1行のDataFrameに変換して返す
        return pd.DataFrame([record])
   
    def _format_race_data_(self,target_date="2024-04-01"):
        """レース番組表から結果を予測する用"""
        name = self.generate_date_list(start_date=target_date, end_date=target_date)[0]
        df_B = pd.read_csv(self.folder+f"\\B_csv\\B{name}.csv", encoding='shift-jis')
        
        date = "20" + name[:2] + "-" + name[2:4] + "-" + name[4:6]
        
        df_B['日付'] = date
        
        # 1年前の日付を取得
        three_months_ago = (datetime.strptime(name, "%y%m%d") - timedelta(days=366)).strftime("%Y-%m-%d")
        # 1日前の日付を取得
        one_day_ago = (datetime.strptime(name, "%y%m%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        # 基本データの取得
        base_df = self.get_base_data(start_date=three_months_ago, end_date=one_day_ago, venue="全国") 
        # データを結合する
        merged_df = pd.merge(df_B, base_df[['登録番号', '艇番', '平均ST','全体平均ST']], on=['登録番号', '艇番'], how='left')
        
        # 1着率を計算
        merged_df['1着率'] = base_df['勝利回数'] / base_df['出走数']
        # 2着率を計算
        merged_df['2着率'] = base_df['2着回数'] / base_df['出走数']
        # 3着率を計算
        merged_df['3着率'] = base_df['3着回数'] / base_df['出走数']
        # その他
        merged_df['逃げ率'] = base_df['逃げ'] / base_df['出走数']
        merged_df['逃し率'] = base_df['逃し'] / base_df['出走数']
        merged_df['差し率'] = base_df['差し'] / base_df['出走数']
        merged_df['まくり率'] = base_df['まくり'] / base_df['出走数']
        merged_df['まくり差し率'] = base_df['まくり差し'] / base_df['出走数']
        merged_df['差され率'] = base_df['差され'] / base_df['出走数']
        merged_df['まくられ率'] = base_df['まくられ'] / base_df['出走数']
        merged_df['まくり差され率'] = base_df['まくり差され'] / base_df['出走数']
        
        # 日付を一番左のカラムに移動
        cols = merged_df.columns.tolist()
        cols = ['日付'] + [col for col in cols if col != '日付']
        merged_df = merged_df[cols]
        
        return merged_df

class ModelTrainer:
    """モデル訓練を担当するクラス"""
    def __init__(self):
        self.models = {}

    def train_binary_lgbm(self, X, y, df, categorical_indices):
        """LightGBMを使用した二値分類モデルの訓練（内部メソッド）"""
        # ======== データ分割 ========
        # 元のインデックスを保持
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.2, random_state=42
        )
        
        # ======== LightGBMモデル構築 ========
        # パラメータ設定
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=7,
            num_leaves=63,
            min_data_in_leaf=20,
            bagging_fraction = 0.8,
            bagging_freq = 5,
            verbose=-1,
            random_seed=777
        )

        # モデル訓練
        model.fit(X_train, y_train,
            categorical_feature=categorical_indices,
            eval_set=[(X_test, y_test)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True),
                       lgb.log_evaluation(100)]
        )

        # ======== 評価 ========
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # クラス1の確率を取得
        y_pred = (y_pred_prob > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"\nモデル評価 (LightGBM):")
        print(f"- 精度: {accuracy:.4f}")
        print(f"- AUC: {auc:.4f}")
        print(f"- 適合率: {precision:.4f}")
        print(f"- 再現率: {recall:.4f}")

        # ======== 特徴量重要度の可視化 ========
        lgb.plot_importance(model, figsize=(20, 12),max_num_features=20)
        plt.show()
        
        # ======== 予測分布の可視化 ========
        print("\n=== 予測結果の詳細分析 ===")
        print("混同行列:")
        print(confusion_matrix(y_test, y_pred))

        return model

    def train_multiclass_lgbm(self, X, y, df):
        """LightGBMを使用した多クラス分類モデルの訓練（内部メソッド）"""
        # ======== データ分割 ========
        # 元のインデックスを保持
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y-1, df.index, test_size=0.2, random_state=42
        )

        # ======== モデル構築（LightGBM） ========
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=6,
            boosting_type='gbdt',#'gbdt
            learning_rate=0.05,#0.05
            n_estimators=1000,#100
            max_depth=8,
            num_leaves=31,#31
            min_data_in_leaf=8,
            feature_fraction=0.8,
            class_weight='balanced',
            verbose=-1
        )
                
        verbose_eval=50
        # モデル学習
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50,verbose=True),
                            lgb.log_evaluation(verbose_eval)])

        # ======== 特徴量重要度の可視化 ========
        lgb.plot_importance(model,figsize=(6,30))
        plt.show()
        
        # ======== 評価 ========
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"LightGBMモデルの多クラス分類精度: {acc:.4f}")
        print("\n=== 予測結果の詳細分析 ===")
        print("混同行列:")
        print(classification_report(y_test, y_pred, digits=3))
        print("クラスごとの精度:")
        print(confusion_matrix(y_test, y_pred))
        
        return model

    def train_multiclass_lgbm_exclusion_one(self, X, y, df,categorical_indices):
        """1号艇が1着でないデータのみを使って多クラス分類モデルを訓練"""
        # 1号艇を除外する場合
        mask = (y != 1)  # 2-6号艇のみ
        X_filtered = X[mask]
        y_filtered = y[mask]
        df_filtered = df.iloc[mask]
        
        # データ分割（元のコードと同様）
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_filtered, y_filtered, df_filtered.index, test_size=0.2, random_state=42
        )
        # ======== モデル構築（LightGBM） ========
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=5,
            boosting_type='gbdt',
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=-1,
            num_leaves=63,
            min_data_in_leaf=20,
            bagging_fraction = 0.75,
            bagging_freq = 5,
            verbose=-1,
            class_weight='balanced'
        )
        
        # 学習と予測
        model.fit(X_train, y_train-2, eval_set=[(X_test, y_test-2)],
                    categorical_feature=categorical_indices,
                    callbacks=[lgb.early_stopping(stopping_rounds=50,verbose=True),
                            lgb.log_evaluation(100)]) 
               
        # ======== 特徴量重要度の可視化 ========
        lgb.plot_importance(model,figsize=(20, 12) ,max_num_features=20)
        plt.show()               

        # ======== 評価 ========
        y_pred = model.predict(X_test) + 2
        acc = accuracy_score(y_test, y_pred)
        print(f"LightGBMモデルの多クラス分類精度: {acc:.4f}")
        
        print("\n=== 予測結果の詳細分析 ===")
        print("混同行列:")
        print(confusion_matrix(y_test, y_pred))
        print("\nクラスごとの精度:")
        print(classification_report(y_test, y_pred))
        
        return model

    def train_multiclass_lgbm_target_1st(self, X, y, df, target_1st_num=1):
        """
        1着が指定した艇のデータのみを使って多クラス分類モデルを訓練
        （yには2着の艇番号が入っている前提）
        """
        # 1着情報を取得（dfから1着の艇番号が入っている列を指定）
        first_place = df['1着艇'].values

        # 1着が指定艇番号のもののみマスク
        mask = (first_place == target_1st_num)
        X_filtered = X[mask]
        y_filtered = y[mask]  # このyは2着の艇番号
        df_filtered = df.iloc[mask]

        # 2着候補の艇番号リスト（1着艇は除く）
        candidate_boats = sorted([x for x in set(y_filtered) if x != target_1st_num])
        num_classes = len(candidate_boats)

        # 艇番号を 0,1,2,... にマッピング
        label_mapping = {boat: idx for idx, boat in enumerate(candidate_boats)}
        y_mapped = np.array([label_mapping[boat] for boat in y_filtered])

        # 以降は元の処理と同じ（データ分割・モデル訓練・評価）
        # ================================================
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_filtered, y_mapped, df_filtered.index, test_size=0.2, random_state=42
        )

        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=num_classes,
            boosting_type='gbdt',
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=-1,
            num_leaves=63,
            min_data_in_leaf=20,
            bagging_fraction = 0.75,
            bagging_freq = 5,
            verbose=-1,
            class_weight='balanced'
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(-1)
            ]
        )

        # ======== 特徴量重要度の可視化 ========
        #lgb.plot_importance(model,figsize=(20, 12),max_num_features=20)
        #plt.show()
        
        # 評価（元の艇番号に戻して表示）
        y_pred_mapped = model.predict(X_test)
        inverse_mapping = {idx: boat for idx, boat in enumerate(candidate_boats)}
        y_pred = np.array([inverse_mapping[idx] for idx in y_pred_mapped])
        y_test_original = np.array([inverse_mapping[idx] for idx in y_test])

        print(f"1着が {target_1st_num} 号艇のレースのみ使用")
        print(f"2着候補: {candidate_boats}")
        print(f"データ数: {len(X_filtered)}")
        print(f"分類精度: {accuracy_score(y_test_original, y_pred):.4f}")
        print("混同行列:")
        print(confusion_matrix(y_test_original, y_pred))
        print("\nクラスごとの精度:")
        print(classification_report(y_test_original, y_pred))
        return model

class BettingStrategyEvaluator:
    """ベッティング戦略の評価を担当するクラス"""
    def Win_calculate_return_rate(self, df, df_odds, threshold={1: 1.0,2: 0.8,3: 0.8,4: 0.3,5: 0.2,6: 0.1}, bet_amount=100):
        """単勝回収率算出"""
        # レース場フィルタリングを追加
        #target_stadiums = ['大村', '徳山', '芦屋', '下関', '尼崎', '住之江']
        #filtered_df = df[df['レース場'].isin(target_stadiums)].copy()
        filtered_df = df
        
        # オッズデータと結合（単勝のみをフィルタリング）
        odds_filtered = df_odds[df_odds['舟券種'] == '単勝'].copy() 
        bets_df = pd.merge(filtered_df,odds_filtered,on=['日付', 'レース場', 'レース番号'],how='left')
        
        # 高信頼度のベットをフィルタリング
        high_conf_bets = []
        for _, row in bets_df.iterrows():
            for boat_num in range(1, 7):
                win_rate = row.get(f'{boat_num}号艇勝利確率', 0)
                boat_threshold = threshold.get(boat_num, 0.5)
                
                if win_rate >= boat_threshold:
                    # ベット対象として行をコピーし、予想艇番号を設定
                    bet_row = row.copy()
                    bet_row['1着艇予想'] = boat_num
                    bet_row['predicted_patterns'] = f"艇番{boat_num}"
                    bet_row['won'] = (bet_row['1着艇'] == boat_num)  # 的中フラグを追加
                    high_conf_bets.append(bet_row)
        
        if not high_conf_bets:
            print("ベット対象となるレースがありませんでした")
            return 0.0
        
        high_conf_bets_df = pd.DataFrame(high_conf_bets)
        
        # クロス集計表を表示
        print(pd.crosstab(
            high_conf_bets_df['1着艇予想'], 
            high_conf_bets_df['1着艇'], 
            margins=True, 
            margins_name="合計"
        ))
        
        # 回収率計算
        total_bet = len(high_conf_bets_df) * bet_amount
        correct_bets = high_conf_bets_df[high_conf_bets_df['won']]
        total_return = correct_bets['払戻金'].sum() * (bet_amount / 100)
        roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0.0
        
        print(f"\nベットタイプ: 単勝")
        print(f"閾値: {threshold}")
        print(f"ベット件数: {len(high_conf_bets_df)} レース/{len(bets_df)}レース")
        print(f"的中数: {len(correct_bets)} 回")
        print(f"的中率: {len(correct_bets)/len(high_conf_bets_df):.2%}")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}")
        
        # 的中レース詳細
        if len(correct_bets) > 0:
            print("\n【的中レース詳細】")
            for _, row in correct_bets.iterrows():
                print(
                    f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 艇番{row['1着艇予想']} → 結果 艇番{row['1着艇']} "
                    f"配当 {row['払戻金']}円 "
                    f"予測確率: {row[f'{row['1着艇予想']}号艇勝利確率']:.2f}"
                )
        else:
            print("\n的中したレースはありませんでした")
            
            return roi  

    def Duble_calculate_return_rate(self, df, df_odds, bet_amount=100):
        #target_stadiums = ['大村', '徳山', '芦屋', '下関', '尼崎', '住之江']
        #filtered_df = df[df['レース場'].isin(target_stadiums)].copy()
        filtered_df = df

        # オッズデータとマージ
        odds_filtered = df_odds[df_odds['舟券種'] == "２連単"].copy()
        bets_df = pd.merge(filtered_df, odds_filtered, on=['日付', 'レース場', 'レース番号'], how='left').copy()
        # 初期化
        bets_df['won'] = False
        bets_df['predicted_patterns'] = ""
        bets_df['bet_count'] = 0
        bets_df['bet_combinations'] = ""
        bets = 0

        # メイン処理
        for idx, row in bets_df.iterrows():
            P_1st = [row[f'{i}号艇勝利確率'] for i in range(1,7)]
            P_2nd = {
                i: [row[f'{i}号艇が1着のとき{j}号艇の2着確率'] 
                for j in range(1,7) if j != i]
                for i in range(1,7)
            }
            # 2連単の全組み合わせの確率計算
            pattern_probs = []
            for i in range(1, 7):
                for j in range(1, 7):
                    if i == j:
                        continue
                    prob_1st = P_1st[i-1]
                    # 2着確率はP_2nd[i]の中でjに対応するインデックスは j-1 ただしj > iの場合はインデックスが変わるため調整
                    idx_2nd = j-1 if j < i else j-2
                    prob_2nd = P_2nd[i][idx_2nd]
                    pattern_probs.append(((i, j), prob_1st * prob_2nd))

            # 確率の高い順にソート
            pattern_probs.sort(key=lambda x: x[1], reverse=True)

            # ベット候補の抽出（例: 確率上位3つにベット）
            top_n = 3
            bet_patterns = pattern_probs[:top_n]

            # ベット数をカウント
            bets_df.at[idx, 'bet_count'] = len(bet_patterns)
            # ベットパターンを文字列で格納
            combinations = ", ".join([f"{p[0][0]}-{p[0][1]}" for p in bet_patterns])
            bets_df.at[idx, 'predicted_patterns'] = ", ".join([f"{p[0][0]}-{p[0][1]}" for p in bet_patterns])
            bets_df.at[idx, 'bet_combinations'] = bet_patterns
            
            actual = row['組合せ']
            bets_df.at[idx, 'won'] = actual in combinations

            # 合計ベット数を加算
            if len(bet_patterns) != 0:
                bets += 1
            
        # 回収率計算
        total_bet = bets_df['bet_count'].sum() * bet_amount
        total_return = bets_df[bets_df['won']]['払戻金'].sum()
        roi = total_return / total_bet if total_bet > 0 else 0
        
        # 結果表示
        print(f"\nベットタイプ: 2連単")
        print(f"ベット件数: {bets} レース/{len(bets_df)}レース")
        print(f"総ベット数: {bets_df['bet_count'].sum()} 点")
        print(f"的中数: {bets_df['won'].sum()} 回")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}") 
        
        # 的中レース詳細
        if bets_df['won'].sum() > 0:
            print("\n【的中レース詳細】")
            won_races = bets_df[bets_df['won']].copy()
            
            print("\n【的中した3連単パターン】")
            for _, row in won_races.iterrows():
                print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 {row['predicted_patterns']} → 結果 {row['組合せ']} "
                    f"配当 {row['払戻金']}円 "
                    f"予測確率:{[float(f'{row[f"{i}号艇勝利確率"]:.2f}') for i in range(1,7)]}")
        else:
            print("\n的中したレースはありませんでした")     
        
    def Trifecta_calculate_return_rate(self, df, df_odds, bet_amount=100):        
        # レース場フィルタリングを追加
        target_stadiums = ['大村', '徳山', '芦屋', '下関', '尼崎', '住之江']
        filtered_df = df[df['レース場'].isin(target_stadiums)].copy()
        #filtered_df = df

        # オッズデータとマージ
        odds_filtered = df_odds[df_odds['舟券種'] == "３連単"].copy()
        bets_df = pd.merge(filtered_df, odds_filtered, on=['日付', 'レース場', 'レース番号'], how='left').copy()
        
        # 初期化
        bets_df['won'] = False
        bets_df['predicted_patterns'] = ""
        bets_df['bet_count'] = 0
        bets_df['strategy_used'] = ""
        bets_df['bet_combinations'] = ""
        bets = 0

        # メイン処理
        for idx, row in bets_df.iterrows():
            P_1st = [row[f'{i}号艇勝利確率'] for i in range(1,7)]
            P_2nd = {
                i: [row[f'{i}号艇が1着のとき{j}号艇の2着確率'] 
                for j in range(1,7) if j != i]
                for i in range(1,7)
            }
            
            combinations, strategy_name = self._create_formation1(P_1st, P_2nd)
            
            if len(combinations) != 0:
                bets += 1
            
            # 記録
            bets_df.at[idx, 'strategy_used'] = strategy_name
            bets_df.at[idx, 'predicted_patterns'] = ", ".join(combinations)
            bets_df.at[idx, 'bet_count'] = len(combinations)
            bets_df.at[idx, 'bet_combinations'] = combinations
            
            # 的中チェック
            actual = row['組合せ']
            bets_df.at[idx, 'won'] = actual in combinations
 
        # 回収率計算
        total_bet = bets_df['bet_count'].sum() * bet_amount
        total_return = bets_df[bets_df['won']]['払戻金'].sum()
        roi = total_return / total_bet if total_bet > 0 else 0
        
        
        # 戦術別の出現数と割合を計算
        strategy_counts = bets_df['strategy_used'].value_counts().reset_index()
        strategy_counts.columns = ['戦術名', '出現回数']
        strategy_counts['割合'] = (strategy_counts['出現回数'] / len(bets_df) * 100).round(1)

        # 結果表示
        print("\n【戦術使用頻度】")
        print(strategy_counts.to_string(index=False))

        # あるいはシンプルに
        print("\n【戦術別出現回数】")
        print(bets_df['strategy_used'].value_counts())
        
        # 結果表示
        print(f"\nベットタイプ: 3連単")
        print(f"ベット件数: {bets} レース/{len(bets_df)}レース")
        print(f"総ベット数: {bets_df['bet_count'].sum()} 点")
        print(f"的中数: {bets_df['won'].sum()} 回")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}")
        
        # 的中レース詳細
        if bets_df['won'].sum() > 0:
            print("\n【的中レース詳細】")
            won_races = bets_df[bets_df['won']].copy()
            
            print("\n【的中した3連単パターン】")
            for _, row in won_races.iterrows():
                print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 {row['predicted_patterns']} → 結果 {row['組合せ']} "
                    f"配当 {row['払戻金']}円 " f"戦術 {row['strategy_used']} "
                    f"予測確率:{[float(f'{row[f"{i}号艇勝利確率"]:.2f}') for i in range(1,7)]}")
        else:
            print("\n的中したレースはありませんでした")
        
        return bets_df, roi

    def Trifecta_jissen(self, df):
        bets_df = df.copy()  # 元のデータフレームをコピー
        
        # 初期化（全レコードに対して行う）
        bets_df['won'] = False
        bets_df['predicted_patterns'] = ""
        bets_df['strategy_used'] = ""
        bets_df['bet_combinations'] = ""
        
        # 各行に対して処理
        for idx, row in bets_df.iterrows():
            try:
                P_1st = [row[f'{i}号艇勝利確率'] for i in range(1,7)]
                P_2nd = {
                    i: [row[f'{i}号艇が1着のとき{j}号艇の2着確率'] 
                    for j in range(1,7) if j != i]
                    for i in range(1,7)
                }

                combinations, strategy_name = self._create_formation2(P_1st, P_2nd)

                # 記録
                bets_df.at[idx, 'strategy_used'] = strategy_name
                bets_df.at[idx, 'predicted_patterns'] = ", ".join(combinations)
                bets_df.at[idx, 'bet_combinations'] = combinations
                
            except Exception as e:
                print(f"行 {idx} の処理中にエラーが発生しました: {str(e)}")
                continue
            
        for _, row in bets_df.iterrows():
            print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 {row['predicted_patterns']}" f" 戦術 {row['strategy_used']} "
                    f"予測1着確率:{row[f'1号艇勝利確率']:.2%} {row[f'2号艇勝利確率']:.2%} {row[f'3号艇勝利確率']:.2%} {row[f'4号艇勝利確率']:.2%} {row[f'5号艇勝利確率']:.2%} {row[f'6号艇勝利確率']:.2%} -"
                    f"\n1が1着のときの2着確率[2,3,4,5,6]: {row[f"1号艇が1着のとき2号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき3号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき4号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき5号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき6号艇の2着確率"]:.2%}"
                    f"\n2が1着のときの2着確率[1,3,4,5,6]: {row[f'2号艇が1着のとき1号艇の2着確率']:.2%} {row[f'2号艇が1着のとき3号艇の2着確率']:.2%} {row[f'2号艇が1着のとき4号艇の2着確率']:.2%} {row[f'2号艇が1着のとき5号艇の2着確率']:.2%} {row[f'2号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n3が1着のときの2着確率[1,2,4,5,6]: {row[f'3号艇が1着のとき1号艇の2着確率']:.2%} {row[f'3号艇が1着のとき2号艇の2着確率']:.2%} {row[f'3号艇が1着のとき4号艇の2着確率']:.2%} {row[f'3号艇が1着のとき5号艇の2着確率']:.2%} {row[f'3号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n4が1着のときの2着確率[1,2,3,5,6]: {row[f'4号艇が1着のとき1号艇の2着確率']:.2%} {row[f'4号艇が1着のとき2号艇の2着確率']:.2%} {row[f'4号艇が1着のとき3号艇の2着確率']:.2%} {row[f'4号艇が1着のとき5号艇の2着確率']:.2%} {row[f'4号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n5が1着のときの2着確率[1,2,3,4,6]: {row[f'5号艇が1着のとき1号艇の2着確率']:.2%} {row[f'5号艇が1着のとき2号艇の2着確率']:.2%} {row[f'5号艇が1着のとき3号艇の2着確率']:.2%} {row[f'5号艇が1着のとき4号艇の2着確率']:.2%} {row[f'5号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n6が1着のときの2着確率[1,2,3,4,5]: {row[f'6号艇が1着のとき1号艇の2着確率']:.2%} {row[f'6号艇が1着のとき2号艇の2着確率']:.2%} {row[f'6号艇が1着のとき3号艇の2着確率']:.2%} {row[f'6号艇が1着のとき4号艇の2着確率']:.2%} {row[f'6号艇が1着のとき5号艇の2着確率']:.2%}"
            )
        return bets_df

    def _create_formation1(self, P_1st, P_2nd, threshold=0.2, hole_threshold=0.12):
        formations = []
        strategy_names = []  # 使用された戦略名を記録
        
        entropy = sum([-p * (p and (p > 0) and np.log(p)) for p in P_1st])
        
        sorted_1st = sorted([(p, i+1) for i, p in enumerate(P_1st)], reverse=True)
        top1_p, top1 = sorted_1st[0]
        top2_p, top2 = sorted_1st[1]
        top3_p, top3 = sorted_1st[2]
        
        top1_2nd_prob_boat_pairs = [(prob, boat) for boat, prob in zip([x for x in range(1, 7) if x != top1],P_2nd[top1])]
        sorted_top1_2nd = sorted(top1_2nd_prob_boat_pairs, key=lambda x: x[0], reverse=True)
        top2_2nd_prob_boat_pairs = [(prob, boat) for boat, prob in zip([x for x in range(1, 7) if x != top2],P_2nd[top2])]
        sorted_top2_2nd = sorted(top2_2nd_prob_boat_pairs, key=lambda x: x[0], reverse=True)
        top3_2nd_prob_boat_pairs = [(prob, boat) for boat, prob in zip([x for x in range(1, 7) if x != top3],P_2nd[top3])]
        sorted_top3_2nd = sorted(top3_2nd_prob_boat_pairs, key=lambda x: x[0], reverse=True)
        
        top1_2nd_top1_p, top1_2nd_top1 = sorted_top1_2nd[0]
        top1_2nd_top2_p, top1_2nd_top2 = sorted_top1_2nd[1]
        top1_2nd_top3_p, top1_2nd_top3 = sorted_top1_2nd[2]
        
        top2_2nd_top1_p, top2_2nd_top1 = sorted_top2_2nd[0]
        top2_2nd_top2_p, top2_2nd_top2 = sorted_top2_2nd[1]
        top2_2nd_top3_p, top2_2nd_top3 = sorted_top2_2nd[2]
        
        top3_2nd_top1_p, top3_2nd_top1 = sorted_top3_2nd[0]
        top3_2nd_top2_p, top3_2nd_top2 = sorted_top3_2nd[1]
        top3_2nd_top3_p, top3_2nd_top3 = sorted_top3_2nd[2]
        
        if entropy > 1.6: 
            # エントロピーが高すぎるから、賭けない　→　予測不能
            return [], strategy_names
        if (top1_p - top2_p) > threshold:
            # 1号艇と2号艇の差が顕著
            if (top1==1 and top1_p < 0.6) or (top1==2 and top1_p < 0.5):
                if (top1_2nd_top1_p - top1_2nd_top2_p)>threshold:
                    strategy_names.append("1-1-2")
                    formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top2}")
                    formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top3}")
                else:
                    strategy_names.append("1-1=1")
                    formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top2}")
                    formations.append(f"{top1}-{top1_2nd_top2}-{top1_2nd_top1}")
            else:
                strategy_names.append("1-1-1")
                formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top2}")
            # 穴狙い 456のうちどれかの勝率が12％以上ある場合
            top4s = [t+1 for t in sorted(range(6), key=lambda x: -P_1st[x])[:4]]
            for dark in range(5, 7):
                if P_1st[dark-1] > hole_threshold and (dark in top4s):
                    strategy_names.append("Dark Horse Pursuit (Win Prob)")
                    dark_2nd_prob_boat_pairs = [(prob, boat) for boat, prob in zip([x for x in range(1, 7) if x != dark],P_2nd[dark-1])]
                    sorted_dark_2nd = sorted(dark_2nd_prob_boat_pairs, key=lambda x: x[0], reverse=True)
                    dark_2nd_top1_p, dark_2nd_top1 = sorted_dark_2nd[0]
                    dark_2nd_top2_p, dark_2nd_top2 = sorted_dark_2nd[1]
                    dark_2nd_top3_p, dark_2nd_top3 = sorted_dark_2nd[2]
                    formations.append(f"{dark}-{dark_2nd_top1}-{dark_2nd_top2}")
                    formations.append(f"{dark}-{dark_2nd_top2}-{dark_2nd_top1}")
                    formations.append(f"{dark}-{dark_2nd_top1}-{dark_2nd_top3}")
                    formations.append(f"{dark}-{dark_2nd_top2}-{dark_2nd_top3}")
        elif top2_p > 0.2:
            # 2つの艇が拮抗
            strategy_names.append("Twin Peaks Strategy")
            formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top2}")
            formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top3}")
            formations.append(f"{top2}-{top2_2nd_top1}-{top2_2nd_top2}")
            formations.append(f"{top2}-{top2_2nd_top1}-{top2_2nd_top3}")
                    
        elif ((top2_p - top3_p) <= threshold and top3_p > 0.12):
            # 3つの艇が拮抗
            strategy_names.append("Triple Threat Strategy")
            formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top2}")
            formations.append(f"{top1}-{top1_2nd_top2}-{top1_2nd_top1}")
            formations.append(f"{top2}-{top2_2nd_top1}-{top2_2nd_top2}")
            formations.append(f"{top2}-{top2_2nd_top2}-{top2_2nd_top1}")
            formations.append(f"{top3}-{top3_2nd_top1}-{top3_2nd_top2}")
            formations.append(f"{top3}-{top3_2nd_top2}-{top3_2nd_top1}")

        # Remove duplicates and sort
        formations = list(set(formations))
        
        return formations, strategy_names

    def _create_formation2(self, P_1st, P_2nd):
        """ 与えられた確率データに基づき3連単の組み合わせを生成 """
        strategies = []
        combinations = []
        
        # 1. 基本データ準備
        boat_numbers = list(range(1, 7))  # 1-6号艇
        sorted_boats = sorted(zip(P_1st, boat_numbers), key=lambda x: -x[0])
        
        # 2. 各艇のインデックスマッピング（1号艇→0ではなく1のまま）
        top1_p, top1 = sorted_boats[0]
        top2_p, top2 = sorted_boats[1]
        top3_p, top3 = sorted_boats[2]
        
        # 3. 安全な確率取得関数
        def get_second_prob(first, second):
            """ P_2ndから安全に2着確率を取得 """
            if first not in P_2nd:
                return 0
            second_list = [b for b in boat_numbers if b != first]
            try:
                idx = second_list.index(second)
                return P_2nd[first][idx]
            except (ValueError, IndexError):
                return 0
        
        # 4. 戦略1: 1番人気を外した組み合わせ
        if top1_p > 0.35:  # 1番人気が35%以上
            # 2-3番人気を1着に
            for first in [top2, top3]:
                # 2着候補（1番人気優先）
                seconds = sorted(
                    [(get_second_prob(first, b), b) for b in boat_numbers if b != first],
                    reverse=True
                )
                
                # 1番人気を2着に配置
                if top1 != first:
                    second = top1
                    # 3着候補（2着確率10%以上）
                    thirds = [
                        b for b in boat_numbers 
                        if b not in [first, second] 
                        and get_second_prob(second, b) > 0.1
                    ][:2]
                    
                    for third in thirds:
                        combo = f"{first}-{second}-{third}"
                        combinations.append(combo)
                        strategies.append("Top1 Exclusion")
        
        # 5. 戦略2: ダークホース活用（4-6番人気）
        dark_horses = [b for p, b in sorted_boats[3:] if p > 0.07]  # 7%以上
        for first in dark_horses[:2]:  # 最大2艇まで
            seconds = sorted(
                [(get_second_prob(first, b), b) for b in boat_numbers if b != first],
                reverse=True
            )[:2]  # 2着候補トップ2
            
            for sec_p, second in seconds:
                if sec_p < 0.1:
                    continue
                    
                # 3着は1番人気or2番人気
                thirds = [b for b in [top1, top2] if b not in [first, second]][:1]
                for third in thirds:
                    combo = f"{first}-{second}-{third}"
                    combinations.append(combo)
                    strategies.append("Dark Horse")
        
        # 6. デフォルト戦略（最低1組み合わせ）
        if not combinations:
            combo = f"{top1}-{top2}-{top3}"
            combinations.append(combo)
            strategies.append("Default")
        
        return list(set(combinations)), list(set(strategies))
    
class BoatraceML:
    """メインクラス - 各コンポーネントを統合"""
    def __init__(self, folder):
        self.folder = folder
        self.data_compiler = DataCompiler(folder)
        self.model_trainer = ModelTrainer()
        self.evaluator = BettingStrategyEvaluator()
    
    def run_pipeline(self):
        #race_df = self.data_compiler.compile_race_data()
        #odds_df = self.data_compiler.compile_odds_data()
        #race_df.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\race_df.csv",index=False, encoding='shift-jis')
        #odds_df.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\odds_df.csv",index=False, encoding='shift-jis')
        # データ読み込み
        race_df= pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\race_df.csv", encoding='shift-jis')
        
        # データ分割
        split = -4186
        train_df = race_df[:split]
        test_df = race_df[split:].copy()
        
        # === 1着予想 ===
        # 1号艇は勝てるか？
        # 前処理器の初期化
        preprocessor = DataCompiler(folder)
        
        # 特徴量前処理（全モデル共通）
        X = preprocessor.preprocess_features(race_df)
        
        # バイナリ分類（1号艇が1着かどうか）
        y_binary = preprocessor.get_binary_target(race_df, boat_num=1, top_num=1)
                
        # 前処理情報の保存
        preprocessor.save_preprocessor('preprocessor.pkl')
        
        model_one = self.model_trainer.train_binary_lgbm(X[:split], y_binary[:split], train_df, preprocessor.categorical_indices)
        
        # 1号艇が負けたとき勝つのは？
        y_multi = self.data_compiler.get_multiclass_target(race_df, top_num=1) 
        model_defeat_one = self.model_trainer.train_multiclass_lgbm_exclusion_one(X[:split], y_multi[:split], train_df, preprocessor.categorical_indices)
        
        # 確率計算
        p1 = model_one.predict_proba(X[split:])[:, 1]  # 1号艇の1着確率
        p2to6 = model_defeat_one.predict_proba(X[split:])  # 1号艇以外の1着確率
        
        # 最終確率計算
        final_probs = np.zeros((len(X[split:]), 6))
        final_probs[:, 0] = p1 * 0.83
        final_probs[:, 1:6] = (1 - p1 * 0.83).reshape(-1, 1) * p2to6
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)  # 正規化
        top6_indices = np.argsort(-final_probs, axis=1) + 1  # +1で1-basedの艇番号に
        
        # 各艇番の予測確率を追加
        for boat_num in range(1, 7):
            test_df.loc[:, f'{boat_num}号艇勝利確率'] = final_probs[:, boat_num-1]
            test_df[f'{boat_num}着艇予想'] = top6_indices[:, boat_num-1]
        
        # === 2着予想 ===
        model_twos = {}
        for num in range(1, 7):
            y_mullti = self.data_compiler.get_multiclass_target(race_df, top_num=2)
            model_two = self.model_trainer.train_multiclass_lgbm_target_1st(X[:split], y_mullti[:split], train_df, target_1st_num=num)
            probs_two = model_two.predict_proba(X[split:])
            
            candidate_boats = [i for i in range(1, 7) if i != num]
            predicted_boats = [candidate_boats[idx] for idx in np.argmax(probs_two, axis=1)]
            
            test_df[f'{num}号艇が1着のとき2着艇予想'] = predicted_boats
            for idx, boat_num in enumerate(candidate_boats):
                test_df[f'{num}号艇が1着のとき{boat_num}号艇の2着確率'] = probs_two[:, idx]
            model_twos[num] = model_two
        
        return test_df,model_one,model_defeat_one,model_twos
        
    def run_pipeline_6class(self):
        race_df = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\race_df.csv", encoding='shift-jis')
        
        X, y, _, _ = self.data_compiler.preprocess_for_multiclass(race_df)
        model = self.model_trainer.train_multiclass_lgbm(X, y, race_df)

    def run_pipeline_ellipsis(self):
        odds_df = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\odds_df.csv", encoding='shift-jis')
        result_df = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\result_df.csv", encoding="shift_jis")
        # ベット
        #self.evaluator.Win_calculate_return_rate(result_df, odds_df, bet_amount=100)
        #self.evaluator.Duble_calculate_return_rate(result_df, odds_df, bet_amount=100)
        self.evaluator.Trifecta_calculate_return_rate(result_df, odds_df, bet_amount=100)

    def run_pipeline_jissen(self,model_one,model_defeat_one,model_twos,target_date="2024-04-01", place='大村', race_no=1,weather='晴',wind_dir='東',wind_spd=0,wave_hgt=1, Exhibition_time={1:7.00, 2:7.00, 3:7.00, 4:7.00, 5:7.00, 6:7.00}):
        new_df = self.data_compiler.compile_race_data_B(target_date,place,race_no,weather,wind_dir,wind_spd,wave_hgt,Exhibition_time)
        loaded_preprocessor = DataCompiler.load_preprocessor('preprocessor.pkl',self.folder)
        X_new = loaded_preprocessor.preprocess_features(new_df)
                
        # ベットタイム
        p1 = model_one.predict_proba(X_new)[:, 1]  # 1号艇の1着確率
        p2to6 = model_defeat_one.predict_proba(X_new)  # 1号艇以外の1着確率
        
        # 最終確率計算（1号艇の確率を0.83で減衰）
        final_probs = np.zeros((len(X_new), 6))
        final_probs[:, 0] = p1 * 0.83
        final_probs[:, 1:6] = (1 - p1 * 0.83).reshape(-1, 1) * p2to6
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)  # 正規化
        top6_indices = np.argsort(-final_probs, axis=1) + 1  # +1で1-basedの艇番号に
        
        # 各艇番の予測確率を追加
        for boat_num in range(1, 7):
            new_df.loc[:, f'{boat_num}号艇勝利確率'] = final_probs[:, boat_num-1]
            new_df[f'{boat_num}着艇予想'] = top6_indices[:, boat_num-1]
                
        # === 2着予想 ===
        for num in range(1, 7):
            probs_two = model_twos[num].predict_proba(X_new)
            
            candidate_boats = [i for i in range(1, 7) if i != num]
            predicted_boats = [candidate_boats[idx] for idx in np.argmax(probs_two, axis=1)]
            
            new_df[f'{num}号艇が1着のとき2着艇予想'] = predicted_boats
            for idx, boat_num in enumerate(candidate_boats):
                new_df[f'{num}号艇が1着のとき{boat_num}号艇の2着確率'] = probs_two[:, idx]
        
        self.evaluator.Trifecta_jissen(new_df)

if __name__ == "__main__":
    # ==================変更すべき欄==================
    folder = "C:\\Users\\msy-t\\boatrace-ai\\data"
    # ==================変更してもOK==================
    boatrace_ml = BoatraceML(folder)
    result_df,model_one,model_defeat_one,model_twos = boatrace_ml.run_pipeline()
    
    # モデルを保存する
    result_df.to_csv(folder+"\\agg_results\\result_df.csv", index=False, encoding="shift_jis")
    #os.makedirs('saved_models', exist_ok=True)
    #joblib.dump(model_one, 'saved_models/model_one.joblib')
    #joblib.dump(model_defeat_one,'saved_models/model_defeat_one.joblib')
    #joblib.dump(model_twos, 'saved_models/model_twos_dict.pkl')
    # テスト
    #boatrace_ml.run_pipeline_ellipsis()
    
    # モデルの読み込み
    model_one = joblib.load('saved_models/model_one.joblib')
    model_defeat_one = joblib.load('saved_models/model_defeat_one.joblib')
    model_twos = joblib.load('saved_models/model_twos_dict.pkl')
    scaling_params = joblib.load('saved_models/scaling_params.pkl')
    
    # 入力を受け取る
    #input_str = input("展示タイムを入力(スペースで区切る): ")
    #numbers = list(map(float, input_str.split()))
    #Exhibition_time = {1: numbers[0],2: numbers[1],3: numbers[2],4: numbers[3],5: numbers[4],6: numbers[5]}
    
    #boatrace_ml.run_pipeline_jissen(model_one,model_defeat_one,model_twos,mean,std,columns,target_date="2025-03-26", place='常滑', race_no=5, Exhibition_time=Exhibition_time)