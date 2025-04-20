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
from datetime import datetime, timedelta
from keras import models, layers, regularizers, callbacks
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
import japanize_matplotlib
from itertools import permutations
import joblib

from analyzer import BoatraceAnalyzer

class DataCompiler(BoatraceAnalyzer):
    """データ収集と前処理を担当するクラス"""
    def __init__(self, folder):
        super().__init__(folder)
        self.folder = folder
    
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

                winner_teiban = int(winner_row['艇番'])
                second_teiban = int(second_row['艇番'])
                third_teiban = int(third_row['艇番'])

                # 舟券学習用データ構造
                record = {
                    '日付': date,
                    'レース場': place,
                    'レース番号': race_no,
                    '1着艇': winner_teiban,
                    '2着艇': second_teiban,
                    '3着艇': third_teiban
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
                        f'{prefix}_2-3着率': b['2-3着率'],
                        f'{prefix}_逃し率': b['逃し率'],
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

    def compile_race_data_B(self, target_date="2024-04-01", place='大村', race_no=1, Exhibition_time={1:7.00, 2:7.00, 3:7.00, 4:7.00, 5:7.00, 6:7.00}):
        """指定したレースのデータをコンパイルして1行にまとめる"""
        # CSV読み込み
        df = self.format_race_data(target_date=target_date)
        
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
            'レース番号': race_no
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
                f'{prefix}_2-3着率': b['2-3着率'],
                f'{prefix}_逃し率': b['逃し率'],
            })
        
        # 1行のDataFrameに変換して返す
        return pd.DataFrame([record])
    
    def preprocess_for_multiclass(self, df, top_num = 1):
        """多クラス分類用の前処理"""
        # 1着が何号艇になるかどうか?
        # ======== ラベル作成 ========
        if top_num == 1:
            y = df['1着艇'].values
        elif top_num == 2:
            y = df['2着艇'].values
        elif top_num == 3:
            y = df['3着艇'].values
        
        # ======== 特徴量前処理 ========
        # 不要列の削除
        df = df.drop(columns=['日付', 'レース番号', '1着艇', '2着艇', '3着艇'])

        # カテゴリ変数のダミー変換
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # 欠損値処理・型変換
        X = df.fillna(0).astype('float32')

        # ======== 正規化 ========
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / (std + 1e-7)
        
        return X,y,mean,std
    
    def preprocess_for_binary(self, df, boat_number=1, is_place="Win"):
        """二値分類用の前処理"""
        # 指定艇が1着か?2着以内か?3着以内か?
        # ======== ラベル作成 ========
        if is_place == 'Win':
            y = df[['1着艇']].apply(lambda x: 1 if boat_number in x.values else 0, axis=1).values
        elif is_place == 'place':
            y = df[['1着艇', '2着艇']].apply(lambda x: 1 if boat_number in x.values else 0, axis=1).values
        elif is_place == 'show':
            y = df[['1着艇', '2着艇', '3着艇']].apply(lambda x: 1 if boat_number in x.values else 0, axis=1).values

        # ======== 特徴量前処理 ========
        df = df.drop(columns=['日付', 'レース番号', '1着艇', '2着艇', '3着艇'])

        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        X = df.fillna(0).astype('float32')

        # ======== 正規化 ========
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / (std + 1e-7)
        
        return X, y, mean, std

    def preprocess_for_top3_prediction(self, df_race, winner_boat=1):
        """2-3着を予測用前処理"""
        # 指定艇が1着の場合に、2着または3着に入る艇を予測する前処理
        # ======== ラベル作成 ========
        # 1号艇が1着のレースのみ抽出
        df_winner = df_race[df_race['1着艇'] == winner_boat].copy()
        
        # 特徴量から1号艇関連を除外
        X = df_winner.drop(columns=['日付', 'レース番号', '1着艇', '2着艇', '3着艇'])
        
        # カテゴリ変数の処理
        X = pd.get_dummies(X, columns=[col for col in X.columns if X[col].dtype == 'object'], drop_first=True)
        X = X.fillna(0).astype('float32')
        
        # 残りの艇番
        remaining_boats = [b for b in range(1, 7) if b != winner_boat]
        
        # 各艇について「2着または3着に入ったか」のラベル作成
        y = {}
        for boat in remaining_boats:
            y[boat] = ((df_winner['2着艇'] == boat) | (df_winner['3着艇'] == boat)).astype(int).values
        
        return X, y, remaining_boats, df_winner

    def preprocess_all(self,df,mean,std):
        # ======== 特徴量前処理 ========
        df = df.drop(columns=['日付', 'レース番号'])

        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        X = df.fillna(0).astype('float32')

        # ======== 正規化 ========
        X = (X - mean) / (std + 1e-7)
        
        return X
       
class ModelTrainer:
    """モデル訓練を担当するクラス"""
    def __init__(self):
        self.models = {}

    def train_top3_models(self, X, y, df, remaining_boats, winner_boat=1):
        """
        各艇ごとに2-3着に入るかどうかのバイナリ分類モデルを訓練
        """
        models = {}
        results = {}
        for boat in remaining_boats:
            print(f"\n=== {boat}号艇が2-3着に入るか予測 ===")
            X_filtered = X
            
            # データ分割
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y[boat], df_filtered.index, test_size=0.2, random_state=42, stratify=y[boat]
            )
            
            # モデル訓練（クラス不均衡対策あり）
            model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                class_weight='balanced',
                learning_rate=0.05,
                n_estimators=200
            )
            model.fit(X_train, y_train, 
                    eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(30)])
            
            # 評価
            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred))
            
            # 結果保存
            models[boat] = model
            results[boat] = {
                'y_test': y_test,
                'y_pred': y_pred,
                'idx_test': idx_test
            }
        
        return models, results

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
        #lgb.plot_importance(model,figsize=(6,30))
        #plt.show()
        
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

    def train_multiclass_keras(self, X, y, df):
        """Kerasを使用した多クラス分類モデルの訓練（内部メソッド）"""
        # ======== データ分割 ========
        # 元のインデックスを保持
        y_onehot = to_categorical(y-1, num_classes=6)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y_onehot, df.index, test_size=0.2, random_state=42
        )

        # ======== モデル構築 ========
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],),
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax')  # 6クラス分類
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc', multi_label=True)]
        )

        # ======== コールバック ========
        callbacks_list = [callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                     callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
        ]

        # ======== 学習 ========
        history = model.fit(X_train, y_train,
                            epochs=200,
                            batch_size=128,
                            validation_split=0.2,
                            callbacks=callbacks_list,
                            verbose=1)

        # ======== 評価 ========
        results = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nKerasモデルの多クラス分類精度:")
        print(f"- 精度: {results[1]:.4f}")
        print(f"- AUC: {results[2]:.4f}")
        print(f"- 適合率: {results[3]:.4f}")
        print(f"- 再現率: {results[4]:.4f}")

        # ======== 可視化 ========
        import matplotlib.pyplot as plt
        import japanize_matplotlib
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='Training accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.show()
        
        # ======== 予測結果のDataFrame作成 ========
        y_pred_prob = model.predict(X_test)  # 各艇番の予測確率 (shape: [n_samples, 6])
        
        # 元のDataFrameから対応する行を取得
        original_data = df.loc[idx_test]
        
        # 結果DataFrame作成
        result_df = pd.DataFrame({
            'original_index': idx_test,  # 元のDataFrameのインデックス
            '日付': original_data['日付'].values,
            'レース場': original_data['レース場'].values,
            'レース番号': original_data['レース番号'].values,
            '正解1着艇': original_data['1着艇'].values,  # 元のdfから取得
        })
        
        # 各艇番の予測確率を追加
        for boat_num in range(1, 7):
            result_df[f'{boat_num}号艇勝利確率'] = y_pred_prob[:, boat_num-1]
        
        # 予測情報追加
        result_df['予測1着艇'] = np.argmax(y_pred_prob, axis=1) + 1
        result_df['予測確信度'] = np.max(y_pred_prob, axis=1)
        result_df['正解'] = (result_df['予測1着艇'] == result_df['正解1着艇']).astype(int)
        
        return model, X_test, y_test, result_df    

    def train_binary_lgbm(self, X, y, df, boat_number,is_place):
        """LightGBMを使用した二値分類モデルの訓練（内部メソッド）"""
        # ======== データ分割 ========
        # 元のインデックスを保持
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.2, random_state=42
        )
        
        if boat_number==1 and is_place=='Win':
            class_weights = {0: sum(y == 0) / len(y),1: sum(y == 1) / len(y)}
        else:
            class_weights = "balanced" 
        
        # ======== LightGBMモデル構築 ========
        # パラメータ設定
        model = lgb.LGBMClassifier(
            objective='binary',
            class_weight=class_weights,
            metric='binary_logloss',
            learning_rate=0.01,
            n_estimators=1000,#100
            max_depth=-1,
            num_leaves=31,
            min_data_in_leaf=8,
            feature_fraction=0.6,
            verbose=-1
        )

        # モデル訓練
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True),lgb.log_evaluation(50)]
        )

        # ======== 評価 ========
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # クラス1の確率を取得
        y_pred = (y_pred_prob > 0.4).astype(int)

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
        #lgb.plot_importance(model, figsize=(6, 12),max_num_features=20)
        #plt.show()
        
        # ======== 予測分布の可視化 ========
        print("\n=== 予測結果の詳細分析 ===")
        print("混同行列:")
        # 分かりやすい形式で結果表示
        cm = confusion_matrix(y_test, y_pred)
        print(f"1と予測した件数: {cm[1][0] + cm[1][1]}")
        print(f"  ├─ 実際に1だった件数: {cm[1][1]} (True Positive)")
        print(f"  └─ 実際は0だった件数: {cm[1][0]} (False Positive)")
        print(f"\n0と予測した件数: {cm[0][0] + cm[0][1]}")
        print(f"  ├─ 実際に0だった件数: {cm[0][0]} (True Negative)")
        print(f"  └─ 実際は1だった件数: {cm[0][1]} (False Negative)")
        print("\nクラスごとの精度:")
        print(classification_report(y_test, y_pred))

        return model

    def train_binary_keras(self, X, y, df):
        """Kerasを使用した二値分類モデルの訓練（内部メソッド）"""
        # ======== データ分割 ========
        # 元のインデックスを保持
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.2, random_state=42
        )

        # ======== モデル構築 ========
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X.shape[1],),
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        # ======== コールバック ========
        callbacks = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
        ]

        # ======== 学習 ========
        history = model.fit(X_train, y_train,
                            epochs=100,
                            batch_size=64,
                            validation_split=0.2,
                            callbacks=callbacks,
                            verbose=1)

        # ======== 評価 ========
        results = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n1号艇モデル評価:")
        print(f"- 精度: {results[1]:.4f}")
        print(f"- AUC: {results[2]:.4f}")
        print(f"- 適合率: {results[3]:.4f}")
        print(f"- 再現率: {results[4]:.4f}")

        # ======== 可視化 ========
        import matplotlib.pyplot as plt
        import japanize_matplotlib
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='Training accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.show()
        
        # ======== 確率出力 ========
        y_pred_prob = model.predict(X_test).flatten()  # 各サンプルの1号艇勝利確率（0〜1）

        original_data = df.loc[idx_test]
        # DataFrame化して確認
        result_df = pd.DataFrame({
            'original_index': idx_test,
            '日付': original_data['日付'].values,
            'レース場': original_data['レース場'].values,
            'レース番号': original_data['レース番号'].values,
            f'1号艇勝利確率': y_pred_prob,
            '正解ラベル（1=勝ち）': y_test
        })
        
        return model,X_test,y_test,result_df

    def train_multiclass_lgbm_exclusion(self, X, y, df, exclude_nums=[1]):
        """
        指定した2艇を除外して多クラス分類モデルを訓練
        Args:
            exclude_nums: 除外する艇番号のリスト（例: [1, 2] → 1号艇と2号艇を除外）
        """
        # 指定号艇を除外
        mask = ~np.isin(y, exclude_nums)  # 除外艇以外を選択
        X_filtered = X[mask]
        y_filtered = y[mask]
        df_filtered = df.iloc[mask]
        
        # 除外後の艇番号リスト（例: exclude_nums=[1,2] → [3,4,5,6]）
        candidate_boats = sorted(list(set(y_filtered)))
        num_classes = len(candidate_boats)
        
        # 艇番号を 0,1,2,... にマッピング
        label_mapping = {boat: idx for idx, boat in enumerate(candidate_boats)}
        y_mapped = np.array([label_mapping[boat] for boat in y_filtered])
        
        # データ分割
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_filtered, y_mapped, df_filtered.index, test_size=0.2, random_state=42
        )
        
        # ======== モデル構築（LightGBM） ========
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=num_classes,  # 動的に設定
            boosting_type='gbdt',
            learning_rate=0.05,
            n_estimators=1000,
            max_depth=8,
            num_leaves=31,
            min_data_in_leaf=8,
            feature_fraction=0.8,
            verbose=-1,
            class_weight='balanced'
        )
        
        # 学習
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(50)
            ]
        )
        
        # ======== 評価 ========
        y_pred_mapped = model.predict(X_test)
        
        # 予測結果を元の艇番号に逆変換
        inverse_mapping = {idx: boat for idx, boat in enumerate(candidate_boats)}
        y_pred = np.array([inverse_mapping[idx] for idx in y_pred_mapped])
        y_test_original = np.array([inverse_mapping[idx] for idx in y_test])
        
        # 精度評価
        acc = accuracy_score(y_test_original, y_pred)
        print(f"LightGBMモデルの多クラス分類精度: {acc:.4f}")
        print(f"除外艇: {exclude_nums}, 有効艇: {candidate_boats}")
        
        print("\n=== 予測結果の詳細分析 ===")
        print("混同行列:")
        print(confusion_matrix(y_test_original, y_pred))
        print("\nクラスごとの精度:")
        print(classification_report(y_test_original, y_pred))
        
        return model

class BettingStrategyEvaluator:
    """ベッティング戦略の評価を担当するクラス"""
    def Win_calculate_return_rate(self, df, df_odds, threshold={1: 0.7,2: 0.6,3: 0.6,4: 0.5,5: 0.3,6: 0.3}, bet_amount=100):
        """単勝回収率算出"""
        # オッズデータと結合（単勝のみをフィルタリング）
        odds_filtered = df_odds[df_odds['舟券種'] == '単勝']
            
        merged_df = pd.merge(df,odds_filtered,on=['日付', 'レース場', 'レース番号'],how='left')
        
        # 閾値以上の予測のみを選択
        high_conf_bets = merged_df[merged_df.apply(self._filter_high_confidence_bets, axis=1,threshold=threshold)]
        
        print(pd.crosstab(high_conf_bets['1着艇予想'], high_conf_bets['1着艇'], margins=True, margins_name="合計"))
        
        # 回収率計算
        total_bet = len(high_conf_bets) * bet_amount
        total_return = high_conf_bets[high_conf_bets['1着艇'] == high_conf_bets['1着艇予想']]['払戻金'].sum() * (bet_amount / 100)
        roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
        
        print(f"\nベットタイプ: 単勝")
        print(f"閾値: {threshold}")
        print(f"ベット件数: {len(high_conf_bets)} レース/{len(merged_df)}レース")
        print(f"的中数: {len(high_conf_bets[high_conf_bets['1着艇'] == high_conf_bets['1着艇予想']])} 回")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}")
        won_race = high_conf_bets[high_conf_bets['1着艇'] == high_conf_bets['1着艇予想']].copy()
        print("\n【的中レース詳細】")
        for i,row in won_race.iterrows():
            print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                f"予測 {row['1着艇予想']} → 結果 {row['組合せ']} "
                f"配当 {row['払戻金']}円 "
                f"予測確率:{row[f'{row['1着艇予想']}号艇勝利確率']:.2%}")
        return roi    

    def Trifecta_calculate_return_rate_ex(self, df, df_odds, threshold={1: 0.7,2: 0.6,3: 0.6,4: 0.5,5: 0.3,6: 0.3}, bet_amount=100):
        """改良版3連単回収率計算関数"""
        # オッズデータのフィルタリングとマージ
        odds_filtered = df_odds[df_odds['舟券種'] == "３連単"].copy()
        merged_df = pd.merge(df, odds_filtered, on=['日付', 'レース場', 'レース番号'], how='left').copy()
        
        # 閾値以上の予測のみ選択
        mask = merged_df.apply(self._filter_high_confidence_bets, axis=1, threshold=threshold)
        bets_df = merged_df.loc[mask].copy()
        # 条件1: 1着予想が1 かつ (2着予想が2 または 3) を除外
        #exclude_condition = (bets_df['1着艇予想'] == 1)|(bets_df['1着艇予想'] == 2)

        # 条件を満たさない行のみを残す
        #bets_df = bets_df[~exclude_condition]
        
        # 初期化
        bets_df['won'] = False
        bets_df['predicted_patterns'] = ""
        bets_df['bet_count'] = 0
        bets_df['strategy_used'] = ""
        
        for idx, row in bets_df.iterrows():
            try:
                # 艇番と確率を取得
                top1 = int(row['1着艇予想'])
                top2 = int(row['2着艇予想'])
                top3 = int(row['3着艇予想'])
                top4 = int(row['4着艇予想'])
                top5 = int(row['5着艇予想'])
                top6 = int(row['6着艇予想'])

                # 各艇の勝利確率
                top_rates = {
                    top1: row[f'{top1}号艇勝利確率'],
                    top2: row[f'{top2}号艇勝利確率'],
                    top3: row[f'{top3}号艇勝利確率'],
                    top4: row[f'{top4}号艇勝利確率'],
                    top5: row[f'{top5}号艇勝利確率'],
                    top6: row[f'{top6}号艇勝利確率']
                }

                # 条件付き2着予想
                top1to2 = int(row[f'{top1}号艇が1着のとき2着艇予想'])
                top1to2_rate = row[f'{top1}号艇が1着のとき{top1to2}号艇の2着確率']

                top2 = top1to2
                top_rates[top2] = top1to2_rate

                # 3着予測（1着と2着が確定した状態）
                exclude_nums = [top1, top2]
                remaining_boats = [b for b in range(1, 7) if b not in exclude_nums]
                # 残り艇を確率降順にソート
                sorted_boats = sorted(remaining_boats, key=lambda x: top_rates[x], reverse=True)

                # 確率が高い順にtop3~top6を割り当て
                top3 = sorted_boats[0] if len(sorted_boats) > 0 else 0
                top4 = sorted_boats[1] if len(sorted_boats) > 1 else 0
                top5 = sorted_boats[2] if len(sorted_boats) > 2 else 0
                top6 = sorted_boats[3] if len(sorted_boats) > 3 else 0
                
                # 動的戦略選択
                strategy = self._select_optimal_strategy(
                    top1, top2, top3, top_rates
                )
                
                bets_df.at[idx, 'strategy_used'] = strategy
                                
                # 組み合わせ生成
                predicted_combinations = self._generate_optimized_combinations(
                    top1, top2, top3, top4, top5, top6, strategy
                )
                bets_df.at[idx, 'predicted_patterns'] = ", ".join(predicted_combinations)
                bets_df.at[idx, 'bet_count'] = len(predicted_combinations)
                
                # 的中チェック
                actual_3tan = row['組合せ']
                bets_df.at[idx, 'won'] = actual_3tan in predicted_combinations
                    
            except (ValueError, TypeError) as e:
                print(f"エラーが発生しました: {e}")
                continue
        
        # 賭け金と払戻金計算
        total_bet = bets_df['bet_count'].sum() * bet_amount
        bets_df.loc[:, 'return'] = np.where(bets_df['won'], bets_df['払戻金'] * (bet_amount / 100), 0)
        total_return = bets_df['return'].sum()
        roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
        
        # 結果表示
        print(f"\nベットタイプ: 3連単")
        print(f"閾値: {threshold}")
        print(f"ベット件数: {len(bets_df)} レース/{len(df)}レース")
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
                    f"予測確率:{row[f'{int(row['1着艇予想'])}号艇勝利確率']:.2%}")
        else:
            print("\n的中したレースはありませんでした")

    def Trifecta_jissen(self, df, threshold={1: 0.7, 2: 0.6, 3: 0.6, 4: 0.5, 5: 0.3, 6: 0.3}):
        # 閾値以上かチェック
        mask = df.apply(self._filter_high_confidence_bets, axis=1, threshold=threshold)
        bets_df = df.copy()  # 元のデータフレームをコピー（閾値に関係なく全て処理）
        
        # 閾値を満たさない場合に警告を表示（ただし処理は続行）
        if len(df.loc[mask]) == 0:
            print("注意: 信頼度が閾値を下回っていますが、予測を続行します")
        
        # 初期化（全レコードに対して行う）
        bets_df['won'] = False
        bets_df['predicted_patterns'] = ""
        bets_df['strategy_used'] = ""
        bets_df['confidence_warning'] = ~mask  # 閾値を下回った場合にTrue
        
        # 各行に対して処理
        for idx, row in bets_df.iterrows():
            try:
                # 艇番と確率を取得
                top1 = int(row['1着艇予想'])
                top2 = int(row['2着艇予想'])
                top3 = int(row['3着艇予想'])
                top4 = int(row['4着艇予想'])
                top5 = int(row['5着艇予想'])
                top6 = int(row['6着艇予想'])

                # 各艇の勝利確率
                top_rates = {
                    top1: row[f'{top1}号艇勝利確率'],
                    top2: row[f'{top2}号艇勝利確率'],
                    top3: row[f'{top3}号艇勝利確率'],
                    top4: row[f'{top4}号艇勝利確率'],
                    top5: row[f'{top5}号艇勝利確率'],
                    top6: row[f'{top6}号艇勝利確率']
                }

                # 条件付き2着予想
                top1to2 = int(row[f'{top1}号艇が1着のとき2着艇予想'])
                top1to2_rate = row[f'{top1}号艇が1着のとき{top1to2}号艇の2着確率']

                top2 = top1to2
                top_rates[top2] = top1to2_rate

                # 3着予測（1着と2着が確定した状態）
                exclude_nums = [top1, top2]
                remaining_boats = [b for b in range(1, 7) if b not in exclude_nums]
                # 残り艇を確率降順にソート
                sorted_boats = sorted(remaining_boats, key=lambda x: top_rates.get(x, 0), reverse=True)

                # 確率が高い順にtop3~top6を割り当て
                top3 = sorted_boats[0] if len(sorted_boats) > 0 else 0
                top4 = sorted_boats[1] if len(sorted_boats) > 1 else 0
                top5 = sorted_boats[2] if len(sorted_boats) > 2 else 0
                top6 = sorted_boats[3] if len(sorted_boats) > 3 else 0
                
                # 動的戦略選択
                strategy = self._select_optimal_strategy(
                    top1, top2, top3, top_rates
                )
                
                bets_df.at[idx, 'strategy_used'] = strategy
                                
                # 組み合わせ生成
                predicted_combinations = self._generate_optimized_combinations(
                    top1, top2, top3, top4, top5, top6, strategy
                )
                bets_df.at[idx, 'predicted_patterns'] = ", ".join(predicted_combinations)
                
            except Exception as e:
                print(f"行 {idx} の処理中にエラーが発生しました: {str(e)}")
                continue
        for _, row in bets_df.iterrows():
            print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 {row['predicted_patterns']}" f" 戦術 {row['strategy_used']} "
                    f"予測確率:{row[f'{int(row['1着艇予想'])}号艇勝利確率']:.2%}")
        
        return bets_df
        
    def _filter_high_confidence_bets(self, row, threshold=None):
        """
        艇番号ごとに異なる閾値でベット対象をフィルタリング
        thresholdがNoneの場合はデフォルト値を使用
        thresholdがdictの場合は艇番号ごとの閾値を適用
        """
        # デフォルト閾値（艇番号ごとに設定）
        default_thresholds = {
            1: 0.7,   # 1号艇は0.7以上
            2: 0.6,
            3: 0.6,
            4: 0.5,
            5: 0.3,
            6: 0.3    # 6号艇は0.3以上
        }
        
        # 閾値がdictで指定されていない場合はデフォルトを使用
        if not isinstance(threshold, dict):
            threshold = default_thresholds
        
        predicted_boat = row['1着艇予想']
        win_rate = row[f'{predicted_boat}号艇勝利確率']
        
        # 該当艇の閾値を取得（存在しない場合は0.5をデフォルト）
        boat_threshold = threshold.get(predicted_boat, 0.5)
        
        return win_rate >= boat_threshold
    
    def _select_optimal_strategy(self, top1, top2, top3, top_rates):
        """確率分布に基づく戦略自動選択ロジック（改良版）"""
        # 主要艇の確率を抽出
        p1 = top_rates[top1]
        p2 = top_rates[top2]
        p3 = top_rates[top3]
        rest_prob = 1.0 - p1 - p2 - p3
        
        # 確率の特徴を計算
        dominance_ratio = p1 / (p2 + 1e-6)  # 1着と2着の確率比（ゼロ除算防止）
        top3_concentration = p3 / (p3 + rest_prob + 1e-6)
        entropy = -sum(p * np.log(p) for p in top_rates.values() if p > 0)  # 確率分布のエントロピー
        
        # 戦略決定ロジック
        if p1 > 0.7:
            return "default"  # 1着が圧倒的 → 最小限
        
        elif p1 > 0.5:
            if dominance_ratio > 3.0:
                if top3_concentration > 0.7:
                    return "top1_balanced"  # 1着優位・3着集中
                return "top1_spread"  # 1着優位だが下位が分散
            
            if p2 + p3 > 0.6:
                return "top23_competitive"  # 2-3着が拮抗
            return "top1_balanced"
        
        elif p1 + p2 > 0.7:
            if p3 > 0.25:
                return "top3_darkhorse"  # 3着に穴馬
            return "top2_strong"  # 2着が有力
        
        elif entropy > 1.5:  # 確率分布が平坦
            return "longshot"  # 波乱予想
        else:
            return "conservative"  # 標準戦略

    def _generate_optimized_combinations(self, top1, top2, top3, top4, top5, top6, strategy):
        """拡張戦略パターン"""
        base = [top1, top2, top3, top4, top5, top6]
        strategies = {
            "minimal": [
                f"{top1}-{top2}-{top3}"
            ],
            "default": [
                f"{top1}-{top2}-{top3}",f"{top1}-{top3}-{top2}"
            ],
            
            "box": [
                f"{top1}-{top2}-{top3}", f"{top1}-{top3}-{top2}",
                f"{top2}-{top1}-{top3}", f"{top2}-{top3}-{top1}",
                f"{top3}-{top1}-{top2}", f"{top3}-{top2}-{top1}"
            ],
            "top1_balanced": [
                f"{top1}-{top2}-{top3}", f"{top1}-{top3}-{top2}",
                f"{top1}-{top2}-{top4}", f"{top1}-{top4}-{top2}",
                f"{top1}-{top3}-{top4}", f"{top1}-{top4}-{top3}"
            ],
            "top1_spread": [
                f"{top1}-{top2}-{top3}", f"{top1}-{top2}-{top4}",
                f"{top1}-{top3}-{top2}", f"{top1}-{top3}-{top4}",
                f"{top1}-{top4}-{top2}", f"{top1}-{top4}-{top3}"
            ],
            "top2_strong": [
                f"{top1}-{top2}-{top3}", f"{top2}-{top1}-{top3}",
                f"{top1}-{top2}-{top4}", f"{top2}-{top1}-{top4}",
                f"{top2}-{top3}-{top1}", f"{top2}-{top3}-{top4}"
            ],
            "top23_competitive":[
                f"{top1}-{top2}-{top3}",f"{top1}-{top3}-{top2}",
                f"{top2}-{top1}-{top3}",f"{top3}-{top1}-{top2}",
            ],
            "top3_darkhorse": [
                f"{top1}-{top2}-{top3}", f"{top1}-{top3}-{top2}",
                f"{top3}-{top1}-{top2}", f"{top3}-{top2}-{top1}",
                f"{top2}-{top3}-{top1}", f"{top3}-{top1}-{top4}"
            ],
            "longshot": [
                f"{top1}-{top2}-{top3}", f"{top1}-{top3}-{top2}",
                f"{top2}-{top1}-{top3}", f"{top2}-{top3}-{top1}",
                f"{top3}-{top1}-{top2}", f"{top3}-{top2}-{top1}",
                f"{top4}-{top1}-{top2}", f"{top4}-{top2}-{top1}"
            ],
            "conservative": [
                f"{top1}-{top2}-{top3}", f"{top1}-{top3}-{top2}",
                f"{top2}-{top1}-{top3}", f"{top3}-{top1}-{top2}"
            ]
        }
        return strategies.get(strategy, [f"{top1}-{top2}-{top3}"])

class BoatraceML:
    """メインクラス - 各コンポーネントを統合"""
    def __init__(self, folder):
        self.data_compiler = DataCompiler(folder)
        self.model_trainer = ModelTrainer()
        self.evaluator = BettingStrategyEvaluator()
            
    def run_pipeline(self):
        # データ収集
        #df_race = self.data_compiler.compile_race_data()
        #df_odds = self.data_compiler.compile_odds_data()
        #df_race.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_race.csv",index=False, encoding='shift-jis')
        #df_odds.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_odds.csv",index=False, encoding='shift-jis')
        # データ読み込み
        df_race = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_race.csv", encoding='shift-jis')
        
        # データ分割
        split = -3000
        train_df = df_race[:split]
        test_df = df_race[split:].copy()
        
        # === 1着予想 ===
        # 1号艇は勝てるか？
        X, y, mean, std = self.data_compiler.preprocess_for_binary(df_race, boat_number=1, is_place='Win')
        model_one = self.model_trainer.train_binary_lgbm(X[:split], y[:split], train_df,boat_number=1, is_place='Win')

        # 1号艇が負けたとき勝つのは？
        X, y, _, _ = self.data_compiler.preprocess_for_multiclass(df_race, top_num=1)
        model_defeat_one = self.model_trainer.train_multiclass_lgbm_exclusion(X[:split], y[:split], train_df, exclude_nums=[1])
        
        # ベットタイム
        p1 = model_one.predict_proba(X[split:])[:, 1]  # 1号艇の1着確率
        p2to6 = model_defeat_one.predict_proba(X[split:])  # 1号艇以外の1着確率
        
        # 最終確率計算（1号艇の確率を0.83で減衰）
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
            X, y, _, _ = self.data_compiler.preprocess_for_multiclass(df_race, top_num=2)
            model_two = self.model_trainer.train_multiclass_lgbm_exclusion(X[:split], y[:split], train_df, exclude_nums=[num])
            probs_two = model_two.predict_proba(X[split:])
            
            candidate_boats = [i for i in range(1, 7) if i != num]
            predicted_boats = [candidate_boats[idx] for idx in np.argmax(probs_two, axis=1)]
            
            test_df[f'{num}号艇が1着のとき2着艇予想'] = predicted_boats
            for idx, boat_num in enumerate(candidate_boats):
                test_df[f'{num}号艇が1着のとき{boat_num}号艇の2着確率'] = probs_two[:, idx]
            model_twos[num] = model_two
                
        return test_df,model_one,model_defeat_one,model_twos,mean,std
        
    def run_pipeline_6class(self):
        df_race = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_race.csv", encoding='shift-jis')
        df_odds = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_odds.csv", encoding='shift-jis')
        
        X, y, _, _ = self.data_compiler.preprocess_for_multiclass(df_race)
        model = self.model_trainer.train_multiclass_model(X, y, df_race,model_type="lgbm")

    def run_pipeline_ellipsis(self):
        df_odds = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_odds.csv", encoding='shift-jis')
        test_df = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\test_df.csv", encoding="shift_jis")
        # ベット
        threshold={1: 0.7,2: 0.6,3: 0.6,4: 0.5,5: 0.3,6: 0.3}
        self.evaluator.Win_calculate_return_rate(test_df, df_odds, threshold, bet_amount=100)
        self.evaluator.Trifecta_calculate_return_rate_ex(test_df, df_odds,threshold, bet_amount=100)

    def run_pipeline_jissen(self,model_one,model_defeat_one,model_twos,mean,std,target_date="2024-04-01", place='大村', race_no=1, Exhibition_time={1:7.00, 2:7.00, 3:7.00, 4:7.00, 5:7.00, 6:7.00}):
        df = self.data_compiler.compile_race_data_B(target_date, place, race_no, Exhibition_time)
        #df.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df.csv", index=False, encoding="shift_jis")
        #df = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df.csv", encoding="shift_jis")
        X = self.data_compiler.preprocess_all(df,mean,std)
        
        # ベットタイム
        p1 = model_one.predict_proba(X)[:, 1]  # 1号艇の1着確率
        p2to6 = model_defeat_one.predict_proba(X)  # 1号艇以外の1着確率
        
        # 最終確率計算（1号艇の確率を0.83で減衰）
        final_probs = np.zeros((len(X), 6))
        final_probs[:, 0] = p1 * 0.83
        final_probs[:, 1:6] = (1 - p1 * 0.83).reshape(-1, 1) * p2to6
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)  # 正規化
        top6_indices = np.argsort(-final_probs, axis=1) + 1  # +1で1-basedの艇番号に
        
        # 各艇番の予測確率を追加
        for boat_num in range(1, 7):
            df.loc[:, f'{boat_num}号艇勝利確率'] = final_probs[:, boat_num-1]
            df[f'{boat_num}着艇予想'] = top6_indices[:, boat_num-1]
                
        # === 2着予想 ===
        for num in range(1, 7):
            probs_two = model_twos[num].predict_proba(X)
            
            candidate_boats = [i for i in range(1, 7) if i != num]
            predicted_boats = [candidate_boats[idx] for idx in np.argmax(probs_two, axis=1)]
            
            df[f'{num}号艇が1着のとき2着艇予想'] = predicted_boats
            for idx, boat_num in enumerate(candidate_boats):
                df[f'{num}号艇が1着のとき{boat_num}号艇の2着確率'] = probs_two[:, idx]
        
        self.evaluator.Trifecta_jissen(df,threshold={1: 0.7,2: 0.6,3: 0.6,4: 0.5,5: 0.3,6: 0.3})

    def compiling_and_preprocess_and_train_lane_data(self):
        """1見すると、正答率85%あるが、1号艇を1それ以外は0と予測するだけで達成できるから意味なし"""
        merged_csv_folder = os.path.join(self.folder, "merged_csv")
        all_files = [os.path.join(merged_csv_folder, f) for f in os.listdir(merged_csv_folder) if f.endswith('.csv')]

        all_dataframes = []
        for filepath in all_files:
            filename = os.path.basename(filepath)
            file_date = "20" + filename[:2] + "-" + filename[2:4] + "-" + filename[4:6]

            try:
                df = pd.read_csv(filepath, encoding='shift-jis')
            except Exception as e:
                print(f"読み込みエラー: {filepath} → {e}")
                continue

            # 必須列チェック
            if '着順' not in df.columns:
                print(f"'着順' 列が見つかりません → {filepath}")
                continue
        
            df['日付'] = file_date
            df = df.rename(columns={'着順': 'Rank'})

            cols = [
                '日付', 'レース場', 'レース番号', '艇番', '年齢', '体重', '級別',
                '全国勝率', '全国2連対率', '当地勝率', '当地2連対率',
                'モーター番号', 'モーター2連対率', 'ボート番号', 'ボート2連対率',
                '展示タイム','平均ST',
                'Rank'
            ]

            # 必要な列がすべて揃ってるか確認してから処理
            if all(col in df.columns for col in cols):
                df_out = df[cols].copy()
                all_dataframes.append(df_out)
            else:
                print(f"列が足りない → {filepath}")

        # すべてのDataFrameをまとめて返す
        if all_dataframes:
            df = pd.concat(all_dataframes, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # ========================
        #  学習用前処理
        # ========================
        # ラベル変換：Rank==1 → 1、それ以外 → 0
        df['Rank'] = (df['Rank'] == 1).astype(int)

        # 不要な列を除外
        df = df.drop(['日付', 'レース番号'], axis=1)

        # カテゴリ列のエンコード（級別など）
        df = pd.get_dummies(df, columns=["レース場","級別"], drop_first=True)

        # 残りの欠損値をすべて0で埋める（例外的に）
        df = df.fillna(0)

        # 型を揃える
        df = df.astype("float32")
        
        # 特徴量・目的変数に分ける
        X = df.drop(columns=["Rank"]).values
        y = df["Rank"].values

        # ========================
        # 学習用データ分割
        # ========================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 正規化
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # ========================
        # シンプルなモデル構築
        # ========================
        model = models.Sequential()
        model.add(layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))  # バイナリ出力

        model.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])
        
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(X_train, y_train,
                            epochs=50,
                            batch_size=256,
                            validation_split=0.2,
                            callbacks=[early_stopping],
                            verbose=1)

        # ========================
        # モデル評価
        # ========================hb
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # ========================
        # 可視化（lossとaccuracy）
        # ========================
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='Training acc')
        plt.plot(history.history['val_accuracy'], label='Validation acc')
        plt.title('Accuracy')
        plt.legend()
        plt.show()
 
if __name__ == "__main__":
    # ==================変更すべき欄==================
    folder = "C:\\Users\\msy-t\\boatrace-ai\\data"
    # ==================変更してもOK==================
    boatrace_ml = BoatraceML(folder)
    test_df,model_one,model_defeat_one,model_twos,mean,std = boatrace_ml.run_pipeline()    
    
    # モデルを保存する
    test_df.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\test_df.csv", index=False, encoding="shift_jis")
    #os.makedirs('saved_models', exist_ok=True)
    #joblib.dump(model_one, 'saved_models/model_one.joblib')
    #joblib.dump(model_defeat_one,'saved_models/model_defeat_one.joblib')
    #joblib.dump(model_twos, 'saved_models/model_twos_dict.pkl')
    #joblib.dump({'mean': mean, 'std': std}, 'saved_models/scaling_params.pkl')
    
    # テスト
    boatrace_ml.run_pipeline_ellipsis()
    
    # モデルの読み込み
    model_one = joblib.load('saved_models/model_one.joblib')
    model_defeat_one = joblib.load('saved_models/model_defeat_one.joblib')
    model_twos = joblib.load('saved_models/model_twos_dict.pkl')
    scaling_params = joblib.load('saved_models/scaling_params.pkl')
    mean = scaling_params['mean']
    std = scaling_params['std']
    
    # 入力を受け取る
    input_str = input("展示タイムを入力(スペースで区切る): ")
    numbers = list(map(float, input_str.split()))
    Exhibition_time = {1: numbers[0],2: numbers[1],3: numbers[2],4: numbers[3],5: numbers[4],6: numbers[5]}
    
    boatrace_ml.run_pipeline_jissen(model_one,model_defeat_one,model_twos,mean,std,target_date="2024-04-20", place='戸田', race_no=6, Exhibition_time=Exhibition_time)
    
    
    