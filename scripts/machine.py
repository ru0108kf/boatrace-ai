import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from optuna.integration import LightGBMPruningCallback
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from collections import Counter
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from keras import models, layers, regularizers, callbacks
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
import japanize_matplotlib

from analyzer import BoatraceAnalyzer

class BoatraceML(BoatraceAnalyzer):
    def __init__(self, folder):
        super().__init__(folder)
        self.folder = folder
        self.model = None
      
    def compiling_race_data(self):
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
                winner_row = group[group['着順'] == 1]
                second_row = group[group['着順'] == 2]
                third_row = group[group['着順'] == 3]

                winner_teiban = int(winner_row['艇番'].values[0])
                second_teiban = int(second_row['艇番'].values[0])
                third_teiban = int(third_row['艇番'].values[0])

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

    def Compiling_odds_data(self):
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

    def preprocess_multiclass(self,df):
        """1着が何号艇になるかどうかの前処理"""
        # ======== ラベル作成 ========
        y = df['1着艇'].values
        
        # ======== 特徴量前処理 ========
        # 不要列の削除
        df = df.drop(columns=['日付', 'レース番号', '1着艇', '2着艇', '3着艇'])

        # カテゴリ変数のダミー変換
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # 欠損値処理・型変換
        df = df.fillna(0).astype('float32')
        X = df

        # ======== 正規化 ========
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / (std + 1e-7)
        
        return X,y,mean,std

    def preprocess_binary(self, df, boat_number=1,is_place="first"):
        """
        前処理関数
        :param df: 入力データフレーム
        :param boat_number: 対象艇番号 (1-6)
        :return: X(特徴量), y(ラベル), mean(平均), std(標準偏差)
        """
        # ======== ラベル作成：指定艇が1着か?2着以内か?3着以内か? ========
        if is_place == 'first':
            df['target'] = df[['1着艇']].apply(lambda x: 1 if boat_number in x.values else 0, axis=1)
        elif is_place == 'second':
            df['target'] = df[['1着艇', '2着艇']].apply(lambda x: 1 if boat_number in x.values else 0, axis=1)
        elif is_place == 'third':
            df['target'] = df[['1着艇', '2着艇', '3着艇']].apply(lambda x: 1 if boat_number in x.values else 0, axis=1)

        # ======== 特徴量前処理 ========
        df = df.drop(columns=['日付', 'レース番号', '1着艇', '2着艇', '3着艇'])

        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        df = df.fillna(0).astype('float32')

        X = df.drop(columns=['target'])
        y = df['target'].values

        # ======== 正規化 ========
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / (std + 1e-7)
        
        return X, y, mean, std

    def preprocess_for_2nd_3rd_prediction(self, df_race, winner_boat=1):
        """
        指定した艇が1着の場合の2-3着予測用の前処理
        
        Parameters:
        ----------
        df_race : pd.DataFrame
            レースデータ
        winner_boat : int
            1着の艇番（1-6）
        
        Returns:
        -------
        X : pd.DataFrame
            特徴量
        y_2nd : np.array
            2着艇のラベル（0-4）
        y_3rd : np.array
            3着艇のラベル（0-4）
        remaining_boats : list
            残りの艇番リスト
        mean : pd.Series
            特徴量の平均
        std : pd.Series
            特徴量の標準偏差
        """
        # 1. 指定艇が1着のレースのみ抽出
        df_winner = df_race[df_race['1着艇'] == winner_boat].copy()
        
        if len(df_winner) == 0:
            print(f"{winner_boat}号艇が1着のレースデータがありません")
            return None, None, None, None, None, None
        
        print(f"\n=== {winner_boat}号艇が1着の場合の前処理 ===")
        print(f"対象レース数: {len(df_winner)}レース")
        
        # 2. 特徴量前処理（1着艇関連の特徴量を除外）
        X = df_winner.drop(columns=['日付', 'レース番号', '1着艇', '2着艇', '3着艇'])
        
        # 勝ち艇関連の特徴量を除外
        cols_to_drop = [col for col in X.columns if f"{winner_boat}号艇" in col]
        X = X.drop(columns=cols_to_drop)
        
        # カテゴリ変数の処理
        categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        X = X.fillna(0).astype('float32')
        
        # 正規化
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / (std + 1e-7)
        
        # 3. ラベル作成（2着と3着の艇番、勝ち艇を除く）
        remaining_boats = [b for b in range(1, 7) if b != winner_boat]
        
        # 2着艇のラベル（0-4に変換）
        y_2nd = df_winner['2着艇'].apply(lambda x: remaining_boats.index(x)).values
        # 3着艇のラベル（0-4に変換）
        y_3rd = df_winner['3着艇'].apply(lambda x: remaining_boats.index(x)).values
        
        return X, y_2nd, y_3rd, remaining_boats, mean, std, df_winner

    def preprocess_for_top3_prediction(self, df_race, winner_boat=1):
        """
        指定艇が1着の場合に、2着または3着に入る艇を予測する前処理
        
        Returns:
        -------
        X : 特徴量
        y : 2着または3着に入るかどうかのラベル (0 or 1)
        remaining_boats : 残りの艇番リスト
        """
        # 1号艇が1着のレースのみ抽出
        df_winner = df_race[df_race['1着艇'] == winner_boat].copy()
        
        # 特徴量から1号艇関連を除外
        X = df_winner.drop(columns=['日付', 'レース番号', '1着艇', '2着艇', '3着艇'])
        cols_to_drop = [col for col in X.columns if f"{winner_boat}号艇" in col]
        X = X.drop(columns=cols_to_drop)
        
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

    def train_multiclass_Keras(self, X, y, df):    
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

    def train_multiclass_LGBM(self, X, y, df):
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
            n_estimators=100,#100
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

        # ======== 評価 ========
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"LightGBMモデルの多クラス分類精度: {acc:.4f}")
        
        print(classification_report(y_test, y_pred, digits=3))
        print(confusion_matrix(y_test, y_pred))
        

        # ======== 予測結果のDataFrame作成 ========
        y_pred_prob = model.predict_proba(X_test)

        # 結果DataFrame作成
        result_df = pd.DataFrame({
            'original_index': idx_test,
            '日付': df.loc[idx_test, '日付'].values,
            'レース場': df.loc[idx_test,'レース場'].values,
            'レース番号': df.loc[idx_test,'レース番号'].values,
            '正解1着艇': df.loc[idx_test,'1着艇'].values,
            '予測1着艇': np.argmax(y_pred_prob, axis=1) + 1,
            '予測確信度': np.max(y_pred_prob, axis=1),
            '正解': (y_pred == df.loc[idx_test, '1着艇'].values - 1).astype(int),
        })

        # 各艇番の予測確率を追加
        for boat_num in range(1, 7):
            result_df[f'{boat_num}号艇勝利確率'] = y_pred_prob[:, boat_num-1]

        # 各艇番ごとの予測評価
        for boat_num in range(1, 7):
            predicted_as_boat = (result_df['予測1着艇'] == boat_num)
            
            if predicted_as_boat.sum() > 0:
                accuracy = result_df.loc[predicted_as_boat, '正解'].mean()
                
                # 実際の正解クラスの分布
                actual_classes = result_df.loc[predicted_as_boat, '正解1着艇']
                class_distribution = actual_classes.value_counts().sort_index()
                class_distribution_pct = actual_classes.value_counts(normalize=True).sort_index()
                
                print(f"\n{boat_num}号艇が1着と予測したデータの評価")
                print(f"- 予測件数: {predicted_as_boat.sum()} レース/{len(y_test)}レース")
                print(f"- 正答率: {accuracy:.2%}")
                
                print("\n実際の正解クラス分布:")
                for cls, count in class_distribution.items():
                    pct = class_distribution_pct[cls]
                    print(f"  {cls}号艇: {count}回 ({pct:.2%})")
            else:
                print(f"\n{boat_num}号艇が1着と予測したデータはありません")
        
        lgb.plot_importance(model,figsize=(6,30))
        #plt.show()

        return model, X_test, y_test, result_df

    def train_multiclass_lgbm_exclusion_one(self, X, y, df):
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
            boosting_type='gbdt',#'gbdt
            learning_rate=0.05,#0.05
            n_estimators=100,#100
            max_depth=8,
            num_leaves=31,#31
            min_data_in_leaf=8,
            feature_fraction=0.8, 
            verbose=-1
        )
        verbose_eval=50
        # 学習と予測（元のコードと同様）
        model.fit(X_train, y_train-2, eval_set=[(X_test, y_test-2)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50,verbose=True),
                            lgb.log_evaluation(verbose_eval)]) 
               
        # ======== 評価 ========
        y_pred = model.predict(X_test) + 2
        acc = accuracy_score(y_test, y_pred)
        print(f"LightGBMモデルの多クラス分類精度: {acc:.4f}")
        
        print("\n=== 予測結果の詳細分析 ===")
        print("混同行列:")
        print(confusion_matrix(y_test, y_pred))
        print("\nクラスごとの精度:")
        print(classification_report(y_test, y_pred))
        
        # ======== 予測結果のDataFrame作成 ========
        y_pred_prob = model.predict_proba(X_test)

        # 結果DataFrame作成
        result_df = pd.DataFrame({
            'original_index': idx_test,
            '日付': df.loc[idx_test,'日付'].values,
            'レース場': df.loc[idx_test,'レース場'].values,
            'レース番号': df.loc[idx_test,'レース番号'].values,
        })
        
        # 各艇番の予測確率を追加
        for boat_num in range(2, 7):
            result_df[f'{boat_num}号艇勝利確率'] = y_pred_prob[:, boat_num-2]

        return model, X_test, y_test, result_df
       
    def train_binary_Keras(self, X, y, df):
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

    def train_binary_lgbm(self, X, y, df, boat_number=1,is_place="first"):
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
        lgb.plot_importance(model, figsize=(6, 30))
        plt.title('Feature Importance')
        #plt.show()
        
        # ======== 確率出力 ========
        if is_place == 'first':
            word = '1着'
        elif is_place == 'second':
            word = '2着以内'
        elif is_place == 'third':
            word = '3着以内'

        # DataFrame化して確認
        result_df = pd.DataFrame({
            'original_index': idx_test,
            '日付': df.loc[idx_test,'日付'].values,
            'レース場': df.loc[idx_test,'レース場'].values,
            'レース番号': df.loc[idx_test,'レース番号'].values,
            f'{boat_number}号艇{word}確率': y_pred_prob,
        })
        
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

        return model, X_test, y_test, result_df

    def train_2nd_3rd_models(self, X, y_2nd, y_3rd, df_filtered, remaining_boats, winner_boat=1):
        """
        2-3着予測モデルの訓練
        """      
            # データ分割
        X_train, X_test, y_2nd_train, y_2nd_test, y_3rd_train, y_3rd_test, idx_train, idx_test = train_test_split(
            X, y_2nd, y_3rd, df_filtered.index, test_size=0.2, random_state=42, stratify=y_2nd
        )
        
        # 2. 2着予測モデル
        model_2nd = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=5,
            metric='multi_logloss',
            learning_rate=0.05,
            n_estimators=200,
            class_weight='balanced',
            verbose=-1
        )
        
        model_2nd.fit(X_train, y_2nd_train, 
                    eval_set=[(X_test, y_2nd_test)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30),
                        lgb.log_evaluation(50)
                    ])
        
        # 3. 3着予測モデル
        model_3rd = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=5,
            metric='multi_logloss',
            learning_rate=0.05,
            n_estimators=200,
            class_weight='balanced',
            verbose=-1
        )
        
        model_3rd.fit(X_train, y_3rd_train,
                    eval_set=[(X_test, y_3rd_test)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30),
                        lgb.log_evaluation(50)
                    ])
        
        # 4. 評価
        y_2nd_pred = model_2nd.predict(X_test)
        y_3rd_pred = model_3rd.predict(X_test)
        
        # 艇番に変換
        y_2nd_test_boats = [remaining_boats[i] for i in y_2nd_test]
        y_2nd_pred_boats = [remaining_boats[i] for i in y_2nd_pred]
        y_3rd_test_boats = [remaining_boats[i] for i in y_3rd_test]
        y_3rd_pred_boats = [remaining_boats[i] for i in y_3rd_pred]
        
        print("\n=== 2着予測精度 ===")
        print(classification_report(y_2nd_test_boats, y_2nd_pred_boats, 
                                target_names=[f"{b}号艇" for b in remaining_boats]))
        
        print("\n=== 3着予測精度 ===")
        print(classification_report(y_3rd_test_boats, y_3rd_pred_boats,
                                target_names=[f"{b}号艇" for b in remaining_boats]))
        
        # 5. 結果DataFrame作成
        y_2nd_prob = model_2nd.predict_proba(X_test)
        y_3rd_prob = model_3rd.predict_proba(X_test)
        
        result_df = pd.DataFrame({
            'original_index': idx_test,
            '日付': df_race.loc[idx_test, '日付'].values,
            'レース場': df_race.loc[idx_test, 'レース場'].values,
            'レース番号': df_race.loc[idx_test, 'レース番号'].values,
            f'実際の1着艇': [winner_boat] * len(idx_test),
            '実際の2着艇': y_2nd_test_boats,
            '予測2着艇': y_2nd_pred_boats,
            '実際の3着艇': y_3rd_test_boats,
            '予測3着艇': y_3rd_pred_boats,
        })
        
        # 各艇の2着・3着確率を追加
        for i, boat in enumerate(remaining_boats):
            result_df[f'{boat}号艇2着確率'] = y_2nd_prob[:, i]
            result_df[f'{boat}号艇3着確率'] = y_3rd_prob[:, i]
                
        return model_2nd, model_3rd, result_df
    
    def train_top3_models(self, X, y, df_filtered, remaining_boats, winner_boat=1):
        """
        各艇ごとに2-3着に入るかどうかのバイナリ分類モデルを訓練
        """
        models = {}
        results = {}
        
        for boat in remaining_boats:
            print(f"\n=== {boat}号艇が2-3着に入るか予測 ===")
            
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

    def calculate_return_rate_multiclass(self, result_df, df_odds, prob_threshold=0.8, bet_type='３連単', bet_amount=100,strategy='default'):
        """
        多クラス分類モデルを使用して舟券予測と回収率計算を行う
        """
            
        # オッズデータをフィルタリング
        odds_filtered = df_odds[df_odds['舟券種'] == bet_type].copy()
                            
        final_df = pd.merge(result_df, odds_filtered,
                            on=['日付', 'レース場', 'レース番号'], how='left')
                
        # 各艇の確率を取得
        boat_probs = []
        for i in range(1, 7):
            boat_probs.append(final_df[f'{i}号艇勝利確率'].values)
        boat_probs = np.column_stack(boat_probs)
                    
        # トップ3の艇番を計算
        top3_boats = np.argsort(boat_probs, axis=1)[:, -3:][:, ::-1] + 1
        final_df['top1_boat'] = top3_boats[:, 0].astype(str)
        final_df['top2_boat'] = top3_boats[:, 1].astype(str)
        final_df['top3_boat'] = top3_boats[:, 2].astype(str)
        final_df['top1_prob'] = np.sort(boat_probs, axis=1)[:, -1]
        final_df['top2_prob'] = np.sort(boat_probs, axis=1)[:, -2]
        final_df['top3_prob'] = np.sort(boat_probs, axis=1)[:, -3]
            
        # ベット条件
        bets_df = final_df[final_df['top1_prob'] >= prob_threshold].copy()

        if bets_df.empty:
            print("条件を満たすレースがありませんでした")
            return 0, pd.DataFrame()
                
        if bet_type == '単勝':
            bets_df['won'] = (bets_df['top1_boat'] == bets_df['組合せ'])
            total_bet = len(bets_df)*bet_amount
        elif bet_type == '３連単':
            # 3連単の的中判定
            bets_df['won'] = False
            bets_df['predicted_patterns'] = ""  # 予測した組み合わせを記録
            
            for idx, row in bets_df.iterrows():
                # トップ3の艇番を取得
                top1 = row['top1_boat']
                top2 = row['top2_boat']
                top3 = row['top3_boat']
                
                # 実際の結果
                actual_3tan = row['組合せ']
                
                # 生成する組み合わせ
                predicted_combinations = self.generate_3tan_combinations(top1, top2, top3, strategy)
                
                # 組み合わせをチェック
                for combo in predicted_combinations:
                    if combo == actual_3tan:
                        bets_df.at[idx, 'won'] = True
                        break
                        
                # 予測パターンを記録
                bets_df.at[idx, 'predicted_patterns'] = ", ".join(predicted_combinations)
                total_bet = len(bets_df)*bet_amount*len(predicted_combinations)
        
        # 払戻金計算（100円単位）
        bets_df['return'] = np.where(bets_df['won'],bets_df['払戻金'] * (bet_amount / 100),0)
        
        total_return = bets_df['return'].sum()
        roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
            
        # 結果表示
        print(f"\nベットタイプ: {bet_type}")
        print(f"確率閾値: {prob_threshold:.0%} 以上")
        print(f"ベット件数: {len(bets_df)} レース/{len(result_df)}レース")
        print(f"的中数: {bets_df['won'].sum()} 回")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}")
        
        # 関数の最後
        if bets_df['won'].sum() > 0:
            print("\n【的中レース詳細】")
            won_races = bets_df[bets_df['won']].copy()
            
            # オッズと予測の詳細も表示
            print("\n【的中した3連単パターン】")
            for _, row in won_races.iterrows():
                print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 {row['predicted_patterns']} → 結果 {row['組合せ']} "
                    f"配当 {row['払戻金']}円 "
                    f"予測確率:{row["top1_prob"]:.2%},{row["top2_prob"]:.2%},{row["top3_prob"]:.2%}")
        else:
            print("\n的中したレースはありませんでした")

        return roi, bets_df

    def generate_3tan_combinations(self, top1, top2, top3, strategy):
        """
        3連単の組み合わせを戦略に基づいて生成
        """
        other_boats = [str(i) for i in range(1, 7) if str(i) not in [top1, top2, top3]]
        combinations = []
        
        if strategy == 'default':
            # 基本パターン: 1-2-3, 1-3-2
            combinations.append(f"{top1}-{top2}-{top3}")
            combinations.append(f"{top1}-{top3}-{top2}")
        
        elif strategy == '1-2-all':
            combinations.append(f"{top1}-{top2}-{top3}")
            combinations.append(f"{top1}-{top3}-{top2}")
            # 1着固定で2着はtop2-3,後は全組み合わせ
            for third in other_boats:
                if third != top3:
                    combinations.append(f"{top1}-{top2}-{third}")
                    combinations.append(f"{top1}-{top3}-{third}")
        
        elif strategy == '1-1=all':
            # 1-2着固定で3着は全組み合わせ
            for third in [top3] + other_boats:
                # 1-2-3, 1-2-X
                combinations.append(f"{top1}-{top2}-{third}")
                # 2-1-3, 2-1-X
                combinations.append(f"{top2}-{top1}-{third}")
        
        elif strategy == 'box':
            # トップ3が入っている全ての組み合わせ
            combinations.append(f"{top1}-{top2}-{top3}")
            combinations.append(f"{top1}-{top3}-{top2}")
            combinations.append(f"{top2}-{top1}-{top3}")
            combinations.append(f"{top2}-{top3}-{top1}")
            combinations.append(f"{top3}-{top1}-{top3}")
            combinations.append(f"{top3}-{top2}-{top1}")
            
        elif strategy == '2-all-2':
            combinations.append(f"{top1}-{top2}-{top3}")
        
        # 重複を削除
        return list(set(combinations))

    def calculate_return_rate_binaly(self, result_df, df_odds, threshold=0.7, bet_type='単勝', bet_amount=100,boat_number=1):
        """
        バイナリモデルから回収率を計算する関数
        
        Parameters:
        ----------
        original_df : pd.DataFrame
            元のデータフレーム（日付、レース番号などの情報を含む）
        result_df : pd.DataFrame
            モデルの予測結果（original_indexを含む）
        df_odds : pd.DataFrame
            オッズデータ
        threshold : float
            ベットする確率の閾値
        bet_type : str
            賭ける舟券の種類（'単勝', '複勝'など）
        bet_amount : int
            1レースあたりの賭け金
        """
        
        # オッズデータと結合（単勝のみをフィルタリング）
        odds_filtered = df_odds[df_odds['舟券種'] == bet_type]
        
        # 各号艇のオッズのみを取得（単勝の場合）
        if bet_type == '単勝':
            odds_filtered = odds_filtered[odds_filtered['組合せ'] == f'{boat_number}']
        elif bet_type == '複勝':
            odds_filtered = odds_filtered[odds_filtered['組合せ'] == f'{boat_number}']
        
        merged_df = pd.merge(result_df,odds_filtered,
            on=['日付', 'レース場', 'レース番号'],how='left')
        
        # 閾値以上の予測のみを選択
        high_conf_bets = merged_df[merged_df[f'{boat_number}号艇確率'] >= threshold]
        
        # 回収率計算
        total_bet = len(high_conf_bets) * bet_amount
        total_return = high_conf_bets[high_conf_bets[f'{boat_number}号艇正解ラベル'] == 1]['払戻金'].sum()*(bet_amount/100)
        roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
        
        print(f"\nベットタイプ: {bet_type}")
        print(f"閾値: {threshold:.0%} 以上")
        print(f"ベット件数: {len(high_conf_bets)} レース/{len(merged_df)}レース")
        print(f"的中数: {len(high_conf_bets[high_conf_bets[f'{boat_number}号艇正解ラベル'] == 1])} 回")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}")
        
        return roi, high_conf_bets

    def calculate_return_rate_multibinaly(self, results_df, df_odds, threshold=0.7, bet_amount=100):
        """
        マルチクラスモデルとバイナリモデルから回収率を計算する関数
        """
        # オッズデータと結合（単勝のみをフィルタリング）
        odds_filtered = df_odds[df_odds['舟券種'] == '３連単']              
        merged_df = pd.merge(results_df,odds_filtered,on=['日付', 'レース場', 'レース番号'],how='left')
        
        # 各艇の1着確率を取得
        boat_probs_first_place = []
        boat_probs_top_three = []
        for i in range(1, 7):
            boat_probs_first_place.append(merged_df[f'{i}号艇勝利確率'].values)
            boat_probs_top_three.append(merged_df[f'{i}号艇3着以内確率'].values)
        boat_probs_first_place = np.column_stack(boat_probs_first_place)
        boat_probs_top_three = np.column_stack(boat_probs_top_three)
        
        # 1. top_boat (1着) を取得 (0-based indexと仮定)
        top_boats = np.argmax(boat_probs_first_place, axis=1)

        # 2. 各レースについて、1着と2着を予想
        second_boats = []
        third_boats = []
        forth_boats = []

        for i in range(len(top_boats)):
            # 現在のレースのtop_three確率を取得
            race_probs = boat_probs_top_three[i].copy()
            
            # 1. 1着艇を取得し、その確率を0にマスク
            top_boat = top_boats[i]
            race_probs[top_boat] = 0
            
            # 2. 2着艇を取得 (残りの中で最大確率)
            second_boat = np.argmax(race_probs)
            race_probs[second_boat] = 0  # 2着艇もマスク
            
            # 3. 3着艇を取得 (さらに残りの中で最大確率)
            third_boat = np.argmax(race_probs)
            race_probs[third_boat] = 0
            
            # 4. 4着艇を取得 (さらに残りの中で最大確率)
            forth_boat = np.argmax(race_probs)
            race_probs[forth_boat] = 0
            
            second_boats.append(second_boat)
            third_boats.append(third_boat)
            forth_boats.append(forth_boat)

        # 結果をDataFrameに追加 (号艇番号は1-basedにする場合 +1)
        merged_df['top1_boat'] = top_boats + 1  # 1-6で表現
        merged_df['top2_boat'] = np.array(second_boats) + 1
        merged_df['top3_boat'] = np.array(third_boats) + 1
        merged_df['top1_prob'] = np.sort(boat_probs_first_place, axis=1)[:, -1]
        
        # ベット条件
        bets_df = merged_df[merged_df['top1_prob'] >= threshold].copy()

        if bets_df.empty:
            print("条件を満たすレースがありませんでした")
            return 0, pd.DataFrame()
        
        # 3連単の的中判定
        bets_df['won'] = False
        bets_df['predicted_patterns'] = ""  # 予測した組み合わせを記録
        
        for idx, row in bets_df.iterrows():
            # トップ3の艇番を取得
            top1 = row['top1_boat']
            top2 = row['top2_boat']
            top3 = row['top3_boat']
            
            # 実際の結果
            actual_3tan = row['組合せ']
            
            # 生成する組み合わせ
            predicted_combinations = self.generate_3tan_combinations(top1, top2, top3, strategy='box')
            
            # 組み合わせをチェック
            for combo in predicted_combinations:
                if combo == actual_3tan:
                    bets_df.at[idx, 'won'] = True
                    break
                    
            # 予測パターンを記録
            bets_df.at[idx, 'predicted_patterns'] = ", ".join(predicted_combinations)
            total_bet = len(bets_df)*bet_amount*len(predicted_combinations)
            
        # 払戻金計算（100円単位）
        bets_df['return'] = np.where(bets_df['won'],bets_df['払戻金'] * (bet_amount / 100),0)
        
        total_return = bets_df['return'].sum()
        roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
        
        print("\nベットタイプ: ３連単")
        print(f"確率閾値: {threshold:.0%} 以上")
        print(f"ベット件数: {len(bets_df)} レース/{len(result_df)}レース")
        print(f"的中数: {bets_df['won'].sum()} 回")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}")
        
        # 関数の最後
        if bets_df['won'].sum() > 0:
            print("\n【的中レース詳細】")
            won_races = bets_df[bets_df['won']].copy()
            
            # オッズと予測の詳細も表示
            print("\n【的中した3連単パターン】")
            for _, row in won_races.iterrows():
                print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 {row['predicted_patterns']} → 結果 {row['組合せ']} "
                    f"配当 {row['払戻金']}円 "
                    f"予測確率:{row["top1_prob"]:.2%}")
        else:
            print("\n的中したレースはありませんでした")
        
        return roi, bets_df

    def _calculate_return_rate(self, result_df, df_odds, winner_boat):
        """回収率計算の内部関数"""
        print("\n=== 回収率計算（3連単）===")
        
        # 予測した2-3着の組み合わせ
        result_df['predicted_pattern'] = result_df.apply(
            lambda x: f"{winner_boat}-{x['予測2着艇']}-{x['予測3着艇']}", axis=1)
        
        # オッズデータと結合
        merged = pd.merge(result_df, 
                        df_odds[df_odds['舟券種'] == '３連単'],
                        on=['日付', 'レース場', 'レース番号'], 
                        how='left')
        
        # 的中判定
        merged['hit'] = merged['predicted_pattern'] == merged['組合せ']
        
        if len(merged) > 0:
            hit_rate = merged['hit'].mean()
            total_bet = len(merged) * 100  # 100円単位
            total_return = merged[merged['hit']]['払戻金'].sum()
            roi = (total_return - total_bet) / total_bet
            
            print(f"ベット数: {len(merged)} レース")
            print(f"的中数: {merged['hit'].sum()} 回")
            print(f"的中率: {hit_rate:.2%}")
            print(f"総賭け金: {total_bet:,} 円")
            print(f"総払戻金: {total_return:,} 円")
            print(f"回収率: {roi:.2%}")
            
            # 的中したレースを表示
            if merged['hit'].sum() > 0:
                print("\n【的中レース】")
                hits = merged[merged['hit']]
                for _, row in hits.iterrows():
                    print(f"{row['日付']} {row['レース場']}{row['レース番号']}R: "
                        f"予測 {row['predicted_pattern']} → 結果 {row['組合せ']} "
                        f"(配当 {row['払戻金']}円)")
        else:
            print("オッズデータが見つかりませんでした")

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
    BoatraceML = BoatraceML(folder = "C:\\Users\\msy-t\\boatrace-ai\\data")
    # レースデータとオッズデータのコンパイル
    #df_race = BoatraceML.compiling_race_data()
    #df_odds = BoatraceML.Compiling_odds_data()
    #df_race.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_race.csv", index=False, encoding="shift_jis")
    #df_odds.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_odds.csv", index=False, encoding="shift_jis")
    df_race = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_race.csv", encoding='shift-jis')
    df_odds = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_odds.csv", encoding='shift-jis')
    
    # モデル訓練
    # 1号艇が逃げるか?
    X, y, mean, std = BoatraceML.preprocess_binary(df_race, boat_number=1, is_place='first')
    model, X_test, y_test, result_df = BoatraceML.train_binary_lgbm(X, y, df_race, boat_number=1)
    
    # 逃げれなかった場合の2-6号艇のなかでどれが1着になるか？
    X,y,mean,std = BoatraceML.preprocess_multiclass(df_race)
    model,X_test,y_test,result_df = BoatraceML.train_multiclass_lgbm_exclusion_one(X,y,df_race)
    
    # 1号艇が逃げた場合
    # 2-3着を予測する
    # 前処理
    X, y, remaining_boats, df_filtered = BoatraceML.preprocess_for_top3_prediction(df_race, winner_boat=1)

    # 訓練（各艇ごとにモデル作成）
    models, results = BoatraceML.train_top3_models(X, y, df_filtered, remaining_boats)    
    
    # 1号艇が逃げれなかった場合
    # 2号艇が1着のとき2-3着を予測する
    # 3号艇が1着のとき2-3着を予測する
    # 4号艇が1着のとき2-3着を予測する
    # 5号艇が1着のとき2-3着を予測する
    # 6号艇が1着のとき2-3着を予測する