import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pickle

from boatrace.analyzer import BoatraceAnalyzer

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

    def compile_race_data(self,target_places=None):
        """レースデータをコンパイル"""
        merged_csv_folder = os.path.join(self.folder, "merged_csv")
        all_files = [os.path.join(merged_csv_folder, f) for f in os.listdir(merged_csv_folder) if f.endswith('.csv')]

        all_dataframes = []
        for filepath in all_files:
            # CSV読み込み
            df = pd.read_csv(filepath,encoding="shift-jis")
            
            # レース場でフィルタリング
            if target_places and 'レース場' in df.columns:
                df = df[df['レース場'].isin(target_places)]
            if df.empty:
                continue

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
        o_csv_folder = os.path.join(self.folder, "O_csv")
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