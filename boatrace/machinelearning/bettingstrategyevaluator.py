import os
import pandas as pd
import numpy as np

class BettingStrategyEvaluator:
    """ベッティング戦略の評価を担当するクラス"""
    def Win_calculate_return_rate(self, df, df_odds, threshold={1: 0.7,2: 0.4,3: 0.3,4: 0.2,5: 0.1,6: 0.1}, bet_amount=100,venues=None):
        """単勝回収率算出"""
        # レース場フィルタリング
        if venues is not None:
            # venues = ['大村', '徳山', '芦屋', '下関', '尼崎', '住之江']
            df = df[df['レース場'].isin(venues)].copy()
        
        # オッズデータと結合（単勝のみをフィルタリング）
        odds_filtered = df_odds[df_odds['舟券種'] == '単勝'].copy() 
        bets_df = pd.merge(df,odds_filtered,on=['日付', 'レース場', 'レース番号'],how='left')
        
        # 高信頼度のベットをフィルタリング
        high_conf_bets = self._create_bet_strategy(bets_df,threshold)
        
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
    
    def _create_bet_strategy(self,bets_df,threshold):
        # 高信頼度のベットをフィルタリング
        high_conf_bets = []
        for _, row in bets_df.iterrows():
            best_bet_found = False
            # 各艇の勝利確率とその艇番号をリストに格納
            probabilities = []
            for boat_num in range(1, 7):
                win_rate = row.get(f'{boat_num}号艇勝利確率', 0)
                probabilities.append({'boat_num': boat_num, 'win_rate': win_rate})
            
            # 勝利確率が高い順にソート
            probabilities.sort(key=lambda x: x['win_rate'], reverse=True)
            
            # 確率が高い順に、閾値を超えているかチェック
            for prob_info in probabilities:
                boat_num = prob_info['boat_num']
                win_rate = prob_info['win_rate']
                boat_threshold = threshold.get(boat_num, 0.0) # 閾値が設定されていない艇番の場合のデフォルト値を0.0に修正
                
                if win_rate >= boat_threshold:
                    # ベット対象として行をコピーし、予想艇番号を設定
                    bet_row = row.copy()
                    bet_row['1着艇予想'] = boat_num
                    bet_row['predicted_patterns'] = f"艇番{boat_num}"
                    bet_row['won'] = (bet_row['1着艇'] == boat_num)  # 的中フラグを追加
                    high_conf_bets.append(bet_row)
                    best_bet_found = True
                    break # このレースではこれ以上ベットしない
        return high_conf_bets

    def Duble_calculate_return_rate(self, df, df_odds, bet_amount=100,venues=None):
        # レース場フィルタリング
        if venues is not None:
            # venues = ['大村', '徳山', '芦屋', '下関', '尼崎', '住之江']
            df = df[df['レース場'].isin(venues)].copy()

        # オッズデータとマージ
        odds_filtered = df_odds[df_odds['舟券種'] == "２連単"].copy()
        bets_df = pd.merge(df, odds_filtered, on=['日付', 'レース場', 'レース番号'], how='left').copy()
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
            top_n = 1
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
        """
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
        """
        
    def Trifecta_calculate_return_rate(self, df, df_odds, all_odds, bet_amount=100, venues = None):     
        # レース場フィルタリング
        if venues is not None:
            # venues = ['宮島']
            df = df[df['レース場'].isin(venues)].copy()

        # オッズデータとマージ
        odds_filtered = df_odds[df_odds['舟券種'] == "３連単"].copy()
        bets_df = pd.merge(df, odds_filtered, on=['日付', 'レース場', 'レース番号'], how='left').copy()
        bets_df = pd.merge(bets_df, all_odds.copy(), on=['日付', 'レース場', 'レース番号'], how='left').copy()
        
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
            P_3rd = {
                (i,j): [row[f'{i}号艇1着&{j}号艇2着のとき{k}号艇の3着確率'] 
                    for k in range(1,7) if k not in [i,j]]
                for i in range(1,7)
                for j in range(1,7)
                if i != j
            }
            
            combinations,model_probability,expected_value,strategy_name = self._create_formation_top_strategy(P_1st, P_2nd, P_3rd, row)
            
            if len(combinations) != 0:
                bets += 1
            
            # 記録
            bets_df.at[idx, 'strategy_used'] = strategy_name
            bets_df.at[idx, 'predicted_patterns'] = ", ".join(combinations)
            bets_df.at[idx, 'bet_count'] = len(combinations)
            bets_df.at[idx, 'bet_combinations'] = combinations
            bets_df.at[idx, 'expected_value'] = ", ".join([f"{x:.2f}" for x in expected_value])
            bets_df.at[idx, 'model_probability'] = ", ".join([f"{x*100:.2f}%" for x in model_probability])
            
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
                    f"配当 {row['払戻金']}円 " f"戦術 {row['strategy_used']} " f"期待値 {row['expected_value']} " f"確率 {row['model_probability']}")
        else:
            print("\n的中したレースはありませんでした")
        
        return bets_df, roi
        
    def _create_formation_top_strategy(self, P_1st, P_2nd, P_3rd, all_odds_for_race, expected_value_threshold=1.0, probability_threshold=0.003, sort="e", n_top_combos=3):
        """
        1着、2着、3着の確率と各組み合わせのオッズを考慮し、期待値か確率の高い順に上位n個の三連単組み合わせを返す
        expected_value_threshold: この閾値未満の期待値の組み合わせは除外する
        probability_threshold: この閾値未満の総合確率の組み合わせは除外する
        sort: p(確率),e(期待値)
        n_top_combos: 返す上位の組み合わせの数
        """
        combo_mp_ev = []  # (組み合わせ, 確率, 期待値) を保存するリスト
        strategy_name = f"上位{n_top_combos}組み合わせ"

        try:
            # 1着候補を選定 (確率>0のもの)
            first_candidates = [i for i in range(1, 7) if P_1st[i - 1] > 0]

            for first in first_candidates:
                # 2着候補を選定 (1着艇を除く)
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]

                second_candidates = [j for j in range(1, 7)
                                     if j != first and
                                     len(valid_second_probs) > j - 1 and
                                     valid_second_probs[j - 1] > 0]

                for second in second_candidates:
                    # 3着候補を選定 (1着艇と2着艇を除く)
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]

                    third_candidates = [k for k in range(1, 7)
                                        if k not in [first, second] and
                                        len(valid_third_probs) > k - 1 and
                                        valid_third_probs[k - 1] > 0]

                    for third in third_candidates:
                        # 組み合わせの総合確率を計算
                        current_probability = (P_1st[first - 1] *
                                               P_2nd[first][second - 1] *
                                               P_3rd[(first, second)][third - 1])
                        
                        # 確率の閾値のフィルター
                        if current_probability >= probability_threshold:
                            # オッズを取得
                            combo_str = f"{first}-{second}-{third}"
                            # ミスでスペース入れないとだめ
                            odds = all_odds_for_race.get(f" {combo_str}", 0) # オッズがない場合は0とする
                            # 期待値を計算: 総合確率 * オッズ
                            current_expected_value = current_probability * float(odds)
                            # 期待値のフィルター
                            if current_expected_value >= expected_value_threshold:
                                combo_mp_ev.append((combo_str, current_probability, current_expected_value))

            # ソート
            n = 1 if sort == "p" else 2
            combo_mp_ev.sort(key=lambda x: x[n], reverse=True)

            # 上位n個の組み合わせを取得
            top_combos = [combo for combo, mp, ev in combo_mp_ev[:n_top_combos]]
            model_probability = [mp for combo, mp, ev in combo_mp_ev[:n_top_combos]]
            expected_value = [ev for combo, mp, ev in combo_mp_ev[:n_top_combos]]

            if top_combos:
                return top_combos, model_probability,expected_value,strategy_name
            else:
                return [],[],[], "有効な組み合わせなし"

        except Exception as e:
            print(f"Error : {str(e)}")
            print(all_odds_for_race)
            return [],[],[], "エラー発生"

    def _create_formation_biased_odds_strategy(self, P_1st, P_2nd, P_3rd, all_odds_for_race, expected_value_threshold=1.0, n_top_combos=6, min_prob_advantage=0.005, min_odds=10.0,alpha = 0):
        """
        1着、2着、3着の確率と各組み合わせのオッズを考慮し、
        「モデル確率が市場確率より高く、かつ期待値が閾値を超える」組み合わせを、
        期待値の高い順に上位n個の三連単組み合わせを返す戦略。
        
        expected_value_threshold: この閾値未満の期待値の組み合わせは除外する（過小評価されていると判断しても、期待値が低すぎる場合は除外）
        n_top_combos: 返す上位の組み合わせの数
        min_prob_advantage: モデル確率が市場確率より優位であると判断する最小の差（例: 0.01 = 1%の差）
        min_odds: オッズの最低値（低オッズの組み合わせは除外）
        alpha: 調整係数 (0 < alpha < 1)。alphaが小さいほど確率の高い組み合わせを重視。
        """
        combo_results = []  # (組み合わせ, 期待値, モデル確率, 市場確率) を保存するリスト
        strategy_name = f"上位{n_top_combos}組み合わせ（市場歪み狙い、差{min_prob_advantage*100:.1f}%以上）"

        try:
            first_candidates = [i for i in range(1, 7) if P_1st[i - 1] > 0]

            for first in first_candidates:
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]

                second_candidates = [j for j in range(1, 7)
                                     if j != first and
                                     len(valid_second_probs) > j - 1 and
                                     valid_second_probs[j - 1] > 0] 

                for second in second_candidates:
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]

                    third_candidates = [k for k in range(1, 7)
                                        if k not in [first, second] and
                                        len(valid_third_probs) > k - 1 and
                                        valid_third_probs[k - 1] > 0]

                    for third in third_candidates:
                        combo_str = f"{first}-{second}-{third}"

                        odds = all_odds_for_race.get(f" {combo_str}", 0) 
                        
                        # オッズが有効な数値で、かつ最低オッズ以上であることを確認
                        if not isinstance(float(odds), (int, float)) or float(odds) <= min_odds:
                            continue

                        model_probability = (P_1st[first - 1] *
                                             P_2nd[first][second - 1] *
                                             P_3rd[(first, second)][third - 1])

                        # 市場の確率 (控除率を考慮しない単純な逆数)
                        market_probability = 1.0 / float(odds)

                        current_expected_value = model_probability * float(odds)
                        adjusted_expected_value = current_expected_value * (model_probability ** alpha)

                        # モデル確率が市場確率より指定した差以上高い
                        if (model_probability - market_probability >= min_prob_advantage and
                            adjusted_expected_value >= expected_value_threshold): # かつ期待値も閾値以上
                            
                            combo_results.append((combo_str, adjusted_expected_value, model_probability, market_probability))

            combo_results.sort(key=lambda x: x[2], reverse=True) # 確率の高い順

            top_combos = [combo for combo, ev, mp, mkp in combo_results[:n_top_combos]]

            if top_combos:
                return top_combos, strategy_name
            else:
                return [], "有効な組み合わせなし"

        except Exception as e:
            print(f"Error in _create_formation_biased_odds_strategy: {str(e)}")
            return [], "エラー発生"

    def Trifecta_practice(self, df, all_odds):
        bets_df = df.copy()  # 元のデータフレームをコピー
        
        # 初期化（全レコードに対して行う）
        bets_df['won'] = False
        bets_df['predicted_patterns'] = ""
        bets_df['strategy_used'] = ""
        bets_df['bet_combinations'] = ""
        bets_df['ev'] = ""
        bets_df['mp'] = ""
        
        # 各行に対して処理
        for idx, row in bets_df.iterrows():
            try:
                P_1st = [row[f'{i}号艇勝利確率'] for i in range(1,7)]
                P_2nd = {
                    i: [row[f'{i}号艇が1着のとき{j}号艇の2着確率'] 
                    for j in range(1,7) if j != i]
                    for i in range(1,7)
                }
                P_3rd = {
                    (i,j): [row[f'{i}号艇1着&{j}号艇2着のとき{k}号艇の3着確率'] 
                        for k in range(1,7) if k not in [i,j]]
                    for i in range(1,7)
                    for j in range(1,7)
                    if i != j
                }

                combinations,ev,mp = self._create_formation_practice(P_1st, P_2nd, P_3rd, all_odds)

                # 記録
                bets_df.at[idx, 'predicted_patterns'] = ", ".join(combinations)
                bets_df.at[idx, 'bet_combinations'] = combinations
                bets_df.at[idx, 'ev'] = ev
                bets_df.at[idx, 'mp'] = mp
                
            except Exception as e:
                print(f"行 {idx} の処理中にエラーが発生しました: {str(e)}")
                continue
            
        for _, row in bets_df.iterrows():
            print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"\n予測 {row['predicted_patterns']}"
                    f"\n期待値 {row['ev']}"
                    f"\nモデル確率 {row['mp']}"
                    f"\n予測1着確率:{row[f'1号艇勝利確率']:.2%} {row[f'2号艇勝利確率']:.2%} {row[f'3号艇勝利確率']:.2%} {row[f'4号艇勝利確率']:.2%} {row[f'5号艇勝利確率']:.2%} {row[f'6号艇勝利確率']:.2%} -"
                    f"\n1が1着のときの2着確率[2,3,4,5,6]: {row[f"1号艇が1着のとき2号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき3号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき4号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき5号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき6号艇の2着確率"]:.2%}"
                    f"\n2が1着のときの2着確率[1,3,4,5,6]: {row[f'2号艇が1着のとき1号艇の2着確率']:.2%} {row[f'2号艇が1着のとき3号艇の2着確率']:.2%} {row[f'2号艇が1着のとき4号艇の2着確率']:.2%} {row[f'2号艇が1着のとき5号艇の2着確率']:.2%} {row[f'2号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n3が1着のときの2着確率[1,2,4,5,6]: {row[f'3号艇が1着のとき1号艇の2着確率']:.2%} {row[f'3号艇が1着のとき2号艇の2着確率']:.2%} {row[f'3号艇が1着のとき4号艇の2着確率']:.2%} {row[f'3号艇が1着のとき5号艇の2着確率']:.2%} {row[f'3号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n4が1着のときの2着確率[1,2,3,5,6]: {row[f'4号艇が1着のとき1号艇の2着確率']:.2%} {row[f'4号艇が1着のとき2号艇の2着確率']:.2%} {row[f'4号艇が1着のとき3号艇の2着確率']:.2%} {row[f'4号艇が1着のとき5号艇の2着確率']:.2%} {row[f'4号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n5が1着のときの2着確率[1,2,3,4,6]: {row[f'5号艇が1着のとき1号艇の2着確率']:.2%} {row[f'5号艇が1着のとき2号艇の2着確率']:.2%} {row[f'5号艇が1着のとき3号艇の2着確率']:.2%} {row[f'5号艇が1着のとき4号艇の2着確率']:.2%} {row[f'5号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n6が1着のときの2着確率[1,2,3,4,5]: {row[f'6号艇が1着のとき1号艇の2着確率']:.2%} {row[f'6号艇が1着のとき2号艇の2着確率']:.2%} {row[f'6号艇が1着のとき3号艇の2着確率']:.2%} {row[f'6号艇が1着のとき4号艇の2着確率']:.2%} {row[f'6号艇が1着のとき5号艇の2着確率']:.2%}"
            )
        return bets_df

    def _create_formation_practice(self, P_1st, P_2nd, P_3rd, all_odds_for_race, expected_value_threshold=0.0, n_top_combos=30, min_prob_advantage=0.001):
        combo_results = []  # (組み合わせ, 期待値, モデル確率, 市場確率) を保存するリスト
        try:
            first_candidates = [i for i in range(1, 7) if P_1st[i - 1] > 0]

            for first in first_candidates:
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]

                second_candidates = [j for j in range(1, 7)
                                     if j != first and
                                     len(valid_second_probs) > j - 1 and
                                     valid_second_probs[j - 1] > 0] 

                for second in second_candidates:
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]

                    third_candidates = [k for k in range(1, 7)
                                        if k not in [first, second] and
                                        len(valid_third_probs) > k - 1 and
                                        valid_third_probs[k - 1] > 0]

                    for third in third_candidates:
                        combo_str = f"{first}-{second}-{third}"

                        odds = all_odds_for_race.get(f" {combo_str}", 0) 
                        
                        model_probability = (P_1st[first - 1] *
                                             P_2nd[first][second - 1] *
                                             P_3rd[(first, second)][third - 1])

                        # 市場の確率 (控除率を考慮しない単純な逆数)
                        market_probability = 1.0 / float(odds)

                        current_expected_value = model_probability * float(odds)
                        
                        combo_results.append((combo_str, current_expected_value, model_probability, market_probability))

            combo_results.sort(key=lambda x: x[2], reverse=True) # 期待値の高い順

            top_combos = [combo for combo, ev, mp, mkp in combo_results[:n_top_combos]]
            top_ev = [ev for combo, ev, mp, mkp in combo_results[:n_top_combos]]
            top_mp = [mp for combo, ev, mp, mkp in combo_results[:n_top_combos]]

            if top_combos:
                return top_combos,top_ev,top_mp
            else:
                return [],[],[]

        except Exception as e:
            print(f"Error in _create_formation_biased_odds_strategy: {str(e)}")
            return [],[],[]