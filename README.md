# biosim — バイオリアクター物質収支シミュレーター

菌体増殖・代謝物生産・基質消費・溶存酸素消費の4つのサブモデルを組み合わせ、常微分方程式（ODE）による物質収支をバッチ／流加培養（fed-batch）／連続培養（chemostat）の3つの運転モードで解くシミュレーターです。各サブモデルは戦略パターン（抽象基底クラス＋レジストリ）で差し替え可能に設計されています。コアはUI非依存の純粋なPythonライブラリで、Streamlit製GUIはその薄いラッパーです。

- コアライブラリ: `src/biosim/`
- GUI（起動ページ）: `app/Home.py`
- GUI（シミュレーション）: `app/pages/1_simulation.py`
- GUI（実験データフィッティング）: `app/pages/2_fitting.py`
- GUI（パラメータ予測とベイズ最適化）: `app/pages/3_bayes_prediction.py`
- テスト: `tests/`（pytest, 107件）
- 動作デモ: `examples/batch_monod_example.py`

## セットアップ

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## 実行

```bash
# ライブラリ単体のデモ
.venv/bin/python examples/batch_monod_example.py

# GUI
.venv/bin/streamlit run app/Home.py

# テスト
.venv/bin/python -m pytest
```

---

## 実測データとの比較

GUIのサイドバー下部「実測データとの比較」から、1つのCSVファイルをアップロードしてシミュレーション結果と重ね描きできます（`src/biosim/experimental_data.py`）。列名は固定スキーマでパースされ、必須の `t` 列がない場合は `ExperimentalDataError` を出します。

| 列名 | 必須/任意 | 内容 |
|---|---|---|
| `t` | 必須 | 時間 (h) |
| `X` | 任意 | 菌体濃度・DCW (g/L) |
| `OD` | 任意 | 濁度。`X` 列が無い場合のみ `X = OD変換係数 * OD` として `X` に換算 |
| `S` | 任意 | 基質濃度 (g/L) |
| `P` | 任意 | 生産物濃度 (g/L) |
| `OTR` | 任意 | 酸素移動速度。`OxygenWithKLa`（kLaモデル）使用時のみシミュレーション側に `OTR` 列が存在 |

`X`/`S`/`P`/`OTR`/`OD` はすべて任意で、1つのCSVに好きな組み合わせで含められます。存在しない列は自動的にプロットからスキップされます。ライブラリを直接使う場合は `load_experimental_csv(path, od_conversion_factor=1.0)` を呼び、返された DataFrame を `SimulationResults.to_plotly_figure(experimental_data=...)` に渡します。

---

## 実験データフィッティング

GUIの「Fitting」ページ（`app/pages/2_fitting.py`）から、実測データに対して増殖・生産・基質・酸素の各モデルパラメータを推定できます（`src/biosim/fitting.py`）。

- **複数バッチ対応**: 実測CSVを複数アップロードすると、同一のモデル・パラメータ設定を各バッチに独立にフィットし、バッチごとの推定結果を比較できます（ファイル名からバッチ名を自動生成、重複時は自動で連番付与）。
- **パラメータごとの固定/フィット選択**: 4カテゴリ（増殖・生産・基質・酸素）それぞれについて使用するモデルを選び、各パラメータを「固定値」か「自由(推定)」（初期推定値＋上下限）のどちらにするか個別に指定できます。`Yps` のような任意パラメータは「未使用 (None)」も選べます。
- **フィッティング方式**: 閉形式回帰ではなく、候補パラメータで実際にODEをシミュレーションし、実測時刻に補間した値との残差を `scipy.optimize.least_squares` で最小化する汎用的な simulate-and-compare 方式です。そのため `GompertzGrowth` のような力学的ODE形のモデルにもそのままフィッティングできます。
- **CSVエクスポート**: フィッティング結果はワイド形式でエクスポートできます。列は `batch`（バッチ名）、`{カテゴリ}_model`（使用モデル名。例: `growth_model` = `gompertz`）、`{カテゴリ}_{パラメータ名}`（固定値またはフィット値の最終値。例: `growth_mu_max`）、`cost`（正規化残差のRMSE）、`success`（収束フラグ）です。

ライブラリを直接使う場合は `ModelSpec`/`ParameterSpec` でモデルとパラメータ設定を組み立て、`fit_batch(...)` に実測データ（`load_experimental_csv` の戻り値と同じスキーマの DataFrame）を渡します。複数バッチの `FitResult` は `fit_results_to_dataframe(...)` でまとめてCSV化できます。

---

## パラメータ予測とベイズ最適化

GUIの「Bayes Prediction」ページ（`app/pages/3_bayes_prediction.py`）から、温度・pHなどの実験条件とフィッティング結果を紐づけた学習データをもとに、条件からモデルパラメータを予測し、次に試すべき実験条件を提案できます（`src/biosim/parameter_prediction.py`, `src/biosim/experiment_design.py`）。

### 条件→パラメータのベイズ回帰予測

`fit_results_to_dataframe` が出力するワイド形式CSV（`batch`, `{category}_model`, `{category}_{param}`, ...）に温度・pHなどの条件列を追加したものを学習データとして、**パラメータ列ごとに独立したガウス過程回帰（GP）** を学習します（`fit_parameter_models`）。1点の新しい条件に対する予測は `predict_parameters` が平均・標準偏差・信頼区間（デフォルト95%）を返します。

- データ不足（非欠損行数が `min_rows` 未満）や分散がほぼゼロのパラメータ列は、そのパラメータだけスキップされ（`skipped_parameters` に理由が記録される）、他のパラメータの学習は継続します。
- 複数パラメータ間の相関（例: `mu_max` が高いバッチでは他のパラメータも高くなる傾向がある、など）を学習データのGP残差から推定し（`residual_correlation`）、`sample_parameters` で相関を保ったまま複数パラメータを同時にモンテカルロサンプリングできます。ペアごとの共通データが少なすぎる場合は無相関にフォールバックします（`correlation_fallback_pairs`）。

### 次の実験点の提案（シミュレーションベースのベイズ最適化）

最適化の対象は `mu_max` のような予測パラメータそのものではなく、そのパラメータで実際にシミュレーションを実行して得られる**生産物の絶対量**（濃度ではなく質量、`P(t_end) * V(t_end)`）です（`suggest_next_experiment`）。

1. 各過去バッチについて、そのバッチの**実際のフィット済みパラメータ値**（GP予測ではない）でシミュレーションを実行し、生産物絶対量の実績値を求めます（`compute_historical_objectives`）。この中の最良値が `f_best` になります。
2. 探索範囲内の候補条件（条件変数が2つ以下なら格子、3つ以上なら `scipy.stats.qmc.LatinHypercube`）それぞれについて、GPの同時分布から相関を保ったパラメータをモンテカルロサンプリングし、サンプルごとにシミュレーションして生産物絶対量の分布（平均・標準偏差）を推定します（`evaluate_condition`）。個々のサンプルのシミュレーションが失敗しても、そのサンプルだけを除外して継続します。
3. 各候補の Expected Improvement `EI(x) = (μ(x)-f_best-ξ)Φ(Z) + σ(x)φ(Z)` を計算し（`ξ` は `f_best` に対する相対値）、最も有望な1点を提案します。

パレートフロントを探す多目的最適化ではなく、相関を考慮した不確実性伝播の上に立つ**単一目的**のベイズ最適化です。

---

## 温度・pH 条件と基質律速モデル

温度・pH は ODE の状態ではなく、フィードと同じ**時間の入力** `T(t)`, `pH(t)` として与えます（`src/biosim/conditions.py`）。`BioreactorSimulation(..., condition_profile=ConditionProfile(temperature_fn, pH_fn))` で渡し、省略時は標準条件（30 ℃ / pH 5.0）です。`shift_value(30, 24, 48)`（48 h で 30→24 ℃）、`ramp_value`、`piecewise_value` が使えます。`fit_batch(..., condition_profile=...)` も同様で、バッチごとに自分の条件履歴を渡します。

温度に依存するモデル（`condition_aware = True`）だけが現在の条件を受け取ります。既存モデルは従来どおりのシグネチャで動きます。

| モデル | レジストリ名 | 内容 |
|---|---|---|
| `PirtMonodGrowth` | `growth: pirt_monod` | `qS = qSmax(T)·S/(Ks+S)`、`μ = Y(T)·(qS − m(T,pH))`。μmax を持たず、温度の効果は Y と m に現れる |
| `GrowthDrivenSubstrate` | `substrate: growth_driven` | 基質消費を持たない。`pirt_monod` が `dS/dt = −qS·X` を定めるため、必ず対で使う（組み合わせを誤ると構築時にエラー） |
| `TemperatureLuedekingPiretProduct` | `product: luedeking_piret_temp` | `dP/dt = α·max(dX/dt,0) + β(T)·X − k_d(T)·P` |

温度依存は基準温度（30 ℃）で再パラメータ化した Arrhenius 型 `k(T) = k_ref·exp[−Ea/R·(1/T − 1/T_ref)]`（Ea は kJ/mol）、Y は温度の1次、m の pH 補正は pH 5.0 まわりの2次式です。定速フィードで S≈0 のとき、総菌体量 `x = X·V` は `x(t) = x_ss − (x_ss − x0)·exp(−Y·m·t)`、`x_ss = F·S_f/m` の解析解に一致します（`tests/test_temperature_models.py`）。実行例は `examples/temperature_shift_example.py`。

### 誘導剤の添加と β の誘導依存

誘導剤は**瞬時に終濃度になる添加イベント**として与えます（`src/biosim/inducer.py`）。`BioreactorSimulation(..., inducer_events=[InducerDose(56.0, 300.0)])`、標準レシピ（32/56/80 h に 300 µM）は `standard_induction()` です。`fit_batch(..., inducer_events=...)` も同様に、バッチごとの添加履歴を渡します。

- 誘導剤の濃度 `I`（µM）が状態に加わり、添加の間はフィードによる希釈だけで下がります（`I·V` が保存される。分解は無し）。添加時刻で積分を区切って再開するので、不連続を `solve_ivp` が跨ぎません。出力の時刻はそのままで、添加時刻の値は添加後の値です。
- `mode="set"`（既定）は「終濃度 C まで引き上げる」（残っている誘導剤も C に数える）、`mode="add"` は「いまの濃度に C を足す」です。実験で「終濃度300 µM」をどちらの意味で運用しているかに合わせて選びます。
- `InducibleLuedekingPiretProduct`（`product: luedeking_piret_induced`）: `β(T, C) = A(T)·[β_basal + β_max·Cⁿ/(EC50ⁿ + Cⁿ)]`。無誘導の基礎発現 `β_basal`、飽和時の上乗せ `β_max`、半飽和濃度 `EC50`、Hill 係数 `hill_n`（1 で双曲線）。誘導剤イベントはこのモデルでだけ使えます（他のモデルに渡すとエラー）。

**未実装**: 温度シフト直後の順応ラグと、誘導から β が立ち上がるまでの遅れ（応答は瞬時と仮定）、β/k_d への pH 依存。

---

## 操作ログ（Excel）と複数培養の同時フィッティング

操作ログと分析値を 1 つの Excel（1 シート = 1 培養。シート名が培養名。`batch` 列があれば 1 シート内で培養を分割）にまとめて読み込みます。空欄は「値なし」です。

| 列 | 内容 |
|---|---|
| `time` | 経過時間 [h]（日時でも可。先頭行からの経過時間に変換） |
| `temperature`, `ph` | 設定値。空欄行は変更なし（次の入力までステップ保持。最初の入力前は 30 ℃ / pH 5.0） |
| `induction` | その行で添加する誘導剤濃度 [µM]（1 行 1 回の添加。空欄/0 = 添加なし） |
| `feed` | フィード速度 [L/h]（同上ステップ保持） |
| `DCW`(`X`), `OD`, `S`, `P` | 分析値。OD は `od_to_dcw` で DCW に換算（DCW が空欄の行のみ） |

```python
from biosim import load_operation_log, fit_joint, fit_staged, FitStage

cultures = load_operation_log("culture_log.xlsx", initial_conditions=ic, S_feed=100.0)
result = fit_joint(cultures, model_specs)          # 全培養で共通パラメータを推定
print(result.estimates, result.flags)

staged = fit_staged(cultures, model_specs, [       # 段階フィット
    FitStage("30℃ 成長", ["growth.Yxs_ref", "growth.ms_ref"], ["standard", "recipe2"], ["X"]),
    FitStage("温度", ["growth.Ea_ms"], ["standard", "T24", "T36"], ["X"]),
    FitStage("生産", ["product.beta_max", "product.EC50"], ["standard", "Cu150", "Cu600"], ["P"]),
])
```

残差は列ごとに共通スケール（全培養での最大値、または `sigma`）で正規化します。`result.flags` は、相関 ≥ 0.95 のパラメータ対、境界張り付き、相対標準誤差が大きいパラメータ、残差数不足、条件数過大を警告します。

**識別性の注意**: このモデルは x に対して線形なので、フィード量を変えても Y と m は分離できません（同じ x_ss = F·S_f/m, τ = 1/(Y·m) の組を再現するため）。分離を助けるのは温度変化（特に高温側へのシフト）で、培養数を増やすのはノイズの平均化にしかなりません。また実データでは Y は OD→DCW 換算係数と交絡します。

## 理論背景

### 1. 統一物質収支の枠組み

反応器を完全混合槽（CSTR: Continuous Stirred-Tank Reactor）とみなすと、任意の溶質濃度 `C` に対する一般物質収支は次の通りです。

```
d(V·C)/dt = r(C)·V + F_in·C_feed − F_out·C
```

- `V`: 作動体積 (L)
- `r(C)`: 反応速度（濃度の時間変化率, g/L/h）
- `F_in`, `F_out`: 流入・流出流量 (L/h)
- `C_feed`: 流入液中の濃度

積の微分則 `d(VC)/dt = V·dC/dt + C·dV/dt` と全体の体積収支 `dV/dt = F_in − F_out` を代入すると、`F_out` の項が完全に相殺され、次の非常にシンプルな形に帰着します（本シミュレーターの核となる関係式）。

```
dC/dt = r(C) + (F_in/V)·(C_feed − C)
dV/dt = F_in − F_out
```

この1本の式に対して `F_in(t)`, `F_out(t)`, `C_feed` の与え方を変えるだけで、3つの運転モードすべてを表現できます（実装: `src/biosim/simulation.py` の `_rhs`、モード別の流量供給は `src/biosim/operation_modes.py`）。

| 運転モード | F_in | F_out | 帰結 |
|---|---|---|---|
| バッチ (`Batch`) | 0 | 0 | `dC/dt = r(C)`（反応項のみ）、`dV/dt = 0` |
| 流加培養 (`FedBatch`) | `F(t)`（ユーザー指定プロファイル） | 0 | `dC/dt = r(C) + (F(t)/V)(C_feed−C)`、体積は単調増加 |
| 連続培養 (`Chemostat`) | `D·V` | `D·V` | `dC/dt = r(C) + D(C_feed−C)`（教科書通りのケモスタット式）、`V` は一定 |

状態ベクトルは `[X, S, P, (C_O2), V]`（`X`: 菌体, `S`: 基質, `P`: 生産物, `C_O2`: 溶存酸素 [任意], `V`: 体積）。`C_O2` は酸素モデルが供給ダイナミクス（kLa）をサポートする場合のみ状態に追加され、そうでない場合は次元を持たず、累積OURのみ診断値として報告されます（`src/biosim/state.py` の `StateLayout` が可変長の状態ベクトルを動的に管理）。

流加培養のフィードプロファイル `F(t)` はGUI上で `constant`（一定）・`step`（単一ステップ）・`exponential`（指数関数、`mu_set` 一定を狙うfeed-forward）・`csv`（多段階ステップ、任意個数のブレークポイント）から選べます。`csv` は `time`（h）・`feed_rate`（L/h）の2列を持つCSVをアップロードし、ブレークポイント間を階段状（ステップホールド）で保持します。最初のブレークポイントより前はCSV1行目の値、最後のブレークポイントより後は最終行の値をそのまま保持します（`stepwise_feed`/`load_feed_profile_csv`、`src/biosim/operation_modes.py`・`src/biosim/feed_profile.py`）。

各カテゴリのモデルは以下の順序で呼び出され、依存関係が解決されます: **増殖 → 生産 → 基質 → 酸素**（生産・基質・酸素の速度式は増殖モデルが計算した `dX/dt` に依存するため）。

---

### 2. 菌体増殖モデル (`src/biosim/models/growth.py`)

全ての増殖モデルは比増殖速度 `μ (1/h)` を返すインターフェースに統一されており、`dX/dt = μ·X` はオーケストレーター側（`simulation.py`）で計算されます。

#### Monod式（基質律速）

```
μ(S) = μ_max · S / (Ks + S)
```

- `μ_max`: 最大比増殖速度 (1/h)
- `Ks`: 半飽和定数 (g/L) — `μ = μ_max/2` となる基質濃度

古典的なMichaelis-Menten型の飽和曲線で、`S >> Ks` で `μ→μ_max`、`S→0` で `μ→0`。

#### Logistic式（環境容量律速）

```
μ(X) = μ_max · (1 − X/Xmax)
```

- `Xmax`: 環境収容力（最大菌体濃度）(g/L)

基質濃度に依存せず、菌体濃度自体が収容力に近づくにつれ増殖速度が線形に減衰するモデル。`X = Xmax` で `μ=0`、`X > Xmax` では `μ<0`（減衰）となり、これはクリップせずそのまま許容しています。

#### Gompertz式（力学的ODE形）

```
μ(X) = μ_max · ln(Xmax / X)
```

⚠️ **注意**: これはODE（物質収支）に組み込むための**力学的Gompertz式**であり、実験データの終点フィッティングに使われる閉形式回帰曲線 `X(t) = Xmax·exp(−exp(...))` とは異なります。上式を `dX/dt = μ·X` に代入すると `dX/dt = μ_max·X·ln(Xmax/X)` となり、これがGompertz型の非対称S字増殖曲線を与える標準的な力学モデルです。simulate-and-compare方式の[実験データフィッティング](#実験データフィッティング)機能は、閉形式解を必要としないため、この力学的形のまま `μ_max`/`Xmax` を実測データから推定できます。

境界値の扱い: `X ≤ 0` または `X ≥ Xmax` のとき `ln` が未定義／負になるため、実装では `μ=0` にクリップしています。

#### Monod式（基質＋酸素律速、`MonodGrowthO2`）

```
μ(S, C_O2) = μ_max · S/(Ks+S) · C_O2/(Ko2+C_O2)
```

- `μ_max`: 最大比増殖速度 (1/h), 典型範囲 0.1-1.0
- `Ks`: 基質の半飽和定数 (g/L), 典型範囲 0.01-1.0
- `Ko2`: 溶存酸素の半飽和定数 (mg/L), 典型範囲 0.01-0.5, デフォルト 0.2

通常のMonod式に溶存酸素濃度による律速項を掛け合わせた二重Monod型です。通気・撹拌能力（kLa）が酸素需要に追いつかない場合、実際の培養では溶存酸素濃度の低下とともに増殖速度自体が鈍化しますが、`MonodGrowth`（基質律速のみ）は`C_O2`を一切参照しないためこの現象を表現できません。酸素律速による増殖阻害を再現したい場合はこちらを選びます。**`C_O2`を状態として追跡する`OxygenWithKLa`と組み合わせる必要があり**、`OxygenDemandOnly`と組み合わせると`BioreactorSimulation`構築時に`InvalidParameterError`になります。

---

### 3. 代謝物生産モデル (`src/biosim/models/product.py`)

#### Luedeking-Piret式（増殖連動＋非連動）

```
dP/dt = α·(dX/dt) + β·X
```

- `α`: 増殖連動生産係数 (g生産物 / g菌体) — 増殖速度に比例する生産（一次代謝物によくみられる）
- `β`: 非増殖連動生産係数 (1/h) — 菌体濃度に比例する生産（二次代謝物・定常期の生産をモデル化）

`α=0` で純粋な非増殖連動型、`β=0` で純粋な増殖連動型になります。

#### NoProduct

生産物を追跡しない場合の恒等モデル（`dP/dt = 0`）。

---

### 4. 基質消費モデル (`src/biosim/models/substrate.py`)

#### 収率係数＋維持代謝モデル

```
dS/dt = −(1/Yxs)·(dX/dt) − ms·X   [ − (1/Yps)·(dP/dt) ]
```

- `Yxs`: 基質に対する菌体収率 (g菌体 / g基質)
- `ms`: 維持代謝係数 (g基質 / g菌体 / h) — 増殖に使われず細胞の維持のみに消費される基質
- `Yps`（任意）: 基質に対する生産物収率 (g生産物 / g基質) — 設定した場合のみ生産物消費項を追加（デフォルトは無効）

これは古典的な Pirt の式（維持代謝を考慮した収率モデル）です。なお、`ms` の項は基質濃度に依存せず菌体濃度にのみ比例するため、基質が枯渇した後もこの項自体は消費を続けようとします。ただし `BioreactorSimulation`（`src/biosim/simulation.py` の `_rhs`）が、`S <= 0` かつ正味の変化率が負になる場合にその変化率を0にクランプするため、実際のシミュレーション結果では `S` は0で底打ちし、負にはなりません。

---

### 5. 酸素消費モデル (`src/biosim/models/oxygen.py`)

#### 需要のみモデル (`OxygenDemandOnly`)

```
OUR = −(1/Yxo2)·(dX/dt) − mo2·X
```

- `Yxo2`: 酸素に対する菌体収率 (g菌体 / g O2)
- `mo2`: 維持代謝酸素係数 (g O2 / g菌体 / h)

酸素が律速にならないと仮定する場合に使用し、溶存酸素濃度 `C_O2` は状態変数として持たず、累積OUR（酸素摂取速度）のみを診断値として報告します。

#### kLa供給付きモデル (`OxygenWithKLa`)

需要式は同じ（OUR）に加えて、供給側（OTR: Oxygen Transfer Rate）を次式でモデル化し、溶存酸素濃度 `C_O2` を状態変数として追跡します。

```
OTR = kLa · (Cs* − C_O2)
dC_O2/dt = OTR + OUR + (F_in/V)·(C_O2_feed − C_O2)
```

- `kLa`: 総括容量物質移動係数 (1/h) — 通気・撹拌条件を反映するパラメータ
- `Cs*`: 飽和溶存酸素濃度 (mg/L)

`C_O2 < Cs*` のとき酸素供給（`OTR>0`）、過飽和（`C_O2 > Cs*`）のときは負（脱気側）になります。

基質と同様、需要側（OUR、特に維持代謝項 `-mo2*X`）は `C_O2` の値に関わらず作用しようとしますが、`BioreactorSimulation` が `C_O2 <= 0` かつ正味の変化率が負の場合に変化率を0にクランプするため、`C_O2` も0で底打ちし負にはなりません。ただしこれは供給不足を検知して`C_O2`の値自体を非負に保つ仕組みに過ぎず、増殖速度 `μ` 自体を酸素で律速するには増殖モデル側で `MonodGrowthO2`（上記「2. 菌体増殖モデル」参照）を選ぶ必要があります（既定の `MonodGrowth`/`Logistic`/`Gompertz` は酸素非依存のままです）。

---

## パラメータ選択と拡張性

各具象モデルは `@dataclass` で実装され、デフォルト値と物理的に妥当な範囲がdocstringに明記されています。パラメータの妥当性（正値・非負制約）は `__post_init__` で検証され、不正な値は `InvalidParameterError` を送出します。

モデルの選択は `src/biosim/models/registry.py` の `dict[str, type]` レジストリを介して行われ、新しい増殖モデル（例: Contois式）を追加する場合は、

1. `GrowthModel` を継承した `@dataclass` を実装
2. `GROWTH_MODELS` 辞書に1行追加

するだけでよく、シミュレーションオーケストレーターやGUIには一切変更が不要です（GUIのセレクトボックスはレジストリのキーから自動生成されます）。

## 数値解法

`scipy.integrate.solve_ivp` を `method="LSODA"`（デフォルト, `rtol=1e-6, atol=1e-9`）で使用しています。求解に失敗した場合（`sol.success=False`）は `IntegrationError` を送出します。

## 既知の制限（v1スコープ）

- 基質(`S`)・溶存酸素(`C_O2`)は、枯渇後は変化率がゼロにクランプされ0で底打ちします。ただし代謝物生産モデル（`LuedekingPiretProduct`）の非増殖連動項 `beta*X` は基質・酸素の枯渇によってゲートされないため、菌体(`X`)が存在する限り生産物(`P`)は枯渇後も増え続けます。これは既知の簡略化で、今後の課題です。
- 増殖速度 `μ` を溶存酸素で律速するには `MonodGrowthO2` を明示的に選ぶ必要があります。既定の `MonodGrowth`/`Logistic`/`Gompertz` は酸素非依存のままです。
- 実験データフィッティングは、同一バッチグループ内の全バッチで初期条件・運転モードを共有する前提です（バッチごとに異なる初期条件を自動設定する機能は含みません）。
- 複数シナリオの重ね合わせ比較・DB永続化・認証などはGUIのv1スコープ外です。
- 流加プロファイルCSV（`csv` プロファイル）でブレークポイントが密（時間間隔が非常に短い箇所が多数）な場合、ステップ状の不連続性によりLSODAの積分が遅くなることがあります。`max_step` 等のソルバーチューニングは現状GUI/APIから公開していません。
- パラメータ予測・次の実験点提案も、実験データフィッティングと同じく学習データ内の全バッチで同一モデル・同一初期条件/運転モードを共有する前提です。次の実験点の提案は候補数×モンテカルロサンプル数だけシミュレーションを実行するため、条件変数が多い・格子解像度やサンプル数を大きくすると計算時間が伸びます。
