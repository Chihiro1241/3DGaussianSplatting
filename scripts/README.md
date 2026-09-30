# scripts/

3DGS / 4DGS の学習・描画・評価の入口と、実験ごとのスクリプト群。

| 場所 | 中身 |
| --- | --- |
| 直下 | 3DGS / 4DGS 共通の評価（`evaluate.py`） |
| `rendering/` | チェックポイントからの描画・動画化・FPS 計測 |
| `3DGS/` | 単一シーンの学習（`train_3d.py`）と、論文再現ベンチマーク（`output/3DGS/benchmark_report/`）の実行・集計・図表 |
| `4DGS/` | 動的シーンの学習（`train_4d.py`）と 4D ランの後処理。`4DGS/warmstart/` は warm-start の比較と iteration 数探索、`4DGS/tracking/` は点追跡とその評価 |

## 3DGS/

| スクリプト | 用途 |
| --- | --- |
| `3DGS/train_3d.py` | 単一シーンの学習 |
| `3DGS/benchmark/dry_run.py` | ベンチマーク設定の事前検証。学習はせず `manifest.json` を書く |
| `3DGS/benchmark/run.py` | `manifest.json` の全21シーンを順に学習・評価（中断したランは再開） |
| `3DGS/benchmark/report.py` | ベンチマーク結果のCSV/JSON/Markdown集計（`run.py` の最後にも呼ばれる） |
| `3DGS/benchmark/qualitative.py` | 定性比較図（GT / 7K / 30K）の生成 |
| `3DGS/figures/` | 論文原稿用の表と図（下記） |

### 3DGS/figures/ — 論文原稿用の表と図

`output/3DGS/<dataset>/<scene>/` 配下の学習ログから、論文原稿用の表と図を生成する4本。
`3DGS/benchmark/run.py` による全21シーンの学習が完了していることが前提。

| スクリプト | 用途 | 入力 | 出力 |
| --- | --- | --- | --- |
| `collect_scene_memory.py` | シーン別のGaussian数・メモリ使用量を集計 | `training_telemetry.json`, `vram_samples_*.json`, `validation_summary.json` | `output/3DGS/benchmark_report/report/scene_memory.csv` |
| `make_scene_memory_table.py` | 上記CSVをLaTeX表に整形 | `scene_memory.csv` | `output/3DGS/benchmark_report/report/scene_memory_table.tex` |
| `plot_gaussian_count.py` | Gaussian数の推移を作図（2パネル、8×8インチ） | `train_log.jsonl` | `output/3DGS/benchmark_report/report/gaussian_count_plot.pdf` |
| `plot_loss_psnr.py` | 損失とPSNRの推移を別々の図に作図（各2パネル、8×8インチ） | `train_log.jsonl` | `output/3DGS/benchmark_report/report/loss_plot.pdf`, `psnr_plot.pdf` |

対象シーンの一覧（`SCENES`）、データセットの順序と配色（`DATASETS`, `COLOURS`）、
パネル分け、入出力の場所は `scenes.py` にまとめてあり、4本ともここから読む。
シーンを足したり図の体裁を変えたりするときは `scenes.py` だけを直せば、表と図が揃ったまま変わる。

#### 実行順序

`make_scene_memory_table.py` のみ `collect_scene_memory.py` の出力に依存する。
作図2本は互いに独立で、順不同・並列実行してよい。

```bash
# すべてリポジトリルートから実行する（入出力パスが相対パスのため）
python scripts/3DGS/figures/collect_scene_memory.py      # 1. CSVを生成
python scripts/3DGS/figures/make_scene_memory_table.py   # 2. CSVからLaTeX表を生成（1 の後）
python scripts/3DGS/figures/plot_gaussian_count.py       # 3. 以下2本は順不同
python scripts/3DGS/figures/plot_loss_psnr.py
```

#### 依存パッケージ

`collect_scene_memory.py` と `make_scene_memory_table.py` は標準ライブラリのみで動作する。

作図2本は `matplotlib` と `numpy` を必要とする。`matplotlib` は `pyproject.toml` の依存に
含まれていないため、別途インストールするか使い捨ての venv を用意する。

```bash
python -m venv /tmp/plotenv && /tmp/plotenv/bin/pip install matplotlib
/tmp/plotenv/bin/python scripts/3DGS/figures/plot_gaussian_count.py
```

日本語ラベルの描画に `Noto Sans CJK JP` を使用する。未インストールの環境では
`DejaVu Sans` にフォールバックし、日本語が豆腐（□）になる。

## その他のスクリプト

| スクリプト | 用途 |
| --- | --- |
| `evaluate.py` | PSNR / SSIM / D-SSIM / MS-SSIM / LPIPS の評価。`--checkpoint` でチェックポイントから、`--render-dir` で描画済み画像から |
| `rendering/render_3d.py` | 学習済みチェックポイントからの描画 |
| `rendering/render_4d.py` | 4D ラン（`frame_NNNN/` ごとのチェックポイント）を 1 プロセスで全フレーム描画 |
| `rendering/make_video.py` | `render_4d.py` の連番 PNG をカメラごとの mp4 に |
| `rendering/benchmark_fps.py` | チェックポイントの描画 FPS（ラスタライズ 1 回の時間）を実測 |
| `4DGS/train_4d.py` | 動的シーンをフレームごとに学習（前フレームから warm-start） |
| `4DGS/train_4d_regularized.py` | `train_4d.py` に Dynamic 3D Gaussians の正則化を足したもの（下記） |
| `4DGS/rebuild_manifest_4d.py` | 4D ランの `frames_4d.json` を `frame_NNNN/checkpoints` から作り直す（再開すると前半が消えるため） |
| `4DGS/warmstart/warmstart_trainer.py` | warm-start あり（`train_4d.py`）/ なし（`train_3d.py` をフレームごと）を同条件で回すドライバ。`runs/4DGS_*.sh` の学習段 |
| `4DGS/warmstart/warmstart_iteration_sweep.py` | warm-start の 1 フレームあたり iteration 数を振って比較 |
| `4DGS/warmstart/plot_warmstart_sweep.py` | 上の `results.csv` から図と飽和/ドリフト分析を生成 |
| `4DGS/tracking/track_points.py` | 4D ランからクエリ点（3D 点か 2D 画素）の軌跡を作る（下記） |
| `4DGS/tracking/evaluate_tracking.py` | 予測軌跡と正解軌跡から 3D / 2D 追跡指標（MTE, δ, Survival）を出す |
| `4DGS/tracking/plot_tracks.py` | 2D 軌跡を 1 視点の画像に重ねる（予測は青、正解は赤） |

## 4DGS/warmstart/ — warm-start の iteration 数探索

動的シーンを 1 フレームずつ学習し、2 フレーム目以降を前フレームの Gaussian から
warm-start するとき、1 フレームに何 iteration 割くべきかを決めるための 2 本。

```bash
# frame 1 は既存の 30,000 iter チェックポイントを全条件で共有する
python scripts/4DGS/warmstart/warmstart_iteration_sweep.py \
    --data data/dynamic/neu3d/cook_spinach/converted_4d \
    --frame1-checkpoint output/4DGS/neu3d/cook_spinach/cook_spinach_baseline_30k/\
frame_0001/checkpoints/iteration_00030000.pt \
    --output output/4DGS/neu3d/cook_spinach/warmstart_sweep_stage1 \
    --iters 0 100 250 500 1000 2000 5000 \
    --start-frame 2 --end-frame 4 --render-backend cuda

python scripts/4DGS/warmstart/plot_warmstart_sweep.py \
    --sweep output/4DGS/neu3d/cook_spinach/warmstart_sweep_stage1 \
    --data data/dynamic/neu3d/cook_spinach/converted_4d
```

`--iters 0` は学習せず frame 1 のモデルを全フレームで評価する下限、
`--scratch` は各フレームを独立に通常学習する上限。評価は必ず held-out カメラ
（COLMAP ローダーが画像名ソート順で 8 枚ごとに除外するもの）だけで行う。

`results.csv` は `(条件, フレーム)` ごとに 1 行。中断しても同じ引数で再実行すれば
記録済みの行は飛ばして続きから再開する。

設定と git commit hash は `sweep_metadata.json` と各条件の `run_metadata.json`
に残る。

**注意**: 学習に使う `configs/neu3d/warmstart_sweep.yaml` は frame 2 以降専用。
密度制御・不透明度リセット・progressive SH・解像度 warm-up をすべて無効にして
ガウシアン数を固定し、位置 LR を固定値にしてある（iteration 数を変えたときに
学習率スケジュールまで変わるのを防ぐため）。これらの上書きは carry-over した
フレームにしか効かないので、frame 1 の静的学習の挙動は変わらない。

## 4DGS/train_4d_regularized.py — Dynamic 3D Gaussians の正則化

前フレームの結果を次フレームの初期値にする従来の方式（`train_4d.py`）はそのまま残し、
別スクリプト `train_4d_regularized.py` で Luiten et al., *Dynamic 3D Gaussians:
Tracking by Persistent Dynamic View Synthesis* (3DV 2024) の物理的正則化を足して学習する。
引数・データ読み込み・出力の構成は `train_4d.py` と共通（`train_4d.py` の関数を
import して使う）で、違いは正則化だけ。正則化の本体は
`extensions/4dgs/dynamic_regularization.py`。

設定ファイルには `dynamic_regularization` セクションを書き `enabled: true` にする
（例: `configs/neu3d/dynamic_regularization.yaml`）。取り違え防止のため、
`enabled: true` の設定を `train_4d.py` に渡すと、逆に無効の設定を
`train_4d_regularized.py` に渡すとエラーになる。セクションを省略した既存の
設定は従来どおり `train_4d.py` で動く。

```bash
# frame 1 は密度制御ありで別に学習したチェックポイントを渡す
python scripts/4DGS/train_4d_regularized.py \
    --data data/dynamic/neu3d/cook_spinach/converted_4d \
    --config configs/neu3d/dynamic_regularization.yaml \
    --output output/4DGS/neu3d/cook_spinach/dynamic_regularization \
    --start-frame 2 \
    --carry-over-checkpoint output/4DGS/neu3d/cook_spinach/cook_spinach_baseline_30k/\
frame_0001/checkpoints/iteration_00030000.pt \
    --render-backend cuda --disable-training-evaluation
```

carry-over したフレーム（frame 2 以降）で、画像損失に次の項が加わる。
近傍グラフは基準フレーム（frame 1 の最終状態）で 1 回だけ作り、以後使い回す。

| 項 | 意味 | 既定の重み |
| --- | --- | --- |
| `rigid` | 近傍の相対位置が、自分の回転で運ばれた剛体として動く（局所剛性） | 4.0 |
| `rotation` | 近傍どうしの回転の変化量が等しい | 4.0 |
| `isometry` | 近傍との距離が基準フレームの距離から変わらない（長期等長性） | 2.0 |
| `color` | DC 色が前フレームから変わらない | 0.01 |

近傍は各ガウシアンの中心から近い `num_neighbors`（既定 20）個、重みは
`exp(-neighbor_weight_lambda * d^2)`。重み・近傍数・重み関数は論文実装
（`JonathonLuiten/Dynamic3DGaussians` の `train.py`）と同じ値で、重みは
論文実装どおり平方根の内側に入る。さらに論文実装にならって次の 2 つも行う
（どちらも設定で切れる）。

- `velocity_initialization`: frame t を `mu_{t-1} + (mu_{t-1} - mu_{t-2})`
  （回転も同様に外挿して正規化）から始める。frame 2 は速度がまだ無いので従来どおり。
- `freeze_opacity_and_scale`: 不透明度とスケールの Adam 学習率を 0 にして
  frame 1 の値に固定する。

**制約と注意**

- 近傍グラフがガウシアン集合を固定するので、`adaptive_density_control` と
  `opacity_reset` は `false` でなければならない（違えば起動時にエラー）。1 本の
  設定を全フレームで共有するため、frame 1 は別に学習して `--start-frame 2` で渡す。
- 論文の前景/背景分離とそれに依存する項（背景固定・床・セグメンテーション描画）、
  カメラごとの色補正は、セグメンテーションマスクが無いので実装していない。
  正則化は全ガウシアンにかかる。
- `neighbor_weight_lambda` の 2000 は CMU Panoptic のメートル単位を前提にした値。
  COLMAP のスケールはシーンごとに違うので、近傍距離の典型値で重みが 0 に
  潰れていないか確認して調整すること。
- `train_log.jsonl` の `loss_total` は従来どおり画像損失だけ。正則化は
  `loss_regularization`（重み付き合計）と `loss_reg_<項>`（重みを掛ける前）に出る。
- `frames_4d.json` の各フレームに `dynamic_regularization` /
  `velocity_initialized` / `frozen_parameter_groups` が記録される。

**一括実行**: `runs/4DGS_regularized.sh <scene> <end_frame>` が学習から描画・評価・
動画・可視化・eval.md 生成までを `runs/4DGS_warmstart.sh` と同じ手順で回す。frame 1 は
学習せず、既存ラン（既定 `output/4DGS/neu3d/<scene>/<scene>_baseline_30k`、`FRAME1_RUN` で
変更可）の `frame_0001` をラン直下にシンボリックリンクして使う。再実行すれば完了済みの
最終フレームから自動で再開する。

**再開**: `<output>/dynamic_regularization/` に近傍グラフ（`neighbor_graph.pt`）と
次フレーム用の速度の起点（`motion_origin.pt`、毎フレーム上書き）が残る。途中で
止まったランは、同じ `--output` に対して `--start-frame F --carry-over-checkpoint
<frame F-1 の最終チェックポイント>` で再開すれば、中断しなかった場合と同じ結果に
なる。別ディレクトリの状態から再開するときは `--regularization-state DIR` を渡す。
`--start-frame 2` だけは状態が無くても、渡されたチェックポイントから近傍グラフを作る。

## 4DGS/tracking/ — 点追跡と追跡評価（Dynamic 3D Gaussians の Table 1）

Luiten et al. の Sec. 3 "Tracking with Dynamic 3D Gaussians" の方法で 4D ランから点の軌跡を作り、
Table 1 の 3D / 2D 追跡指標で評価する。追跡は `extensions/4dgs/point_tracking.py`、
指標は `extensions/4dgs/tracking_metrics.py` にあり、指標側は他手法の軌跡も同じ形式で評価できる。

```bash
# cam00 の 40 px 格子を frame 1 から追跡する
python scripts/4DGS/tracking/track_points.py \
    --run_dir output/4DGS/neu3d/cook_spinach/cook_spinach_dynreg_7k_250_300f \
    --data_dir data/dynamic/neu3d/cook_spinach/converted_4d \
    --query-camera cam00 --query-grid 40 \
    --out output/4DGS/neu3d/cook_spinach/cook_spinach_dynreg_7k_250_300f/tracking/tracks_cam00.npz

# 正解があるシーンの評価（シーンごとに --scene 名前 予測 正解）
python scripts/4DGS/tracking/evaluate_tracking.py \
    --scene juggle pred/juggle.npz gt/juggle.npz --unit-to-cm 100 --output-csv tracking.csv
```

入出力の `.npz` のキーは各スクリプトの先頭に書いてある。

**追跡の方法**

- 3D クエリは論文どおり。基準フレームで影響 `f_i(p) = sigmoid(o_i) exp(-½ (p-μ_i)ᵀ Σ_i⁻¹ (p-μ_i))` が
  最大のガウシアンのローカル座標で点を表し、各フレームでそのガウシアンと一緒に動かす。
  全ガウシアンで `f < --background-threshold`（既定 0.5）なら静的背景として固定する。
  最大の探索は近似なしの総当たりを GPU で分割して行う。
- 2D クエリは中央値深度で 3D 点にする。中央値深度は、光線の透過率が 0.5 を切るガウシアンの中心深度で、
  そのガウシアンを担当にする。論文の文面（平均深度と f の最大）ではなく、公式の深度描画器
  （`diff-gaussian-rasterization-w-depth`）と著者の説明（Dynamic3DGaussians issue #20）に合わせた。
  学習済みのガウシアンは平たく、6 割は不透明度が 0.5 未満なので、f ≥ 0.5 の規則では
  cook_spinach で 98% の点が背景になるため。
- カメラは全フレームで動かないものとし、基準フレームのカメラで投影する。

**指標の定義**

公式の評価コードは公開されていないので、PointOdyssey の参照評価（PIPs++ の `test_on_pod.py`）の、
論文当時の版に合わせた。

- 2D の誤差は x を 256/W 倍、y を 256/H 倍した「正規化 px」で測る。3D は cm（`--unit-to-cm`）。
- δ は閾値 1, 2, 4, 8, 16 ごとに、誤差が閾値未満の (軌跡, フレーム) の割合を出して平均する。frame 0 は除く。
- Survival は、誤差が閾値を超える最初のフレームまでの割合を軌跡とフレームで平均する。
  閾値は 2D が 50 正規化 px（現行の PIPs++ は 16）、3D が 50 cm。frame 0 を含む。
- MTE は軌跡ごとに誤差の中央値を取り、軌跡で平均する。frame 0 を含む。
- 有効なフレームは、正解が画像内（`[1, W-2] × [1, H-2]`）にあるフレーム。遮蔽されたフレームも数える。
- 2D はシーンの全カメラの軌跡をまとめて 1 つとして指標を出す。表の Mean はシーン平均
  （論文 Table 1 の Mean 列の値から検算して一致）。

## evaluate.py — 画質評価

| モード | 入力 | 出力 |
| --- | --- | --- |
| チェックポイント (`--checkpoint --data --split --output`) | チェックポイント + データセット（その場で描画） | 画像ごとの PSNR / SSIM / LPIPS を JSON |
| 画像 (`--render-dir --gt-dir --dataset`) | 描画済み PNG + GT PNG | `--output-csv` (`metrics.csv`) / `--json-out` (`summary.json`、カメラ別) / `--per-frame-csv` (`per_frame.csv`) |

```bash
python scripts/evaluate.py --dataset nerf_synthetic \
    --render-dir output/3DGS/nerf_synthetic/lego/renders \
    --gt-dir     data/static/nerf_synthetic/lego/test \
    --output-csv output/3DGS/nerf_synthetic/lego/results/metrics.csv
```

画像モードは、学習を再実行せずに測り直したいときや、4D のフレーム系列（`rendering/render_4d.py` の出力）を
まとめて測るときに使う。`--dataset` で指標の組み合わせが決まる（`dnerf` / `nerf_synthetic` / `colmap` は
PSNR・SSIM・LPIPS、`neu3d` は PSNR・D-SSIM・LPIPS、`hypernerf` は PSNR・MS-SSIM）。

- **実装はどちらのモードも共通。** PSNR は `evaluation/metrics.py`、SSIM / D-SSIM は `training/losses.py`、
  LPIPS は `LPIPSMetric`（VGG）を使うので、同じ画像なら同じ数値になる。MS-SSIM だけは本体に実装が無いので
  torchmetrics を使う（`pip install -e ".[eval]"`）。
- **旧 `eval/evaluate.py` の数値とは比べない。** 旧版は torchmetrics の SSIM を使っていたので、過去の
  `metrics.csv` / `per_frame.csv` とは SSIM / D-SSIM が小数第 2〜3 位で一致しない。
- **画像の対応付け**（`evaluation/images.py: collect_image_pairs`）は「相対パス一致 → ファイル名 (stem) 一致」の順。
  `render_3d.py` は画像名のベース名だけをフラットに書くので、GT がサブディレクトリにあっても
  ファイル名で引ける。NeRF Synthetic の `test/` に混ざる `*_depth_*` / `*_normal_*` は GT から除外する。
  4D はフレーム間でファイル名 (`cam00.png` など) が重なるので、`render_4d.py` が GT を
  `gt/frame_NNNN/` にミラーして相対パスで一致させる。
- **`--rgba-background` は学習 config の `data.rgba_background` と揃える。** 食い違うと透明背景の色が変わり、
  PSNR が大きく落ちる。
- D-SSIM は `1 - SSIM`（本体の `dssim_loss`）。文献によっては `(1 - SSIM) / 2` を指すので、他の数値と比べるときは注意。
- 完全一致で `+inf` になった PSNR は平均から除く。
