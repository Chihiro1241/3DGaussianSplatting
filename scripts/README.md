# scripts/

論文再現ベンチマーク（`output/3DGS/benchmark_report/`）に関するスクリプト群。

## レポート成果物の生成

`output/3DGS/<dataset>/<scene>/<scene>/` 配下の学習ログから、論文原稿用の表と図を生成する4本。
`run_paper_benchmark.py` による全21シーンの学習が完了していることが前提。

| スクリプト | 用途 | 入力 | 出力 |
| --- | --- | --- | --- |
| `collect.py` | シーン別のGaussian数・メモリ使用量を集計 | `training_telemetry.json`, `vram_samples_*.json`, `validation_summary.json` | `output/3DGS/benchmark_report/report/scene_memory.csv` |
| `make_table.py` | 上記CSVをLaTeX表に整形 | `scene_memory.csv` | `output/3DGS/benchmark_report/report/scene_memory_table.tex` |
| `make_plot.py` | Gaussian数の推移を作図（2パネル、8×8インチ） | `train_log.jsonl` | `output/3DGS/benchmark_report/report/gaussian_count_plot.pdf` |
| `make_loss_psnr_plot.py` | 損失とPSNRの推移を別々の図に作図（各2パネル、8×8インチ） | `train_log.jsonl` | `output/3DGS/benchmark_report/report/loss_plot.pdf`, `psnr_plot.pdf` |

### 実行順序

`make_table.py` のみ `collect.py` の出力に依存する。作図2本は互いに独立で、順不同・並列実行してよい。

```bash
# すべてリポジトリルートから実行する（入出力パスが相対パスのため）
python scripts/collect.py          # 1. CSVを生成
python scripts/make_table.py       # 2. CSVからLaTeX表を生成（collect.py の後）
python scripts/make_plot.py        # 3. 以下2本は順不同
python scripts/make_loss_psnr_plot.py
```

### 依存パッケージ

`collect.py` と `make_table.py` は標準ライブラリのみで動作する。

作図2本は `matplotlib` と `numpy` を必要とする。`matplotlib` は `pyproject.toml` の依存に
含まれていないため、別途インストールするか使い捨ての venv を用意する。

```bash
python -m venv /tmp/plotenv && /tmp/plotenv/bin/pip install matplotlib
/tmp/plotenv/bin/python scripts/make_plot.py
```

日本語ラベルの描画に `Noto Sans CJK JP` を使用する。未インストールの環境では
`DejaVu Sans` にフォールバックし、日本語が豆腐（□）になる。

### 図の体裁を変更する場合

作図2本はシーンとデータセットの対応（`SCENES`）と配色（`COLOURS`）を各ファイルで
重複定義している。3つの図を並べたときに体裁が揃うよう、片方だけを編集せず
2本の両方に同じ変更を反映すること。

## その他のスクリプト

| スクリプト | 用途 |
| --- | --- |
| `train.py` | 単一シーンの学習 |
| `train_4d.py` | 動的シーンをフレームごとに学習（前フレームから warm-start） |
| `warmstart_trainer.py` | warm-start あり（`train_4d.py`）/ なし（`train.py` をフレームごと）を同条件で回すドライバ。`runs/4DGS_*.sh` の学習段 |
| `rebuild_manifest_4d.py` | 4D ランの `frames_4d.json` を `frame_NNNN/checkpoints` から作り直す（再開すると前半が消えるため） |
| `rendering/render_3d.py` | 学習済みチェックポイントからの描画 |
| `rendering/render_4d.py` | 4D ラン（`frame_NNNN/` ごとのチェックポイント）を 1 プロセスで全フレーム描画 |
| `rendering/make_video.py` | `render_4d.py` の連番 PNG をカメラごとの mp4 に |
| `rendering/benchmark_fps.py` | チェックポイントの描画 FPS（ラスタライズ 1 回の時間）を実測 |
| `evaluate.py` | PSNR / SSIM / D-SSIM / MS-SSIM / LPIPS の評価。`--checkpoint` でチェックポイントから、`--render-dir` で描画済み画像から |
| `warmstart_iteration_sweep.py` | warm-start の 1 フレームあたり iteration 数を振って比較 |
| `plot_warmstart_sweep.py` | 上の `results.csv` から図と飽和/ドリフト分析を生成 |
| `run_paper_benchmark.py` | 全21シーンのベンチマーク実行 |
| `paper_benchmark_dry_run.py` | ベンチマーク設定の事前検証 |
| `generate_paper_benchmark_report.py` | ベンチマーク結果のCSV/JSON/Markdown集計 |
| `generate_paper_benchmark_qualitative.py` | 定性比較図（GT / 7K / 30K）の生成 |

## warm-start の iteration 数探索

動的シーンを 1 フレームずつ学習し、2 フレーム目以降を前フレームの Gaussian から
warm-start するとき、1 フレームに何 iteration 割くべきかを決めるための 2 本。

```bash
# frame 1 は既存の 30,000 iter チェックポイントを全条件で共有する
python scripts/warmstart_iteration_sweep.py \
    --data data/dynamic/neu3d/cook_spinach/converted_4d \
    --frame1-checkpoint output/4DGS/neu3d/cook_spinach/cook_spinach_baseline_30k/\
frame_0001/checkpoints/iteration_00030000.pt \
    --output output/4DGS/neu3d/cook_spinach/warmstart_sweep_stage1 \
    --iters 0 100 250 500 1000 2000 5000 \
    --start-frame 2 --end-frame 4 --render-backend cuda

python scripts/plot_warmstart_sweep.py \
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
