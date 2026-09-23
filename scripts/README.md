# scripts/

論文再現ベンチマーク（`output/3DGS/benchmark_report/`）に関するスクリプト群。

## レポート成果物の生成

`output/3DGS/<dataset>/<scene>/<scene>/` 配下の学習ログから、論文原稿用の表と図を生成する5本。
`run_paper_benchmark.py` による全21シーンの学習が完了していることが前提。

| スクリプト | 用途 | 入力 | 出力 |
| --- | --- | --- | --- |
| `collect.py` | シーン別のGaussian数・メモリ使用量を集計 | `training_telemetry.json`, `vram_samples_*.json`, `validation_summary.json` | `output/3DGS/benchmark_report/report/scene_memory.csv` |
| `make_table.py` | 上記CSVをLaTeX表に整形 | `scene_memory.csv` | `output/3DGS/benchmark_report/report/scene_memory_table.tex` |
| `make_plot.py` | Gaussian数の推移を作図（2パネル、8×8インチ） | `train_log.jsonl` | `output/3DGS/benchmark_report/report/gaussian_count_plot.pdf` |
| `make_loss_plot.py` | 損失の推移を作図（2パネル、8×8インチ） | `train_log.jsonl` | `output/3DGS/benchmark_report/report/loss_plot.pdf` |
| `make_loss_psnr_plot.py` | 損失とPSNRの推移を作図（2×2パネル、12×8インチ） | `train_log.jsonl` | `output/3DGS/benchmark_report/report/loss_psnr_plot.pdf` |

### 実行順序

`make_table.py` のみ `collect.py` の出力に依存する。作図3本は互いに独立で、順不同・並列実行してよい。

```bash
# すべてリポジトリルートから実行する（入出力パスが相対パスのため）
python scripts/collect.py          # 1. CSVを生成
python scripts/make_table.py       # 2. CSVからLaTeX表を生成（collect.py の後）
python scripts/make_plot.py        # 3. 以下3本は順不同
python scripts/make_loss_plot.py
python scripts/make_loss_psnr_plot.py
```

### 依存パッケージ

`collect.py` と `make_table.py` は標準ライブラリのみで動作する。

作図3本は `matplotlib` と `numpy` を必要とする。`matplotlib` は `pyproject.toml` の依存に
含まれていないため、別途インストールするか使い捨ての venv を用意する。

```bash
python -m venv /tmp/plotenv && /tmp/plotenv/bin/pip install matplotlib
/tmp/plotenv/bin/python scripts/make_plot.py
```

日本語ラベルの描画に `Noto Sans CJK JP` を使用する。未インストールの環境では
`DejaVu Sans` にフォールバックし、日本語が豆腐（□）になる。

### 図の体裁を変更する場合

作図3本はシーンとデータセットの対応（`SCENES`）と配色（`COLOURS`）を各ファイルで
重複定義している。3つの図を並べたときに体裁が揃うよう、片方だけを編集せず
3本すべてに同じ変更を反映すること。

## その他のスクリプト

| スクリプト | 用途 |
| --- | --- |
| `train.py` | 単一シーンの学習 |
| `render.py` | 学習済みチェックポイントからの描画 |
| `evaluate.py` | PSNR / SSIM / LPIPS の評価 |
| `run_paper_benchmark.py` | 全21シーンのベンチマーク実行 |
| `paper_benchmark_dry_run.py` | ベンチマーク設定の事前検証 |
| `generate_paper_benchmark_report.py` | ベンチマーク結果のCSV/JSON/Markdown集計 |
| `generate_paper_benchmark_qualitative.py` | 定性比較図（GT / 7K / 30K）の生成 |
| `diagnose_adc_differential.py` | 適応的密度制御の差分診断 |
| `diagnose_official_renderer_gradient.py` | 公式実装との勾配比較診断 |
