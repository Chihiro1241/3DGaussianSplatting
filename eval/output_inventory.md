# output/ ディレクトリ調査レポート

調査日: 2026-09-15 / 調査者: Claude Code

> **2026-09-15 更新: `3DGS` / `4DGS` の 2 分類へ整理を実施済み。**
> 本レポート中のパス表記は特記なき限り**整理前**のもの。新旧の対応は「整理後のディレクトリ構成」を参照。
> **削除は一切行っていない**（`mv` による移動と、参照パスの書き換えのみ）。

## 整理後のディレクトリ構成

```
output/  (237 GB)
├─ 3DGS/                  52 GB   静的シーン（データセット別）
│  ├─ mipnerf360/               36 GB   9 scenes: bicycle bonsai counter flowers garden
│  │  ├─ runs/<scene>/                    kitchen room stump treehill
│  │  └─ qualitative/<scene>/
│  ├─ deepblending/            6.5 GB   2 scenes: drjohnson playroom
│  ├─ tandt/                   4.2 GB   2 scenes: train truck
│  ├─ nerf_synthetic/          3.1 GB   8 scenes: chair drums ficus hotdog lego
│  │                                              materials mic ship
│  ├─ implementation/          2.7 GB   実装検証 run015–022 ※データセット混在のため分割せず
│  ├─ blender/                  85 MB   合成シーン run001–014
│  └─ benchmark_report/         23 MB   全 21 シーン横断の集計
│     ├─ report/                          *.md / *.pdf / *.tex / figures/
│     ├─ results/                         benchmark_results.{csv,json} 等
│     ├─ manifest.json
│     └─ qualitative_index.json           （旧 qualitative/index.json）
│
├─ 4DGS/                 185 GB   動的シーン（データセット別）
│  ├─ neu3d/                   174 GB   Neu3D coffee_martini (16 dirs)
│  │  ├─ warmstart_neu3d_full/             58 GB   2,000iter warm-start 本番 (102f)
│  │  ├─ warmstart_neu3d_7000_trial/       21 GB
│  │  ├─ warmstart_fixed_gaussian_noreset/ 15 GB
│  │  ├─ warmstart_fixed_gaussian/         15 GB
│  │  ├─ baseline_neu3d_7000_trial/        14 GB
│  │  ├─ warmstart_neu3d_4000_trial/       13 GB
│  │  ├─ warmstart_fixed_gaussian_full/    13 GB   Gaussian固定 本番 (frame 2–54)
│  │  ├─ neu3d_coffee_martini_frame1/      11 GB   単一フレーム 30k リファレンス
│  │  ├─ baseline_neu3d_full/             5.7 GB   2,000iter baseline 本番 (100f)
│  │  ├─ baseline_neu3d_4000_trial/       5.0 GB
│  │  ├─ warmstart_neu3d_trial/           3.1 GB
│  │  ├─ baseline_neu3d_trial/            2.4 GB
│  │  ├─ warmstart_neu3d_7000_full/       2.2 GB
│  │  ├─ fixed_gaussian_frame1/           244 MB   ← *_full の frame_0001 実体
│  │  ├─ renders/                         206 MB
│  │  └─ neu3d_coffee_martini_smoke/      4.9 MB
│  ├─ hypernerf/               6.8 GB   4 scenes: train/ renders/ videos/
│  └─ dnerf/                   3.9 GB   8 scenes: experiments/ train/ renders/ videos/
│
└─ baseline_30k/         803 MB   🔴 実行中のため top 階層に据え置き
```

分類基準:

- **第1階層 = 手法** — 3DGS（静的シーン） / 4DGS（動的シーン）
- **第2階層 = データセット** — `data/` 配下の名称に揃えた
  - 3DGS: `mipnerf360` / `deepblending` / `tandt` / `nerf_synthetic` / `blender`
  - 4DGS: `neu3d`（coffee_martini） / `dnerf`（+ `dnerf_chunks`） / `hypernerf`
- **第3階層 = 役割** — 3DGS は `runs/` `qualitative/`、4DGS は `train/` `renders/` `videos/` `experiments/`

データセットの判定は推測ではなく、各 `frames_4d.json` / `driver.log` / `manifest.json` に記録された
`data/...` パスで確認した。

### 分割しなかったもの

| 対象 | 理由 |
|---|---|
| `3DGS/implementation/` | `run015_overnight_cpu_validation` が `drjohnson_cpu_regression_10` / `lego_cpu_regression_20` / `truck_cpu_regression_10` を 1 run 内に含むなどデータセット横断。run 番号順に意味がある検証シリーズのため現状維持 |
| `3DGS/benchmark_report/` | `benchmark_results.csv`（43 行 = 21 シーン × 7K/30K）や横断プロットなど、データセット別に割れない集計 |
| `4DGS/neu3d/` 配下 | 全 16 ディレクトリが単一シーン `coffee_martini`。これ以上のデータセット分割は不可 |

旧 `paper_3dgs/` `paper_4dgs/` は解体し、空になったディレクトリのみ `rmdir` した
（いずれもファイル 0 個を確認済み。**データの削除は一切していない**）。

`baseline_30k` は実行中（PID 3832、frame 1..300 × 30,000 iter、完了見込み約 55 時間）のため移動していない。
**ジョブ完了後に `mv output/baseline_30k output/4DGS/neu3d/` を実施すること。**
その際は `output/baseline_30k/frames_4d.json` 内の相対パスと、
`eval/baseline_30k_status.sh` / `eval/run_baseline_30k_pipeline.sh` / `eval/make_videos_4d.py` の
`output/baseline_30k` `output/renders/baseline_30k` `output/videos/baseline_30k` も併せて更新が必要。

### 整理に伴って実施した参照パスの更新

| 対象 | 件数 | 内容 |
|---|---:|---|
| `frames_4d.json`（+ `.bak101`） | 15 ファイル | 埋め込まれた相対パスを更新。全件 JSON 妥当性を検証済み |
| `manifest.json` / `results/dry_run.json` | 2 ファイル | `output_root` と 21 件 ×2 の `estimated_output_path`。**移動前の古い絶対パス `output/paper_3dgs` が残っていたのを併せて修正** |
| スクリプト（`.py` / `.sh`） | 延べ 27 ファイル | ハードコードパス、`SCENES` のパス表記、docstring の使用例 |

主なロジック変更:

```diff
# scripts/{collect,make_plot,make_loss_plot,make_loss_psnr_plot}.py
- RUNS = Path("output/3DGS/paper_3dgs/runs")
- SCENES = [("mipnerf360_bicycle", "Mip-NeRF360", "bicycle"), ...]
+ ROOT_3DGS = Path("output/3DGS")
+ SCENES = [("mipnerf360/runs/bicycle", "Mip-NeRF360", "bicycle"), ...]

# scripts/{generate_paper_benchmark_qualitative,generate_paper_benchmark_report,
#          paper_benchmark_dry_run}.py
+ DATASET_DIR = {"Mip-NeRF360": "mipnerf360", "Tanks&Temples": "tandt",
+                "Deep Blending": "deepblending", "Synthetic NeRF": "nerf_synthetic"}
- run_dir = root / "runs" / slug
+ run_dir = root_3dgs / DATASET_DIR[dataset] / "runs" / scene
```

**⚠️ `eval/run_all.sh` は環境変数名が変わった。** レンダリング出力が
`renders/<dataset>/<scene>` から `<dataset>/renders/<scene>` になったため、
単一 renders ルートを各手法のルートに組み替えた:

```diff
- RENDER_3DGS="${RENDER_3DGS:-$REPO_ROOT/output/3DGS/paper_3dgs/renders}"
- RENDER_4DGS="${RENDER_4DGS:-$REPO_ROOT/output/4DGS/paper_4dgs/renders}"
-     "$RENDER_3DGS/db/$scene"
+ ROOT_3DGS="${ROOT_3DGS:-$REPO_ROOT/output/3DGS}"
+ ROOT_4DGS="${ROOT_4DGS:-$REPO_ROOT/output/4DGS}"
+     "$ROOT_3DGS/deepblending/renders/$scene"
```

`RENDER_3DGS=...` / `RENDER_4DGS=...` で上書きしていた場合は `ROOT_3DGS=...` / `ROOT_4DGS=...` へ変更が必要。
`db` → `deepblending` に名称も統一した。

動作確認:

- 実行: `scripts/collect.py`（21 シーン）、`scripts/make_table.py`（21 シーン）
- 静的検証: 4 本の `SCENES` 計 84 パスの `train_log.jsonl` 実在、`DATASET_DIR` が 3 本で一致、
  manifest の 21 シーン × `runs`/`qualitative` 計 42 パスの実在、`estimated_output_path` 42 件の実在
- 構文: `py_compile` 9 本、`bash -n` 1 本、`frames_4d.json` 全件の JSON パース
- `scripts/make_plot.py` 等 3 本は **matplotlib 未インストールのため未実行**（環境側の問題で、本整理とは無関係）

**未更新（要判断）**: 各 `driver.log` は当時の実行記録なので履歴として旧パスのまま残した。
`README.md` / `eval/README.md` / `scripts/README.md` / `document/3DGS_実装設計書.md` 内の
output パス参照（約 20 箇所）も未更新。

---

## data/ の整理

`output/` と対になるよう `data/` も **static / dynamic** に分類した（合計 55 GB）。

```
data/  (55 GB)
├─ static/                19 GB   静的シーン
│  ├─ mipnerf360/              17 GB   9 scenes
│  ├─ nerf_synthetic/         1.3 GB   8 scenes
│  ├─ tandt_db/               740 MB   tandt/ + db/
│  └─ blender/                 35 MB   camera_poses_blender.json + view_000〜099.png
├─ dynamic/              5.9 GB   動的シーン
│  ├─ hypernerf/              5.6 GB   4 scenes
│  ├─ dnerf/                  260 MB   8 scenes
│  └─ dnerf_chunks/           172 KB   lego（フレーム分割版）
└─ neu3d/                 31 GB   🔴 実行中ジョブが読むため据え置き
```

`data/` 直下に散らばっていた `camera_poses_blender.json` と `view_*.png`（100 枚）は
Blender デモシーンのデータルートを兼ねていた（`camera_file` は `--data` 指定ディレクトリの直下を見る）。
`data/static/blender/` にまとめたため、**以後この シーンは `--data data/static/blender` で指定する**
（旧 run001–014 は `--data data` を使っていた）。

`data/neu3d`（31 GB）は実行中ジョブ（PID 3832）が
`data/neu3d/coffee_martini/converted_4d/frame_NNNN` を読み続けているため移動していない。
**ジョブ完了後に `mv data/neu3d data/dynamic/` を実施すること。**
その際は `eval/run_all.sh` の `$BASE_GT/neu3d/$scene/images`、
`output/4DGS/neu3d/*/frames_4d.json` の `"source"`（計 346 件のうち neu3d 分）、
`eval/convert_neu3d.py` / `eval/undistort_neu3d.py` の docstring も併せて更新が必要。

### data/ 移動に伴う参照パスの更新

| 対象 | 件数 | 内容 |
|---|---:|---|
| `manifest.json` / `results/dry_run.json` | 2 ファイル | `dataset_path` 21 件 ×2 を `data/static/<dataset>/<scene>` に更新 |
| `frames_4d.json`（dnerf lego） | 1 ファイル | `"source"` を `data/dynamic/dnerf_chunks/lego/frame_NNNN` に更新 |
| `eval/run_all.sh` | 1 ファイル | `$BASE_GT/<dataset>/...` → `$BASE_GT/{static,dynamic}/<dataset>/...` |
| `eval/evaluate.py` / `eval/convert_hypernerf.py` | 2 ファイル | docstring |

`scripts/paper_benchmark_dry_run.py` の `_discover()` は `data_root.rglob("*")` で
再帰的にシーンルートを探すため、階層が深くなっても**変更不要**（動作確認済み）。

動作確認:

- `manifest.json` / `dry_run.json` の `dataset_path` 計 42 件、`data_root` の実在
- `output/` 配下の全 `frames_4d.json` の `"source"` 計 **346 件すべて実在**
- `run_all.sh` の GT パス 7 種の実在（`nerf_synthetic` `mipnerf360` `tandt` `db` `dnerf` `hypernerf` `blender`）
- `_discover()` 相当の探索再現 — 21 シーン全て発見

**既知の事象（今回の整理とは無関係）**:

- `run_all.sh` の `$BASE_GT/neu3d/$scene/images` は元から存在しない。
  実際の Neu3D は `data/neu3d/coffee_martini/{converted,converted_4d,converted_nerf}/` の形で、
  `images/` という GT ディレクトリは未整備。
- `_discover()` は `lego` を `data/dynamic/dnerf/lego` と `data/static/nerf_synthetic/lego` の
  2 箇所で検出し AMBIGUOUS になる。`rglob` は深さに依存しないため**移動前から同じ状態**で、
  manifest 上の `synthetic_lego` は `AVAILABLE` として確定済み。

---

## ディレクトリ一覧

サイズは重複 inode（ハードリンク）を 1 回だけ数えた実容量。

| ディレクトリ名 | サイズ | frames | range | iter | 最終ckpt | 中間ckpt | 推定内容 |
|---|---:|---:|---|---:|---:|---:|---|
| `baseline_30k` | 0.00 GB | 1 | 1 | 30000 | 0.00 | 0.00 | 🔴 **実行中**（30k baseline, 1..300） |
| `baseline_neu3d_4000_trial` | 4.93 GB | 10 | 1–10 | 4000 | 1.67 | **3.26** | 4,000iter baseline 試行 |
| `baseline_neu3d_7000_trial` | 13.00 GB | 10 | 1–10 | 7000 | 2.43 | **10.57** | 7,000iter baseline 試行 |
| `baseline_neu3d_full` | 5.61 GB | 100 | 1–100 | 2000 | 5.61 | 0.00 | 2,000iter baseline 本番（100f 完走） |
| `baseline_neu3d_trial` | 2.31 GB | 10 | 1–10 | 5000 | 2.31 | 0.00 | 5,000iter baseline 試行（最初期） |
| `blender` | 0.08 GB | 0 | – | 5000 | 0.07 | 0.00 | 9/8 以前の Blender/合成データ実験 run001–014 |
| `fixed_gaussian_frame1` | 0.24 GB | 1 | 1 | 7000 | 0.24 | 0.00 | ⚠️ **`warmstart_fixed_gaussian_full` の frame_0001 実体**（下記参照） |
| `implementation` | 2.66 GB | 0 | – | 3000 | 1.97 | 0.19 | 9/8 以前の実装検証 run015–022 + diagnostics(1.4GB) |
| `neu3d_coffee_martini_frame1` | 10.81 GB | 0 | – | 30000 | 0.41 | **10.40** | 単一フレーム 30k リファレンス（snapshots 付き） |
| `neu3d_coffee_martini_smoke` | 0.00 GB | 0 | – | 30000 | 0.00 | 0.00 | スモークテスト（9/14 11:49） |
| `paper_3dgs` | **48.94 GB** | 0 | – | 30000 | 28.36 | **20.43** | 3DGS 論文再現ベンチ（mipnerf360/deepblending 等 runs 49GB） |
| `paper_4dgs` | 10.56 GB | 0 | – | 30000 | 5.85 | **3.95** | 4DGS 論文再現（train 6.9G / experiments 3.0G / renders / videos, mp4 24本） |
| `renders` | 0.43 GB | 0 | – | – | 0.00 | 0.00 | `fixed_gaussian` の test split レンダリング 156枚 + GT ミラー |
| `warmstart_fixed_gaussian` | 14.51 GB | 9 | 2–10 | 7000 | 2.07 | **12.43** | Gaussian固定 warm-start **reset有** 試行 |
| `warmstart_fixed_gaussian_full` | 12.03 GB | 53 | 2–54 | 7000 | 12.02 | 0.00 | Gaussian固定 warm-start 本番（52f 評価、frame53 で中断） |
| `warmstart_fixed_gaussian_noreset` | 14.51 GB | 9 | 2–10 | 7000 | 2.07 | **12.43** | Gaussian固定 warm-start **reset無** 試行 |
| `warmstart_neu3d_4000_trial` | 12.07 GB | 10 | 1–10 | 4000 | 2.97 | **9.10** | 4,000iter warm-start 試行 |
| `warmstart_neu3d_7000_full` | 2.14 GB | 8 | 1–8 | 7000 | 2.14 | 0.00 | 7,000iter warm-start 本番（frame7/8 で自動中断） |
| `warmstart_neu3d_7000_trial` | 20.82 GB | 10 | 1–10 | 7000 | 2.95 | **17.87** | 7,000iter warm-start 試行 |
| `warmstart_neu3d_full` | **57.58 GB** | 102 | 1–102 | 2000 | 57.58 | 0.00 | 2,000iter warm-start 本番（100f 評価、102f まで学習） |
| `warmstart_neu3d_trial` | 3.03 GB | 10 | 1–10 | 5000 | 3.03 | 0.00 | 5,000iter warm-start 試行（最初期） |
| **合計** | **236.25 GB** | | | | 133.74 | **100.63** | |

### 10 GB 超のディレクトリ（特記）

| ディレクトリ | サイズ | 備考 |
|---|---:|---|
| `warmstart_neu3d_full` | 57.58 GB | 論文の主要結果。Gaussian 数が単調増加（最終 1,532,128）するため 1 フレームあたり最大 ~1.5GB。中間 ckpt はゼロ（`checkpoint_interval == iterations`）なので圧縮余地は小さい |
| `paper_3dgs` | 48.94 GB | 3DGS 論文再現。`runs/` が 49GB、中間 ckpt 20.43 GB |
| `warmstart_neu3d_7000_trial` | 20.82 GB | 試行（10f）。**86% が中間 ckpt** |
| `warmstart_fixed_gaussian` / `_noreset` | 各 14.51 GB | 試行（9f）。**各 86% が中間 ckpt** |
| `baseline_neu3d_7000_trial` | 13.00 GB | 試行（10f）。**81% が中間 ckpt** |
| `warmstart_neu3d_4000_trial` | 12.07 GB | 試行（10f）。**75% が中間 ckpt** |
| `warmstart_fixed_gaussian_full` | 12.03 GB | Gaussian固定本番。中間 ckpt ゼロ |
| `neu3d_coffee_martini_frame1` | 10.81 GB | 単一フレーム 30k。**96% が中間 ckpt**（iteration_1000〜29000 の 29 個） |
| `paper_4dgs` | 10.56 GB | 4DGS 再現。mp4 24 本あり |


---

## ディレクトリ構造

`output/` は大きく **4 系統**に分かれる。第1階層 21 ディレクトリ、全体で 1,129 ディレクトリ（深さ分布: d1=21 / d2=381 / d3=459 / d4=215 / d5=39 / d6=35）。

```
output/  (236 GB)
│
├─ ① Neu3D 4D warm-start 実験（現行, 15 dirs, 約 174 GB）
├─ ② 単一フレーム参照        （2 dirs, 約 10.8 GB）
├─ ③ 論文再現ベンチ          （paper_3dgs / paper_4dgs, 約 59.5 GB）
└─ ④ レガシー実装検証        （blender / implementation, 約 2.7 GB）
```

### ① Neu3D 4D warm-start 実験 — 共通構造

`scripts/train_4d.py` / `eval/warmstart_trainer.py` の出力形式。**該当 15 ディレクトリすべてこの形**。

```
<実験名>/
├─ config.yaml            # 解決済み設定（baseline_* には無い）
├─ driver.log             # フレームごとの1行ログ
├─ frames_4d.json         # フレーム間の引き継ぎ記録
└─ frame_NNNN/            # ← 1フレーム1ディレクトリ（最大 102 個）
   ├─ checkpoints/
   │  ├─ iteration_XXXXXXXX.pt   # checkpoint_interval ごと
   │  └─ latest.pt               # 最終 iteration へのハードリンク
   ├─ config.yaml
   ├─ train_log.jsonl            # 損失・Gaussian 数の時系列
   └─ training_telemetry.json
```

`frame_NNNN/checkpoints/` の中身だけが実験ごとに異なり、これが容量差の正体。

| 型 | 例 | `checkpoints/` の中身 | 実質ファイル数 |
|---|---|---|---|
| 本番系（`*_full`） | `warmstart_neu3d_full/frame_0050` | `iteration_00002000.pt` + `latest.pt`（hardlink） | 1 個 |
| 試行系（`*_trial`） | `warmstart_neu3d_7000_trial/frame_0001` | `iteration_00001000`〜`00007000` の 7 個 + `latest.pt` | **7 個** |

15 ディレクトリの内訳（frame 範囲別）:

| frame 範囲 | ディレクトリ | サイズ | 備考 |
|---|---|---:|---|
| 1–102 | `warmstart_neu3d_full` | 57.6 GB | `frames_4d.json.bak101` あり |
| 1–100 | `baseline_neu3d_full` | 5.6 GB | |
| 2–54 | `warmstart_fixed_gaussian_full` | 12.0 GB | **frame_0001 が無い** |
| 1 | `fixed_gaussian_frame1` | 0.2 GB | **上記の frame_0001 の実体** |
| 1–8 | `warmstart_neu3d_7000_full` | 2.1 GB | |
| 1 | `baseline_30k` | – | 🔴 実行中 |
| 1–10 | `*_trial` × 6（7000/4000/5000 × warm/base） | 約 68 GB | |
| 2–10 | `warmstart_fixed_gaussian` | 14.5 GB | reset 有 |
| 2–10 | `warmstart_fixed_gaussian_noreset` | 14.5 GB | reset 無 |

### ② 単一フレーム参照

`frame_NNNN` 構造を持たない、単一シーン学習の出力形式。

```
neu3d_coffee_martini_frame1/            10.8 GB
├─ checkpoints/   iteration_00001000〜00030000（29個）+ iteration_00030000 + latest.pt
├─ snapshots/     index.json + iteration_*.npz     ← eval/snapshot_viewer.py の入力
├─ eval_test.json
├─ config.yaml / train_log.jsonl / training_telemetry.json

neu3d_coffee_martini_smoke/              4.9 MB   同形式のスモークテスト
```

### ③ 論文再現ベンチ

```
paper_3dgs/                              48.9 GB   ← 静的シーン（3DGS）
├─ runs/<scene>/                         49 GB     ← 21 シーン
│  ├─ checkpoints/iteration_{00007000,00030000}.pt
│  ├─ metrics/test_{00007000,00030000}.json
│  ├─ stdout.log / stderr.log / run_state.json
│  ├─ train_log.jsonl / training_telemetry.json
│  └─ vram_samples_attempt_01.json
├─ qualitative/<scene>/                  126 MB   + index.json
├─ report/   *.md, *.pdf, *.tex, figures/  23 MB
├─ results/  benchmark_results.{csv,json}, dataset_summary.csv,
│            dry_run.json, failures.json          104 KB
└─ manifest.json

paper_4dgs/                              10.6 GB   ← 動的シーン（4DGS）
├─ train/{dnerf,hypernerf}/<scene>/      6.9 GB    ← 12 シーン、paper_3dgs/runs と同形式
├─ experiments/{baseline,warmstart}/lego/ 3.0 GB   ← ① と同じ frame_NNNN 形式
├─ renders/{dnerf,hypernerf}/<scene>/    644 MB
└─ videos/{dnerf,hypernerf}/             123 MB    mp4 24 本
```

シーン一覧:
- `paper_3dgs/runs`: mipnerf360（bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill）, deepblending（drjohnson, playroom）, tandt（train, truck）, synthetic（chair, drums, ficus, hotdog, lego, materials, mic, ship）
- `paper_4dgs`: dnerf（bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex）, hypernerf（broom2, vrig-3dprinter, vrig-chicken, vrig-peel-banana）

### ④ レガシー実装検証（9/8 以前）

```
implementation/                          2.7 GB
├─ diagnostics/                          1.4 GB  comparisons, profiling, reports,
│                                                official_drjohnson_7k,
│                                                drjohnson_pre_adc_formal_run
├─ legacy_smoke/   smoke_* × 10          204 MB
├─ legacy_wrappers/ gpu_probe_20260824, gpu_adc_probe_20260824
├─ _campaigns/five_day_validation_20260824
└─ run015〜run022/  各 COMMAND.txt, RESULT.md, config.yaml, run/ または checkpoints/,
                    diagnostics/density_pre_event_*.{json,pt}, evaluation_*.json

blender/                                 85 MB
└─ run001〜run014/  checkpoints/, metrics/, renders/test/, config.yaml, train_log.jsonl
```

### 補足: レンダリング出力

```
output/renders/fixed_gaussian/           206 MB
├─ renders/frame_0002 〜 frame_0053      52 フレーム × 3 カメラ = 156 枚
└─ gt/     frame_0002 〜 frame_0053      GT ミラー
```
---

## 実験との対応

`eval/results/experiment_summary.csv`（10 実験）を軸に対応付けた。

| 実験（summary.csv の行） | output ディレクトリ | logs | eval/results |
|---|---|---|---|
| 2,000iter warm-start (100f) | `warmstart_neu3d_full`（102f 学習） | `warmstart_full.log` | `warmstart_full/neu3d_coffee_martini.csv`（302行） |
| 2,000iter baseline (100f) | `baseline_neu3d_full` | `baseline_full.log` | `baseline_full/neu3d_coffee_martini.csv`（302行） |
| 4,000iter warm-start (試行, 10f) | `warmstart_neu3d_4000_trial` | `warmstart_4000_trial.log` | （CSV なし / summary.csv のみ） |
| 4,000iter baseline (試行, 10f) | `baseline_neu3d_4000_trial` | `baseline_4000_trial.log` | （CSV なし） |
| 7,000iter warm-start (試行, 10f) | `warmstart_neu3d_7000_trial` | `warmstart_7000_trial.log` | （CSV なし） |
| 7,000iter baseline (試行, 10f) | `baseline_neu3d_7000_trial` | `baseline_7000_trial.log` | （CSV なし） |
| 7,000iter warm-start (本番, 7f) | `warmstart_neu3d_7000_full`（8f 保存） | `warmstart_7000_full.log` | （CSV なし） |
| Gaussian固定 reset有 (試行, 9f) | `warmstart_fixed_gaussian` | `warmstart_fixed_gaussian.log` | （CSV なし / `pixel_diff_cam00.csv` 300行が関連の可能性） |
| Gaussian固定 reset無 (試行, 9f) | `warmstart_fixed_gaussian_noreset` | `warmstart_fixed_gaussian_noreset.log` | （CSV なし） |
| Gaussian固定 warm-start (52f) | `warmstart_fixed_gaussian_full` **+ `fixed_gaussian_frame1`** | `fixed_gaussian_full.log`, `render_fixed_gaussian.log` | `fixed_gaussian.csv`（158行）, `output/renders/fixed_gaussian` |

### summary.csv に載っていない output ディレクトリ

| ディレクトリ | logs | 内容 |
|---|---|---|
| `baseline_30k` | `baseline_30k.log` | 🔴 実行中の 30k baseline |
| `baseline_neu3d_trial` / `warmstart_neu3d_trial` | `baseline_trial.log` / `warmstart_trial.log`, `warmstart_summary.log` | 5,000iter の最初期試行（10f）。summary.csv 未掲載 |
| `neu3d_coffee_martini_frame1` | （該当ログなし） | 単一フレーム 30k リファレンス。`eval_test.json`, `snapshots/` あり |
| `neu3d_coffee_martini_smoke` | （該当ログなし） | スモークテスト |
| `paper_3dgs` | – | 3DGS 論文再現ベンチ（9/11 以前、静的シーン） |
| `paper_4dgs` | `dnerf_batch.log`, `hypernerf_batch.log` | 4DGS 再現 → `dnerf_*.csv`（8本）, `hypernerf_*.csv`（4本）, `summary.txt` |
| `blender`, `implementation` | `warmstart_lego.log`, `warmstart_batch.log`, `baseline_lego_frame_000*.log`（全て空） | 9/8 以前の実装検証・診断 |
| `renders` | `render_fixed_gaussian.log` | Gaussian固定のレンダリング成果物 |

---

## 不明・要確認のディレクトリ

- **`fixed_gaussian_frame1`（0.24 GB）— 要注意**
  `warmstart_fixed_gaussian_full/` には **frame_0001 が存在しない**（frame_0002 から始まる）。
  `fixed_gaussian_frame1/driver.log` に
  `ValueError: existing resolved config does not match this run: output/warmstart_fixed_gaussian_full/config.yaml`
  というエラーがあり、config 不一致で本来の出力先に書けず別ディレクトリに退避した形跡。
  **52フレーム Gaussian固定実験の起点フレームなので、単独では削除できない可能性が高い。**
  再学習・再レンダリングとの関係を人間が確認すること。

- **`warmstart_neu3d_full` の frame_0101 / frame_0102**
  評価は 100 フレームまで（CSV 302 行 = 100f × 3 cam + ヘッダ）だが、学習は 102 フレームまで進んでいる。
  `frames_4d.json.bak101` という bak ファイルもあり、中断・再開の痕跡。扱いは要判断。

- **`neu3d_coffee_martini_frame1`（10.81 GB）**
  単一フレーム 30,000 iter のリファレンス。`eval_test.json` と `snapshots/`（npz）があり
  `eval/snapshot_viewer.py` / `eval/extract_snapshots.py` の入力と思われるが、
  summary.csv にも logs にも対応エントリがなく、現在も使うのか不明。

- **`implementation/diagnostics`（1.4 GB）**
  9/8 以前の診断データ。個別の中身は未確認。

- **`paper_3dgs` / `paper_4dgs`（合計 59.5 GB）**
  3DGS/4DGS 論文再現ベンチ。現在の warm-start 研究とは別系統で、
  `scripts/run_paper_benchmark.py` 等が参照する。成果物（report/qualitative/videos）は小さいが
  `runs/`・`train/` が大半を占める。

- **空ログ 5 本**: `logs/baseline_lego_frame_000{1..5}.log`（全て 0 バイト、9/8）

---

## 削除候補（判断は人間が行う / 本レポートでは削除していない）

### A. 中間チェックポイントのみの削除 — 最大 約 100 GB

各 `checkpoints/` から**最終 iteration 以外の `iteration_*.pt` を削除**する案。
最終成果（`latest.pt` とそのハードリンク先）と `train_log.jsonl` / `training_telemetry.json` は残るため、
指標・グラフの再生成には影響しない。学習途中からの再開のみができなくなる。

| 対象 | 削減量 |
|---|---:|
| `paper_3dgs` | 20.43 GB |
| `warmstart_neu3d_7000_trial` | 17.87 GB |
| `warmstart_fixed_gaussian` | 12.43 GB |
| `warmstart_fixed_gaussian_noreset` | 12.43 GB |
| `baseline_neu3d_7000_trial` | 10.57 GB |
| `neu3d_coffee_martini_frame1` | 10.40 GB |
| `warmstart_neu3d_4000_trial` | 9.10 GB |
| `paper_4dgs` | 3.95 GB |
| `baseline_neu3d_4000_trial` | 3.26 GB |
| `implementation` | 0.19 GB |
| **合計** | **約 100.6 GB** |

> ⚠️ `neu3d_coffee_martini_frame1` は `snapshots/*.npz` が iteration ごとの解析用途に見えるため、
> ckpt を消すと snapshot 再生成ができなくなる可能性がある。要確認。

### B. 試行（trial）ディレクトリごとの削除 — 約 70 GB

本番実験に置き換わった 10 フレーム規模の試行。
ただし summary.csv に結果が記載されているため、**再現性が必要なら残すべき**。
`train_log.jsonl` だけ別途退避してから消す、という選択肢もある。

- `warmstart_neu3d_7000_trial` (20.82 GB)
- `warmstart_fixed_gaussian` (14.51 GB) / `warmstart_fixed_gaussian_noreset` (14.51 GB)
- `baseline_neu3d_7000_trial` (13.00 GB)
- `warmstart_neu3d_4000_trial` (12.07 GB)
- `baseline_neu3d_4000_trial` (4.93 GB)
- `warmstart_neu3d_trial` (3.03 GB) / `baseline_neu3d_trial` (2.31 GB) — 5,000iter の最初期試行、summary.csv にも未掲載で最も削除しやすい

### C. 明らかに小さい・無害な整理

- `neu3d_coffee_martini_smoke`（4.9 MB）— スモークテストの残骸
- `logs/baseline_lego_frame_000{1..5}.log` — 0 バイトの空ログ 5 本
- `output/blender/` 配下の **macOS AppleDouble ゴミファイル `._*` が 89 個**（計 0.3 MB）。
  容量は無視できるが `._latest.pt` `._iteration_00005000.pt` など本物の ckpt と紛らわしいため、
  整理時に併せて削除する候補（`find output -name '._*'` で列挙できる）

### D. 別系統として退避を検討

- `paper_3dgs`（48.94 GB）/ `paper_4dgs`（10.56 GB）/ `blender`（85 MB）/ `implementation`（2.66 GB）
  いずれも 9/11 以前の、現在の 4D warm-start 研究とは別系統。
  外部ストレージへ退避すれば **約 62 GB** が空く。成果物（`report/`, `qualitative/`, `videos/`, `*.csv`）は
  合計 1 GB 未満なので、それだけ残して `runs/` `train/` `experiments/` を退避する案もある。

### 削除してはいけないもの

- 🔴 `output/baseline_30k/` — **実行中**
- `output/warmstart_neu3d_full`, `output/baseline_neu3d_full`, `output/warmstart_fixed_gaussian_full` — 論文の主要結果（中間 ckpt もゼロなので圧縮余地なし）
- `output/fixed_gaussian_frame1` — `warmstart_fixed_gaussian_full` の frame_0001 実体
- `eval/results/`, `eval/loss_logs*/`, `eval/loss_plots/`, `logs/` — いずれも小容量で再現困難

---

## 2026-09-19 追記: baseline 30k / 7k の追加と eval/ の整理

本レポート本文は 2026-09-15 時点 (output/ 237 GB) の記述。その後に以下を実施した。
**削除は一切行っていない**（`mv` と参照パスの書き換えのみ）。

### output/ への追加・移動

| 対象 | 移動元 | 移動先 | 容量 |
|---|---|---|---|
| baseline 30,000 iter × 300 フレーム | `output/baseline_30k` | `output/4DGS/neu3d/baseline_30k` | 122 GB |
| baseline 7,000 iter × 300 フレーム | `output/baseline_7k` | `output/4DGS/neu3d/baseline_7k` | 72 GB |
| レンダリング 12 ラン | `output/renders/*` | `output/4DGS/neu3d/renders/` | 3.3 GB |
| 動画 5 ラン | `output/videos/*` | `output/4DGS/neu3d/videos/` | 436 MB |
| レンダリング 3 ラン | `eval/renders/*` | `output/4DGS/neu3d/renders/` | 711 MB |

`output/` 直下は `3DGS` / `4DGS` の 2 つのみ。総容量 **436 GB**。

### eval/ の整理

- `eval/loss_logs_<variant>/` 11 個 → `eval/loss_logs/<variant>/` に集約
  （素の `eval/loss_logs/` は `eval/loss_logs/_default/` へ）
- `eval/renders/` (711 MB) → 上表のとおり `output/` 配下へ（成果物は output に統一）
- `eval/__pycache__/` を削除（再生成物）
- eval/ 直下のディレクトリ 17 → 5 (`gaussian_viz` `logs` `loss_logs` `loss_plots` `results`)

スクリプトの `.py` / `.sh` は **移動していない**。`evaluate_per_frame.py` が
`from eval.evaluate import ...` でパッケージパスに依存し、`.sh` / `.md` から
`eval/*.py` が約 40 箇所参照されているため、サブディレクトリへ移すと
両方が壊れる。整理の利得に見合わないと判断した。

### 書き換えた参照

移動で切れるパス参照を全て更新した（旧パス参照 0 件を確認済み）。

- `output/baseline_{30k,7k}` → `output/4DGS/neu3d/baseline_{30k,7k}`
- `output/renders/`, `output/videos/` → `output/4DGS/neu3d/` 配下
- `eval/loss_logs_<variant>` → `eval/loss_logs/<variant>`
- `eval/renders/` → `output/4DGS/neu3d/renders/`
- `eval/loss_logger.py` の `--out_dir` 既定値 → `eval/loss_logs/_default`
  （集約前と同じ場所に書かれるよう等価に保つ）

`frames_4d.json` は `output/<run>/frame_NNNN` という相対パスを保持しており移動で
無効になるため、`eval/rebuild_manifest_4d.py` で両ラン分を再生成した。

> 注: `run_full_neu3d_100.sh` / `run_trial_*.sh` / `make_compare_runs_video.py` は
> 本整理の実施者が作成したものではないが、旧パスを参照しており移動で壊れるため
> 参照のみ書き換えた。ロジックには手を入れていない。
