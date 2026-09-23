# eval/ — レンダリング画像ベースのオフライン評価

`scripts/render.py` が書き出した PNG と GT PNG を突き合わせて PSNR / SSIM /
MS-SSIM / D-SSIM / LPIPS を計算し、データセット単位で集計する。

## ツール索引

工程順。**自動**列が ● のものは `runs/3DGS.sh` / `runs/4DGS_baseline.sh` /
`runs/4DGS_warmstart.sh` が実行するので、個別に叩く必要は普通ない。
○ は手で使う道具。

### 1. データ変換（学習の前処理）

以下 3 本は `eval/` ではなく `data/` 直下に置いてある。

| ツール | 用途 | 主な入力 → 出力 | 自動 |
|---|---|---|---|
| `data/convert_neu3d.py` | Neu3D を COLMAP 形式へ | `--colmap_dir --frames_dir` → `--out_dir` | ● |
| `data/undistort_neu3d.py` | Neu3D の魚眼歪み補正 | `--frames_dir --colmap_dir` → `--out_dir` | ● |
| `data/convert_hypernerf.py` | HyperNeRF を COLMAP 形式へ | `--scene_dir` → `--out_dir` | ○ |

### 2. 学習・描画

| ツール | 用途 | 主な入力 → 出力 | 自動 |
|---|---|---|---|
| `warmstart_trainer.py` | 4D の warm-start 学習ドライバ | `--source_path --config` → `--output_dir` | ● |
| `render_4d.py` | 4D ランを 1 プロセスで全フレーム描画 | `--run_dir --data_dir` → `--out_dir` (`renders/` `gt/`) | ● |
| `benchmark_fps.py` | 描画 FPS の実測 | `--run_dir --data_dir` → 標準出力 | ○ |
| `rebuild_manifest_4d.py` | 壊れた `frames_4d.json` を実体から再生成 | `--run_dir` → 同ファイル | ○ |

### 3. 指標の算出

| ツール | 用途 | 主な入力 → 出力 | 自動 |
|---|---|---|---|
| `evaluate.py` | 画像対から PSNR/SSIM/D-SSIM/MS-SSIM/LPIPS | `--render_dir --gt_dir` → `--output_csv` (`metrics.csv`) と `--json_out` (`summary.json`、カメラ別) | ● |
| `evaluate_per_frame.py` | 同上をフレーム × カメラで展開 | 同上 → `--output_csv` (`per_frame.csv`) | ● |
| `gaussian_count_trend.py` | ガウシアン数・時間・VRAM の推移 | `--run_dir` → `--output_csv` (`gaussian_counts.csv`) | ● |
| `loss_logger.py` | `train_log.jsonl` を損失 CSV へ変換 | `--run_dir` → `--out_dir` (`loss_logs/`) | ○ |

### 4. 集計・比較・レポート

| ツール | 用途 | 主な入力 → 出力 | 自動 |
|---|---|---|---|
| `summarize.py` | 全データセットを論文値と並べる | `--results_dir output` → 標準出力 | ○ |
| `compare_runs.py` | 2 ランを突き合わせる（カメラ別 / フレームブロック別） | `--a --b` (`per_frame.csv` でも `metrics.csv` でも可) → 標準出力 | ○ |
| `summarize_warmstart.py` | warm-start と baseline の収束比較 | `--warm_dir --baseline_dir` (+ `--warm_run --baseline_run`) → 標準出力 | ○ |
| `make_eval_report.py` | ラン 1 本の `eval.md` を生成 | `--run_dir` (+ `--compare`) → `<run_dir>/eval.md` | ● |

### 5. 可視化・動画

| ツール | 用途 | 主な入力 → 出力 | 自動 |
|---|---|---|---|
| `visualize_gaussians.py` | チェックポイントのガウシアン分布を投影 | `--ckpt_dir` → `--out_dir` (PNG + HTML) | ● |
| `gaussian_viz_report.py` | 上の出力を 1 枚の HTML にまとめる | `--viz_dir` → `--out` | ● |
| `plot_loss.py` | 損失曲線の比較 HTML | `--log_dir --baseline_dir` → `--out_html` | ○ |
| `extract_snapshots.py` | 学習過程のスナップショット (npz) を抽出 | `--run` → npz | ○ |
| `snapshot_viewer.py` | フレーム × iteration の 2 軸ビューワー (streamlit) | `--run` → ブラウザ | ○ |
| `make_videos_4d.py` | 描画結果を mp4 に | `--render_root` → `--out_dir` | ● |
| `make_compare_runs_video.py` | 2 ランを並べた比較動画 | `--a_root --b_root --gt_root` → `--out_dir` | ○ |

`archive/` には退役したシェルスクリプトが置いてある（`run_all.sh` ほか）。
実行経路は `runs/` に移したので、過去の実行記録としてのみ残している。

### 統合の履歴

同じことをする道具が分かれていたので、以下を 1 本にまとめた。

| 廃止 | 統合先 |
|---|---|
| `summarize_warmstart_full.py` | `summarize_warmstart.py`（`--block` で試行用/本番用を切り替え） |
| `summarize_metrics_4d.py` | `compare_runs.py`（CSV の列から形式を自動判別） |
| `summarize_camera_metrics.py` | `evaluate.py --json_out` |

## リポジトリ本体の評価との使い分け

本体には既にチェックポイント直結の評価経路がある。**数値の正式な記録はそちらが正**。

- `scripts/evaluate.py` → `gaussian_splatting.evaluation.runner.evaluate_camera_set`
  （チェックポイント + データセットから直接 PSNR/SSIM/LPIPS-VGG を計算し JSON 出力）
- `scripts/run_paper_benchmark.py` / `generate_paper_benchmark_report.py`
  （21 シーンの学習〜評価〜CSV 集計を通しで実行）

`eval/` は **既に書き出された画像だけがある場合**（他実装の出力との比較、
学習を再実行せずに指標を測り直したい場合、4D の frame 系列を後段でまとめて
測る場合）に使う補助経路である。SSIM 実装が本体 (`training/losses.py`) と
torchmetrics で異なるため、**本体の数値と小数第 2〜3 位で一致しないことがある**。
LPIPS backbone は本体に合わせて既定 `vgg`（`--lpips_net` で変更可）。

## 依存関係

```bash
pip install -e ".[eval]"     # torchmetrics[image] + scikit-image
```

torchmetrics が無い場合は scikit-image へフォールバックする（その場合
MS-SSIM と LPIPS は `nan` になる）。

## 使い方

```bash
# 1. test view を書き出す
python scripts/render.py \
    --data data/nerf_synthetic/lego \
    --checkpoint output/3DGS/nerf_synthetic/lego/checkpoints/iteration_00030000.pt \
    --split test --render-backend cuda \
    --output output/3DGS/nerf_synthetic/lego/renders

# 2. 評価
python eval/evaluate.py \
    --dataset nerf_synthetic \
    --render_dir output/3DGS/nerf_synthetic/lego/renders \
    --gt_dir     data/nerf_synthetic/lego/test \
    --output_csv output/3DGS/nerf_synthetic/lego/results/metrics.csv

# 3. 一括実行 + 集計
bash eval/archive/run_all.sh
python eval/summarize.py --results_dir output
```

環境変数 `BASE_RENDER` / `BASE_GT` / `OUTPUT_DIR` / `DEVICE` / `PYTHON` で
`archive/run_all.sh` のパスを上書きできる。

## 実行スクリプトは `runs/` の 3 本

学習から評価までの通し実行は、`runs/` の次の 3 本に集約した。
`eval/` にあったシェルスクリプトは `eval/archive/` へ退避してある（削除はしていない）。

| スクリプト | 用途 |
|---|---|
| `../runs/3DGS.sh` | 静的シーン（nerf_synthetic / mipnerf360 / tandt / deepblending） |
| `../runs/4DGS_baseline.sh` | Neu3D をフレームごとに独立学習（warm-start なし） |
| `../runs/4DGS_warmstart.sh` | Neu3D を warm-start で学習（前フレームを引き継ぐ） |

いずれも `STAGES` で工程を絞れ、`DRY_RUN=1` でコマンドの確認だけができる。

## 学習過程ビューワー (フレーム × iteration)

`snapshot_viewer.py` は、ガウシアンの**中心座標**と**不透明度**、そして**個数**の
推移を、フレーム軸と iteration 軸の2軸でアニメーション再生する。
レンダリング画像ではなくパラメータそのものを見るため、densify/prune が
どこで何を増やしたか、warm-start でガウシアンが実際に動いているのか止まって
いるのかを直接確認できる。

| ファイル | 役割 |
|---|---|
| `extract_snapshots.py` | 既存の `checkpoints/iteration_*.pt` から `snapshots/*.npz` を抽出 |
| `snapshot_viewer.py` | Streamlit + Plotly のビューワー本体 |

### 依存関係

```bash
pip install -e ".[viz]"     # streamlit + plotly
```

### スナップショットの作り方

**A. 学習中に記録する**（`scripts/train.py` と `scripts/train_4d.py` の両方で使える）

```bash
python scripts/train_4d.py \
    --data data/neu3d/coffee_martini --config configs/neu3d/base.yaml \
    --output output/scene_4d \
    --snapshot-interval 250 \
    --snapshot-iterations 0,50,100 \
    --snapshot-max-points 20000
```

| オプション | 意味 |
|---|---|
| `--snapshot-interval N` | N iteration ごとに記録。`0`（既定）で無効＝従来どおりの挙動 |
| `--snapshot-iterations LIST` | interval とは別に必ず記録する iteration。序盤を細かく見たいとき |
| `--snapshot-max-points K` | 1コマあたり K 点へ等間隔で間引く（既定 20000）。`0` で全ガウシアン |

出力は `<run>/snapshots/iteration_XXXXXXXX.npz` と `index.json`。4D ランでは
フレームごとに `frame_XXXX/snapshots/` ができ、これがそのまま再生の2軸になる。
学習開始時点（iteration 0）と最終 iteration は interval に関わらず必ず記録する。

**B. 学習済みの run から後から抽出する**

`--snapshot-interval` を付け忘れた run や、すでに完了している `output/` 配下の
run は、チェックポイントから抽出できる。再学習は不要。

```bash
python eval/extract_snapshots.py --run output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial
python eval/extract_snapshots.py --run output/4DGS/neu3d/coffee_martini/neu3d_coffee_martini_frame1 --stride 2
```

`--run` は単一シーンの run でも 4D run root でもよい（`frame_*/` を自動で走査）。
`checkpoint_interval` の粒度でしかコマが取れない点だけ A より粗い。

### ビューワーの起動

```bash
streamlit run eval/snapshot_viewer.py -- --run output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial
```

`--run` の前の `--` は必須（Streamlit 自身の引数と区別するため）。省略した場合は
サイドバーの入力欄にパスを入れる。

再生軸は3通り。単一フレームの run では `iteration` のみになる。

| 再生軸 | 動き |
|---|---|
| `iteration` | フレームを固定し、学習の進行を再生する |
| `frame` | iteration を固定し、時系列方向を再生する |
| `frame x iteration` | 全グリッドをフレーム優先で連結して再生する |

`frame` 軸では、そのフレームに同じ iteration が無ければ最も近いコマで代替する
（フレームごとに iteration 数が違う構成に対応するため）。

再生は Plotly のネイティブ animation で行うため、再生中にサーバとの往復が
発生しない。視点はドラッグで回転でき、コマが進んでも維持される。

### データ量

パラメータ3種だけを float32 で持つため、チェックポイントより桁違いに軽い。
実測（`output/4DGS/neu3d/coffee_martini/neu3d_coffee_martini_frame1`、30 コマ、60 万ガウシアン、20000 点へ間引き）:

| | サイズ |
|---|---|
| `checkpoints/*.pt` 30 本 | 約 3.1 GB |
| `snapshots/*.npz` 30 コマ | 8.3 MB |

### 座標軸について

最適化はシーン本体から数百単位離れたガウシアンを毎回わずかに残す。真の
min/max に軸を合わせるとシーン本体が点に潰れるため、`index.json` には各軸の
中央 99% も `robust_bounds` として記録し、ビューワーは既定でこちらを使う。
サイドバーの「外れ値を除いて軸を決める」で切り替えられる。軸の固定を外すと
コマごとに自動スケールし、再生中に視点が飛ぶ。

### 間引きと個数の関係

`--snapshot-max-points` による間引きは**描画用**であり、`index.json` の
`num_gaussians` には常に間引き前の真の個数が入る。したがって個数プロットは
間引き設定に関わらず正確である。間引きは等間隔ストライド（乱択ではない）で、
コマ間で同じ領域を引き続けるため再生時のちらつきが小さい。ただし densify/prune
で配列の並びは変わるので、点の対応は厳密ではない。

## 画像の対応付け

`render.py` は `image_name` の**ベース名のみ**をフラットに書き出す
（`evaluation/runner.py: render_camera_set`）。一方 GT はデータセット形式ごとに
配置が異なる：

| 形式 | GT パス |
|---|---|
| NeRF Synthetic | `data/nerf_synthetic/<scene>/test/r_*.png` |
| COLMAP | `<scene>/images{,_2,_4}/` |
| Blender (`data/` 直下) | `data/view_*.png` |

そのため `collect_image_pairs` は「相対パス一致 → ベース名一致」の順で解決する。
NeRF Synthetic の `test/` に混在する `r_*_depth_*.png` / `r_*_normal_*.png` は
GT 索引から除外する。

## データセット定義と既知の注意点

- **本リポジトリは 4D-GS (Wu et al., 2024) の実装ではない。**
  `extensions/4dgs/` はフレームごとに 3DGS を回してガウシアンパラメータを
  次フレームへ引き継ぐ方式で、HexPlane 時空間エンコーダも多頭変形デコーダも持たない。
  `dnerf` / `hypernerf` / `neu3d` の指標定義と論文値は**到達目標として**
  置いてあり、対応データも現時点で `data/` に存在しない。
- **D-SSIM の定義**：本スクリプトは `1 - SSIM` を用いる。文献によっては
  `(1 - SSIM) / 2` を指す。論文 Table 3 の値と比較する際は定義差に注意。
- PSNR が完全一致で `+inf` になった画像は平均から除外する。

## Neu3D (Neural 3D Video) — coffee_martini

`data/convert_neu3d.py` が Neu3D を本体の読める形へ変換する。全手順:

### 0. 前提

`ffmpeg` と `colmap` は conda env `3dgs` に入れてある
(`conda install -n 3dgs -c conda-forge colmap`)。
このビルドは CUDA 無しなので SIFT は CPU 実行になる (18 枚なら数十秒)。

### 1. 取得

公式は GitHub Releases 配布 (S3/aws cli は不要)。

```bash
mkdir -p data/neu3d && cd data/neu3d
curl -L -o coffee_martini.zip \
  https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip
python3 -c "import zipfile; zipfile.ZipFile('coffee_martini.zip').extractall('.')"
```

中身: `cam{00,01,02,04..14,16,18,19,20}.mp4` の **18 台** (cam03/15/17 は公式が
不良ストリームとして除外済み) と `poses_bounds.npy` (LLFF 形式, (18,17))。
2704x2028 / 300 フレーム / 30fps。

### 2. フレーム展開 (約 21 GB)

```bash
for mp4 in data/neu3d/coffee_martini/cam*.mp4; do
  cam=$(basename "$mp4" .mp4)
  mkdir -p "data/neu3d/coffee_martini/frames/$cam"
  ffmpeg -y -loglevel error -i "$mp4" -start_number 1 \
    "data/neu3d/coffee_martini/frames/$cam/frame_%04d.png"
done
```

`-q:v 1` は PNG では無効 (MJPEG 用) なので付けない。PNG は元々可逆。

### 3. COLMAP (第 1 フレームの多視点画像)

公式 3DGS の `convert.py` と同じく、SIMPLE_RADIAL で解いてから
`image_undistorter` で PINHOLE 化する。本体の COLMAP ローダーは
**PINHOLE のみ対応**なのでこの手順は必須。

```bash
mkdir -p data/neu3d/coffee_martini/colmap_input
for d in data/neu3d/coffee_martini/frames/*/; do
  cp "${d}frame_0001.png" "data/neu3d/coffee_martini/colmap_input/$(basename $d).png"
done

C=data/neu3d/coffee_martini/colmap
colmap feature_extractor --database_path $C/database.db \
  --image_path data/neu3d/coffee_martini/colmap_input \
  --ImageReader.single_camera 0 --ImageReader.camera_model SIMPLE_RADIAL \
  --SiftExtraction.use_gpu 0
colmap exhaustive_matcher --database_path $C/database.db --SiftMatching.use_gpu 0
mkdir -p $C/sparse && colmap mapper --database_path $C/database.db \
  --image_path data/neu3d/coffee_martini/colmap_input --output_path $C/sparse
colmap image_undistorter --image_path data/neu3d/coffee_martini/colmap_input \
  --input_path $C/sparse/0 --output_path $C/undistorted --output_type COLMAP
```

実測 (coffee_martini): 18/18 登録、点群 6,859 点、平均再投影誤差 0.80px。
推定焦点距離 1454-1468px は `poses_bounds.npy` の 1460.75 と 0.5% 以内で一致する。

### 4. 変換

```bash
python data/convert_neu3d.py \
  --colmap_dir   data/neu3d/coffee_martini/colmap/undistorted/sparse \
  --images_dir   data/neu3d/coffee_martini/colmap/undistorted/images \
  --out_dir      data/neu3d/coffee_martini/converted \
  --nerf_out_dir data/neu3d/coffee_martini/converted_nerf \
  --downscale 2 \
  --poses_bounds data/neu3d/coffee_martini/poses_bounds.npy
```

**出力が 2 系統ある理由**は `convert_neu3d.py` の docstring を読むこと。要点:
`load_dataset` は `transforms_train.json` を先に見るので、同一ディレクトリに
両方置くと COLMAP 経路が遮られ **SfM 点群が捨てられる**。
学習には `converted/` (COLMAP 配置) を使うこと。

`--poses_bounds` は検証専用。カメラ中心の正規化距離行列を公式ポーズと比べる。
coffee_martini では平均相対誤差 0.0023 / 最大 0.0105 だった。

### 5. 学習 (第 1 フレーム)

```bash
python scripts/train.py \
  --data   data/neu3d/coffee_martini/converted \
  --config configs/neu3d/base.yaml \
  --output output/4DGS/neu3d/coffee_martini/neu3d_coffee_martini_frame1 \
  --image-directory images_2 \
  --render-backend cuda
```

* `--image-directory images_2` = 1341x1006 (Neu3D 論文系の標準解像度)。
  `features.resolution_warmup` が `data.resolution_scale=1.0` を要求するため、
  縮小は読み込み時ではなく `--downscale 2` で事前に済ませてある。
* `--render-backend cuda` は**必須級**。reference (PyTorch) 実装はこの解像度で
  4.5 秒/iter (30k で約 37 時間)、CUDA は 0.009 秒/iter で約 500 倍速い。
* 分割は本体の 8 枚ごと固定規則により train 15 台 / test 3 台
  (cam00, cam09, cam19)。cam00 は公式が held-out 指定している中央参照カメラ。

### 未了 / 注意

* 歪み補正は第 1 フレームの 18 枚にしか掛かっていない。全 300 フレームを使う
  場合は同じ intrinsics で残り 5382 枚も undistort する必要がある。
* `converted_nerf/` は NeRF Synthetic 形式だが SfM 点群を使えない。
  形式互換の確認用で、品質比較には使わないこと。

## Neu3D warm-start 実験 (coffee_martini)

### 全フレームの undistort

第 1 フレームと**同じ `colmap image_undistorter` バイナリ**を全フレームに適用する。
OpenCV で書き直さないこと: image_undistorter は出力解像度をカメラごとに
「黒縁最小」で決めており (2674x2005 〜 2698x2023)、再現し損ねると sparse の
寸法と画素がずれる。

```bash
python data/undistort_neu3d.py \
    --frames_dir data/neu3d/coffee_martini/frames \
    --colmap_dir data/neu3d/coffee_martini/colmap/sparse/0 \
    --sparse_dir data/neu3d/coffee_martini/colmap/undistorted/sparse \
    --out_dir    data/neu3d/coffee_martini/converted_4d \
    --downscale  2 --workers 8 \
    --verify_against data/neu3d/coffee_martini/converted/images_2
```

* `--colmap_dir` は **歪みあり** (SIMPLE_RADIAL) — image_undistorter の入力。
* `--sparse_dir` は **undistort 済み** (PINHOLE) — 各 frame_NNNN/sparse/0 に置く方。
  本体の COLMAP ローダーは PINHOLE しか受け付けないので、ここを取り違えると
  `unsupported COLMAP camera model 'SIMPLE_RADIAL'` で落ちる。
* `--verify_against` は第 1 フレームを再生成して既存 images_2/ と
  バイト比較する。18/18 一致することを確認済み。

出力配置 (scripts/train_4d.py が期待する形):

```
converted_4d/
  frame_0001/{sparse/0, images/cam00.png ...}   <- sparse は実体
  frame_0002/{sparse/0 -> ../../frame_0001/sparse/0, images/...}
  ...
```

カメラが静止したリグなのでポーズはフレーム間で不変。sparse は 1 つを共有する。

### warm-start あり / なし

フレーム間の受け渡しは既存の `scripts/train_4d.py` + `extensions/4dgs/trainer_4d.py` が
実装済み。**ガウシアンパラメータのみ引き継ぎ、optimizer の Adam モーメント・
位置 LR スケジュール・ADC 統計は毎フレーム作り直す**
(`build_frame_training_state`、検査は `assert_frame_state_is_reset`)。
よって新しい学習ループは書かず、`eval/warmstart_trainer.py` は既存の入口を
呼ぶだけのドライバにしてある。`scripts/train.py` も `scripts/train_4d.py` も未変更。

```bash
# warm-start あり
python eval/warmstart_trainer.py \
    --source_path data/neu3d/coffee_martini/converted_4d \
    --output_dir  output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial \
    --config      configs/neu3d/trial_5000.yaml \
    --start_frame 1 --end_frame 10 \
    --image-directory images --render-backend cuda

# baseline (フレームごとに独立学習)
python eval/warmstart_trainer.py ... --no_warmstart
```

`scripts/train_4d.py --subsequent-frame-iterations` は **2 フレーム目以降にしか効かない**
(`_frame_config` が `carried_over` のときだけ上書きする)。第 1 フレームも含めて
揃えたいので、`training.iterations` を目的の値にした設定ファイルを渡す方式に
している (`configs/neu3d/trial_5000.yaml`)。

注意: その設定では `densify_until_iteration: 15000` が学習長 5000 を上回るため
密度制御が最後まで止まらない。warm-start / baseline の両方に等しく効くので
比較の妥当性は保たれるが、ガウシアン数の推移は必ず報告すること。

### 損失曲線

```bash
python eval/loss_logger.py --run_dir output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial \
    --out_dir output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial/loss_logs
python eval/loss_logger.py --run_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_trial \
    --out_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_trial/loss_logs
python eval/plot_loss.py \
    --log_dir      output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial/loss_logs \
    --baseline_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_trial/loss_logs \
    --out_html     output/4DGS/neu3d/coffee_martini/loss_plots/neu3d_coffee_martini_trial.html
```

`plot_loss.py` の「収束 iter」(損失が初期値の 10% に落ちた iter) は
**run 内の相対量**であり run 間で直接比較できない。warm-start は初期損失
そのものが低いためで、スクリプト自身も HTML に注記を出す。

### 数値サマリー

```bash
python eval/summarize_warmstart.py \
    --warm_dir     output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial/loss_logs \
    --baseline_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_trial/loss_logs \
    --warm_run     output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial \
    --baseline_run output/4DGS/neu3d/coffee_martini/baseline_neu3d_trial
```

**「収束 iter = 損失が初期値の 10%」は Neu3D では使えない**。`log_interval: 100`
なので最初の記録点が iter 100 で、そこまでに損失が大きく落ちきっており、
その 10% には両アームとも到達しない (= 「到達せず」と出る)。
iter 数を決めるときは代わりに「自身の最終損失に対する途中 iter の比」を見ること。

**frame_0001 はノイズ下限の測定に使える**。第 1 フレームは warm-start / baseline
どちらも SfM 点群から同一 config・同一 seed で学習するので、理論上は同一結果に
なるはずだが、CUDA ラスタライザ backward の atomicAdd が非決定なため一致しない。
coffee_martini では損失で 0.0006 (2.8%)、ガウシアン数で 0.9% ずれた。
これより小さい差は有意でないと見なすこと。

## 300 フレーム実験 (2000 iter/frame)

設定は `configs/neu3d/warmstart_2000.yaml` (neu3d/base.yaml から
`iterations: 2000` / `densify_until_iteration: 1500` / `checkpoint_interval: 2000`)。
checkpoint_interval を 2000 にしたのは 1 フレーム 2 個だと 600 フレームで
120GB になるため。学習結果には影響せず、両アームで同一。

```bash
# warm-start あり (約 7-8 時間)
python eval/warmstart_trainer.py \
    --source_path data/neu3d/coffee_martini/converted_4d \
    --output_dir  output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full \
    --config      configs/neu3d/warmstart_2000.yaml \
    --start_frame 1 --end_frame 300 \
    --image-directory images --render-backend cuda \
    --disable-training-evaluation

# baseline (約 3 時間)
python eval/warmstart_trainer.py ... --no_warmstart \
    --output_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_full
```

### 途中再開

```bash
# 書きかけのフレームディレクトリを先に消す (exist_policy=error で落ちるため)
python3 -c "
import json, shutil
from pathlib import Path
d=json.load(open('output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full/frames_4d.json'))
done={int(r['frame']) for r in d['frames'] if r['status']=='COMPLETED'}
for p in sorted(Path('output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full').glob('frame_*')):
    if int(p.name.split('_')[1]) not in done: shutil.rmtree(p)
"
python eval/warmstart_trainer.py ... \
    --start_frame 48 \
    --carry_over_checkpoint output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full/frame_0047/checkpoints/latest.pt
```

`--carry_over_checkpoint` を**忘れると再開フレームが SfM 点群から始まり**、
引き継ぎの連鎖が切れる。再開後のログが `carried_over, <前フレームの終了数>
Gaussians` になっていることを必ず確認すること。

### ガウシアン数が飽和しない件 (既知)

2000 iter では `opacity_reset_interval: 3000` が学習長を上回るため
**不透明度リセットが一度も発火しない**。標準 3DGS はリセットで不要な
ガウシアンを低不透明度にしてから刈り取るので、この圧が失われる。
実測 (frame 35): 初回 densify で 3,892 除去した後は 100〜240/step まで落ち、
作成 2,068/step に追いつかない。

結果として warm-start は +7,000/frame で線形に増え続け、frame 300 で
約 290 万に達する (baseline は毎フレーム約 84,000 で一定)。

`opacity_reset_interval: 1000` を試した実測では、増加は 35% 遅くなるだけで
止まらず、品質は下がった (frame 12 で PSNR 36.67 vs 38.31)。

**このため損失・PSNR を報告するときは必ずガウシアン数を併記すること。**
差を「初期値が良いこと」だけに帰属させると、30 倍の容量差を見落とす。

### フェーズ 2: 損失集計

```bash
python eval/loss_logger.py --run_dir output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full \
    --out_dir output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full/loss_logs
python eval/loss_logger.py --run_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_full \
    --out_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_full/loss_logs
python eval/plot_loss.py \
    --log_dir      output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full/loss_logs \
    --baseline_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_full/loss_logs \
    --out_html     output/4DGS/neu3d/coffee_martini/loss_plots/neu3d_coffee_martini_full.html
python eval/summarize_warmstart.py \
    --warm_dir     output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full/loss_logs \
    --baseline_dir output/4DGS/neu3d/coffee_martini/baseline_neu3d_full/loss_logs \
    --warm_run     output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full \
    --baseline_run output/4DGS/neu3d/coffee_martini/baseline_neu3d_full
```

### フェーズ 3: 画質評価

`scripts/render.py` はチェックポイント 1 つを取る設計なので、300 フレーム x
2 アームには `eval/render_4d.py` を使う (同じ `render_camera_set` を 1 プロセスで回す)。

```bash
python eval/render_4d.py --run_dir output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full \
    --data_dir data/neu3d/coffee_martini/converted_4d \
    --out_dir  eval/renders/warmstart_full --split test
python eval/evaluate.py --dataset neu3d \
    --render_dir eval/renders/warmstart_full/renders \
    --gt_dir     eval/renders/warmstart_full/gt \
    --output_csv output/4DGS/neu3d/coffee_martini/warmstart_neu3d_full/results/metrics.csv --device cuda
```

**GT をフラットに置いてはいけない**。300 フレームすべてが cam00/cam09/cam19 と
同じ名前なので、evaluate.py のペア照合が stem に落ちて 300 候補となり
「一意に定まらない」として全件スキップされる。render_4d.py は GT を
`gt/frame_NNNN/camXX.png` とフレーム構造でミラーし、相対パス照合が
効くようにしている。

なお evaluate.py の neu3d プリセットが出すのは MS-SSIM ではなく **D-SSIM**。

## 100 フレーム warm-start 実験の結果（2,000 iter/frame）

300 フレームの予定だったが、warm-start 側の Gaussian 数が加速的に増加して
frame 125-135 付近で OOM する見込みだったため、100 フレームで打ち切った。
両アームとも `configs/neu3d/warmstart_2000.yaml`（同一 config）、frames 1-100。

### フェーズ 2: 損失

| | warm-start | baseline |
|---|---|---|
| iter 500  | 0.0105 | 0.0989 |
| iter 1000 | 0.0127 | 0.0885 |
| iter 1500 | 0.0115 | 0.0486 |
| iter 2000 | 0.0111 | 0.0308 |
| frame 000-049 収束損失 | 0.0125 | 0.0310 |
| frame 050-099 収束損失 | 0.0097 | 0.0306 |
| Gaussian 数（frame 099）| 1,532,128 | 82,697（平均）|
| 1 フレーム平均時間 | 76.0 秒 | 39.9 秒 |

プロット: `output/4DGS/neu3d/coffee_martini/loss_plots/neu3d_coffee_martini_full.html`

### フェーズ 3: 画質（test split = cam00 / cam09 / cam19、300 枚）

| カメラ | PSNR↑ warm/base | D-SSIM↓ warm/base | LPIPS↓ warm/base |
|---|---|---|---|
| cam00 | 27.341 / 27.048 | 0.087 / 0.095 | 0.213 / 0.245 |
| cam09 | 22.201 / 22.883 | 0.184 / 0.172 | 0.270 / 0.309 |
| cam19 | 21.195 / 25.223 | 0.183 / 0.146 | 0.295 / 0.298 |
| 全体  | **23.579 / 25.051** | 0.152 / 0.137 | 0.259 / 0.284 |

`evaluate.py` の neu3d プリセットが出すのは MS-SSIM ではなく **D-SSIM**。

### 結論: warm-start は学習損失を下げるが test 画質は劣化する

20 フレームブロックの PSNR 推移（warm / base / 差）:

| frame | cam00 | cam09 | cam19 |
|---|---|---|---|
| 1-20   | 28.57 / 26.92 / **+1.65** | 22.71 / 22.89 / -0.17 | 24.97 / 25.23 / -0.26 |
| 21-40  | 27.52 / 27.08 / +0.43 | 22.42 / 22.86 / -0.44 | 22.58 / 25.18 / -2.60 |
| 41-60  | 27.28 / 27.12 / +0.16 | 22.20 / 23.04 / -0.84 | 21.06 / 25.28 / -4.21 |
| 61-80  | 26.95 / 27.04 / -0.10 | 21.96 / 22.81 / -0.85 | 19.53 / 25.26 / -5.73 |
| 81-100 | 26.40 / 27.08 / -0.68 | 21.71 / 22.82 / -1.11 | 17.83 / 25.16 / **-7.33** |

baseline は全フレームで平坦（各カメラ ±0.3 dB 以内）なのに対し、warm-start は
3 カメラすべてで単調に劣化する。これは誤差の蓄積であって、個々のフレームが
難しくなっているのではない（baseline が平坦なことがその証拠）。

原因は `opacity_reset_interval: 3000 > iterations: 2000` で opacity reset が
一度も発火せず、densify の生成に対して prune 圧力が効かないこと。frame 35 の
計測では densify 1 ステップあたり生成 約2,068 に対し prune 100-240 だった。
不要な Gaussian が毎フレーム持ち越されて train view に過適合し、held-out view で
崩れる。`opacity_reset_interval: 1000` を 12 フレームで試したが、増加は 35% 遅く
なるだけで画質はむしろ悪化した（frame 12 PSNR 36.67 vs 38.31）。

注意点:
- 容量が約 18.5 倍違う（1.53M vs 82.7k）ので、損失差は初期化の質だけに帰属できない。
- frame_0001 は両アームとも SfM 点群から同一条件で学習するが、CUDA ラスタライザ
  backward の atomicAdd 非決定性で損失 0.0006 (2.8%)・Gaussian 数 0.9% ずれる。
  これがノイズ下限で、これを下回る差は有意ではない。

## iter 数変更の 10 フレーム試行（4,000 / 7,000 iter）

`opacity_reset_interval` は `configs/neu3d/base.yaml` の実値が 3000。2,000 iter では
reset が一度も発火しないため、iter 数を上げて発火させ蓄積が止まるか確かめた。

作成した config（neu3d/base.yaml からの差分は各 2 行だけ。reset 間隔は既定値のまま）:

| | 2000（既存）| 4000 | 7000 |
|---|---|---|---|
| `iterations` | 2000 | 4000 | 7000 |
| `densify_until_iteration` | 1500 | 3500 | 6500 |
| `opacity_reset_interval` | 3000 | 3000 | 3000 |
| reset 発火回数 | 0 | 1 (iter 3000) | 2 (iter 3000, 6000) |

### Gaussian 数（frame 10 時点、10 フレーム試行）

| | baseline 定常値 | warm-start frame 10 | 比 | frame 10 の増分 |
|---|---|---|---|---|
| 2,000 iter | 83.6k | 468,316 | 5.6× | +15,000 |
| 4,000 iter | 244.2k | 573,056 | 2.35× | +25,174 |
| 7,000 iter | 358.8k | 466,599 | **1.30×** | **+4,293** |

7,000 iter の warm-start 増分（frame 5 以降）: +2,630 / +4,508 / +165 / +6,566 /
+6,568 / +4,293（平均 +4,122）。baseline 自身のフレーム間変動が −8,941〜+5,844
なので、残存ドリフトは baseline のノイズ帯に収まっている。

4,000 iter は不十分。frame 5 以降が +30.5k→+31.3k→+25.7k→+26.4k→+28.7k→+25.2k
とほぼ横ばいで、減衰は frame 2-4 で止まる。一方 baseline は約 245k に安定する
ので、reset 自体は機能しており問題は warm-start の持ち越しに固有。

### 損失（10 フレーム平均）

| iter | 4000 warm/base | 7000 warm/base |
|---|---|---|
| 1000 | 0.0224 / 0.0827 | 0.0216 / 0.0812 |
| 2000 | 0.0160 / 0.0287 | 0.0156 / 0.0278 |
| 3000 | 0.0163 / 0.0336 | 0.0154 / 0.0322 |
| 3100 | 0.0201 / 0.0249 | 0.0185 / 0.0239 |
| 6000 | — | 0.0159 / 0.0257 |
| 6100 | — | 0.0222 / 0.0266 |
| 最終 | 0.0153 / 0.0195 | 0.0153 / 0.0234 |

iter 3000→3100、6000→6100 で損失が跳ねるので reset の発火は確認済み。

収束損失の 10 フレーム平均: 4,000 iter は warm 0.0153 / base 0.0195（差 −0.0042）、
7,000 iter は warm 0.0153 / base 0.0234（差 −0.0081）。

**7,000 iter のノイズ下限は大きい。** frame 1 は両アームとも SfM 点群から同一
条件で学習するのに warm 0.0338 / base 0.0257（差 +0.0081、31%）ずれた。Gaussian
数は 349,198 / 349,364（0.05% 差）でモデル規模はほぼ同じなので、これは iter 6000
の reset からの回復途中で測っているために atomicAdd 非決定性が増幅されたもの。
2,000 iter のノイズ下限 0.0006 (2.8%) とは桁が違う。warm/base の損失差 −0.006〜
−0.016 はこの下限をかろうじて超える程度でしかない。

### 所要時間（1 フレーム平均）

| | warm-start | baseline |
|---|---|---|
| 4,000 iter | 102.4 秒 | 85.3 秒 |
| 7,000 iter | 183.0 秒 | 155.4 秒 |

プロット: `output/4DGS/neu3d/coffee_martini/loss_plots/neu3d_4000_trial.html`, `output/4DGS/neu3d/coffee_martini/loss_plots/neu3d_7000_trial.html`

## フレーム間ピクセル差分（cam00, 全300フレーム）

`output/4DGS/neu3d/coffee_martini/results/pixel_diff_cam00.csv` / `output/4DGS/neu3d/coffee_martini/loss_plots/pixel_diff_cam00.html`

平均 1.5903 / 中央値 1.5701 / 最大 2.0087 (frame 10) / 最小 1.4701 (frame 116) /
標準偏差 0.0887（平均の 5.6%）。動き量はフレーム間でほとんど変わらない。

Gaussian 増分との相関（2,000 iter 100 フレーム実行、frame 4 以降 n=97）:

| | 値 |
|---|---|
| 素の相関 | +0.2683 |
| フレーム番号トレンド除去後 | **+0.6522** |
| ピクセル差分の傾き | −0.0027 /frame（100 フレームで −0.27）|
| Gaussian 増分の傾き | +77.0 /frame（100 フレームで +7,702）|

フレームごとの揺らぎは動きで説明できる（r=+0.65）が、トレンドは逆向き。
ピクセル差分は frame 2-100 平均 1.5990 に対し frame 201-300 平均 1.5976 と一定なのに
Gaussian 増分だけが増える。蓄積はシーン内容ではなくアルゴリズム側の要因。

なお frame 1-10 はシーン中で最も動きが大きい区間（1.79-2.01 対 全体中央値 1.57）。
10 フレーム試行は平均より厳しい条件で測っていることになる。

## Gaussian 数固定（densify/prune 無効）

`densify_until_iteration: 0` は `config.py` の
`densify_until_iteration > densify_from_iteration` 検証に弾かれて使えない。
`schedules.py` では densify と prune が同一イベント (`run_density_control_event`) なので
「densify だけ無効・prune は維持」は config では実現できない。
`features.adaptive_density_control: false` で両方まとめて止める。

frame 1 だけ densify を有効にするのはコード改変なしでできる。`warmstart_trainer.py`
の `--start_frame 2 --carry_over_checkpoint <frame_0001 の latest.pt>` を使う。

| config | densify/prune | opacity reset |
|---|---|---|
| `neu3d/warmstart_7000.yaml` | 59 回 (600-6400) | 3000, 6000 |
| `neu3d/fixed_gaussian.yaml` | 0 回 | 3000, 6000 |
| `neu3d/fixed_gaussian_noreset.yaml` | 0 回 | 無効 |

10 フレーム試行の結果（frame 1 = 349,198 を共通の出発点にして frames 2-10）:

| | Gaussian 数 | 収束損失 平均 | 秒/frame |
|---|---|---|---|
| 固定 reset無 | 349,198 固定 | **0.0120** | **176.7** |
| 固定 reset有 | 349,198 固定 | 0.0136 | 205.6 |
| 7,000 iter 通常 ADC | 466,599 (frame 10) | 0.0133 | 183.0 |
| 7,000 iter baseline | 358,797 | 0.0232 | 155.4 |

prune が動かない状態での opacity reset は有害。iter 3100 で 0.0186 対 0.0126、
iter 6100 で 0.0194 対 0.0133 と reset 直後に悪化し、最終損失も 13% 劣る。

## 重要: 7,000 iter の「安定」は再現しない

`checkpoint_interval` 以外が同一の 2 つの実行（1000 と 7000。checkpoint 保存は
`torch.get_rng_state()` 等で RNG を読むだけで消費しないため最適化には影響しない）が、
全く違う Gaussian 数の軌跡を描いた。

| frame | 10フレーム試行 | 増分 | 300フレーム実行 | 増分 |
|---|---|---|---|---|
| 1 | 349,198 | — | 353,884 | — |
| 2 | 406,479 | +57,281 | 413,711 | +59,827 |
| 3 | 427,421 | +20,942 | 438,193 | +24,482 |
| 4 | 441,869 | +14,448 | 456,862 | +18,669 |
| 5 | 444,499 | **+2,630** | 472,248 | **+15,386** |
| 6 | 449,007 | **+4,508** | 498,383 | **+26,135** |
| 7 | 449,172 | **+165** | 516,929 | **+18,546** |

frame 1 の 1.3% 差（CUDA atomicAdd 非決定性）が ADC を通じて増幅され、frame 7 で
15% 差になる。ADC の軌跡は初期の微小差に対してカオス的で、10 フレーム試行で
観測した「frame 5 以降 +4k/frame で安定」は再現性のある性質ではなく、たまたま
引いた 1 本の軌跡だった。300 フレーム実行は監視により frame 7 で自動中断
(直近3フレーム平均 +20,022 > 閾値 8,000)。

Gaussian 数固定の構成はこの問題の影響を受けない。数が定義上一定なので、
ADC のカオス的挙動による暴走が起こりえない。

## Gaussian 数固定の 300 フレーム展開 → frame 53 で劣化を検出

構成: frame 1 のみ通常 ADC (`neu3d/warmstart_7000_full.yaml`) で学習して
`output/4DGS/neu3d/coffee_martini/fixed_gaussian_frame1/` に置き、frames 2 以降は
`neu3d/fixed_gaussian_noreset_full.yaml` (densify/prune・opacity reset とも無効) で
そこから持ち越す。`scripts/train_4d.py` は実行ディレクトリの `config.yaml` と一致しない
config での再開を拒否するので、frame 1 は別ディレクトリに置く必要がある。

Gaussian 数は frames 2-53 の全フレームで 350,659 固定。設計どおり蓄積はゼロ。

### 収束損失の推移（10 フレームブロック平均）

| 区間 | Gaussian固定 | baseline(独立学習) | ピクセル差分 |
|---|---|---|---|
| frame 2-11 | 0.0119 | 0.0316 | 1.8879 |
| frame 12-21 | 0.0102 | 0.0310 | 1.6990 |
| frame 22-31 | 0.0108 | 0.0307 | 1.6203 |
| frame 32-41 | 0.0117 | 0.0311 | 1.5386 |
| frame 42-51 | **0.0181** | 0.0305 | 1.5554 |
| frame 52-53 | **0.0199** | 0.0306 | 1.5433 |

frames 2-31 は 0.0102-0.0119 で平坦、frame 40 付近から悪化が始まり frame 53 で
0.0216。直近 15 フレーム平均が序盤 15 フレーム平均の 1.53 倍になり監視が中断した。

**これは内容の難しさではない。** 毎フレーム SfM 点群から独立に学習する baseline は
frames 2-100 で 0.0304-0.0316 と完全に平坦（トレンドなし）で、ピクセル差分も
1.89 → 1.54 と減少している。同じ区間で Gaussian 固定だけが 1.7 倍に悪化する。
frame 1 の Gaussian 配置が 40 フレーム程度でシーンに追随できなくなる。

### 監視条件について

当初の「収束損失が前フレームの 1.5 倍」は使えない。フレーム間の状態を一切持たない
7,000 iter baseline でも frame 7 で 1.52 倍、frame 9 で 1.41 倍が出る。内容の
難易度変動であって劣化ではない。実際 frame 9 の 1.70 倍は独立した 2 実行で再現し、
どちらも frame 10 で 0.0098 に戻る。

単調悪化を直接測る条件に変更した:
  1. 傾向: 直近 15 フレーム平均が最初の 15 フレーム平均の 1.5 倍超
  2. 破局: 1 フレームが それまでの中央値の 3 倍超（自然変動の実測最大は 1.70 倍）

この条件は frame 30 の時点で 0.97 倍（発火せず）、frame 53 で 1.53 倍（発火）と、
狙いどおり動いた。

### Gaussian 固定の画質評価（frames 2-53、test = cam00/cam09/cam19、156枚）

| カメラ | PSNR↑ | D-SSIM↓ | LPIPS↓ |
|---|---|---|---|
| cam00 | 26.489 | 0.0999 | 0.2284 |
| cam09 | 21.037 | 0.2084 | 0.2888 |
| cam19 | 22.600 | 0.1878 | 0.2996 |
| 全体 | 23.375 | 0.1654 | 0.2723 |

10 フレームブロックの PSNR（全体）: frame 2-11 = 24.49 / 12-21 = 24.49 /
22-31 = 24.32 / 32-41 = 22.94 / 42-51 = 21.28 / 52-53 = 20.12。
損失で見た劣化開始（frame 40 付近）と一致する。frame 31 までは完全に平坦。

全実験の一覧は `output/4DGS/neu3d/coffee_martini/results/experiment_summary.csv`。
