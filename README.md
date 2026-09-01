# 3D Gaussian Splatting

`document/3DGS_実装設計書.md`と`document/3DGS_定式化.tex`に対応した、
検証優先のPyTorch参照実装です。raw Gaussianパラメータの初期化から、投影、
degree-3実球面調和関数、前方から後方へのsplat合成、学習、評価、
チェックポイントおよび公式形式PLYの入出力までを含みます。

検証用のPyTorch参照rendererに加え、GraphDecoの公式
`diff-gaussian-rasterization`を利用するCUDA backendを備えます。公式実装と同じ
Adaptive Density Control（ADC）のstate transition、progressive SH、解像度warm-up、
COLMAPおよびSynthetic NeRF loaderを含み、原論文の全21 sceneで7K/30K評価を
完走したreproduction baselineです。OpenGLリアルタイムviewerは含みません。

## 環境構築

CPython 3.11または3.12を使用してください。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorchは`2.4以上2.7未満`を対象とします。CUDA版が必要な場合は、環境に
対応したwheelをPyTorch公式配布元から導入してください。`runtime.device: auto`
ではCUDAを優先し、利用できなければCPUを使用します。

### CUDA rasterizer

Paper benchmarkで使用した外部extensionは
`graphdeco-inria/diff-gaussian-rasterization`の次のcommitです。

```text
59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
```

CUDA toolkitとPyTorch CUDA wheelを用意した環境で、次のようにbuildします。

```bash
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
git checkout 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d

# GCC 13ではcuda_rasterizer/rasterizer_impl.hのinclude群へ
# `#include <cstdint>` を追加してからbuildする。
python -m pip install --no-build-isolation .
```

検証済みbaseline環境はCPython 3.11.15、PyTorch 2.6.0+cu124、CUDA 12.4、
GCC/G++ 13.3です。extensionを利用できない環境では`reference` backendのtestsは
実行できますが、paper benchmarkは`cuda` backendを必須とします。

## データ

プロジェクト直下の`data/`へ以下を配置します。`data/`はGit管理対象外です。

```text
data/
├── camera_poses_blender.json
├── view_000.png
├── ...
└── view_099.png
```

画像はBlenderから出力した800×800 RGBA、姿勢は資料所定のcamera-to-world
JSONを想定します。RGBAは設定した黒または白背景へ合成して`[0,1]`の
`(3,H,W)`テンソルとして読み込みます。

## 実行

```bash
python scripts/train.py \
  --data data \
  --config configs/default.yaml \
  --output output/runs/run001_example

python scripts/render.py \
  --data data \
  --checkpoint output/runs/run001_example/checkpoints/latest.pt \
  --split test \
  --output output/runs/run001_example/renders/test

python scripts/evaluate.py \
  --data data \
  --checkpoint output/runs/run001_example/checkpoints/latest.pt \
  --split test \
  --output output/runs/run001_example/metrics/evaluation.json
```

学習再開時はチェックポイントの全乱数状態とカメラ選択状態を復元します。

```bash
python scripts/train.py \
  --data data \
  --config configs/default.yaml \
  --output output/runs/run001_example \
  --resume output/runs/run001_example/checkpoints/latest.pt
```

## Adaptive Density Control

ADCを有効にすると、可視Gaussianのscreen-space位置勾配と最大半径をGaussian
indexごとに蓄積します。density-control eventではsmallかつhigh-gradientな
Gaussianをcloneし、largeかつhigh-gradientなGaussianを2個のchildへsplitし、
low-opacityまたは過大なGaussianをpruneします。event終了時に統計windowをreset
するため、Gaussian数`N`は学習中に動的に変化します。periodic opacity resetは
ADCとは独立して有効化できます。

既定値と同じ設定でADCとopacity resetを有効にする例は次のとおりです。

```yaml
features:
  adaptive_density_control: true
  opacity_reset: true
  progressive_sh_degree: false

density_control:
  densify_from_iteration: 500
  densify_until_iteration: 15000
  densification_interval: 100
  position_gradient_threshold: 0.0002
  percent_dense: 0.01
  prune_opacity_threshold: 0.005
  opacity_reset_interval: 3000
  opacity_reset_maximum: 0.01
  prune_screen_radius_threshold: 20.0
  prune_world_scale_fraction: 0.1
```

既定scheduleでは、統計を`iteration < 15000`で収集し、density-control eventを
`iteration > 500`、`iteration < 15000`、かつ`iteration % 100 == 0`のときに
実行します。最初のeventは600、最後は14900です。opacity resetは15000より前に
3000反復間隔で実行し、白背景では`densify_from_iteration`（既定500）でも特別に
実行します。screen-spaceおよびworld-space size pruningは、
`iteration > opacity_reset_interval`のdensity-control eventでのみ有効です。

clone/split後のscreen-radius statistics reset順序はpaper-era公式実装に合わせて
います。densification postfix後のfinal pruneは、過去windowの`max_radii2D`を
参照しません。gradient accumulator、denominator、screen radiusはevent終了時に
次window用のzero stateになります。

## Checkpointとresume

新規保存されるversion 2 checkpointには、dynamic Gaussian model、Adam state、
scheduler state、Python・NumPy・PyTorchの乱数状態、カメラsampling状態、および
screen-space density statisticsが含まれます。このため100反復の統計window途中で
中断しても、蓄積済み統計を保持して再開できます。version keyを持たないlegacy
version 1 fixed-Gaussian checkpointは、ADCとopacity resetをともに無効のままなら
resumeできます。

## Logging

`train_log.jsonl`の通常recordには、event処理後の`gaussian_count`を記録します。
density-control event発生時は、次のfieldも追加します。

- `density_control_event`
- `density_num_gaussians_before`, `density_num_gaussians_after`
- `density_num_cloned`
- `density_num_split_parents`, `density_num_children_created`
- `density_num_pruned_total`
- `density_num_low_opacity`, `density_num_large_screen`, `density_num_large_world`

opacity reset発生時は、`opacity_reset`、`opacity_num_clamped`、
`opacity_reset_maximum`を追加します。event/reset反復は通常の`log_interval`外でも
1 recordだけ出力します。

## 原論文benchmark

`data/`へMip-NeRF360、Tanks&Temples、Deep Blending、Synthetic NeRFを配置後、
学習前監査を実行します。dry-runはdataset path、split、native resolution、初期化、
background、warm-up、SH scheduleを記録し、学習は開始しません。

```bash
python scripts/paper_benchmark_dry_run.py
```

manifestが`READY`であることを確認してから、逐次runnerを開始します。各sceneは
独立subprocessで0→30Kを1回だけ実行し、7K/30K checkpointを保持します。完了済み
sceneはskipし、中断runはcheckpointからresumeします。

```bash
python scripts/run_paper_benchmark.py \
  --manifest output/paper_benchmark/manifest.json \
  --continue-on-oom

python scripts/generate_paper_benchmark_report.py \
  --manifest output/paper_benchmark/manifest.json

python scripts/generate_paper_benchmark_qualitative.py \
  --manifest output/paper_benchmark/manifest.json
```

`output/`にはcheckpoint、metrics、VRAM telemetry、CSV/JSON、Markdown report、
定性的renderが生成されます。これらは大容量のためGit管理対象外です。

CPU参照ラスタライザは検証用途では有用ですが、高解像度30K benchmarkには低速です。

## テスト

```bash
python -m pytest
```

テストには数式単体テスト、解析ヤコビアン・autograd・数値微分の比較、
投影と合成の統合テスト、チェックポイントおよびPLY往復テスト、
TeXの全116式ラベルの追跡確認が含まれます。

CUDA extensionが利用可能な環境では、CUDA smoke/integration testsも自動的に
実行されます。利用できない環境では該当testsのみskipされます。
