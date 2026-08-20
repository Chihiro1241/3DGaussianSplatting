# 3D Gaussian Splatting

`document/3DGS_実装設計書.md`と`document/3DGS_定式化.tex`に対応した、
検証優先のPyTorch参照実装です。raw Gaussianパラメータの初期化から、投影、
degree-3実球面調和関数、前方から後方へのsplat合成、学習、評価、
チェックポイントおよび公式形式PLYの入出力までを含みます。

検証優先の参照実装としてGaussian単位のPythonループを使用します。CUDA/C++
拡張、タイルベースラスタライザ、progressive SH degree、OpenGLリアルタイム
viewerは未実装です。Adaptive Density Control（ADC）はoptional featureとして
実装済みですが、既定では無効であり、従来どおりGaussian数を固定して学習します。

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
  --output output/run001

python scripts/render.py \
  --data data \
  --checkpoint output/run001/checkpoints/latest.pt \
  --split test \
  --output output/run001/renders/test

python scripts/evaluate.py \
  --data data \
  --checkpoint output/run001/checkpoints/latest.pt \
  --split test \
  --output output/run001/metrics/evaluation.json
```

学習再開時はチェックポイントの全乱数状態とカメラ選択状態を復元します。

```bash
python scripts/train.py \
  --data data \
  --config configs/default.yaml \
  --output output/run001 \
  --resume output/run001/checkpoints/latest.pt
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

既定の1000 Gaussian・800×800画像はCPU参照ラスタライザでは低速です。
まず縮小設定や少数反復で動作確認してください。設計上も30,000反復の完走は
この初期実装の完了条件ではありません。

## テスト

```bash
python -m pytest
```

テストには数式単体テスト、解析ヤコビアン・autograd・数値微分の比較、
投影と合成の統合テスト、チェックポイントおよびPLY往復テスト、
TeXの全116式ラベルの追跡確認が含まれます。
