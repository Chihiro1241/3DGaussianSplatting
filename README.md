# 3D Gaussian Splatting

`document/3DGS_実装設計書.md`と`document/3DGS_定式化.tex`に対応した、
検証優先のPyTorch参照実装です。raw Gaussianパラメータの初期化から、投影、
degree-3実球面調和関数、前方から後方へのsplat合成、学習、評価、
チェックポイントおよび公式形式PLYの入出力までを含みます。

初期実装のため、Gaussian単位のPythonループを使用します。CUDA拡張、
タイルラスタライザ、適応的密度制御、OpenGL可視化ツールは対象外です。

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
