# {{TITLE}}

<!-- このファイルは eval/make_eval_report.py が生成する。
     human マーカーが付いた節 (定性的評価 / AIによる初見) は人が書く節であり、
     再生成しても上書きされない。機械的に取得できなかった値は「要追記」と書かれる。
     生成日時: {{GENERATED_AT}} -->

## 実行環境
- マシン : {{MACHINE}}
- GPU : {{GPU}}
- CPU : {{CPU}}
- メモリ : {{MEMORY}}
- ストレージ : {{STORAGE}}
- フレームワーク : {{FRAMEWORK}}
- カーネル : {{KERNEL}}
- ソート : {{SORT}}
- ビューア : {{VIEWER}}

## 使用したデータセット
- {{DATASET_NAME}} / シーン {{SCENE}}
  - データパス : {{DATA_PATH}}
  - 画像ディレクトリ : {{IMAGE_DIRECTORY}}
  - 画像枚数 : {{IMAGE_SPLIT}}
  - 解像度 : {{RESOLUTION}}
  - 背景色 : {{BACKGROUND}}
  - train / test split : {{TEST_SPLIT}}

## 実験方法
{{METHOD}}

| 項目 | 設定値 |
|---|---|
| iterations | {{ITERATIONS}} |
| 評価 milestone | {{MILESTONES}} |
| 初期化 | {{INITIALIZATION}} |
| 初期 Gaussian 数 | {{INITIAL_GAUSSIANS}} |
| SH degree | {{SH_DEGREE}} |
| densify_from / until / interval | {{DENSIFY_SCHEDULE}} |
| opacity_reset_interval | {{OPACITY_RESET_INTERVAL}} |
| 密度制御イベント / 不透明度リセット | {{ADC_EVENT_COUNTS}} |
| position lr (initial → final) | {{POSITION_LR}} |
| その他 lr (sh_dc / sh_rest / opacity / scale / quaternion) | {{OTHER_LR}} |
| D-SSIM weight | {{LAMBDA_DSSIM}} |

密度制御イベントと不透明度リセットの回数は `train_log.jsonl` の発火 iteration を数えた実測値である
(発火条件は `src/gaussian_splatting/training/schedules.py` の `density_control_schedule`)。

## 再現情報
- Git branch : {{GIT_BRANCH}}
- Commit : {{GIT_COMMIT}}
- Working tree : {{GIT_DIRTY}}
- 実行コマンド : {{COMMAND}}
- 実行時の historical command : {{HISTORICAL_COMMAND}}
- 実行期間 : {{RUN_PERIOD}}
- config : {{CONFIG_PATH}}

## 評価

チェックポイント評価 (`metrics/test_<iteration>.json`、test view {{TEST_VIEW_COUNT}} 枚):

| iteration | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Gaussian数 | 学習時間 | checkpoint |
|---|---|---|---|---|---|---|
{{MILESTONE_ROWS}}

画像ベース評価 (`eval/evaluate.py`、描画済み PNG 対 GT): {{IMAGE_EVAL}}

学習経過:

| iteration | loss | PSNR (学習ビュー) | Gaussian数 |
|---|---|---|---|
{{PROGRESS_ROWS}}

VRAM : {{VRAM}}
異常終了・NaN/Inf/OOM : {{ANOMALIES}}

## 出力ファイル
{{ARTIFACTS}}

## 定性的評価
<!-- human -->
要追記

## AIによる初見
<!-- human -->
要追記
