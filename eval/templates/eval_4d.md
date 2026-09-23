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
- {{DATASET_NAME}}
  - {{SOURCE_VIDEO}}
  - {{FRAME_COUNT}} frames に分解して使用する
  - 解像度 {{RESOLUTION}}
  - カメラ数は {{CAMERA_COUNT}} 台
  - test view は {{TEST_VIEWS}} ({{TEST_EVERY}})
  - 評価対象は {{EVAL_IMAGE_COUNT}} 枚

## 実験方法
{{METHOD}}
{{SFM_NOTE}}

| 項目 | 設定値 |
|---|---|
| iterations | {{ITERATIONS}} |
| 初期化 | {{INITIALIZATION}} |
| 初期 Gaussian 数 | {{INITIAL_GAUSSIANS}} |
| SH degree | {{SH_DEGREE}} |
| 背景色 | {{BACKGROUND}} |
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
- 実行期間 : {{RUN_PERIOD}}
- config : {{CONFIG_PATH}}

## 評価
| | test PSNR ↑ | D-SSIM ↓ | LPIPS ↓ | Gaussian数 | train time | VRAM (peak reserved) | 出力容量 |
|---|---|---|---|---|---|---|---|
{{RESULT_ROWS}}

カメラ別 test PSNR:

{{CAMERA_TABLE}}

フレーム別の推移: {{FRAME_TREND}}

Gaussian 数: 平均 {{GAUSSIAN_MEAN}} (最小 {{GAUSSIAN_MIN}} / 最大 {{GAUSSIAN_MAX}} / 標準偏差 {{GAUSSIAN_STD}})
学習ビュー PSNR (各フレーム最終 iteration): {{TRAIN_VIEW_PSNR}}
異常終了・NaN/Inf/OOM: {{ANOMALIES}}

## 出力ファイル
{{ARTIFACTS}}

## 定性的評価
<!-- human -->
要追記

## AIによる初見
<!-- human -->
要追記
