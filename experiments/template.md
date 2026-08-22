# runXXX 実験記録

## 1. 概要

- Run:
- 実行日時:
- 状態:
- 実験目的:
- 前run:
- 比較対象:

### 今回の変更点

-

## 2. 再現情報

### Git

- Branch:
- Commit:
- Working tree:
- Push済み:

### 実行環境

- OS:
- GPU:
- CUDA:
- PyTorch:
- Python:
- device:
- dtype:
- seed:

## 3. データセット

- データセット:
- 学習画像数:
- テスト画像数:
- 解像度:
- train/test split:
- 背景色:

## 4. 学習条件

### Gaussian

| 項目 | 設定値 |
| --- | --- |
| 初期Gaussian数 | |
| 初期化方法 | |
| SH degree | |
| 初期opacity | |

### Optimization

| 項目 | 設定値 |
| --- | --- |
| Iterations | |
| Optimizer | |
| position learning rate | |
| SH learning rate | |
| opacity learning rate | |
| scale learning rate | |
| rotation learning rate | |
| D-SSIM weight | |

### Adaptive Density Control

| 項目 | 設定値 |
| --- | --- |
| 有効 / 無効 | |
| densify開始iteration | |
| densify終了iteration | |
| densify interval | |
| gradient threshold | |
| clone条件 | |
| split条件 | |
| split scale factor | |
| opacity prune threshold | |
| opacity reset interval | |
| max screen radius | |

## 5. 実行コマンド

## 6. 学習経過

| Iteration | Loss | PSNR [dB] | Gaussian数 | Elapsed [s] | VRAM [MiB] | 備考 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |

### Adaptive Density Controlの経過

| Iteration | Before | Clone | Split | Prune | After | 備考 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |

## 7. 最終結果

### 定量評価

- Mean PSNR:
- Mean SSIM:
- Mean LPIPS:
- 最終Gaussian数:
- 総学習時間:
- Peak VRAM:

### 異常

- NaN:
- Inf:
- CUDA OOM:
- Exception:

## 8. 出力ファイル

## 9. 結果・考察

### 良かった点

-

### 問題点

-

### 前runとの比較

-

### 考察

-

## 10. 次の実験

-
