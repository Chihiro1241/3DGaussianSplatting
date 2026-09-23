# configs/

学習設定ファイル。`gaussian_splatting.config.load_config` が読む。

## 構成

```
configs/
├── default.yaml          動作確認・単体テスト用（tests/ が直接読む。移動不可）
├── hypernerf.yaml        HyperNeRF (vrig)
├── paper_benchmark/      論文再現ベンチマーク（静的シーン 21 本）
│   ├── real.yaml           Mip-NeRF360 / Tanks&Temples / Deep Blending
│   └── synthetic.yaml      NeRF Synthetic
└── neu3d/                Neu3D coffee_martini の 4D 実験（base.yaml からの派生）
    ├── base.yaml
    ├── baseline_7k.yaml   baseline_30k.yaml
    ├── warmstart_2000.yaml  warmstart_2000_reset.yaml
    ├── warmstart_4000.yaml  warmstart_7000.yaml  warmstart_7000_full.yaml
    ├── fixed_gaussian.yaml  fixed_gaussian_noreset.yaml
    ├── fixed_gaussian_noreset_full.yaml
    └── trial_5000.yaml      trial_5000_fixed.yaml
```

## ローダーの制約（重要）

`config.py` は **strict** 実装で、**全キーが必須・継承なし**。欠損キーも未知キーも
`ConfigError` になる。

そのため「差分だけを書いた設定ファイル」は作れず、`neu3d/` の 13 本は
`base.yaml` の全文コピーに数行の変更を入れたものになっている。**共通部分の
括り出しはローダーを変えない限り不可能**。代わりに、各ファイルの冒頭コメントと
下表で「どこが違うか」を追えるようにしてある。

設定を 1 本増やすときは、派生元を丸ごとコピーして冒頭コメントに差分を書くこと。

## `neu3d/base.yaml` からの差分一覧

| ファイル | base.yaml からの差分 | 使った実験 |
|---|---|---|
| `neu3d/base.yaml` | — （30,000 iter の基準設定） | 単一フレーム 30k リファレンス |
| `neu3d/baseline_30k.yaml` | `checkpoint_interval` 1000→30000 | 300f baseline 30k（実測 65 時間） |
| `neu3d/baseline_7k.yaml` | `iterations` 30000→7000 / `densify_until_iteration` 15000→6500 / `checkpoint_interval` 1000→7000 | 300f baseline 7k |
| `neu3d/warmstart_2000.yaml` | `iterations` 30000→2000 / `densify_until_iteration` 15000→1500 / `checkpoint_interval` 1000→2000 | 100f warm-start 本番（両アーム共用） |
| `neu3d/warmstart_2000_reset.yaml` | ↑ に加え `opacity_reset_interval` 3000→1000 | リセット間隔の対処案（12f 試行・不採用） |
| `neu3d/warmstart_4000.yaml` | `iterations` 30000→4000 / `densify_until_iteration` 15000→3500 | iter 数試行（reset 1 回・不十分） |
| `neu3d/warmstart_7000.yaml` | `iterations` 30000→7000 / `densify_until_iteration` 15000→6500 | iter 数試行（reset 2 回・有効） |
| `neu3d/warmstart_7000_full.yaml` | ↑ に加え `checkpoint_interval` 1000→7000 | 7k warm-start 本番 |
| `neu3d/fixed_gaussian.yaml` | `iterations` 30000→7000 / `densify_until_iteration` 15000→6500 / `adaptive_density_control` → false | ガウシアン数固定 試行 |
| `neu3d/fixed_gaussian_noreset.yaml` | ↑ に加え `opacity_reset` → false | ADC + リセット無効 試行 |
| `neu3d/fixed_gaussian_noreset_full.yaml` | ↑ に加え `checkpoint_interval` 1000→7000 | ガウシアン固定 本番（frame 2 以降） |
| `neu3d/trial_5000.yaml` | `iterations` 30000→5000 / `evaluation_interval` 30000→5000 / `checkpoint_interval` 1000→5000 | 10f 予備試行（密度制御に欠陥あり・下記） |
| `neu3d/trial_5000_fixed.yaml` | ↑ に加え `densify_until_iteration` 15000→4500 | 10f 予備試行（修正版） |

その他のファイル（派生元が `neu3d/base.yaml` でないもの）:

| ファイル | 差分 |
|---|---|
| `paper_benchmark/real.yaml` | `neu3d/base.yaml` と**設定値が完全に同一** |
| `paper_benchmark/synthetic.yaml` | real.yaml から背景を白へ（`rgba_background` black→white / `rendering.background` →`[1,1,1]`）+ `random_initial_sh_dc` → true |
| `hypernerf.yaml` | real.yaml から `aabb_half_extent` `[1.3]×3`→`[0.6]×3` のみ |
| `default.yaml` | base.yaml から 10 キー（`device` auto / `num_gaussians` 20000 / `iterations` 5000 / `log_interval` 10 / `evaluation_interval` 1000 / `checkpoint_interval` 5000 / `aabb_half_extent` `[1.0]×3` / `progressive_sh_degree`・`resolution_warmup` false / `save_rendered_images` true） |

## 設定値が同一のファイル（別名として残しているもの）

削除していない。eval/README.md の実験記録とコマンド例が、どの名前で回したかを
そのまま辿れるようにするため。

| 組 | 関係 |
|---|---|
| `neu3d/base.yaml` ＝ `paper_benchmark/real.yaml` | Neu3D 実験の派生元を名前で示すための別名 |
| `neu3d/baseline_7k.yaml` ＝ `neu3d/warmstart_7000_full.yaml` | baseline / warm-start の両アームが同一設定で回るため |

## 既知の落とし穴

- **`neu3d/trial_5000.yaml` は `densify_until_iteration` (15000) が学習長 (5000) を
  上回る。** 密度制御が最後まで止まらない。warm-start / baseline の両アームに
  等しく効くので比較自体は成立するが、ガウシアン数の推移を必ず併記すること。
  修正版が `neu3d/trial_5000_fixed.yaml`。
- **`neu3d/warmstart_2000.yaml` は `opacity_reset_interval` (3000) が学習長 (2000) を
  上回る。** 不透明度リセットが一度も発火せず、prune 圧が効かない。warm-start で
  ガウシアン数が飽和せず線形に増え続ける原因。詳細は eval/README.md。
- **`checkpoint_interval` は 4D 実験では容量に直結する。** 300 フレーム × 1 フレーム
  あたりのチェックポイント数で効くので、本番用の設定は学習長と同値にしてある
  （学習結果には影響しない）。
- **`default.yaml` のパスは動かせない。** `tests/` の 20 以上のファイルが
  `configs/default.yaml` を直接参照している。

## 旧パスとの対応（2026-09-19 整理）

削除・設定値の変更は行っていない。移動と改名、および参照パスの書き換えのみ。

| 旧 | 新 |
|---|---|
| `configs/paper_benchmark_real.yaml` | `configs/paper_benchmark/real.yaml` |
| `configs/paper_benchmark_synthetic.yaml` | `configs/paper_benchmark/synthetic.yaml` |
| `configs/neu3d.yaml` | `configs/neu3d/base.yaml` |
| `configs/neu3d_baseline_7k.yaml` | `configs/neu3d/baseline_7k.yaml` |
| `configs/neu3d_baseline_30k.yaml` | `configs/neu3d/baseline_30k.yaml` |
| `configs/neu3d_warmstart_2000.yaml` | `configs/neu3d/warmstart_2000.yaml` |
| `configs/neu3d_warmstart_2000_reset.yaml` | `configs/neu3d/warmstart_2000_reset.yaml` |
| `configs/neu3d_warmstart_4000.yaml` | `configs/neu3d/warmstart_4000.yaml` |
| `configs/neu3d_warmstart_7000.yaml` | `configs/neu3d/warmstart_7000.yaml` |
| `configs/neu3d_warmstart_7000_full.yaml` | `configs/neu3d/warmstart_7000_full.yaml` |
| `configs/neu3d_fixed_gaussian.yaml` | `configs/neu3d/fixed_gaussian.yaml` |
| `configs/neu3d_fixed_gaussian_noreset.yaml` | `configs/neu3d/fixed_gaussian_noreset.yaml` |
| `configs/neu3d_fixed_gaussian_noreset_full.yaml` | `configs/neu3d/fixed_gaussian_noreset_full.yaml` |
| `configs/neu3d_5000.yaml` | `configs/neu3d/trial_5000.yaml` |
| `configs/neu3d_5000_fixed.yaml` | `configs/neu3d/trial_5000_fixed.yaml` |
| `configs/default.yaml` | 変更なし |
| `configs/hypernerf.yaml` | 変更なし |

`scripts/` `eval/` の参照は更新済み。**既に書き出された実行ログと生成物
（`eval/logs/*.log`、`eval/loss_plots/*.html`、`output/` 配下）は実行時の記録
なので旧パスのまま残してある。**
