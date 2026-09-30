# eval/ — 学習結果の図・レポート・可視化

学習や描画が書き出したログ・チェックポイントを読んで、人が見る形にする道具を置く。
指標の実装は `src/gaussian_splatting/evaluation/`、描画と評価の入口は `scripts/`
（`scripts/rendering/`、`scripts/evaluate.py`。使い方は `scripts/README.md`）、
学習から評価までの通し実行は `runs/` にある。

```
eval/
├── plot/           学習ログから推移の図 (HTML) を作る。CSV も出せる
├── report/         ラン 1 本の eval.md を生成する
└── visualization/  ガウシアン分布の投影と、学習過程のビューワー
```

## ツール一覧

**自動**が ● のものは `runs/4DGS_baseline.sh` / `runs/4DGS_warmstart.sh` / `runs/4DGS_regularized.sh`
（`make_eval_report.py` は `runs/3DGS.sh` も）が実行する。○ は手で使う。

| ツール | 用途 | 入力 → 出力 | 自動 |
|---|---|---|---|
| `plot/plot_loss.py` | 損失 / PSNR の推移。任意のランを 1 枚に重ねる | `--run` (複数可) → `--out_html` / `--output_csv_dir` | ○ |
| `plot/plot_gaussian_count.py` | Gaussian 数・学習時間・VRAM の集計と推移。任意のランを 1 枚に重ねる | `--run` (複数可) → 標準出力 / `--output_csv` / `--out_html` | ● |
| `report/make_eval_report.py` | ラン 1 本の実行結果レポート | `--run_dir` (+ `--compare`) → `<run_dir>/eval.md` | ● |
| `visualization/visualize_gaussians.py` | チェックポイントのガウシアン分布を上面/正面/側面に投影 | `--ckpt_dir --frame` → `--out_dir` (PNG + 3D HTML) | ● |
| `visualization/gaussian_viz_report.py` | 上の出力を 1 枚の HTML に並べる | `--viz_dir --frames` → `--out` | ● |
| `visualization/extract_snapshots.py` | 学習済みチェックポイントからスナップショット (npz) を抽出 | `--run` → `<run>/snapshots/` | ○ |
| `visualization/snapshot_viewer.py` | フレーム × iteration の 2 軸で学習過程を再生 (Streamlit) | `--run` → ブラウザ | ○ |

## plot/ — 推移の図

どちらも `--run [ラベル=]ランのディレクトリ` を繰り返すと、指定したランを同じ図に
重ねる。ラベルを省略するとディレクトリ名になる。基準ランや差分は持たず、どのランも
同じ扱い。HTML は Chart.js を CDN から読むので、見るときにネット接続が要る。

```bash
# 損失の推移 (loss_total / loss_l1 / loss_dssim / psnr を --loss_key で選ぶ)
python eval/plot/plot_loss.py \
    --run "30k=output/4DGS/neu3d/coffee_martini/baseline_30k" \
    --run "warm 250=output/4DGS/neu3d/cook_spinach/warmstart_250_300f" \
    --frames 1 51 101 --out_html loss.html

# Gaussian 数・学習時間・VRAM (図の縦軸はボタンで切り替え)
python eval/plot/plot_gaussian_count.py \
    --run "30k=output/4DGS/neu3d/coffee_martini/baseline_30k" \
    --run "warm 250=output/4DGS/neu3d/cook_spinach/warmstart_250_300f" \
    --out_html gaussians.html
```

- `plot_loss.py` は各フレームの `train_log.jsonl` を読む。4D ラン (`frame_NNNN/`) でも
  単一シーンのラン (frame 0 扱い) でもよい。`--frames` は実際のフレーム番号で絞る。
- `plot_gaussian_count.py` は各フレームの `training_telemetry.json` (COMPLETED のみ) を読む。
  4D ランのみ対応。
- CSV (`plot_loss.py --output_csv_dir` / `plot_gaussian_count.py --output_csv`) は
  `--run` が 1 本のときだけ出せる。
- `plot_loss.py` の表の「収束 iter」は、損失がその run の最初の記録値の 10% 以下に
  初めて落ちた iteration。初期損失が違う run 同士では比べられない。また
  `log_interval: 100` だと最初の記録が iter 100 で、そこまでに大きく下がっているため
  到達しない (「—」) ことがある。

## report/ — eval.md

```bash
python eval/report/make_eval_report.py \
    --run_dir output/4DGS/neu3d/cook_spinach/cook_spinach_baseline_7k \
    --compare output/4DGS/neu3d/cook_spinach/cook_spinach_baseline_30k
```

- ラン直下の `frame_*` の有無で 3D / 4D を判別し、`report/templates/eval_{3d,4d}.md` を埋める。
- 評価の数値は `<run_dir>/results/` の `metrics.csv` / `per_frame.csv` / `summary.json`
  から、Gaussian 数・時間・VRAM は `training_telemetry.json` から、密度制御の回数は
  `train_log.jsonl` の実測から取る。
- 4D のデータセット情報 (解像度・カメラ数・初期点群) は `frames_4d.json` の `source` から探す。
  データを移動する前に学習したランでは `source` が古い場所を指すので、`--data_dir` で
  変換済みデータの root を渡す (`runs/4DGS_*.sh` は常に渡す)。
- 機械的に取れなかった値は推測せず「要追記」と書く。`<!-- human -->` の付いた節
  (定性的評価 / AI による初見) は人が書く領域で、再生成しても既存の中身を引き継ぐ。

## visualization/ — ガウシアンの分布と学習過程

### 分布の投影

```bash
python eval/visualization/visualize_gaussians.py \
    --ckpt_dir output/4DGS/neu3d/coffee_martini/baseline_30k/frame_0001 \
    --out_dir  output/4DGS/neu3d/coffee_martini/baseline_30k/gaussian_viz --frame 1
python eval/visualization/gaussian_viz_report.py \
    --viz_dir output/4DGS/neu3d/coffee_martini/baseline_30k/gaussian_viz --frames 1 150 300
```

### 学習過程ビューワー

ガウシアンの中心座標・不透明度・個数の推移を、フレーム軸と iteration 軸の 2 軸で
再生する。画像ではなくパラメータそのものを見るので、densify/prune が何を増やしたか、
warm-start でガウシアンが動いているかを直接確認できる。`pip install -e ".[viz]"` が必要。

スナップショットの作り方は 2 通り。

- **学習中に記録する** (`scripts/train.py` / `scripts/train_4d.py`):
  `--snapshot-interval N` (N iter ごと、既定 0 = 無効)、`--snapshot-iterations 0,50,100`
  (必ず記録する iter)、`--snapshot-max-points K` (1 コマの点数、既定 20000、0 で全点)。
  最初と最後の iteration は必ず記録する。
- **学習済みのランから抽出する**: `python eval/visualization/extract_snapshots.py --run <ラン>`
  (`--stride 2` で間引き)。4D ランなら `frame_*/` を自動で走査する。
  `checkpoint_interval` の粒度でしかコマが取れない。

```bash
streamlit run eval/visualization/snapshot_viewer.py -- --run output/4DGS/neu3d/coffee_martini/warmstart_neu3d_trial
```

- `--run` の前の `--` は必須 (Streamlit 自身の引数と区別するため)。省略するとサイドバーで指定する。
- 再生軸は `iteration` (フレーム固定) / `frame` (iteration 固定。無い iteration は最も近いコマで代替) /
  `frame x iteration` の 3 通り。
- 軸の範囲は既定で各軸の中央 99% (`robust_bounds`) に固定する。シーンから遠く離れた
  少数のガウシアンで本体が潰れないようにするため。サイドバーで切り替えられる。
- 間引きは描画用で、個数のグラフには常に間引き前の真の個数が出る。
