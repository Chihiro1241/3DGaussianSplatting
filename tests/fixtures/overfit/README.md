# Single-view overfit fixture

このfixtureは、固定カメラと4個の固定Gaussianから生成した32×32の教師画像、
学習開始時の異なるGaussianパラメータ、教師Gaussianおよび完全なテスト設定を
保持します。Gaussianには重なりと異なる深度を含みます。

- 乱数シード: `20260807`
- Gaussian数: 4（全1,000反復で固定）
- SH次数: 3
- 背景: 黒
- densification、pruning、不透明度リセット: 無効
- 判定: 最後の50反復の平均損失が最初の50反復の平均損失の50%未満
- 全反復で損失、画像、勾配およびパラメータが有限値であること

教師画像とNPZは次のコマンドで再生成できます。

```bash
PYTHONPATH=src python tests/fixtures/overfit/generate_fixture.py
```

教師画像も同じレンダラで生成されるため、このfixtureはレンダリング、逆伝播、
Adam更新の一貫性を検証します。レンダリング式そのものは、解析ヤコビアン、
数値微分、`gradcheck`、単一Gaussianおよび複数Gaussianのテストで別途検証します。

