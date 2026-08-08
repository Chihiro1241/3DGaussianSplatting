# 3DGS実装時の確認事項と参照記録

## 1. この文書の目的

この文書は、3D Gaussian Splatting（3DGS）の実装に際して確認が必要になった事項、ユーザーとの確認によって確定した事項、および公式リポジトリを参照して解決した技術的な曖昧さを記録する。

実装の主仕様は次の既存資料である。これらのファイルは参照のみとし、今回の実装では変更していない。

- [`document/3DGS_実装設計書.md`](document/3DGS_実装設計書.md)
- [`document/3DGS_Gaussian可視化ツール_仕様書.md`](document/3DGS_Gaussian可視化ツール_仕様書.md)

公式実装は、資料だけでは一意に決められない挙動を確認するための補助資料として使用した。

- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

## 2. ユーザー確認によって確定した事項

### 2.1 実装範囲

- PyTorchを使用した3DGSの初期コア実装を対象とする。
- Gaussianモデル、投影、共分散、SH、ラスタライズ、学習、評価、チェックポイントおよびPLY入出力を実装する。
- 可視化ツールおよび対話的viewerは今回の対象外とする。
- `document/`内の資料は直接参照し、複製や変更を行わない。
- 資料から確定できない事項は、まず公式リポジトリを参照して解決する。公式リポジトリでも一意に決められない場合は、実装前にユーザーへ確認する。

### 2.2 単一学習視点による過学習fixture

結合テスト用fixtureについて、次の条件で実装することをユーザー確認により確定した。

- 固定乱数seedを使用する。
- 教師画像は `32 x 32` とする。
- 重なりと異なる深度を含む4個の固定Gaussianを使用する。
- 教師画像を生成したGaussianとは異なる初期値から学習を開始する。
- Gaussian数、カメラ、背景色、SH次数、学習率をfixture内で固定する。
- densification、pruning、不透明度resetは無効にする。
- 単一視点を1,000反復学習する。
- 最初の50反復の平均損失に対して、最後の50反復の平均損失が50%未満であることを確認する。
- 全反復で損失、描画画像、パラメータおよび勾配に `NaN` または `Inf` がないことを確認する。
- 教師画像の再生成とfixtureの由来を確認できるよう、教師Gaussianも保存する。

fixtureは [`tests/fixtures/overfit/`](tests/fixtures/overfit/) に格納した。実測結果は次のとおりである。

| 指標 | 値 |
| --- | ---: |
| 最初の50反復の平均損失 | `0.06221506` |
| 最後の50反復の平均損失 | `0.00081675` |
| 最後 / 最初 | `0.013128` |

このテストはレンダリング、逆伝播、optimizerによる更新が一貫して動作することを検証する結合テストである。同じレンダラで教師画像を生成しているため、レンダリング式そのものの正しさは、数値微分との勾配比較や単一Gaussianの描画テストで別途検証している。

## 3. 公式リポジトリを参照して確定した事項

### 3.1 画像リサイズ時の補間方法

資料では画像サイズの正規化が必要だったが、Pillowの補間方式を明示的に指定するかは一意でなかった。

公式実装の [`PILtoTorch`](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/general_utils.py#L19-L25) が `pil_image.resize(resolution)` を補間方式の明示なしで呼び出していることを確認し、本実装でもPillowの既定動作を使用した。

該当実装: [`src/gaussian_splatting/data/blender_loader.py`](src/gaussian_splatting/data/blender_loader.py)

### 3.2 学習反復番号と学習率schedule

反復番号を0始まりと1始まりのどちらとして扱うかは、checkpoint再開時のcursorと学習率scheduleに影響する。

公式の [`train.py`](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py#L43-L83) では、初回学習を反復1として最終反復を含む範囲で処理し、その反復番号を学習率更新へ渡している。この規約に合わせ、本実装も公開される学習反復番号を1始まりとした。

該当実装:

- [`src/gaussian_splatting/training/trainer.py`](src/gaussian_splatting/training/trainer.py)
- [`src/gaussian_splatting/training/schedules.py`](src/gaussian_splatting/training/schedules.py)
- [`src/gaussian_splatting/io/checkpoint.py`](src/gaussian_splatting/io/checkpoint.py)

### 3.3 PLYの属性名と保存値

PLYの相互運用性を保つため、公式の [`GaussianModel.construct_list_of_attributes` と `save_ply`](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py#L191-L236) を参照した。

次の属性名を公式形式に合わせた。

- 座標: `x`, `y`, `z`
- 法線placeholder: `nx`, `ny`, `nz`
- SH DC成分: `f_dc_*`
- SH残差成分: `f_rest_*`
- 不透明度raw値: `opacity`
- scale raw値: `scale_*`
- quaternion raw値: `rot_*`

保存対象はactivation後の表示値ではなく、学習パラメータとして保持しているraw値とした。既定の出力形式はbinary little endianで、検査用としてASCII出力も選択できる。

該当実装: [`src/gaussian_splatting/io/ply_io.py`](src/gaussian_splatting/io/ply_io.py)

## 4. 資料に基づいて実装内で確定した事項

次の事項は資料の数式・API定義から実装可能であり、追加確認を必要としなかった。

- quaternionの正規化、scaleの指数parameterization、不透明度のsigmoid parameterization
- world座標からcamera座標への変換と透視投影
- 3次元共分散の構築およびJacobianによるscreen-space共分散への変換
- degree 3までのspherical harmonicsによる色評価
- depth順sortとfront-to-back alpha compositing
- L1とSSIMを組み合わせた学習損失
- Adamのparameter group、位置学習率schedule、checkpoint再開
- PSNRおよびSSIMによる評価

公式実装のCUDA rasterizerをコピーするのではなく、資料にある数式を追跡できるPyTorch実装として構成した。そのため、正確性は式ごとの単体テスト、`torch.autograd.gradcheck`、有限差分比較、描画テストおよび過学習テストを組み合わせて確認している。

## 5. 実装完了時点の検証結果と注意事項

- テスト結果: `100 passed`
- `document/`の差分: なし
- 構文チェック: 成功
- 未実装marker（`TODO`, `NotImplementedError`）: なし
- fixtureを含む全テストで有限値検査: 成功

依存条件は [`pyproject.toml`](pyproject.toml) でPython 3.11以上3.13未満として定義している。ただし、実装時に利用可能だったローカル環境はPython 3.14であり、Python 3.11または3.12の独立環境を追加できなかったため、対象Python環境でのinstallation確認だけは未実施である。
