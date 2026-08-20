# 3D Gaussian Splatting 実装設計書

- 基準資料：`3DGS_定式化.tex`、`3D_Gaussian_Splatting_定式化.pdf`
- 対象：PyTorchによる検証優先の初期実装
- 文書版：1.5
- 作成日：2026年8月6日
- 最終更新日：2026年8月20日

## 1. 文書の目的

本書は、`3DGS_定式化.tex`に記載された3D Gaussian Splatting（3DGS）の定式化を、PythonおよびPyTorchによる初期実装へ落とし込むための実装設計書である。

当初の検証優先baselineでは、数式とコードの対応関係を追跡しやすくすること、および各計算の正しさを単体テストで検証できることを最優先とした。そのためCUDAラスタライザ、タイル単位のカリング・ソートおよび適応的密度制御を対象外とし、Gaussian数を固定してPyTorchのテンソル演算と自動微分を検証した。現在はこのbaselineを既定動作として維持しつつ、Adaptive Density Control（ADC）と不透明度resetを独立したoptional featureとして追加している。

## 2. 参照資料と優先順位

Codexへは、本書、`3DGS_定式化.tex`および`3D_Gaussian_Splatting_定式化.pdf`の3ファイルを必ずまとめて渡す。実装仕様が競合した場合は、次の順序で判断する。

1. 本書に記載されたソフトウェア構成、インターフェース、入出力、実行環境およびテスト仕様
2. `3DGS_定式化.tex`の「初期実装の仕様」およびハイパーパラメータ表
3. `3DGS_定式化.tex`の順伝播、逆伝播、損失関数および評価指標の定式化
4. 3DGS原論文
5. 原論文の公式実装

公式実装は、パラメータの保持方法、最適化器のパラメータグループ、PLY入出力および学習ループの責務分割を確認するために参照する。ただし、初期実装は定式化の検証を目的とするため、公式実装と同等の実行速度は要件としない。

上記3ファイルを一組の実装指示として使用し、未記載事項を実装者の裁量で補ってはならない。実装中に仕様が競合した場合は本節の優先順位に従い、それでも一意に決まらない場合は、該当箇所を`TODO(spec)`として明示して処理を停止する。計算デバイスについては、実行環境の可搬性を確保するため、本書の`runtime.device: auto`を定式化資料中の`cuda`指定より優先する。

## 3. 実装範囲

### 3.1 初期実装に含める機能

- Blenderで生成した多視点画像およびカメラ姿勢JSONの読み込み
- 設定した3次元領域へのGaussianのランダム初期化
- rawパラメータから実パラメータへの変換
- 世界座標からカメラ座標への変換
- Gaussian中心の透視投影
- 3D共分散および2D共分散の計算
- 実球面調和関数によるRGB値の計算
- 描画範囲の計算、深度ソートおよび前方から後方への不透明度合成
- $L_1$損失、D-SSIM損失および全損失の計算
- PyTorchの自動微分による勾配計算
- Adamによるパラメータ更新
- MSE、PSNRおよび平均PSNRの計算
- PLYおよびチェックポイントの保存・読み込み
- 数値微分と自動微分の比較を含む単体テスト

### 3.2 当初のbaselineに含めなかった機能

- Gaussianの複製、分割および除去
- 不透明度の定期的なリセット
- 球面調和関数の最大次数の段階的増加
- CUDA/C++拡張による手動の順伝播・逆伝播
- タイル単位のカリングおよびタイルごとの深度ソート
- 公式ビューアとのネットワーク通信
- OpenGLによるリアルタイム表示

この一覧は初期設計時の範囲を示す歴史的記録である。現在はGaussianの複製・分割・除去と不透明度resetを実装済みであり、3.3節のoptional extensionとして扱う。現在も未実装なのは、progressive SH degree、CUDA/C++拡張、タイルベースラスタライザ、公式viewerとの通信、およびOpenGLリアルタイムviewerである。別資料「3DGS Gaussian可視化ツール 仕様書」で定義する可視化ツールは、引き続き本書が扱う3DGSコア実装には含めない。

### 3.3 現在実装済みのoptional extension

`features.adaptive_density_control=true`では、screen-space位置勾配統計に基づくclone・splitと、opacity・screen-space size・world-space sizeに基づくpruneを実行する。6個のGaussian Parameter、Adam stateおよびdensity statisticsを同じGaussian indexで動的にappend/keepするため、学習中のGaussian数`N`は変化する。`features.opacity_reset=true`では、ADCの有効・無効とは独立に定期的な不透明度resetを実行する。

両feature flagの既定値は`false`である。この場合はstatisticsを生成せず、scene extentを計算せず、ADC由来の乱数を消費せず、当初のfixed-Gaussian baselineと同じ学習経路を使用する。

### 3.4 初期実装の性能上の位置付け

初期ラスタライザは、数式検証および小規模データによる動作確認専用とする。Gaussian単位のPythonループを許容するため、`data/`の全100視点を用いた30,000反復を実用的な時間内に完走させることは初期実装の完了要件に含めない。

CPUでも単体テスト、勾配テストおよび小規模統合テストを実行可能とする。CUDAが利用可能な場合は学習・レンダリングにCUDAを使用する。本格的なデータセットを対象とする学習は、Phase 5のタイル処理またはCUDA/C++拡張を導入した後の要件とする。

## 4. 数式ラベルと関数名の命名規則

### 4.1 基本規則

TeXの計算式ラベルを

```tex
\label{eq:world_to_camera}
```

とした場合、対応するPython関数名を

```python
def world_to_camera(...):
    ...
```

とする。すなわち、原則として`eq:`を除いた文字列と関数名を完全に一致させる。

各関数のdocstringには、対応するTeXラベルを必ず記載する。

```python
def world_to_camera(means_world, rotation_cw, translation_cw):
    """世界座標をカメラ座標へ変換する。

    TeX: eq:world_to_camera
    """
```

### 4.2 ラベルの有効性

`3DGS_定式化.tex`に含まれる`eq:`ラベルは116個であり、すべて重複がなく、`eq:`を除いた文字列はPythonの有効な識別子である。したがって、関数として実装する式については例外的な名前変換を設けない。`eq:covariance_3d`、`eq:covariance_2d`および`eq:covariance_2d_components`も、そのまま同名の関数へ対応させる。

### 4.3 一つの式を独立した関数にしない場合

次の式は、データの定義、他の式の再掲または理論上の連鎖律であり、学習時に独立して呼び出す計算ではない。

- `eq:gaussian_parameters`
- `eq:optimization_problem`
- `eq:rotation_gradient_chain`
- `eq:scale_gradient_chain`
- `eq:sh_gradient_chain`
- `eq:opacity_gradient_chain`
- `eq:position_gradient_chain`
- `eq:backward_world_to_camera`
- `eq:backward_perspective_projection`
- `eq:backward_3d_covariance`
- `eq:backward_2d_covariance`
- `eq:backward_sh_color`
- `eq:pixel_color_backward`
- `eq:transmittance_backward`

これらは独立した本番関数を作らず、対応するデータクラス、順伝播関数または検証用ヤコビアン関数のdocstringから参照する。内容が同一の式を別名の関数として重複実装してはならない。

### 4.4 独立関数を持たない式の対応先

すべての式ラベルをコードから追跡できるよう、残りの定義式および理論式は次の場所から参照する。

| TeXラベル | コード上の対応先 |
|---|---|
| `eq:raw_gaussian_parameters` | `GaussianModel`および`GaussianParameters` |
| `eq:camera_covariance_definition` | `covariance_2d()`内のローカル変数`covariance_camera` |
| `eq:backward_view_direction` | 順伝播と共通の`view_direction()` |
| `eq:pixel_color` | `rasterize_gaussians()`および`stable_color_accumulation()` |
| `eq:transmittance` | `rasterize_gaussians()`および`stable_transmittance_accumulation()` |
| `eq:pixel_opacity` | 数値安定化を含む`implementation_projected_opacity()` |
| `eq:recursive_pixel_color` | `math/jacobians.py`内の検証用ローカル再帰 |
| `eq:screen_opacity_backward` | `implementation_projected_opacity()`の中間値 |
| `eq:screen_covariance_inverse_components` | `inverse_covariance_jacobian()`の入力成分 |
| `eq:screen_covariance_vectors` | `pack_symmetric_2d()`で生成するベクトル |
| `eq:pixel_displacement_backward` | `implementation_projected_opacity()`内の`displacement` |
| `eq:gaussian_weight_backward` | `implementation_projected_opacity()`内の`gaussian_weight` |
| `eq:screen_covariance_determinant` | `safe_2d_covariance_determinant()` |
| `eq:inverse_covariance_vector` | `pack_symmetric_2d(inverse_covariance)` |

これらの対応先のdocstringまたは直前コメントにもTeXラベルを記載する。これにより、独立関数を過度に増やさず、116個すべての式ラベルの所在を追跡可能にする。

## 5. ディレクトリ構成

```text
3dgs/
├── pyproject.toml
├── README.md
├── data/
│   ├── camera_poses_blender.json
│   ├── view_000.png
│   ├── ...
│   └── view_099.png
├── docs/
│   ├── 3DGS_定式化.tex
│   └── 3D_Gaussian_Splatting_定式化.pdf
├── configs/
│   └── default.yaml
├── src/
│   └── gaussian_splatting/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── camera.py
│       │   ├── blender_loader.py
│       │   └── dataset.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── gaussian_model.py
│       │   └── initialization.py
│       ├── math/
│       │   ├── __init__.py
│       │   ├── parameterization.py
│       │   ├── transform.py
│       │   ├── covariance.py
│       │   ├── spherical_harmonics.py
│       │   └── jacobians.py
│       ├── renderer/
│       │   ├── __init__.py
│       │   ├── projection.py
│       │   ├── rasterizer.py
│       │   └── renderer.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── density_control.py
│       │   ├── losses.py
│       │   ├── optimizer.py
│       │   ├── schedules.py
│       │   └── trainer.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py
│       └── io/
│           ├── __init__.py
│           ├── checkpoint.py
│           └── ply_io.py
├── scripts/
│   ├── train.py
│   ├── render.py
│   └── evaluate.py
└── tests/
    ├── unit/
    ├── gradient/
    ├── integration/
    └── fixtures/
```

### 5.1 実行環境と依存関係

パッケージ管理には`pyproject.toml`と`pip`を使用し、Python 3.11以上3.13未満を対象とする。最低限の依存関係を次のように固定する。PyTorchは実行環境に対応するCPU版またはCUDA版を公式配布元から導入してよいが、バージョンは`2.4以上2.7未満`とする。

```toml
[project]
requires-python = ">=3.11,<3.13"
dependencies = [
  "torch>=2.4,<2.7",
  "numpy>=1.26,<3",
  "Pillow>=10,<12",
  "PyYAML>=6,<7",
  "scipy>=1.12,<2",
  "plyfile>=1.0,<2",
]

[project.optional-dependencies]
dev = [
  "pytest>=8,<9",
  "pytest-cov>=5,<7",
]
```

インストールおよびテストの標準コマンドは次のとおりとする。

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## 6. 共通データ表現

### 6.1 型、デバイスおよび座標規約

- 数式関数は`torch.float32`および`torch.float64`を受け付け、出力でも入力dtypeを維持する。異なるdtypeの浮動小数点テンソルを同じ関数へ混在させてはならない。
- 学習および通常のレンダリングは`torch.float32`、`gradcheck`および解析ヤコビアンは`torch.float64`を使用する。
- `runtime.device`が`auto`の場合はCUDAを優先し、利用できなければCPUへフォールバックする。テストはCPUでも実行可能とする。
- RGB値域は`[0, 1]`とする。
- 画像テンソルの形状は`(3, H, W)`とする。
- バッチ学習は行わず、一反復につき一視点を処理する。
- 画像の左上を`(0, 0)`とし、$x$軸を右向き、$y$軸を下向きとする。
- 画素中心は整数座標`(x, y)`として扱う。`+0.5`の画素中心補正は行わない。
- クォータニオンの成分順序は`(qw, qx, qy, qz)`とする。
- 3D対称行列の上三角ベクトル順序は`(00, 01, 02, 11, 12, 22)`とする。
- 2D対称行列の上三角ベクトル順序は`(00, 01, 11)`とする。
- SH係数は$l$の昇順、各$l$について$m=-l,\ldots,l$の昇順とし、`b(l, m) = l**2 + l + m`へ格納する。

### 6.2 `Camera`

`data/camera.py`に次の不変データクラスを定義する。

```python
@dataclass(frozen=True)
class Camera:
    rotation_cw: Tensor       # (3, 3)
    translation_cw: Tensor    # (3,)
    camera_center_world: Tensor  # (3,)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    image: Tensor | None = None  # (3, H, W)
    image_name: str | None = None
```

生成時に`torch.testing.assert_close(translation_cw, -rotation_cw @ camera_center_world, rtol=1e-5, atol=1e-6)`で整合性を検証する。

### 6.3 `GaussianModel`

`model/gaussian_model.py`に`torch.nn.Module`を継承したクラスを定義する。

```python
class GaussianModel(nn.Module):
    means_world: nn.Parameter       # (N, 3)
    raw_quaternions: nn.Parameter   # (N, 4)
    raw_scales: nn.Parameter        # (N, 3)
    raw_opacities: nn.Parameter     # (N, 1)
    sh_dc: nn.Parameter             # (N, 1, 3)
    sh_rest: nn.Parameter           # (N, 15, 3)

    @property
    def sh_coefficients(self) -> Tensor:
        return torch.cat((self.sh_dc, self.sh_rest), dim=1)
```

`means_world`、`sh_dc`および`sh_rest`はそのままレンダリングへ使用する。SH係数は異なる学習率を持つ二つの`nn.Parameter`として必ず保持し、単一の`nn.Parameter`やそのスライスとして保持してはならない。その他のrawパラメータは`raw_parameter_transformations()`を介して実パラメータへ変換する。

6個のParameterの第0次元は常に同じGaussian indexを表し、shape tail、dtypeおよびdeviceを一括検証する。ADC transactionは6個すべてを新しいleaf `nn.Parameter`へ原子的に置換し、個別Parameterの`.data`を用いたshape変更は行わない。`num_gaussians`は現在の第0次元から取得するため、ADC有効時には学習中に変化する。

`model.sh_degree`は初期実装では`3`のみ受け付ける。設定読み込み時と`GaussianModel`生成時の双方で検証し、`3`以外の場合は`ValueError`を送出する。

```python
def transformed_parameters(self) -> GaussianParameters:
    """raw変換とSH係数の連結を行い、レンダラーへ渡す値を返す。"""
```

### 6.4 `GaussianParameters`

変換後のパラメータを次のデータクラスにまとめる。

```python
@dataclass
class GaussianParameters:
    means_world: Tensor     # (N, 3)
    quaternions: Tensor     # (N, 4)
    scales: Tensor          # (N, 3)
    opacities: Tensor       # (N, 1)
    sh_coefficients: Tensor # (N, 16, 3)
```

### 6.5 `RenderResult`

```python
@dataclass
class RenderResult:
    image: Tensor                       # (3, H, W)
    final_transmittance: Tensor         # (H, W)
    projected: ProjectedGaussians       # Gaussian単位の値は(Nv, ...)で保持
    visible_mask: Tensor                # (N,), bool
```

`N`はモデルが保持する全Gaussian数、`Nv`は深度条件と画像領域との交差判定を通過したGaussian数である。`visible_mask`は元の`N`個に対するマスク、`projected.original_indices`は深度順に並べた`Nv`個から元の添字への対応とする。両者は次を満たさなければならない。

```python
torch.equal(
    torch.sort(projected.original_indices).values,
    visible_mask.nonzero().squeeze(1),
)
```

深度が等しい場合は元のGaussian添字を昇順のタイブレークキーとして使用する。Gaussianごとの中間値を`N`個へscatterして`RenderResult`に重複保持してはならない。

`projected.means_screen`はADC統計収集対象の反復だけ`retain_grad()`を呼び出す。backward後の勾配は、depth sort後の行順ではなく`projected.original_indices`を用いて元Gaussian indexへscatterする。

## 7. ファイル別設計

### 7.1 `model/initialization.py`

設定した3次元領域に生成した初期点集合から学習パラメータを生成する。

| 関数 | 対応するTeXラベル | 入力 | 出力 |
|---|---|---|---|
| `implementation_position_initialization()` | `eq:implementation_position_initialization` | 点位置`(N,3)` | 中心`(N,3)` |
| `initial_neighbor_distance()` | `eq:initial_neighbor_distance` | 点位置`(N,3)`, `k=3` | 平均二乗距離`(N,)` |
| `initial_gaussian_scale()` | `eq:initial_gaussian_scale` | 平均二乗距離`(N,)` | スケール`(N,3)` |
| `initial_raw_scale()` | `eq:initial_raw_scale` | スケール`(N,3)` | rawスケール`(N,3)` |
| `initial_raw_quaternion()` | `eq:initial_raw_quaternion` | Gaussian数`N` | rawクォータニオン`(N,4)` |
| `initial_opacity()` | `eq:initial_opacity` | `N`, `alpha_init` | 不透明度`(N,1)` |
| `initial_raw_opacity()` | `eq:initial_raw_opacity` | 不透明度`(N,1)` | raw不透明度`(N,1)` |
| `initial_sh_dc()` | `eq:initial_sh_dc` | RGB`(N,3)` | 0次係数`(N,1,3)` |
| `initial_sh_rest()` | `eq:initial_sh_rest` | `N`, `sh_degree` | 高次係数`(N,15,3)` |

公開関数`generate_initial_points(target, config, generator)`は、JSONの`target`を中心とする軸平行直方体内に`initialization.num_gaussians`個の点を一様乱数で生成する。各軸の半幅は`initialization.aabb_half_extent`とし、乱数列は`runtime.seed`で初期化した`torch.Generator`だけから生成する。初期RGBはすべて`initialization.initial_rgb`とする。

公開関数`initialize_gaussian_model(points, colors, config)`は、上記関数を順に呼び出して`GaussianModel`を返す。近傍点探索には`scipy.spatial.cKDTree`を使用し、`query(k=neighbor_count + 1)`の先頭に含まれる自点を除いて最近傍3点を選択する。点数が4未満の場合は明示的に`ValueError`を送出する。

### 7.2 `math/parameterization.py`

| 関数 | 対応するTeXラベル | 仕様 |
|---|---|---|
| `raw_parameter_transformations()` | `eq:raw_parameter_transformations` | `exp(raw_scales)`、`sigmoid(raw_opacities)`、クォータニオン正規化を行う |

```python
def raw_parameter_transformations(
    raw_quaternions: Tensor,
    raw_scales: Tensor,
    raw_opacities: Tensor,
    epsilon_q: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """TeX: eq:raw_parameter_transformations"""
```

戻り値の順序は`quaternions, scales, opacities`とする。クォータニオンのノルムは`clamp_min(epsilon_q)`で下限を設ける。

### 7.3 `math/transform.py`

| 関数 | 対応するTeXラベル | 入力 | 出力 |
|---|---|---|---|
| `world_to_camera()` | `eq:world_to_camera` | `(N,3)`, `(3,3)`, `(3,)` | `(N,3)` |
| `perspective_projection()` | `eq:perspective_projection` | `(N,3)`, 内部パラメータ | `(N,2)` |
| `projection_jacobian()` | `eq:projection_jacobian` | `(N,3)`, `fx`, `fy` | `(N,2,3)` |
| `view_direction()` | `eq:view_direction` | 中心`(N,3)`, カメラ中心`(3,)` | 単位方向`(N,3)` |

`perspective_projection()`および`projection_jacobian()`は、`z > 0`を満たすGaussianに対してのみ呼び出す。全Gaussianを入力する場合は、呼び出し側が可視マスクを適用する。

### 7.4 `math/covariance.py`

| 関数 | 対応するTeXラベル | 入力 | 出力 |
|---|---|---|---|
| `quaternion_rotation_matrix()` | `eq:quaternion_rotation_matrix` | `(N,4)` | `(N,3,3)` |
| `scale_matrix()` | `eq:scale_matrix` | `(N,3)` | `(N,3,3)` |
| `covariance_3d()` | `eq:covariance_3d` | 回転、スケール | `(N,3,3)` |
| `covariance_2d()` | `eq:covariance_2d` | `J`, `R_cw`, `Sigma` | `(N,2,2)` |
| `covariance_2d_components()` | `eq:covariance_2d_components` | `(N,2,2)` | `(N,3)` |
| `inverse_2d_covariance()` | `eq:inverse_2d_covariance` | `(N,2,2)` | `(N,2,2)` |
| `symmetrized_2d_covariance()` | `eq:symmetrized_2d_covariance` | `(N,2,2)` | `(N,2,2)` |
| `stabilized_2d_covariance()` | `eq:stabilized_2d_covariance` | `(N,2,2)`, `epsilon_cov` | `(N,2,2)` |
| `safe_2d_covariance_determinant()` | `eq:safe_2d_covariance_determinant` | `(N,2,2)`, `epsilon_det` | `(N,)` |
| `stabilized_inverse_2d_covariance()` | `eq:stabilized_inverse_2d_covariance` | `(N,2,2)` | `(N,2,2)` |

実装順序は次のとおりとする。

1. 単位クォータニオンから回転行列を計算する。
2. `Sigma = R @ diag(scales**2) @ R.T`として3D共分散を計算する。
3. `C = R_cw @ Sigma @ R_cw.T`を計算する。
4. `Sigma_2d = J @ C @ J.T`として2D共分散を計算する。
5. 2D共分散を対称化する。
6. `0.3 * I`を加える。
7. 行列式を`1e-8`以上へ制限して逆行列を計算する。

対称行列を上三角ベクトルとして入出力する補助関数`pack_symmetric_3d()`、`unpack_symmetric_3d()`、`pack_symmetric_2d()`および`unpack_symmetric_2d()`を定義する。ただし、主要な計算では可読性を優先し、行列形式を使用する。

### 7.5 `math/spherical_harmonics.py`

| 関数 | 対応するTeXラベル | 仕様 |
|---|---|---|
| `real_sh_degree_3()` | `eq:real_sh_degree_3` | $l=0,1,2,3$の16個の実SH基底を返す |
| `sh_color_theory()` | `eq:sh_color_theory` | SH係数と基底の線形結合のみを計算する |
| `sh_color_implementation()` | `eq:sh_color_implementation` | 線形結合に`0.5`を加え、`clamp_min(0)`を適用する |

```python
def real_sh_degree_3(directions: Tensor) -> Tensor:
    """TeX: eq:real_sh_degree_3. Returns: (..., 16)."""

def sh_color_implementation(
    directions: Tensor,          # (N, 3)
    sh_coefficients: Tensor,     # (N, 16, 3)
    active_degree: int = 3,
) -> Tensor:                     # (N, 3)
    """TeX: eq:sh_color_implementation"""
```

`3DGS_定式化.tex`と同じ実球面調和関数の符号規約を使用する。公式実装の係数配置と一致することを固定入力によるテストで確認する。

### 7.6 `renderer/projection.py`

数式関数を組み合わせ、Gaussianごとの画像平面上の情報を計算する。

| 関数 | 対応するTeXラベル | 役割 |
|---|---|---|
| `visible_depth_condition()` | `eq:visible_depth_condition` | `z > 0`の可視マスクを返す |
| `maximum_2d_covariance_eigenvalue()` | `eq:maximum_2d_covariance_eigenvalue` | 2D共分散の最大固有値を解析式で計算する |
| `gaussian_rendering_radius()` | `eq:gaussian_rendering_radius` | `ceil(3 * sqrt(lambda_max))`を計算する |
| `gaussian_rendering_rectangle()` | `eq:gaussian_rendering_rectangle` | 画像内へクリップした整数矩形を返す |
| `pixel_coordinate_convention()` | `eq:pixel_coordinate_convention` | `(H,W,2)`の画素座標グリッドを生成する |
| `implementation_depth_order()` | `eq:implementation_depth_order` | 深度と元インデックスによる安定ソート順を返す |

`project_gaussians(parameters, camera, config)`は本ファイルの公開統合関数とする。入力には変換済みの`GaussianParameters`だけを受け取り、rawパラメータ変換は行わない。本関数が3D共分散、座標変換、透視投影、2D共分散、SH色、描画半径・矩形、可視判定および深度ソートまでのGaussian単位の前処理を一括して担当する。戻り値は`ProjectedGaussians`と元の`N`個に対する`visible_mask`のタプルとする。

```python
@dataclass
class ProjectedGaussians:
    original_indices: Tensor       # (Nv,), 深度順、同一深度では元添字順
    means_camera: Tensor           # (Nv,3)
    means_screen: Tensor           # (Nv,2)
    depths: Tensor                 # (Nv,)
    covariances_3d: Tensor         # (Nv,3,3)
    covariances_2d: Tensor         # (Nv,2,2)
    inverse_covariances_2d: Tensor # (Nv,2,2)
    colors: Tensor                 # (Nv,3)
    opacities: Tensor              # (Nv,1)
    radii: Tensor                  # (Nv,)
    rectangles: Tensor             # (Nv,4), xmin,xmax,ymin,ymax
```

画像領域と描画矩形が交差しないGaussianは`ProjectedGaussians`から除外する。

```python
def project_gaussians(
    parameters: GaussianParameters,
    camera: Camera,
    config: RenderingConfig,
) -> tuple[ProjectedGaussians, Tensor]:
    """Gaussian単位の全前処理を行う。第2戻り値は(N,)のvisible_mask。"""
```

### 7.7 `renderer/rasterizer.py`

| 関数 | 対応するTeXラベル | 役割 |
|---|---|---|
| `implementation_projected_opacity()` | `eq:implementation_projected_opacity` | マハラノビス距離から画素不透明度を計算する |
| `clamped_projected_opacity()` | `eq:clamped_projected_opacity` | 不透明度を`0.99`以下へ制限する |
| `opacity_contribution_threshold()` | `eq:opacity_contribution_threshold` | `1/255`未満の寄与を0とする |
| `stable_transmittance_accumulation()` | `eq:stable_transmittance_accumulation` | 合成後の候補透過率を計算する |
| `transmittance_termination()` | `eq:transmittance_termination` | 候補透過率が`1e-4`未満か判定する |
| `stable_color_accumulation()` | `eq:stable_color_accumulation` | 打ち切られない場合のみ累積色を更新する |
| `implementation_pixel_color_with_background()` | `eq:implementation_pixel_color_with_background` | 残存透過率と背景色を合成する |

```python
def rasterize_gaussians(
    projected: ProjectedGaussians,
    height: int,
    width: int,
    background: Tensor,
    alpha_max: float = 0.99,
    alpha_min: float = 1.0 / 255.0,
    transmittance_min: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Returns image (3,H,W) and final transmittance (H,W)."""
```

初期実装では、深度順のGaussianごとに描画矩形内の画素をベクトル化して更新する。画素ごとのPythonループは避ける。Gaussian単位のループは許容する。

各Gaussianについて、`test_transmittance = transmittance * (1.0 - alpha_hat)`を先に計算する。`test_transmittance < transmittance_min`となる画素では、そのGaussianを色へ合成せず、その画素を以後の処理対象から外す。閾値以上の画素だけで累積色を更新し、`transmittance = test_transmittance`とする。深度ソート、描画矩形、閾値判定および終了判定自体に関する勾配は計算しない。

### 7.8 `renderer/renderer.py`

```python
class GaussianRenderer(nn.Module):
    def forward(
        self,
        model: GaussianModel,
        camera: Camera,
        retain_screen_grad: bool = False,
    ) -> RenderResult:
        ...
```

処理順序は次のとおりとする。

1. `model.transformed_parameters()`を呼び、rawパラメータを`GaussianParameters`へ変換する。
2. `project_gaussians(parameters, camera, config)`を1回だけ呼び、Gaussian単位の全前処理結果と`visible_mask`を得る。
3. `retain_screen_grad=True`かつ`projected.means_screen.requires_grad`の場合、`projected.means_screen.retain_grad()`を呼ぶ。
4. `rasterize_gaussians()`を呼び、前方から後方への色合成と背景色合成を行う。
5. `image`、`final_transmittance`、`projected`および`visible_mask`から`RenderResult`を構築する。

`GaussianRenderer.forward()`内で、3D共分散、投影、SH色または可視判定を重複して再計算してはならない。

### 7.9 `training/losses.py`

| 関数 | 対応するTeXラベル | 入出力 |
|---|---|---|
| `l1_norm()` | `eq:l1_norm` | RGBベクトルの$L_1$ノルム |
| `l1_loss()` | `eq:l1_loss` | 2画像からスカラー損失 |
| `ssim_statistics()` | `eq:ssim_statistics` | 局所平均、分散および共分散 |
| `ssim_constants()` | `eq:ssim_constants` | `C1`, `C2` |
| `ssim()` | `eq:ssim` | 2画像から平均SSIM |
| `dssim_loss()` | `eq:dssim_loss` | `1 - ssim` |
| `total_loss()` | `eq:total_loss` | $L_1$とD-SSIMの加重和 |

SSIMは、チャネルごとに独立した`11 x 11`、標準偏差`1.5`の正規化ガウス窓を用いる。パディング幅は5、パディング値は0とする。最終値は全画素および全チャネルについて平均する。

```python
@dataclass
class LossResult:
    total: Tensor
    l1: Tensor
    dssim: Tensor
    ssim: Tensor

def total_loss(
    rendered: Tensor,
    target: Tensor,
    lambda_dssim: float = 0.2,
) -> LossResult:
    """TeX: eq:total_loss"""
```

### 7.10 `training/schedules.py`

| 関数 | 対応するTeXラベル | 仕様 |
|---|---|---|
| `position_learning_rate_schedule()` | `eq:position_learning_rate_schedule` | `1.6e-4`から`1.6e-6`まで30,000反復で指数減衰 |

公開される学習反復番号は1始まりとし、`iteration`は`1 <= iteration <= total_iterations`へ制限する。反復1を初回、`total_iterations`を最終反復として扱い、この反復番号をそのまま学習率scheduleへ渡す。`LambdaLR`へ渡す倍率ではなく、現在の学習率そのものを返す。

`PositionLearningRateScheduler`は現在反復を保持し、`step(iteration)`で`means_world`グループの学習率を更新する。`state_dict()`と`load_state_dict()`を実装し、学習再開時に反復位置を復元する。

`density_control_schedule(config, iteration)`はmutableなlast-event状態を持たないpure policyとする。ADC有効時は`iteration < densify_until_iteration`でstatisticsを収集し、`iteration > densify_from_iteration`、`iteration < densify_until_iteration`、かつ`iteration % densification_interval == 0`でeventを実行する。既定値では統計収集は反復14999まで、eventは600から14900まで100反復間隔となる。

opacity resetは独立したfeature flagで制御し、`iteration < densify_until_iteration`かつ`iteration % opacity_reset_interval == 0`で実行する。白背景では`iteration == densify_from_iteration`でも特別にresetする。screen-spaceおよびworld-space size pruningは`iteration > opacity_reset_interval`のdensity eventでだけ有効とする。

`compute_scene_extent(train_cameras)`はtraining camera centerだけを用い、次の公式実装相当の値を返す。evaluation cameraは含めない。

```text
C_bar = mean(C_i)
D = max_i ||C_i - C_bar||_2
scene_extent = 1.1 * D
```

clone/split境界は`percent_dense * scene_extent`、world-space prune閾値は`prune_world_scale_fraction * scene_extent`とする。scene extentはTrainer初期化時に一度だけ計算し、Gaussian数が変化しても再計算しない。

### 7.11 `training/optimizer.py`

`create_optimizer(model, config)`は、次のパラメータグループを持つAdamを生成する。

| グループ名 | 対象 | 学習率 |
|---|---|---:|
| `means_world` | 中心位置 | 反復ごとのスケジュール |
| `sh_dc` | 0次SH係数 | `2.5e-3` |
| `sh_rest` | 高次SH係数 | `1.25e-4` |
| `raw_opacities` | raw不透明度 | `5.0e-2` |
| `raw_scales` | rawスケール | `5.0e-3` |
| `raw_quaternions` | rawクォータニオン | `1.0e-3` |

Adamの設定は`betas=(0.9, 0.999)`、`eps=1e-15`、`weight_decay=0`とする。`sh_dc`と`sh_rest`は、6.3節で確定した別々の`nn.Parameter`をそのまま各パラメータグループへ登録する。`sh_coefficients`プロパティの戻り値やそのスライスを最適化器へ登録してはならない。

各named parameter groupは対象Parameterを1個だけ保持する。Gaussian append時は既存`exp_avg`および`exp_avg_sq`を保持して新規行を0で追加し、keep/prune時はParameterと同一のmaskをmomentにも適用する。splitでは親のmomentを除去し、childのmomentを0から開始する。opacity resetではopacity groupだけを新Parameterへ差し替え、`exp_avg`、`exp_avg_sq`および存在する場合の`max_exp_avg_sq`を全行0へresetする。いずれもAdamのscalar `step`とnamed groupおよび現在の学習率を維持し、optimizerを作り直さない。optimizer state未生成時にも不要なstateを生成せず動作する。

`eq:gradient_descent_update`は更新の概念式であり、本番実装では独立関数を作成せず、`torch.optim.Adam.step()`を使用する。

#### 7.11.1 `training/density_control.py`

`ScreenSpaceDensityStatistics`はGaussianごとに次のshape `(N,)`のTensorを保持する。

- `position_gradient_accumulator`: screen-space位置勾配の$x,y$成分のL2 normの累積値。modelと同じfloating dtype/device。
- `position_gradient_denominator`: 観測回数。`int64`、modelと同じdevice。
- `max_screen_radius`: statistics window内の最大screen radius。`int64`、modelと同じdevice。

平均位置勾配は`position_gradient_accumulator / position_gradient_denominator`とし、未観測でdenominatorが0のGaussianには0を返す。蓄積では`projected.original_indices`を用いてdepth-sort後の行を元Gaussian indexへ戻す。appendしたGaussianの統計は0から開始し、keep/pruneではParameterと同じmaskで既存統計を保持する。

clone、splitおよびpruneの選択条件は実パラメータとevent開始前までの平均勾配から次のように決める。

- Clone: `mean_gradient >= position_gradient_threshold`かつ`max(actual_scale) <= percent_dense * scene_extent`。選択した親の`means_world`、`raw_quaternions`、`raw_scales`、`raw_opacities`、`sh_dc`、`sh_rest`を変更せず、元index順で末尾へ完全コピーする。親は残し、childのAdam momentとstatisticsは0とする。
- Split: `mean_gradient >= position_gradient_threshold`かつ`max(actual_scale) > percent_dense * scene_extent`。各親から2 childを生成する。`epsilon ~ N(0, I)`、`offset_local = actual_scale * epsilon`、`offset_world = R(q) offset_local`、`child_mean = parent_mean + offset_world`とし、childの実scaleを`parent_scale / 1.6`とする。rotation・opacity・SHは親をコピーし、親を削除してchildのAdam momentとstatisticsを0とする。
- Prune: `sigmoid(raw_opacity) < prune_opacity_threshold`、有効時の`max_screen_radius > prune_screen_radius_threshold`、有効時の`max(actual_scale) > prune_world_scale_threshold`のunionを削除する。理由別件数は非排他的に数え、total件数はunionを一度だけ数える。

1回の`run_density_control_event()`は`clone -> split -> prune -> statistics reset`の順に実行し、`1 event = 1 statistics window`とする。cloneとsplitはevent開始前のstatisticsに基づいて相補的なscale条件を使い、新しく追加したGaussianのstatisticsは0であるため同一event内で再densifyしない。pruneはclone/split後の最終集合へ適用する。eventがno-opでも成功時にはstatistics windowをresetする。

不透明度resetでは、実opacityを`alpha_new = min(alpha, opacity_reset_maximum)`とし、既定上限を`0.01`とする。単調性により等価なraw-domain clampを行って新しいleaf `raw_opacities` Parameterへ置換し、opacityのAdam momentsを全行0へresetする。閾値以下で値が変わらないGaussianがあってもmoment resetは実行し、scalar stepは維持する。

model Parameter、optimizer group/state、statisticsは一つのGaussian index transactionとして扱う。入力とstateをmutation前に検証・事前構築し、commit途中で失敗した場合はすべてを復元する。splitとeventでは、乱数消費後の後段処理が失敗した場合にCPUまたは対象CUDA deviceのPyTorch RNG stateもevent開始前へrollbackする。

### 7.12 `training/trainer.py`

```python
class Trainer:
    def train(self) -> None: ...
    def train_step(self, camera: Camera, iteration: int) -> TrainStepResult: ...
    def validate(self, iteration: int) -> EvaluationResult: ...
    def save_checkpoint(self, iteration: int) -> None: ...
```

一反復の処理は次のとおりとする。

1. 未使用の学習視点集合から一視点をランダムに選択する。
2. 中心位置の学習率を更新する。
3. `optimizer.zero_grad(set_to_none=True)`を呼び出す。
4. 対象視点をレンダリングする。
5. 真値画像との損失を計算する。
6. `loss.total.backward()`を呼び出す。
7. 勾配に`NaN`または`Inf`がないことを検査する。
8. schedule対象ならscreen-space density statisticsを蓄積する。
9. schedule対象ならdensity-control eventを実行する。
10. schedule対象ならopacity resetを実行する。
11. Parameterに`NaN`または`Inf`がないことを検査する。
12. `optimizer.step()`を呼び出す。
13. Parameterに`NaN`または`Inf`がないことを再検査する。
14. 損失、PSNR、event後のGaussian数および学習率をログへ記録する。
15. 評価およびcheckpoint保存を行う。

重要部分の順序は`render -> loss -> backward -> statistics accumulate -> density-control event -> opacity reset -> optimizer.step -> logging/evaluation/checkpoint`である。Parameter replacement前の古いgradientは新Parameterへコピーしない。置換されなかったParameterだけが当該反復の通常の`optimizer.step()`対象となる。

全視点を一度ずつ選択した後、視点リストを再度シャッフルする。30,000反復を既定値とする。

新規学習時の反復範囲は`range(1, total_iterations + 1)`とする。ログ、評価結果、チェックポイント名およびチェックポイント内部の`iteration`にも、同じ1始まりの反復番号を記録する。

`log_interval`ごとに学習ログ、`evaluation_interval`ごとに評価、`checkpoint_interval`ごとにチェックポイント保存を行う。最終反復では各間隔に一致しない場合でも、ログ・評価・チェックポイント保存を必ず行う。

### 7.13 `evaluation/metrics.py`

| 関数 | 対応するTeXラベル | 仕様 |
|---|---|---|
| `mse()` | `eq:mse` | 全画素・全RGB成分の二乗誤差平均 |
| `psnr()` | `eq:psnr` | `I_max=1`として画像ごとに計算 |
| `mean_psnr()` | `eq:mean_psnr` | 各テスト画像のPSNRの算術平均 |

`psnr()`はMSEが厳密に0の場合に`torch.inf`を返し、任意の上限値でクリップしない。評価時のみレンダリング画像を`[0,1]`へクリップする。学習時の損失計算前には画像全体のクリップを行わない。

### 7.14 `math/jacobians.py`

本ファイルは、`3DGS_定式化.tex`の逆伝播式を検証するための参照実装であり、通常の学習経路からは呼び出さない。各関数は小規模なテンソルを対象とし、自動微分または有限差分との比較に用いる。

全関数は単一Gaussian・単一画素を対象とし、バッチ次元を受け付けない。入力と出力は`torch.float64`、CPUを必須とする。ヤコビアンの形状は`(出力の自由度, 入力の自由度)`とする。3D対称行列は`(00,01,02,11,12,22)`の6要素、2D対称行列は`(00,01,11)`の3要素へpackしたものを微分対象とする。RGBと位置は通常の成分順、SH係数ベクトルは`(basis, rgb)`をbasis優先で平坦化した順とする。

各関数の引数と戻り値を次のように固定する。表中の`B=(active_degree+1)^2`である。

| 関数 | 引数 | 戻り値 | TeXラベル |
|---|---|---|---|
| `backward_world_to_camera_jacobian` | `rotation_cw (3,3)` | `(3,3)` | `eq:backward_world_to_camera_jacobian` |
| `backward_projection_jacobian` | `point_camera (3,), fx, fy` | `(2,3)` | `eq:backward_projection_jacobian` |
| `covariance_quaternion_matrix_derivative` | `quaternion (4,), scales (3,), component: int` | `(3,3)` | `eq:covariance_quaternion_matrix_derivative` |
| `rotation_qw_jacobian` | `quaternion (4,)` | `(3,3)` | `eq:rotation_qw_jacobian` |
| `rotation_qx_jacobian` | `quaternion (4,)` | `(3,3)` | `eq:rotation_qx_jacobian` |
| `rotation_qy_jacobian` | `quaternion (4,)` | `(3,3)` | `eq:rotation_qy_jacobian` |
| `rotation_qz_jacobian` | `quaternion (4,)` | `(3,3)` | `eq:rotation_qz_jacobian` |
| `covariance_quaternion_jacobian` | `quaternion (4,), scales (3,)` | `(6,4)` | `eq:covariance_quaternion_jacobian` |
| `scale_squared_matrix_derivative` | `scales (3,), component: int` | `(3,3)` | `eq:scale_squared_matrix_derivative` |
| `covariance_scale_matrix_derivative` | `quaternion (4,), scales (3,), component: int` | `(3,3)` | `eq:covariance_scale_matrix_derivative` |
| `covariance_scale_jacobian` | `quaternion (4,), scales (3,)` | `(6,3)` | `eq:covariance_scale_jacobian` |
| `projected_covariance_3d_component_derivative` | `point_camera (3,), rotation_cw (3,3), fx, fy, component: int` | `(2,2)` | `eq:projected_covariance_3d_component_derivative` |
| `projected_covariance_3d_jacobian` | `point_camera (3,), rotation_cw (3,3), fx, fy` | `(3,6)` | `eq:projected_covariance_3d_jacobian` |
| `projection_jacobian_x_derivative` | `point_camera (3,), fx, fy` | `(2,3)` | `eq:projection_jacobian_x_derivative` |
| `projection_jacobian_y_derivative` | `point_camera (3,), fx, fy` | `(2,3)` | `eq:projection_jacobian_y_derivative` |
| `projection_jacobian_z_derivative` | `point_camera (3,), fx, fy` | `(2,3)` | `eq:projection_jacobian_z_derivative` |
| `projected_covariance_position_matrix_derivative` | `covariance_3d (3,3), point_camera (3,), rotation_cw (3,3), fx, fy, component: int` | `(2,2)` | `eq:projected_covariance_position_matrix_derivative` |
| `projected_covariance_position_jacobian` | `covariance_3d (3,3), point_camera (3,), rotation_cw (3,3), fx, fy` | `(3,3)` | `eq:projected_covariance_position_jacobian` |
| `color_sh_component_jacobian` | `direction (3,), basis_index: int` | `(3,3)` | `eq:color_sh_component_jacobian` |
| `color_sh_coefficient_jacobian` | `direction (3,), active_degree: int` | `(3,3B)` | `eq:color_sh_coefficient_jacobian` |
| `view_direction_position_jacobian` | `mean_world (3,), camera_center_world (3,)` | `(3,3)` | `eq:view_direction_position_jacobian` |
| `color_view_direction_jacobian` | `direction (3,), sh_coefficients (B,3)` | `(3,3)` | `eq:color_view_direction_jacobian` |
| `color_world_position_jacobian` | `mean_world (3,), camera_center_world (3,), sh_coefficients (B,3)` | `(3,3)` | `eq:color_world_position_jacobian` |
| `recursive_color_opacity_jacobian` | `color (3,), next_recursive_color (3,)` | `(3,)` | `eq:recursive_color_opacity_jacobian` |
| `pixel_color_recursive_jacobian` | `transmittance: scalar` | `(3,3)` | `eq:pixel_color_recursive_jacobian` |
| `pixel_color_screen_opacity_jacobian` | `transmittance: scalar, color (3,), next_recursive_color (3,)` | `(3,)` | `eq:pixel_color_screen_opacity_jacobian` |
| `pixel_color_color_jacobian` | `transmittance: scalar, screen_opacity: scalar` | `(3,3)` | `eq:pixel_color_color_jacobian` |
| `screen_opacity_parameter_derivative` | `displacement (2,), inverse_covariance (2,2)` | `scalar` | `eq:screen_opacity_parameter_derivative` |
| `gaussian_weight_position_jacobian` | `displacement (2,), inverse_covariance (2,2)` | `(2,)` | `eq:gaussian_weight_position_jacobian` |
| `screen_opacity_position_jacobian` | `opacity: scalar, displacement (2,), inverse_covariance (2,2)` | `(2,)` | `eq:screen_opacity_position_jacobian` |
| `gaussian_weight_inverse_covariance_jacobian` | `displacement (2,), inverse_covariance (2,2)` | `(3,)` | `eq:gaussian_weight_inverse_covariance_jacobian` |
| `inverse_covariance_jacobian` | `covariance_vector (3,)` | `(3,3)` | `eq:inverse_covariance_jacobian` |
| `screen_opacity_covariance_jacobian` | `opacity: scalar, displacement (2,), covariance_vector (3,)` | `(3,)` | `eq:screen_opacity_covariance_jacobian` |
| `pixel_color_opacity_parameter_jacobian` | `transmittance: scalar, color (3,), next_recursive_color (3,), gaussian_weight: scalar` | `(3,)` | `eq:pixel_color_opacity_parameter_jacobian` |
| `pixel_color_position_jacobian` | `transmittance: scalar, color (3,), next_recursive_color (3,), opacity: scalar, displacement (2,), inverse_covariance (2,2)` | `(3,2)` | `eq:pixel_color_position_jacobian` |
| `pixel_color_covariance_jacobian` | `transmittance: scalar, color (3,), next_recursive_color (3,), opacity: scalar, displacement (2,), covariance_vector (3,)` | `(3,3)` | `eq:pixel_color_covariance_jacobian` |

`screen_opacity_backward`、`screen_covariance_inverse_components`、`screen_covariance_vectors`、`pixel_displacement_backward`、`gaussian_weight_backward`、`screen_covariance_determinant`および`inverse_covariance_vector`は中間変数の定義式であるため、独立関数ではなく上表の関数内のローカル変数として実装する。

`component`は、クォータニオンでは`0=qw, 1=qx, 2=qy, 3=qz`、位置・スケールでは`0=x, 1=y, 2=z`とする。範囲外は`ValueError`とする。各関数は対応TeXラベル、引数形状、戻り値形状および成分順をdocstringへ記載する。

### 7.15 データ読み込み・入出力

数式に直接対応しない周辺機能は、責務が重複しないよう次の関数へ集約する。

| ファイル | 関数 | 役割 |
|---|---|---|
| `data/blender_loader.py` | `read_camera_poses_json()` | `camera_poses_blender.json`を読み込み、スキーマを検証する |
| `data/blender_loader.py` | `blender_c2w_to_opencv_w2c()` | Blender形式のcamera-to-world行列をレンダラ用world-to-camera行列へ変換する |
| `data/dataset.py` | `load_blender_dataset()` | JSONとPNGを`Camera`列に変換する |
| `data/dataset.py` | `split_cameras()` | `image_name`順と`data.test_every`により学習・評価視点を分割する |
| `io/checkpoint.py` | `save_checkpoint()` | モデル、Adam状態、反復番号、設定、乱数状態を保存する |
| `io/checkpoint.py` | `load_checkpoint()` | チェックポイントを読み込み、学習再開状態を復元する |
| `io/ply_io.py` | `save_gaussians_ply()` | 学習済みGaussianを公式実装互換の属性名で保存する |
| `io/ply_io.py` | `load_gaussians_ply()` | PLY属性を`GaussianModel`のrawパラメータへ復元する |

Blenderのカメラ情報と画像の変換規約を次のように固定する。

- `transform_matrix_blender_c2w`は、Blender世界座標系におけるcamera-to-world行列である。Blenderカメラのローカル座標系`(x右, y上, -z前)`をOpenCV形式`(x右, y下, z前)`へ変換する。
- `B = diag(1, -1, -1, 1)`とし、`c2w_opencv = c2w_blender @ B`、`w2c_opencv = inverse(c2w_opencv)`とする。`rotation_cw = w2c_opencv[:3,:3]`、`translation_cw = w2c_opencv[:3,3]`、`camera_center_world = c2w_opencv[:3,3]`とする。
- JSONの`coordinate_system`は`Blender: x-y floor plane, z-up`、`camera_transform`は`camera-to-world`と完全一致しなければならない。
- 画像幅`W`、高さ`H`、焦点距離`lens_mm`、センサー幅`sensor_width_mm`から、`fx=fy=lens_mm/sensor_width_mm*W`、`cx=W/2`、`cy=H/2`とする。本データでは`W=H=800`、`lens_mm=35`、`sensor_width_mm=36`のため、`fx=fy=777.777...`、`cx=cy=400`となる。
- Pillowで画像を開いた直後に`ImageOps.exif_transpose()`を適用する。その後、グレースケールはRGBの3チャネルへ複製し、RGB画像はそのまま使用する。
- RGBA画像は、`data.rgba_background`が`black`ならRGBに黒背景を合成し、`white`なら白背景を合成する。それ以外の値は設定エラーとする。
- `resolution_scale`は`0 < resolution_scale <= 1`とする。元画像サイズを`(W,H)`、変換後を`(W',H')`とし、`W'=round(W*scale)`、`H'=round(H*scale)`とする。内部パラメータは`fx*=W'/W`、`cx*=W'/W`、`fy*=H'/H`、`cy*=H'/H`で変換する。
- 画像をリサイズする場合は、公式実装の`PILtoTorch`に合わせて`PIL.Image.resize((W', H'))`を補間方式の明示なしで呼び出し、Pillowの既定の補間方法を使用する。
- JSONの`resolution_x`、`resolution_y`と実ファイルのEXIF補正後サイズが一致しない場合は、暗黙に補正せず`ValueError`を送出する。
- `frames`は`index`の昇順で並べ、`index`の重複や欠番、`file_path`の重複、対応画像の不存在はエラーとする。0始まりのフレーム順位`k`について`k % test_every == 0`を評価用、それ以外を学習用とする。`test_every <= 1`、または学習・評価のいずれかが空になる場合は`ValueError`を送出する。

### 7.16 コマンドラインエントリーポイント

| スクリプト | 主な引数 | 処理 |
|---|---|---|
| `scripts/train.py` | `--data`, `--config`, `--output`, `--resume CHECKPOINT` | データ読み込み、初期化または再開、学習、定期保存 |
| `scripts/render.py` | `--data`, `--checkpoint`, `--split`, `--output` | 指定視点をレンダリングしてPNGへ保存 |
| `scripts/evaluate.py` | `--data`, `--checkpoint`, `--split`, `--output` | 各画像のPSNRと平均PSNRを計算して指定先のJSONへ保存 |

`--resume`は真偽値フラグではなく、再開元の`.pt`チェックポイントパスを受け取る任意引数とする。指定時は新規初期化を行わず、9.2節の全状態を復元する。`evaluate.py`の`--output`は出力JSONのファイルパスを受け取る必須引数とする。各スクリプトの`main()`は引数解析と依存関係の組み立てのみを担当し、数式処理を直接実装しない。

標準実行コマンドは次のとおりとする。

```bash
python scripts/train.py --data data --config configs/default.yaml --output output/run001
python scripts/render.py --data data --checkpoint output/run001/checkpoints/latest.pt --split test --output output/run001/renders/test
python scripts/evaluate.py --data data --checkpoint output/run001/checkpoints/latest.pt --split test --output output/run001/metrics/evaluation.json
```

## 8. 設定ファイル

`configs/default.yaml`に次の設定を記載する。

`config.py`にはYAMLの各トップレベルキーに対応する`RuntimeConfig`、`DataConfig`、`ModelConfig`、`InitializationConfig`、`RenderingConfig`、`LossConfig`、`TrainingConfig`、`AdaptiveDensityControlConfig`、`OutputConfig`および`FeatureConfig`を`@dataclass(frozen=True)`として定義し、それらを束ねる`Config`を定義する。未知のキー、必須キーの欠落または型不一致は読み込み時にエラーとし、暗黙の既定値補完は行わない。

```yaml
runtime:
  device: auto
  dtype: float32
  seed: 0

data:
  camera_file: camera_poses_blender.json
  resolution_scale: 1.0
  rgba_background: black
  test_every: 8

model:
  sh_degree: 3
  epsilon_q: 1.0e-8

initialization:
  num_gaussians: 1000
  aabb_half_extent: [1.0, 1.0, 1.0]
  initial_rgb: [0.5, 0.5, 0.5]
  neighbor_count: 3
  epsilon_scale: 1.0e-7
  opacity: 0.1

rendering:
  background: [0.0, 0.0, 0.0]
  sigma_extent: 3.0
  epsilon_covariance: 0.3
  epsilon_determinant: 1.0e-8
  alpha_max: 0.99
  alpha_min: 0.00392156862745098
  transmittance_min: 1.0e-4
  tile_based: false

loss:
  lambda_dssim: 0.2
  ssim_window_size: 11
  ssim_sigma: 1.5
  ssim_k1: 0.01
  ssim_k2: 0.03

training:
  iterations: 30000
  views_per_iteration: 1
  optimizer: adam
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-15
  weight_decay: 0.0
  position_lr_initial: 1.6e-4
  position_lr_final: 1.6e-6
  sh_dc_lr: 2.5e-3
  sh_rest_lr: 1.25e-4
  opacity_lr: 5.0e-2
  scale_lr: 5.0e-3
  quaternion_lr: 1.0e-3
  log_interval: 10
  evaluation_interval: 1000
  checkpoint_interval: 5000
  save_best_by: mean_psnr

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

output:
  exist_policy: error
  save_rendered_images: true

features:
  adaptive_density_control: false
  opacity_reset: false
  progressive_sh_degree: false
```

## 9. 入出力設計

### 9.1 入力データ

動作確認用データは、プロジェクトルート直下の`data/`に配置する。`src/gaussian_splatting/data/`はデータ読み込みコードであり、実データを置くルート直下の`data/`とは役割が異なる。

```text
data/
├── camera_poses_blender.json
├── view_000.png
├── ...
└── view_099.png
```

`camera_poses_blender.json`には100個の`frames`があり、各フレームの`file_path`、`index`、`azimuth_deg`、`elevation_deg`および`transform_matrix_blender_c2w`を格納する。画像は`view_000.png`から`view_099.png`までの100枚、解像度は`800×800`、各PNGはRGBAである。カメラは方位角20方向×仰角5段階である。COLMAP形式の`cameras.bin`、`images.bin`および`points3D.bin`は使用しない。

### 9.2 チェックポイント

次の内容を保存する。

```python
{
    "checkpoint_version": 2,
    "iteration": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "config": dict,
    "python_random_state": object,
    "numpy_random_state": tuple,
    "torch_cpu_random_state": Tensor,
    "torch_cuda_random_states": list[Tensor] | None,
    "camera_order": list[int],
    "camera_cursor": int,
    "best_mean_psnr": float | None,
    "density_statistics_state": dict[str, Tensor] | None,
}
```

`density_statistics_state`はADC無効時は`None`、有効時は3個のstatistics Tensorを持つmappingとする。model stateのTensor shapeから保存時点の可変`N`を復元し、Adam stateも同じshapeへ復元する。version keyを持たない旧fixed-Gaussian checkpointはversion 1として読み込み、保存側・再開側ともADCとopacity resetが`false`である場合に限り、旧configに`density_control` sectionがなくても互換性を維持する。

保存対象はmodel、optimizer、scheduler、config、Python RNG、NumPy RNG、PyTorch CPU/CUDA RNG、camera order/cursor、best PSNRおよびdensity statisticsである。statisticsはinterval途中の非zero値も保存し、resume後に同じwindowを継続する。scheduleはiterationからpureに決まるため、`last_densification_iteration`や`last_opacity_reset_iteration`は保存しない。

チェックポイントは`checkpoint_interval`反復ごとに`checkpoints/iteration_00005000.pt`の形式で保存し、同じ内容を`checkpoints/latest.pt`へも保存する。検証時の平均PSNRがそれまでの最高値を上回った場合は`checkpoints/best.pt`も更新する。

`best_mean_psnr`は評価を一度も実行していない場合に`None`とし、それ以外は再開前までの最高平均PSNRを保持する。再開時には、上記の全状態を`optimizer.step()`直後の境界へ復元する。保存済み設定と新たに指定された設定が異なる場合は、`output`およびログ間隔以外の差異をエラーとする。同じチェックポイントから再開した最初の1反復について、選択カメラ、損失および更新後パラメータが一致することを要件とする。

チェックポイントの`iteration`は、保存直前に完了した1始まりの反復番号を表す。反復`k`のチェックポイントから再開する場合は、モデル、最適化器、scheduler、乱数状態、`camera_order`および`camera_cursor`を復元した上で、次の反復`k + 1`から処理を再開する。`k >= total_iterations`の場合は追加の更新を行わず、学習完了として扱う。

### 9.3 PLY

学習済みGaussianを公式実装と交換できるよう、`x, y, z`、法線用ダミー値、`f_dc_*`、`f_rest_*`、`opacity`、`scale_*`および`rot_*`を保存する。`opacity`、`scale_*`および`rot_*`には、`GaussianModel`が保持するraw不透明度、rawスケールおよびrawクォータニオンを格納する。0次SH係数は`sh_dc.transpose(1, 2).flatten(1)`、高次SH係数は`sh_rest.transpose(1, 2).flatten(1)`で平坦化し、それぞれ`f_dc_0,...,f_dc_2`および`f_rest_0,...,f_rest_44`の順に保存する。読み込み時は`f_rest_*`を番号順に並べ、`reshape(N, 3, 15).transpose(1, 2)`によって`(N,15,3)`へ戻す。往復変換テストで属性値が保持されることを確認する。

`nx`、`ny`、`nz`は公式形式との互換性を維持するためのplaceholderとして0を保存し、初期実装では計算に使用しない。PLYの既定出力形式は公式実装と同じbinary little endianとする。デバッグおよび内容検査に限り、同一の属性名と保存値を持つASCII形式も選択可能とする。ローダは両形式を読み込めなければならない。

### 9.4 出力ディレクトリ

出力構成を次のように固定する。

```text
output/run001/
├── config.yaml
├── train_log.jsonl
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   └── iteration_XXXXXXXX.pt
├── renders/
│   ├── train/
│   └── test/
└── metrics/
    ├── validation_XXXXXXXX.json
    └── evaluation.json
```

`config.yaml`には実際に使用した解決済み設定を保存する。`train_log.jsonl`は1行1オブジェクトとし、`iteration`、`loss_total`、`loss_l1`、`loss_dssim`、`psnr`、各学習率、event後の`gaussian_count`および`elapsed_seconds`を記録する。density event時は`density_control_event`、`density_num_gaussians_before`、`density_num_gaussians_after`、`density_num_cloned`、`density_num_split_parents`、`density_num_children_created`、`density_num_pruned_total`、`density_num_low_opacity`、`density_num_large_screen`、`density_num_large_world`を追加する。opacity reset時は`opacity_reset`、`opacity_num_clamped`、`opacity_reset_maximum`を追加する。event/reset反復は通常の`log_interval`外でも1行だけ記録する。レンダリング画像名は元画像の拡張子を`.png`へ置換したものとする。評価JSONには画像名ごとのPSNRと`mean_psnr`を保存する。

出力先が既に存在する場合、`output.exist_policy=error`では開始前にエラーとする。`--resume CHECKPOINT`を指定した場合のみ既存ディレクトリを使用でき、既存ファイルを無条件に上書きしてはならない。`evaluate.py`は`--output`で指定されたJSONへ保存し、標準コマンドでは`metrics/evaluation.json`を使用する。学習中の検証結果は`metrics/validation_XXXXXXXX.json`へ保存する。

## 10. 例外処理と数値検査

次の場合は処理を継続せず、原因を含む例外を送出する。

- カメラ内部パラメータが0以下である。
- 画像サイズと真値画像の形状が一致しない。
- `initialization.num_gaussians`が4未満であり、近傍3点を選べない。
- `camera_poses_blender.json`の必須キー、座標系、行列形状、フレーム数または対応画像が仕様と一致しない。
- 入力テンソルの最終次元が仕様と異なる、浮動小数点テンソルのdtypeが`float32`または`float64`以外である、あるいは同一関数の入力dtypeが混在している。
- クォータニオン、共分散、レンダリング画像または損失に`NaN`/`Inf`が含まれる。
- 3D共分散または安定化後の2D共分散が、許容誤差を超えて非対称である。
- ADC有効時にtraining cameraから正のscene extentを計算できない、またはstatisticsのN・dtype・deviceがmodelと一致しない。
- 学習継続が必要なTrainerのGaussian数が0、またはdensity-control eventが全Gaussianをpruneする。

正定値性は`torch.linalg.eigvalsh()`を用いてデバッグ時またはテスト時に検査する。本番の各反復では計算コストを避けるため、既定では実行しない。

## 11. テスト設計

特記がない限り、`float32`の値比較には`torch.testing.assert_close(rtol=1e-5, atol=1e-6)`を使用する。`float64`の解析ヤコビアンと自動微分の比較には`rtol=1e-4, atol=1e-6`を使用する。乱数を使うテストでは先頭で`torch.manual_seed(0)`を設定する。

### 11.1 命名規則

数式単体テストは次の形式とする。

```text
test_<関数名>__<TeXラベルのコロンをアンダースコアに置換>
```

例：

```python
def test_world_to_camera__eq_world_to_camera():
    ...
```

### 11.2 単体テスト

- 単位回転およびゼロ並進で`world_to_camera()`が入力を保持する。
- 光軸上の点が主点`(cx, cy)`へ投影される。
- 単位クォータニオンから単位行列が得られる。
- 等方スケールではクォータニオンを変更しても3D共分散が変化しない。
- 3D共分散が対称半正定値となる。
- 安定化後の2D共分散が対称正定値となる。
- 2D逆共分散と元行列の積が単位行列に一致する。
- 全SH係数が0の場合、実装色が`0.5`となる。
- 0次SH係数の初期化から`initialization.initial_rgb`が復元される。
- `view_000.png`〜`view_099.png`と100個のフレームが一対一に対応し、すべて`800×800`として読み込まれる。
- 先頭カメラをBlender形式から変換したとき、カメラ中心がJSONのcamera-to-world行列の並進成分と一致し、`translation_cw = -rotation_cw @ camera_center_world`を満たす。
- 同一深度の場合、Gaussianの元インデックス順にソートされる。
- `torch.sort(projected.original_indices).values`が`visible_mask.nonzero().squeeze(1)`と一致する。
- `model.sh_degree=3`は受理され、3以外は`ValueError`となる。
- 数式関数が`float32`および`float64`の入力dtypeをそれぞれ維持し、dtypeの混在を拒否する。
- 不透明度が`0.99`で上限処理される。
- `alpha < 1/255`の寄与が0となる。
- 前景の不透明度が1に近いほど背景の寄与が減少する。
- 同一画像間の$L_1$損失とD-SSIM損失が0となる。
- 同一画像間ではMSEが0となり、PSNRが`+inf`となる。
- 候補透過率が`1e-4`未満となるGaussianが、色と確定透過率のどちらにも反映されない。
- PLYの保存と再読み込み後に、全rawパラメータとSH係数が`rtol=1e-5, atol=1e-6`で一致する。

式ラベルの追跡漏れを防ぐため、`docs/3DGS_定式化.tex`から`\\label{eq:...}`を抽出し、設計上の対応表または対象関数のdocstringに全116ラベルが存在することを検査するテストも設ける。

### 11.3 勾配テスト

`torch.autograd.gradcheck()`を使用するテストのみ`float64`で実行し、通常実装は`float32`のままとする。`gradcheck`は`eps=1e-6, atol=1e-6, rtol=1e-4`とする。

- 中心位置から投影中心までの勾配
- クォータニオンから3D共分散までの勾配
- スケールから3D共分散までの勾配
- 3D共分散から2D共分散までの勾配
- 中心位置からSH色までの勾配
- 不透明度、2D中心および2D共分散から画素色までの勾配
- rawスケール、raw不透明度およびrawクォータニオンまで含めた全経路の勾配

解析ヤコビアン、PyTorchの自動微分および中心差分による数値微分の3者を比較する。

### 11.4 統合テスト

1. 単一Gaussianを光軸上に配置すると、画像中心に楕円状の分布が描画される。
2. 二つのGaussianを同一画素上で異なる深度に配置すると、前方から後方へ正しく合成される。
3. `tests/fixtures/overfit/`の固定データを用いて一視点を1,000反復学習し、最後の50反復の平均損失が最初の50反復の平均損失の50%未満となる。
4. 100反復の学習中にパラメータ、画像、損失および勾配へ`NaN`/`Inf`が発生しない。
5. 保存直前とチェックポイント再読み込み直後のレンダリング画像の最大絶対誤差が`1e-6`以下となる。
6. 同一のチェックポイントから再開した次の1反復について、選択されたカメラ添字が一致し、更新後の全パラメータの最大絶対誤差が`1e-6`以下となる。
7. チェックポイント再開前後で`best_mean_psnr`が一致し、再開後の`best.pt`更新判定が同一となる。
8. clone・split・prune・opacity resetの選択境界、Parameter/Adam/statistics同期、failure rollbackおよびno-opを検証する。
9. ADC有効時のcontinuous trainingとmid-window checkpointからのresumeで、可変`N`、全Parameter、optimizer/scheduler、statistics、camera samplingおよびPyTorch RNGが一致する。

#### 11.4.1 単一学習視点による過学習fixture

`tests/fixtures/overfit/`は、次の条件を固定した再生成可能な結合テストfixtureとする。

- 固定乱数seedを使用する。
- 教師画像の解像度を`32 x 32`とする。
- 重なりと異なる深度を含む4個の固定Gaussianから教師画像を生成する。
- 教師画像を生成したGaussianとは異なる初期値から学習を開始する。
- Gaussian数、カメラ、背景色、SH次数および学習率をfixture内で固定する。
- densification、pruningおよび不透明度resetを無効にする。
- 固定した単一視点を1,000反復学習する。
- 最初の50反復の平均損失に対して、最後の50反復の平均損失が50%未満であることを合格条件とする。
- 全1,000反復について、損失、描画画像、全学習パラメータおよび勾配に`NaN`または`Inf`がないことを確認する。
- 教師画像に加えて教師Gaussianも保存し、fixtureの由来と教師画像の再生成手順を確認できるようにする。

このfixtureはレンダリング、逆伝播およびoptimizerによる更新が一貫して動作することを検証する。同じレンダラで教師画像を生成するため、レンダリング式そのものの正しさは、数値微分との勾配比較および単一Gaussianの描画テストで別途検証する。

## 12. 実装順序

### Phase 1：数式単位の順伝播

1. `Camera`および`GaussianModel`
2. rawパラメータ変換
3. 座標変換と透視投影
4. 3D・2D共分散
5. 球面調和関数
6. 単一Gaussianの描画

### Phase 2：複数Gaussianのレンダリング

1. 可視判定
2. 描画半径と矩形
3. 深度ソート
4. 不透明度合成
5. 背景色合成

### Phase 3：学習

1. $L_1$損失とD-SSIM損失
2. Adamのパラメータグループ
3. 学習率スケジュール
4. 一視点過学習
5. 複数視点学習

### Phase 4：入出力と評価

1. BlenderカメラJSONと多視点PNGの読み込み
2. チェックポイント
3. PLY入出力
4. PSNR評価

### Phase 5：高速化および拡張

初期実装の検証後に、プロファイリング結果に基づいて次を段階的に導入する。

1. Gaussian単位処理のチャンク化
2. タイル単位のカリングとソート
3. CUDA/C++拡張
4. SH次数の段階的増加

適応的密度制御と不透明度resetは、このphaseの計画後にoptional PyTorch extensionとして実装済みである。CUDA/C++およびtile rasterizerへの移植は未実装である。

## 13. 完了条件

初期実装は、次の条件をすべて満たした時点で完了とする。

- `docs/3DGS_定式化.tex`の初期実装対象となるすべての主要式に、対応する関数または参照先が存在する。
- 数式関数のdocstringからTeXラベルを追跡できる。
- 主要関数の入出力形状が型注釈またはdocstringに明記されている。
- 単体テスト、勾配テストおよび統合テストがすべて成功する。
- 一視点過学習が11.4節の数値基準を満たす。
- 既定値30,000反復を指定した学習を開始でき、100反復の小規模試験で`NaN`/`Inf`が生じない。初期実装で30,000反復を完走することは完了条件に含めない。
- テスト視点についてPSNRを画像ごとに計算し、平均PSNRを出力できる。
- 同一チェックポイントから11.4節の誤差基準を満たす再現可能なレンダリング結果と次反復の更新結果が得られる。

## 14. 実装時の注意事項

- D-SSIMの重みは、`docs/3DGS_定式化.tex`、原論文および公式実装に合わせて`0.2`とする。
- 公式実装では高速なCUDAラスタライザを使用するため、本設計の初期実装とは浮動小数点演算順序およびカリング方法が異なる。画素値の完全一致ではなく、許容誤差を設定して比較する。
- `clamp`、閾値処理、描画矩形、可視判定、深度ソートおよび早期終了は不連続な処理を含む。勾配テストでは閾値付近の入力を避ける。
- rawパラメータに対する勾配と、TeX本文で示された実パラメータに対する勾配を混同しない。実際の学習では`exp`、`sigmoid`およびクォータニオン正規化を通したrawパラメータの勾配が更新に用いられる。
- 既定のbaselineではGaussian数を固定する。ADC有効時は`nn.Parameter`の追加・削除に伴ってAdam stateとstatisticsを同じindexで再構成する。low-level primitiveは`N=0`を扱えるが、Trainerは学習継続時の`N=0`をfail-fastし、minimum Gaussian countを暗黙に強制しない。

## 15. 参照先

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG, 2023.
- 原論文公式実装: <https://github.com/graphdeco-inria/gaussian-splatting>
- 本プロジェクトの定式化資料: `docs/3DGS_定式化.tex`および`docs/3D_Gaussian_Splatting_定式化.pdf`
