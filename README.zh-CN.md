# GenIM：面向通用材料化学的对称性感知晶体生成

[English](README.md) | **简体中文**

GenIM 是一个用于晶体结构构建、生成、验证、基准评测与筛选的 Python
软件包和命令行工具。0.3 版本新增了可审计的多模型生成后端；0.2 版本不再把
化学体系硬编码为金属间化合物：
氧化物、氮化物、卤化物、碳化物、半导体、元素固体和金属间化合物可以
共用 Hall/Wyckoff—Transformer 流程。

模型以空间群、Wyckoff 位点、离散晶格参数和坐标为序列表示，生成结果
解码为 ASE `Atoms`，随后进行几何/对称性检查、去重，并可选择使用 MLIP
弛豫和 Energy Above Hull 筛选。

## 0.3 版的核心改进

- 新增与模型无关的 `GenerationBackend`、`GenerationConstraints`、
  `GeneratedCandidate` 和 `EnsembleGenerator` 契约；旧名称保留为 0.3 兼容别名。
- `GenIMBackend` 和可选 `MatraBackend` 共用 ASE 转换、结构验证、条件审计
  和跨模型去重流程。
- Matra checkpoint 使用 PyTorch 权重安全模式、SHA256、结构检查和重建模型
  权重精确匹配，不调用历史的非安全便捷加载路径。
- `genim generate-ensemble` 同时输出通过筛选的 CIF、完整 `candidates.jsonl`
  审计记录和按来源统计的 `ensemble-report.json`。
- Matra 可按稳定性、精确元素集合、化学计量、空间群、checkpoint 特有
  Wyckoff 索引和连续凸包目标生成；未支持或无法直接验证的条件会明确写入记录。

## 0.2 版的核心改进

- 默认采用 `ChemistryPolicy(mode="any")`；`metallic` 和
  `intermetallic` 作为显式兼容模式保留。
- `any` 模式的词表可以覆盖全部 118 种真实元素，并支持精确的元素白名单
  和黑名单。
- 新增稳定的 Python API `GenIM.from_checkpoint(...)` 和批量约束采样。
- 新检查点采用带版本的格式，记录 SHA256、训练参数和数据来源；旧 v1
  检查点仍可加载。
- 新训练默认采用 `periodic8` 元素描述符：原子序数、共价半径、质量、
  周期、族以及金属/类金属/非金属指示。旧模型仍使用原来的四维投影，
  不破坏权重兼容性。
- 验证返回可审计的物理/几何指标；`genim benchmark` 报告有效率、唯一率、
  元素覆盖和空间群覆盖。
- 测试和 CI 覆盖通用化学策略、旧检查点兼容、批量生成与原有工作流。

## 科学适用边界

GenIM 给出的是满足表示和快速筛选条件的候选结构，并不自动证明结构可合成、
动力学稳定或处于热力学基态。合理的证据层级是：

1. 语法与解码检查：结构表示可构造；
2. 几何和对称性检查：排除明显不合理候选；
3. 去重和基准评测：衡量生成集内部新颖性与覆盖度；
4. MLIP 弛豫与凸包：提供依赖模型和参考集的快速筛选；
5. 强科学结论仍需 DFT、声子、有限温度分析和实验判断。

尤其要注意：把旧金属间化合物检查点的 `chemistry_mode` 改为 `any`，并不会
使它自动成为可靠的氧化物模型。通用化学必须使用有代表性的跨体系训练集重新
训练。详见 [科学范围](docs/SCIENTIFIC_SCOPE.md)。

## 安装

```powershell
python -m pip install -e .
python -m pip install -e ".[test]"
python -m pip install -e ".[api]"
python -m pip install -e ".[hull]"
python -m pip install -e ".[all]"
```

要求 Python 3.9 或更高版本。

Matra 是可选后端，并采用独立的非商业科研许可证。GenIM 不捆绑 Matra
代码或权重，应从获准来源单独安装：

```powershell
python -m pip install -e ".[matra]"
python -m pip install -e ..\matra-genoa-preview
```

## 化学策略

| 模式 | 元素范围 | 默认最少元素种类 |
|---|---|---:|
| `any` | 全部真实化学元素 | 1 |
| `metallic` | 金属和可选类金属 | 1 |
| `intermetallic` | 旧版金属间化合物范围 | 2 |

还可以用 `allowed_elements` 和 `excluded_elements` 精确限制元素。GenIM 没有
采用含糊的 `inorganic` 自动分类，因为仅从元素集合不能无歧义地判断“无机”。

```yaml
mp_download:
  chemistry_filter: any

preprocess:
  chemistry_mode: any
  seed_all_elements: true
  seed_all_hall: true

generate:
  chemistry_mode: any
  allowed_elements: [Na, Cl, K, Br]
  excluded_elements: null
  random_pool: chemistry
```

恢复旧版行为：

```yaml
generate:
  chemistry_mode: intermetallic
  include_metalloids: true
  random_pool: intermetallic
```

## 离线端到端示例

内置示例同时包含金属间、离子、共价和半导体结构：

```powershell
genim examples-make --out data/examples.jsonl
genim preprocess --in data/examples.jsonl --out data/examples.tokens.pt `
  --seed-all-elements --seed-all-hall --chemistry any
genim train --data data/examples.tokens.pt --out checkpoints/example.pt `
  --steps 200 --element-emb features --element-feature-set periodic8
genim generate --ckpt checkpoints/example.pt --n 10 --out-dir output/cif `
  --chemistry any --nelements-min 1
genim validate --cif-dir output/cif
genim benchmark --cif-dir output/cif --out output/cif/benchmark.json
```

## 使用 Materials Project 训练通用模型

设置 `MP_API_KEY`/`PMG_MAPI_KEY`，或把密钥写入已被 Git 忽略的
`.mp_api_key`：

```powershell
genim mp-download --chemistry any --max-atoms 100 --eah-max 0.5 `
  --nelements-min 1 --nelements-max 5 --limit 50000 --out data/mp.jsonl
genim preprocess --in data/mp.jsonl --out data/mp.tokens.pt `
  --seed-all-elements --seed-all-hall --chemistry any
genim inspect --in data/mp.jsonl --wyckoff
genim inspect --in data/mp.tokens.pt
genim train --data data/mp.tokens.pt --out checkpoints/mp_general.pt `
  --steps 20000 --val-fraction 0.1 `
  --element-emb features --element-feature-set periodic8
```

发布模型时应同时报告化学体系分布、元素频率、空间群覆盖、晶胞大小分布、
数据划分方法以及 MP 下载筛选条件。内置的固定随机种子验证集只是可复现的
软件基线；若要声称外推能力，应另外采用按组成和原型留出的外部测试集。

## 默认生成服务

HTTP 服务带有可直接使用的默认设置，并接受任意有效化学组成。若检测到已配置的
Matra 或 GenIM checkpoint，`backend="auto"` 会优先使用 checkpoint；否则返回明确
标记为非机器学习预测的可复现 algorithmic seed：

```powershell
genim serve --host 127.0.0.1 --port 8000
```

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/generate `
  -ContentType application/json `
  -Body '{"formula":"LiFePO4","spacegroup_number":62,"n":4,"seed":7}'
```

服务提供 `/v1/health`、`/v1/capabilities`、`/v1/generate` 和 `/docs`。
可用 `GENIM_MATRA_CHECKPOINT`、`GENIM_MATRA_SHA256`、
`GENIM_MODEL_CHECKPOINT`、`GENIM_MODEL_SHA256` 配置模型。显式请求不可用的
`matra` 或 `genim` 后端会返回错误，不会把算法种子伪装成 checkpoint 预测。

## Python API

```python
from genim import ChemistryPolicy, GenIM, SamplingConfig

model = GenIM.from_checkpoint("checkpoints/mp_general.pt", device="auto")
results = model.sample(
    config=SamplingConfig(
        n=64, batch_size=16, max_sites=25, min_sites=2,
        temperature=0.9, seed=7,
    ),
    chemistry=ChemistryPolicy(
        mode="any",
        allowed_elements=["Na", "Cl", "K", "Br"],
        min_elements=2,
        max_elements=2,
    ),
)
paths = model.write_valid_cifs(results, "output/alkali_halides")
records = [result.to_record() for result in results]  # 含检查点哈希和验证指标
```

当前批量采样器对每个 token 位置只进行一次模型前向计算，服务接口和大批量
测试可以直接复用。KV cache 是后续性能优化，不影响现有结果与来源记录格式。

## GenIM 与 Matra 联合生成

联合接口把两套模型视为独立候选来源，不拼接或平均不兼容的权重。

```powershell
genim matra-checkpoint-info `
  --ckpt ..\matra-genoa-preview\checkpoints\matra-v02-med.ckpt `
  --expected-sha256 4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa

genim generate-ensemble `
  --genim-ckpt checkpoints\mp_general.pt `
  --matra-ckpt ..\matra-genoa-preview\checkpoints\matra-v02-med.ckpt `
  --matra-sha256 4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa `
  --elements Na Cl --stoichiometry 1 1 --stability stable `
  --n-per-backend 64 --seed 7 --out-dir output\hybrid-nacl
```

Python API：

```python
from genim import (
    EnsembleGenerator, GenerationConstraints, GenerationSettings, GenIMBackend,
    MatraBackend, write_ensemble_run,
)

backends = [
    GenIMBackend.from_checkpoint("checkpoints/mp_general.pt"),
    MatraBackend.from_checkpoint(
        "../matra-genoa-preview/checkpoints/matra-v02-med.ckpt",
        expected_sha256="4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa",
    ),
]
run = EnsembleGenerator(backends).run(
    condition=GenerationConstraints(
        elements=("Na", "Cl"), stoichiometry=(1, 1), stability="stable",
    ),
    config=GenerationSettings(n=64, temperature=0.75, seed=7),
)
write_ensemble_run(run, "output/hybrid-nacl")
```

这里的 `selected` 只表示结构有效、约束未被证伪且在本次运行中唯一，不代表
热力学稳定。每个约束明确报告为 `satisfied`、`violated` 或 `not_evaluated`，并
记录方法和证据级别；稳定性/凸包目标仍需 MLIP 或 DFT 证据。详见
[Matra 集成说明](docs/MATRA_INTEGRATION.md)。

## 组成约束生成

```powershell
genim synth --elements Na Cl --nelements 2 --ratios 1 1
genim synth --elements Fe O --nelements 2 --ratios 2 3
genim synth --elements Li Fe P O --nelements 4 `
  --ratios 1 1 1 4 --ratio-mode ratio
```

`X` 表示元素必须存在、但比例不固定：

```powershell
genim synth --elements Li Fe P O --nelements 4 `
  --ratios 1 X 1 X --ratio-mode ratio
```

## 检查点与可复现性

大数据和权重不提交到 Git，应作为不可变的 GitHub Release 资源发布并记录
SHA256。v2 检查点包含格式版本、模型/词表/分词配置、GenIM 版本、时间、
随机种子、训练步数、token 数据 SHA256、原始数据来源和数据统计。可用
`genim.checkpoints.download_checkpoint(...)` 原子下载并验证哈希。详见
[检查点格式](docs/CHECKPOINT_FORMAT.md)。

```powershell
genim checkpoint-info --ckpt checkpoints/mp_train_fullsg_60.pt `
  --expected-sha256 3777449fe396522c0173aaa699c70c99b4d28e26b436200545f08f86ff28173c
```

历史金属间模型仍可从 [GitHub Releases](https://github.com/XYG-Research/GenIM/releases)
获得，但使用时必须明确它的训练域；经本地核验的文件名、大小、URL 和哈希记录
在 [v0.1.0 资源清单](resources/release-v0.1.0.json) 中。

## 可选科学筛选

```powershell
genim score --conf conf.yml --cif-dir output/cif
genim surface-screen --input-dir output/cif --out-dir output/surfaces
```

MLIP 能量和凸包结果继承模型与参考集的不确定性，不能在没有校准的情况下与
DFT 能量混用。

## 开发与文档

```powershell
python -m pytest -q
```

- [科学范围](docs/SCIENTIFIC_SCOPE.md)
- [检查点格式](docs/CHECKPOINT_FORMAT.md)
- [架构](docs/ARCHITECTURE.md)
- [Matra 集成说明](docs/MATRA_INTEGRATION.md)

## 参考文献

1. [doi:10.1038/s41524-025-01881-2](https://doi.org/10.1038/s41524-025-01881-2)
2. [doi:10.1038/s41524-025-01940-8](https://doi.org/10.1038/s41524-025-01940-8)

## 许可证

[BSD 3-Clause](LICENSE)
