# GenIM：基于 Materials Project 的 intermetallic 结构生成（MVP）

[English](README.md) | **简体中文**

用 Materials Project (MP) 的结构数据做训练集，训练一个"生成式结构构建模型"，快速产生合理的 intermetallic 晶体结构。

本仓库实现一个 **可落地跑通的 MVP**：参考两篇相关工作的核心思想（Wyckoff/对称性表征 + 自回归 Transformer 顺序采样 + 生成后过滤验收，详见文末 [参考文献](#参考文献)），做成可复用的 Python 包与命令行工具。

## 主要能力

- 从 MP API 拉取结构（需要 `MP_API_KEY`）。
- 用 `spglib` 把结构标准化并提取 **Hall number / Wyckoff site** 表征。
- 把每个结构编码成 token 序列，训练一个 **Causal Transformer LM** 做自回归生成。
- 把生成的序列解码回 3D 周期结构（ASE `Atoms`），并做快速验收过滤（最小原子距、重复/冲突、对称性可识别等）。
- 生成阶段默认启用**更贴近物理的几何约束**：连通性检查 + 体积/键长自适应缩放（避免"分裂簇/过稀晶格"）。

## 环境

- Python 3.9+
- 核心依赖：`torch`, `ase`, `spglib`, `requests`, `numpy`, `pyyaml`, `tqdm`, `matplotlib`, `Pillow`
- 可选扩展：`genim[mlip]`（fairchem-core，用于 MLIP 弛豫）、`genim[hull]`（pymatgen + scipy，用于 energy-above-hull 与表面筛选）、`genim[all]` 全功能。

安装（开发态）：

```powershell
python -m pip install -e .
# 全功能：
python -m pip install -e ".[all]"
```

## 配置文件（conf.yml）

本项目把**所有可调参数**集中到 `conf.yml`（仓库根目录已提供一份带默认值的配置）。你也可以用命令生成模板：

```powershell
genim conf-init --out conf.yml
```

常用配置项：

- 训练集抓取：`mp_download.*`（默认支持未来扩展到 `nelements_max: 5`）
- 预处理：`preprocess.*`（可用 `seed_all_elements/seed_all_hall` 提升跨元素/对称性泛化）
- 训练：`train.*`（默认 `element_emb: features` 用周期表特征做元素嵌入/预测）
- 生成：`generate.*`（`n_max`、去重、空间群采样等）
- 生成几何增强：`generate.autoscale_cell`、`generate.prototype_mode`、`validate.max_dist_factor`、`validate.require_connected`

## 快速开始（离线示例）

先用内置的少量示例结构跑通端到端（不需要 MP key）：

```powershell
genim examples-make --out data/examples.jsonl
genim preprocess --in data/examples.jsonl --out data/examples.tokens.pt
genim train --data data/examples.tokens.pt --out checkpoints/example.pt --steps 200
genim generate --ckpt checkpoints/example.pt --n 10 --out-dir output/cif
genim validate --cif-dir output/cif
```

## 最简生成：只指定元素 + 元素数

用户只需要指定"必须包含的元素"以及"总元素种类数"，其余全部从 `conf.yml` 读取：

```powershell
genim synth --elements Fe Si --nelements 2
genim synth --elements Fe Si --nelements 3
```

- `--nelements 2`：只生成只含 Fe/Si 的二元结构
- `--nelements 3`：生成包含 Fe/Si 的三元结构，第三个元素从 intermetallic 元素池随机补齐
- `generate.n_max` 是最大生成数；若唯一结构不足，会自动输出能生成的最大唯一数（不报错）
- 默认开启严格去重：`generate.dedup.*`
- 默认 `generate.prototype_mode: target`：直接在目标元素体系中采样（晶格/键长更合理）；需要更"原型多样性"可改成 `random`
- 默认 `generate.hall_mode: model`（质量优先）；如需覆盖 230 空间群采样，改为 `uniform_230`

### 组成比例的写法

`--ratios` 始终按 `--elements` 的顺序书写，常见写法如下：

```powershell
# 1) nelements == len(elements)：写"原子数比例"
genim synth --elements Ni Fe Co Al --nelements 4 --ratios 1 1 1 1
# 表示 Ni:Fe:Co:Al = 1:1:1:1

# 2) 有些已列元素只要求出现，不固定组成：用 X
genim synth --elements Ni Fe Co Al --nelements 4 --ratios 1 1 X X --ratio-mode ratio
# 表示 Ni:Fe 严格 1:1；Co 和 Al 必须出现，但占比不固定

# 3) 百分比写法：数值位是 total-atom 百分比，X 表示其余元素不固定
genim synth --elements Ni Fe Co Al --nelements 4 --ratios 25 25 X X --ratio-mode percent
# 表示 Ni 占 25 at.%、Fe 占 25 at.%；Co 和 Al 合计占剩余 50 at.%

# 4) nelements > len(elements)，并且想让额外元素也不固定
genim synth --elements Fe Si --nelements 3 --ratios 30 30 --ratio-mode percent
# 表示 Fe 占 30 at.%、Si 占 30 at.%；剩余第 3 个元素合计 40 at.%

# 5) 不限制组成比例
genim synth --elements Fe Si --nelements 3
```

- `--ratios` 的个数必须与 `--elements` 个数一致。
- `X` 表示"该元素必须出现，但组成不固定"。
- `--ratio-mode ratio`：只对数值位施加严格原子数比例约束。
- `--ratio-mode percent`：只对数值位施加 total-atom 百分比约束；所有 `X` 位和额外补齐元素共同分配剩余百分比。
- 默认按 `ratio` 解释；只有你明确写 `--ratio-mode percent` 时，数值位才按百分比解释。
- 建议只在"数值位就是 at.%"时写 `--ratio-mode percent`。

## 数据集覆盖度检查（推荐）

为了做"更通用"的 intermetallic 生成，建议先用 `inspect` 看训练集覆盖度（元素、Hall number、Wyckoff 代表位点数等）：

```powershell
genim inspect --in data/mp_train.jsonl
genim inspect --in data/mp_train.jsonl --wyckoff
genim inspect --in data/mp_train.tokens.pt
```

## 用 Materials Project 做训练集

1) 设置 API key：

```powershell
$env:MP_API_KEY="你的MPKey"   # 或者用 $env:PMG_MAPI_KEY
```

如果你不想在终端历史里留下 key，也可以把 key 写到当前目录的 `.mp_api_key`（一行一个 key），本工具会自动读取。该文件已被 gitignore。

2) 下载（示例：限定元素集合 + 结构大小过滤）：

```powershell
genim mp-download --elements Fe Ni Al --max-atoms 80 --limit 5000 --out data/mp.jsonl
```

3) 预处理/训练/生成同上。

### 生成时的 intermetallic 约束

`genim generate` 默认会做两类生成后过滤：

- 结构几何/对称性快速验收（最小原子距、可识别空间群等）。
- **intermetallic 过滤**：默认要求 **至少 2 种元素**，并排除常见非金属/卤素等（可用 `--include-metalloids` 放开类金属）。

同时默认开启**严格去重**，并把 `--n` 解释为"最多输出多少个**唯一**结构"。如果约束太严格导致达不到 `--n`，程序会输出尽可能多的唯一结构并给出拒绝原因统计（不再抛异常）。

如果你希望更贴近"二元/三元 intermetallic"训练集分布，可在生成时加：

```powershell
genim generate --ckpt checkpoints/mp_train.pt --n 200 --out-dir output/mp_cif --nelements-max 3
```

### "原型生成 → 元素替换"（更通用）

当你希望生成某些训练集中覆盖较少/未覆盖的元素体系时（例如含类金属 Si/Ge 等），更稳妥的做法是：

1) 先让模型生成**结构原型**（空间群/Wyckoff/坐标等）；
2) 再用 `--substitute-elements` 把生成结构中的"元素集合"替换为你指定的元素（保持原型不变）。

示例（生成 Fe–Si 二元原型并替换为 FeSi）：

```powershell
genim generate --ckpt checkpoints/mp_train_fullsg_60.pt --n 20 --out-dir output/demo_FeSi --nelements-min 2 --nelements-max 2 --include-metalloids --substitute-elements Fe Si
```

## 备注（MVP 取舍）

为了先把"落地链路"跑通，本版本对连续变量（晶格参数与 Wyckoff 坐标）采用了 **离散分箱 token**。后续如需对标论文更强的表现，可把坐标/晶格改成连续密度建模（Gaussian embedding + mixture density head 等）。

另外，如果你的目标是"尽可能通用/覆盖更多 intermetallic 体系"，建议：

- `mp-download`：提高 `--limit`，放宽 `--nelements-max/--max-atoms/--eah-max`，并按需加 `--include-metalloids`。
- `preprocess`：按需增大 `--max-sites`（会增加序列长度与训练成本）；也可加 `--seed-all-elements/--seed-all-hall` 以提升跨元素/对称性的可泛化性。
- `train`：可用 `--element-emb features` 开启基于周期表特征的元素嵌入/预测（对未覆盖元素的"原则可生成"更友好）。

## 更新（2026-03-04）

- `mp-download` 新增 `--chemistry any`：用于构建"全空间群覆盖"数据集（不局限 intermetallic）。
- 新增 `sym-seed`：当 MP 缺失某些空间群（例如本环境下缺失 SG=168/207）时，用 spglib 数据库合成最小结构样本补齐 230/230 空间群覆盖（仅用于对称性覆盖/条件化）。
- `synth` 新增 `--n`：覆盖 `conf.yml` 的 `generate.n_max`（依然解释为最大生成数）。
- 验收/过滤增强：`validate.min_dist_factor`（基于 covalent radii 的最小距离因子）+ `generate.dedup.mode: prototype`（更严格去重，避免"晶格略变"的重复）。

## MLIP 弛豫 + Energy Above Hull 打分

`genim score`（别名：`genim mlip`）：用 **eSEN/OMAT24（fairchem-core）** 对生成结构做"晶格 + 原子"快速弛豫，并自动构建同一化学体系的 ML reference set（来自 MP 结构 + 同一 MLIP 能量），计算每个结构的 `Energy Above Hull (eV/atom)`，输出 CSV。
CSV 会额外给出 `composition_ratio`、`composition_percent` 和 `composition_counts`，分别表示约化比例、at.% 和具体原子个数。

示例：

```powershell
genim synth --elements Fe Si --nelements 2
genim score --conf conf.yml --cif-dir output\\Fe-Si_2el

# 或者用更"傻瓜式"的两步别名：
genim gen --elements Fe Si --nelements 2
genim mlip --conf conf.yml --cif-dir output\\Fe-Si_2el
```

> 需要可选扩展：`pip install ".[all]"`（fairchem-core + pymatgen + scipy）。

## 交互模式

不带子命令运行 `genim` 进入交互式菜单。按 Enter 接受默认值（从 `conf.yml` 读取）。

```powershell
genim
```

菜单会打印它读/写的路径，运行一个所选模块后退出。交互模式下的组成控制写成一行 `R/P + values`，例如 `R 1 1 X X` 或 `P 20 20 20 20`。

## 结构快照面板

可用 `genim snapshot-panel` 从生成的 CIF 目录中抽取固定数量的结构做快照面板。默认随机抽取，支持按 ID 区间过滤，每行 6 个，标签格式为 `009-Fe5Co5Ni5Al5`，其中数值是实际晶胞内原子数。每个 tile 用透视晶体视角渲染，同时显示原子与晶胞晶格线。

```powershell
genim snapshot-panel --cif-dir output\\Fe-Co-Ni-Al_4el --n 12
genim snapshot-panel --cif-dir output\\Fe-Co-Ni-Al_4el --n 12 --id-start 9 --id-end 40
```

配置都在 `conf.yml`：

- `ml.*`：选择 eSEN 模型/设备；建议把 `ml.checkpoint` 指向本地 `esen_30m_oam.pt`（避免 HF 拉取/门控）。
- `relax.*`：bulk 弛豫设置（ASE + ExpCellFilter/UnitCellFilter）。
- `hull.*`：MP 参考池抓取 + reference-relax，`stable_threshold_eV_per_atom: 0.2` 对应 200 meV/atom 稳定门槛。

## 预训练权重与数据下载

为保持仓库轻量，训练数据（`data/`）与预训练权重（`checkpoints/`）**不随源码一起提交**，而是作为 [GitHub Releases](https://github.com/XYG-Research/GenIM/releases) 的附件分发：

- `mp_train_fullsg_60.pt`：在 230 空间群全覆盖 + intermetallic 合并集上训练的 Causal Transformer 权重。
- `mp_train.jsonl` / `mp_train.tokens.pt`：MP 抓取的训练结构与其 token 化数据集（源自 Materials Project，遵循其使用条款）。

下载后放回对应目录即可（默认路径见 `conf.yml` 的 `paths.*`）：

```powershell
# 例：把下载的权重放到 checkpoints/ 下
mkdir checkpoints
mv mp_train_fullsg_60.pt checkpoints/
```

你也可以**完全从零复现**：用 `genim mp-download`（需自备 `MP_API_KEY`）抓取数据，再 `preprocess` → `train`，命令见 [`ACCEPTANCE.md`](ACCEPTANCE.md)。

## 参考文献

本项目方法学受以下工作启发（均发表于 *npj Computational Materials*）：

1. DOI: [10.1038/s41524-025-01881-2](https://doi.org/10.1038/s41524-025-01881-2)
2. DOI: [10.1038/s41524-025-01940-8](https://doi.org/10.1038/s41524-025-01940-8)

## License

本项目以 [BSD 3-Clause License](LICENSE) 开源。
