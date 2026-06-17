# 验收报告（GenIM / MP→Intermetallic 结构生成 MVP）

日期：2026-03-04  
工作目录：仓库根目录（`<repo-root>`）

本项目已实现从 Materials Project (MP) 自动抓取训练集 → Wyckoff/对称性表征预处理 → Transformer 训练 → 生成 CIF → 快速验收的端到端流程。

## 1. 训练集抓取（MP）

产物：`data/mp_train.jsonl`

示例命令（本次验收使用的参数）：

```powershell
genim mp-download --out data/mp_train.jsonl --limit 30000 --per-page 500 --max-atoms 80 --eah-max 0.25 --nelements-min 2 --nelements-max 3
```

结果：

- JSONL 行数（结构条目数）：`19494`

说明：

- MP key 通过环境变量 `MP_API_KEY`/`PMG_MAPI_KEY` 或当前目录 `.mp_api_key` 读取（本仓库已在 `.gitignore` 忽略该文件）。

## 2. 预处理（Wyckoff/Hall token 序列）

产物：`data/mp_train.tokens.pt`

示例命令（本次验收使用的参数）：

```powershell
genim preprocess --in data/mp_train.jsonl --out data/mp_train.tokens.pt --max-sites 15 --symprec 1e-2 --coord-bins 96
```

结果（从 `data/mp_train.tokens.pt` 读取）：

- kept/skipped：`19186 / 308`
- token 序列张量：`(19186, 84)`
- vocab size：`697`
- 量化配置：`coord_bins=96, len_bins=180 (2–20 Å), ang_bins=181 (40–140°)`

## 3. 训练（Causal Transformer LM）

产物：`checkpoints/mp_train.pt`

示例命令（本次验收使用的参数）：

```powershell
genim train --data data/mp_train.tokens.pt --out checkpoints/mp_train.pt --steps 3000 --batch 64 --lr 3e-4 --d-model 256 --layers 6 --heads 8
```

Checkpoint 配置（从 `checkpoints/mp_train.pt` 读取）：

- `d_model=256, n_layers=6, n_heads=8, dropout=0.1`
- `vocab_size=697, max_len=84`

## 4. 生成（CIF）

产物目录：`output/mp_cif_bt/`（本次验收生成 200 个 CIF）

示例命令（本次验收使用的参数，约束为二元/三元 intermetallic）：

```powershell
genim generate --ckpt checkpoints/mp_train.pt --n 200 --out-dir output/mp_cif_bt --max-sites 15 --temperature 1.0 --top-k 0 --nelements-min 2 --nelements-max 3
```

结果：

- 生成：`200/200`（attempts=263）
- 组成分布：二元 `90`，三元 `110`

## 5. 快速验收（结构合理性）

示例命令：

```powershell
genim validate --cif-dir output/mp_cif_bt --min-dist 0.7 --symprec 1e-2
```

结果：

- `Validated 200 CIFs: ok=200, fail=0`

## 6. 回归测试

```powershell
pytest -q
```

结果：

- `2 passed`

## 7. 通用性增强（Universal v1）

为了让模型在**元素/对称性空间**上更通用（“原则可生成更多 intermetallic 体系”），本次额外提供了一套 *seeded vocab + feature-based element embedding* 的 checkpoint：

### 7.1 Seeded token 数据集

产物：`data/mp_train_seed.tokens.pt`

```powershell
genim preprocess --in data/mp_train.jsonl --out data/mp_train_seed.tokens.pt --max-sites 15 --symprec 1e-2 --coord-bins 96 --seed-all-elements --include-metalloids --seed-all-hall
genim inspect --in data/mp_train_seed.tokens.pt
```

要点：

- vocab 含 `HALL_1..HALL_530`（530 个 Hall token）
- vocab 含金属+类金属元素 token（98 个 `E_*` token）

### 7.2 通用 checkpoint（元素特征嵌入）

产物：`checkpoints/mp_train_fullsg_60.pt`

```powershell
genim train --data data/mp_train_seed.tokens.pt --out checkpoints/mp_train_fullsg_60.pt --steps 3000 --batch 64 --lr 3e-4 --d-model 256 --layers 6 --heads 8 --element-emb features
```

### 7.3 含类金属体系演示（原型生成 → 元素替换）

产物：`output/demo_FeSi/`

```powershell
genim generate --ckpt checkpoints/mp_train_fullsg_60.pt --n 20 --out-dir output/demo_FeSi --nelements-min 2 --nelements-max 2 --include-metalloids --substitute-elements Fe Si
genim validate --cif-dir output/demo_FeSi
```

结果：`ok=20, fail=0`

## 8. 配置化 & 最简生成（conf.yml + synth）

为满足“参数集中管理 + 用户只需指定元素与元素数”的目标，本次新增：

- 统一配置文件：`conf.yml`（默认值齐全，可直接改）
- 最简生成命令：`genim synth --elements ... --nelements ...`
- 严格去重：生成阶段默认对结构做哈希去重，`--n`/`generate.n_max` 作为“最大输出唯一数”，不足则输出可生成的最大唯一数（不再报错）
- 空间群覆盖：支持 `generate.hall_mode: uniform_230`（均匀采样 230 个空间群的代表 Hall）

示例：

```powershell
genim synth --elements Fe Si --nelements 2
genim synth --elements Fe Si --nelements 3
```

## 9. 全空间群覆盖（230/230）数据集（MP + symmetry seeds）

结论（2026-03-04）：

- 直接使用 MP `materials/summary` + `spacegroup_number` 的 SG-balanced 抓取，在本环境下 **最多覆盖 228/230** 个空间群。
- 缺失的空间群：**168、207**（MP API 返回 0 条记录）。
- 为了满足“程序可生成 230 个空间群”的训练/条件化需求，本项目新增 `genim sym-seed`，用 spglib 数据库合成缺失空间群的最小结构样本（只用于对称性覆盖）。
- 另外，在 `preprocess` 阶段（Wyckoff 代表点抽取）会对个别结构跳过（例如 Wyckoff 代表点数过多/标准化失败）。因此我们额外补齐了 SG=1、8 的 seed，最终保证 token 数据集空间群覆盖 **230/230**。

产物：

- MP SG-balanced（any chemistry）：`data/mp_sg_any_verified_230.jsonl`（228 条）
- Symmetry seeds：`data/sym_seed_168_207.jsonl`、`data/sym_seed_1_8.jsonl`
- 合并后的全覆盖集：`data/mp_sg_any_full230_v2.jsonl`
- token 数据集（max_sites=60）：`data/mp_sg_any_full230_v2_60.tokens.pt`（spacegroups unique=230）

关键命令（可复现）：

```powershell
genim mp-download --out data/mp_sg_any_verified_230.jsonl --chemistry any --balance-spacegroups --per-spacegroup 1 --limit 230 --max-atoms 200 --eah-max 1.0 --nelements-min 1 --nelements-max 5
genim sym-seed --out data/sym_seed_168_207.jsonl --spacegroups 168 207 --elements Fe Si --n-sites 3 --n-per-sg 1
genim sym-seed --out data/sym_seed_1_8.jsonl --spacegroups 1 8 --elements Fe Si --n-sites 3 --n-per-sg 1 --seed 11
cmd /c copy /b data\\mp_sg_any_verified_230.jsonl+data\\sym_seed_168_207.jsonl+data\\sym_seed_1_8.jsonl data\\mp_sg_any_full230_v2.jsonl

genim preprocess --in data/mp_sg_any_full230_v2.jsonl --out data/mp_sg_any_full230_v2_60.tokens.pt --max-sites 60 --symprec 1e-2 --coord-bins 96 --len-bins 180 --len-min 2.0 --len-max 20.0 --ang-bins 181 --ang-min 40.0 --ang-max 140.0 --seed-all-elements --include-metalloids --seed-all-hall
genim inspect --in data/mp_sg_any_full230_v2_60.tokens.pt --top-k 5
```

## 10. 230-SG 通用训练集 + 新 checkpoint

为了让模型真正“见过”全部 230 个空间群（并提升跨空间群的可泛化性），本次把 intermetallic 训练集与上面的 230-SG 覆盖集做了合并，再统一 tokenize（max_sites=60）：

产物：

- 合并 JSONL：`data/mp_train_plus_sg_full230_v2.jsonl`
- 合并 tokens：`data/mp_train_plus_sg_full230_v2_60.tokens.pt`（spacegroups unique=230）
- 新 checkpoint：`checkpoints/mp_train_fullsg_60.pt`

命令：

```powershell
cmd /c copy /b data\\mp_train.jsonl+data\\mp_sg_any_full230_v2.jsonl data\\mp_train_plus_sg_full230_v2.jsonl
genim preprocess --in data/mp_train_plus_sg_full230_v2.jsonl --out data/mp_train_plus_sg_full230_v2_60.tokens.pt --max-sites 60 --symprec 1e-2 --coord-bins 96 --len-bins 180 --len-min 2.0 --len-max 20.0 --ang-bins 181 --ang-min 40.0 --ang-max 140.0 --seed-all-elements --include-metalloids --seed-all-hall
genim inspect --in data/mp_train_plus_sg_full230_v2_60.tokens.pt --top-k 5

genim train --data data/mp_train_plus_sg_full230_v2_60.tokens.pt --out checkpoints/mp_train_fullsg_60.pt --steps 2000 --batch 32 --lr 3e-4 --d-model 256 --layers 6 --heads 8 --dropout 0.1 --element-emb features --seed 7
```

## 11. 生成验收：更严格的几何约束 + 去重 + 配比控制

本次增强点：

- `validate.min_dist_factor`：基于 covalent radii 的最小距离因子（默认 0.75），可显著减少“键长明显不合理”的结构。
- `validate.max_dist_factor`：限制“过稀/过大晶格”（min(d_ij/(r_i+r_j)) 过大）导致的“原子明显不成键”。
- `validate.require_connected: true`：基于化学半径邻接图的连通性检查，避免生成“分裂成多个簇”的非物理结构。
- `generate.autoscale_cell: true`：生成后对晶格做各向同性缩放，使体积/键长指标落入 `validate.*` 约束区间（显著改善初始晶格尺度）。
- `generate.prototype_mode: target`：默认直接在目标元素体系采样（更合理的晶格/键长）；如需“原型多样性”可改为 `random` 再做 substitution。
- `generate.dedup.mode: prototype`：忽略晶格尺度做更严格的原型级去重，避免“同构但晶格略变”的重复结构。
- `genim synth --n`：覆盖 `conf.yml` 的 `generate.n_max`（仍是“最大生成数”）。
- `--ratios`：默认按 `ratio` 解释；如需按 at.% 解释，显式写 `--ratio-mode percent`。可用 `X` 表示该元素组成不固定。

示例：

```powershell
# 最简：只指定元素 + 元素数
genim synth --elements Fe Si --nelements 2

# 严格 1:1
genim synth --elements Fe Si --nelements 2 --ratios 1 1 --n 20

# 30/30/40（允许 ±0.1 的原子分数容差，避免严格 3:3:4 过难）
genim synth --elements Fe Si --nelements 3 --ratios 30 30 --ratio-mode percent --n 5

# 只固定前两个元素，后两个元素必须出现但占比不固定
genim synth --elements Ni Fe Co Al --nelements 4 --ratios 1 1 X X --ratio-mode ratio

# 5 元体系
genim synth --elements Fe Si --nelements 5 --n 20

# 快速验收（生成阶段已内置 validate；这里只做外部复核）
genim validate --cif-dir output/Fe-Si_2el --min-dist 1.5 --symprec 1e-2
```

## 12. generate + substitute 的重复/不匹配问题修复

针对你提到的 `demo_FeSi`（`genim generate --substitute-elements Fe Si`）出现大量重复、以及大量 `substitution_mismatch` 的问题：

- 当提供 `--substitute-elements` 且未显式 `--elements` 限制时，`generate` 现在会：
  - 自动在 intermetallic 元素池中采样同等数量的“原型元素”，并强制前 k 个 site 的元素 token 互异；
  - 同时设置 `min_sites >= k`，减少无效样本；
  - 大幅降低 `substitution_mismatch`，提升有效生成效率。

## 13. MLIP bulk 弛豫 + Energy Above Hull + CSV

目标：对生成结构进行 eSEN/OMAT24（fairchem-core）快速 bulk 弛豫（含晶格 + 原子），并以相同 MLIP 能量自动构建化学体系的 reference hull，计算 `Energy Above Hull`，把结构标号/组成/EAH/稳定性写入 CSV。

命令（可复现）：

```powershell
genim score --conf conf.yml --cif-dir output\\Fe-Si_2el
# alias
genim mlip --conf conf.yml --cif-dir output\\Fe-Si_2el
```

预期产出：

- CSV：`output\\Fe-Si_2el\\ml_score\\score.csv`（默认 out_dir）
- 弛豫后 CIF：`output\\Fe-Si_2el\\ml_score\\relaxed\\*.cif`
- hull cache：`output\\hull_cache\\<chemsys>\\refs.jsonl`（避免重复计算；可用 `hull.force_rebuild: true` 强制重建）
