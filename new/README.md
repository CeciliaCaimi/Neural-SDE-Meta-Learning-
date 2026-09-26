# 本次新生成的实验原始结果

本目录保存 `linear_transport_replacement_2026-09-24` 批次已完成的全部原始评估输出：12 个训练运行、24 个 CIFAR/坐标诊断评估任务，以及 5 个 GMM panel 评估。上传时没有重复运行实验。

| 目录 | 内容 |
|---|---|
| `results/` | 原输出目录全部 68 个文件，含逐 episode/task JSON、数值报告与生成图；`generated_grids/` 另存两张位于原输出目录之外的生成网格 |
| `logs/` | 训练与评估 stdout/stderr，以及 7 份逐步训练指标 JSONL |
| `configs/` | 原始执行计划、12 份配置审计、固定数据划分/变换定义和冻结的 full-FT 预算 |
| `provenance/` | 完成标记、运行状态、12 个 checkpoint 的结构/配置/哈希，以及论文结果对应关系 |
| `statistics/` | 本批次原始记录计算的均值、配对差值、95% 区间，以及 A4/profile 图的统计依据 |
| `figures/` | 当前论文 A4、profile contrast、low-data crossover 图 |

主方法统一为 **correct source + transport**，实际使用 linear transport、J=0。原始 JSON 按生成时的内容原样保留；部分文件同时保存独立 baseline 或 refinement 分支，A3 是单独的 refinement on/off 消融。不能把这些分支当成主方法。

`SEED1001/2002/3003` 是同一 checkpoint 的不同评估随机种子，不代表额外训练种子。`E14_strategies_n60` 等已完成补充输出也完整收录，是否用于当前文稿以 `provenance/manuscript_result_mapping.json` 为准。不同评估包、split 和统计单位应分别使用。

原始记录、日志、配置与生成网格均逐字节复制。`MANIFEST.json` 记录源文件、大小与 SHA256；`FILE_SHA256.txt` 覆盖本目录全部其他文件。局部 `.gitattributes` 保留原始换行，保证 Git 中的字节与源文件哈希一致。

模型权重和数据集图片未放入 Git；checkpoint 身份、结构、配置与 SHA256 已收录。既有 MLP transport 消融属于历史结果，不混入本次新生成原始记录。原计划中未执行的任务在状态文件中明确列出。
