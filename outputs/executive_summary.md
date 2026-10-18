# 基于欧洲 ST-T 心电数据库（EDB）的心肌缺血白盒预警系统阶段性研究总览

**项目主报告**：[`outputs/serial_ecg_ischemia_research_report.md`](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/serial_ecg_ischemia_research_report.md)  
**项目工程目录**：`/Users/bingerliang/antigravity_projects/edb_ischemia_project`  
**核心学术定位**：坚持**全流程内生白盒可解释性**，对比评估“单图无参照分类（Model A）”与“引入患者历史基线的动态预警（Model B）”，揭示动态基线重构模型生理权重的数学机制，并提出下一代解耦式证据检索统一架构。

---

## 核心成果与里程碑摘要

1. **宏观基准表现（盲测 12 位患者，203 样本）**：
   - 在纯物理特征直接拼接（Raw Concat，72 维，无任何人造差值与几何比率）下，内生白盒 **EBM 实现 AUROC 0.8256、AUPRC 0.7166（全场最高）、Balanced Acc 74.11%**，全面超越黑盒 XGBoost（Balanced Acc 63.59%）与线性回归（0.5806）。
   - 证明了加性模型（GA2M）在系列心电图建模上不存在“可解释性税”，其结构归纳偏置更契合跨患者泛化。

2. **全局机制突破（注意力重塑与自发软差分）**：
   - 单图模型（Model A）局限于单时刻静态波形猜测（过度依赖 T 波对称性与 R/S 比例）；
   - 配对模型（Model B）前 10 项特征中有 5 项自发吸纳为历史基线锚点，且 **FAST 算法选出的 15 个交互项中有 13 个为跨时序交互，其中 8 对为同名特征的当前与历史一一配对（`cur_T & prev_T`、`cur_J & prev_J`）**，自发完成了坐标系平移与软差分流形拟合。

3. **微观临床归因（Waterfall 精确局部解析）**：
   - 在 50 例预测不一致样本中，**Model B 成功平反了 34 例由慢性偏移或窦速引起的虚假警报，并将 5 例单图漏诊的微弱真缺血精准挽救**；
   - 在分类一致的样本中，动态基线使得真实缺血的置信度从及格边缘暴增（$0.199 \rightarrow 0.814$），实现了高危鉴别。

4. **批判性反思（基线误导的“三大暗面”）**：
   - 深入归因了 5 例因增加基线反而漏诊的失效样本，提炼出三大陷阱：**基线污染（带有早搏/传导阻滞的基线让模型误以为异常是常态）**、**伪动态漂移（恢复期基线制造虚假报警）**与**长程时效衰退（>1小时生理节律漂移）**。

5. **未来架构规划（解耦式检索黑盒 + 诊断白盒）**：
   - 针对变长历史输入（1 到 $N$ 张心电图）与首诊单图退化，创新性提出前置“证据检索黑盒（仅负责特征门控筛选）”与后置“内生诊断白盒（100% 决定诊断）”，兼顾变长时序容量与严格临床可追溯性，并规划彻底隔离检查频次捷径与物理电轴正交门控（$\Delta \theta_{QRS}^\perp$）。

---
**详细数据、高分辨率 Waterfall 诊断图及论文级图表请查阅**：[`outputs/serial_ecg_ischemia_research_report.md`](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/serial_ecg_ischemia_research_report.md)
