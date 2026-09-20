# 本地语料 · vendored 索引与检索器

## 这是什么

`corpus/` 是 TCM-Mind-RAG 知识数据的**本地 vendored 副本**，配一个确定性检索器。

- `index.json` —— 由 `corpus/vendor.py` 生成，540 个 chunk，带 `corpus_version` 与逐 chunk 证据 ID、文件位置、行范围。
- `retriever.py` —— 只读检索器，无网络、无模型。

## 为什么是副本而不是引用

原项目 `D:\ZYY Project\TCM-Mind-RAG` 是**未修改的依赖**。这里不 import 它、不 shell out 到它、不读取它的运行服务。只有一份带 SHA-256 的副本被查询，因此：

- 结果可复现（同一索引 → 同一检索结果）。
- 引用可核验（`source_file:line_start-line_end` + `corpus_version` + 索引哈希）。
- 原项目结构变化不会静默改变本仓库的结论。

## 刷新语料

```bash
python -m corpus.vendor --source "D:/ZYY Project/TCM-Mind-RAG/backend/data"
```

对相同输入，重跑产生**逐字节相同**的 `index.json`（`tests/test_corpus.py::test_the_corpus_is_reproducible` 会比对已提交的索引）。上游数据目录不在本机时，该测试跳过。

## 检索语义

`LocalCorpus.search(query, species=..., phenotypes=..., limit=5, min_score=0.05)`

- 评分是**确定性字符二元组重合度**，范围 [0, 1]。没有 Embedding 模型，没有向量库，没有 cuVS。
- `min_score` 默认 0.05：重合度为零的 chunk 不是匹配，返回它就是编造引用。
- `species` / `phenotypes` 过滤基于 chunk 标签，标签由 vendor 阶段从文本提取。
- 空查询、`limit < 1`、`min_score` 越界都是 `ValueError`。

`retrieval_score` **只反映字面重合度**，不表示事实确定性。

## 明确的边界：这是 TCM 证型知识，不是植物病理学

语料来自 TCM-Mind-RAG 的 `syndromes.yaml`（20 个证型）、`herb_interactions.yaml`（20 条配伍）与 `synthetic_cases.json`（500 条合成病例）。它是**人体证型与本草药性知识**。

因此：

- 按 `leaf_yellowing` 等植物表型过滤检索，结果**为空**。这是诚实的空结果。
- 本草药性知识**不能**替代栽培病理证据，也**不能**据此确诊病害。
- 视觉分数与检索分数**不组合**成疾病概率。

`tests/test_corpus.py::test_the_corpus_is_TCM_knowledge_not_plant_pathology` 把这条边界钉成测试。

## 使用者

`skills/herbal_knowledge/skill.py` 有两个显式模式：

| 模式 | 行为 |
| --- | --- |
| `fixture` | 已发布合成案例，换任何输入明确失败（`FixtureMismatchError`） |
| `corpus` | 查询本地语料，返回证据 ID、文件位置、语料版本；检索不到返回空 |

模式由调用方显式选择，不会静默切换。`corpus` 模式缺失时 `provenance.corpus_version` / `corpus_sha256` / `retriever` 如实记录检索方式。
