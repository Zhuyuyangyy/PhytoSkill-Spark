# DGX Spark 实测记录

日期：2026-09-21。所有数字来自本次运行，未实测的一律标注。

## 一、节点事实

通过 SSH 探测（`python -m dgx probe`），凭据从 gitignored `.dgx.env` 读取：

| 项 | 实测值 |
| --- | --- |
| 主机名 | `gx10-9ec6` |
| 架构 | `aarch64` |
| 内核 | `6.17.0-1014-nvidia` |
| 系统 | Ubuntu 24.04.4 LTS |
| CPU / 内存 | 20 核 / 119 GiB |
| GPU | **NVIDIA GB10**，驱动 `580.142` |
| CUDA | `V13.0.88`（`/usr/local/cuda/bin/nvcc`） |
| Docker | 29.2.1，`nvidia-container-runtime` 在位 |
| 磁盘 | 916 G 总量，528 G 可用 |
| Python | 3.12.3 |

**`nvidia-smi` 在这台机器上读不到显存容量**：`memory.total` 与 `memory.used` 都是 `[N/A]`。
这是 DGX Spark 的统一内存架构使然，不是查询写错。可用的替代证据是
`nvidia-smi --query-compute-apps` 与 ollama 的 `/api/ps`，后者报告
`size_vram: 33380567612`（约 31 GiB）——这是模型实际占用显存的直接读数。

## 二、节点上已有什么

* **ollama 常驻一个视觉模型**：`modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest`，
  27.3B 参数，Q4_K_M 量化，上下文 262144。`capabilities` 为
  `["completion", "tools", "thinking", "vision"]`。
* CUDA 13.0 工具链完整，`nvcc` 可用。
* Docker 与 nvidia-container-runtime 可用——这意味着真实推理可以关在容器里跑，
  不必改动宿主环境。
* 宿主 Python **没有** torch / transformers / numpy / onnxruntime / tensorrt。
  只有 PIL。所以本轮的视觉路径走 ollama 的 HTTP 接口，而不是自己搭推理栈。

注意：节点上已有两个进程占着 GPU（`llama-server` 约 33 GiB，另有一个约 16 GiB）。
这不是本项目启动的，探测时它们就在跑。

## 三、真实视觉推理

`dgx/vision.py` 把图片 base64 上传，调用 ollama `/api/generate`，拿回自由文本，
解析成 `plant_vision` 契约要求的结构化区域。

第一次往返（2×2 合成图，左黄右绿）：

```
latency_ms: 7345.1
reply: Left half: Yellow / Right half: Green
eval_count: 70
```

这证明管道是真的：模型在读图并给出自己的判断，不是返回预设。

### prompt 是边界的第一道防线

prompt 明确要求：bbox 必须归一化到 0-1、只描述不诊断、没有可见区域就返回空。
解析层是第二道防线：任何区域的文本里出现 `病原` / `病因` / `感染了` /
`确诊` / `caused by` / `pathogen` 等词，该区域被丢弃。

### 踩到的一个真问题：模型不听话归一化

第一次拿到区域时，`_normalise_region` 把四个区域全丢了。原因是模型返回了
**像素坐标**（`[145, 505, 740, 857]`）而不是 0-1。prompt 里写了要求，模型没照做。

修法有两条，都用了：

1. prompt 里把归一化要求加粗成单独一条，并说明"除以图像宽高"。
2. 解析层加防御：只要任一坐标 > 1，就判定整个框是像素空间，按最大值推断画布
   尺寸后缩放。真实观察不该因为模型不听指令而被丢掉。

### 合成图的诚实结论

用 PIL 造了三张"叶片"图，模型对它们的判断是 `image_usable: false`。
用开放式 prompt 追问，模型的回答是：

> No, this is not a photograph of a real plant leaf. It is a digital illustration
> or abstract graphic featuring geometric shapes...

**模型是对的。** 我造的图确实是几何图形，不是叶片照片。所以
`image_usable: false` 是正确判断，不是 bug——接口、解析、GPU 记录全部工作正常。

当换用一个只要求"报告你看到的有色区域"的 prompt 时，模型正常产出了
1 个 `leaf_yellowing` + 3 个 `leaf_spot`，bbox 全部落在 0-1 并通过契约校验。
这说明结构化输出通路是通的，缺的是**真实照片**。

## 四、端到端链路

`python -m dgx.end_to_end <image> <species>` 跑完整链条，
每次提供方调用都经过 AgentShield 中间件：

```
[1/3] plant_vision (live) -> success in 34632 ms
      data_origin=dgx_live_inference dgx_hardware_used=True
      image_usable=False observations=0
[2/3] evidence_fusion -> success
      completeness=partial missing=['environment', 'knowledge']
[3/3] claim-evidence audit -> trust_level=INSUFFICIENT
      supported=0 refused=0 violations=0
```

注意第三步：没有环境来源、没有知识来源、没有观察，审计判 `INSUFFICIENT`。
**这是诚实的降级，不是失败。** 整条链没有为了好看而编造结论。

产物：`artifacts/dgx/end-to-end.json`、`artifacts/dgx/vision-variance.json`。

## 五、接入方式与边界

`plant_vision` 新增 `live` 模式，manifest 里声明：

```json
"supported_modes": ["fixture", "live"],
"live_requirements": {
  "dgx": "SSH access to a DGX Spark node via PHYTO_DGX_* credentials",
  "model": "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest (ollama)",
  "fallback": "none; a missing node or credential is a hard error"
}
```

`fallback: none` 是关键：`live` 模式下 `fixture://` 占位符被明确拒绝，
图片不存在也明确报错。用合成数据回答真实请求就是伪造观察，这条路堵死了。

另外两处 schema 调整，都是为了让诚实的空结果能通过：

* `provenance.model_called` 从 `const false` 改为 `boolean`——live 模式确实调了模型。
* `observations` 去掉 `minItems: 1`——模型看了没找到可报告区域，返回空是合法结果，
  强制非空等于强制编造。

## 六、明确未做

* **没有真实叶片照片**：所有测试图都是 PIL 合成的。模型对它们判不可用是正确行为。
  真实农学数据需要项目所有者提供。
* **没有 YOLO / 实例分割 / TensorRT**：本轮走 ollama，未部署检测模型。
* **没有 Embedding 与 cuVS**：语料检索仍是确定性字符二元组重合度。
* **没有稳定性数据**：模型输出不稳定，同一次输入可能返回区域也可能返回空。
  `artifacts/dgx/vision-variance.json` 记录了这个方差，不掩盖。
* **没有延迟基准**：单次 7–35 秒，受节点上其他进程争用影响，不当作性能指标。

## 七、真实数据集：15 张黄芪图，但内容与预期不同

拿到一个图片目录后，先做的是**普查它到底是什么**，而不是直接跑推理。
`python -m dgx.survey <dir>` 对每张图问同一个开放式问题，记录回答。

结论：**15 张全部是干燥黄芪药材切片（根茎），没有一张是种植植株的叶片。**

模型的原话（节选）：

> The image displays sliced, dried root pieces of Huangqi (Astragalus), a
> processed herbal material, presented in a white dish and a labeled glass jar.

这个发现改变了实现。用叶片黄化的 prompt 去问一盘子根切片，模型要么返回
`image_usable: true, regions: []`，要么编一个覆盖整个盘子的区域——因为图里
没有叶片可描述。仪器必须匹配对象。

因此 `plant_vision` 新增 **`herb` 模式**，观察药材切片真正相关的性状：
断面（裂隙/粉性/致密/空心）、色泽（淡黄/琥珀/深褐）、霉变、虫蛀、片型。
`live` 模式保留给田间/叶片表型。两个模式都在 manifest 的 `mode_semantics` 里
写明了各自观察什么——两个都叫"跑模型"的模式可以观察完全不同的对象，
这个区别必须进签名清单，而不是只写在没人读的 README 里。

### 批量推理结果（`artifacts/dgx/herb-batch.json`）

15 张图，每张一次真实 GPU 推理：

| 指标 | 值 |
| --- | --- |
| 成功 | **14 / 15** |
| 产出观察 | 7 张 |
| 观察总数 | 57 |
| 延迟 | 11.6 s – 167.0 s，均值 54.4 s |
| 识别出的性状 | `colour_pale_yellow` ×31、`cut_surface_dense` ×16、`colour_dark_brown` ×6、`colour_amber` ×2、`slice_irregular` ×2 |
| GPU | 全程 NVIDIA GB10 |

`colour_pale_yellow` 是被识别最多的性状——淡黄色断面正是硫熏切片的典型特征。
这是模型自己从照片里看出来的，没有人告诉它该看什么。

### 过程中修掉的三个真问题

1. **`num_predict=500` 把 JSON 截断**。模型要报告十几个区域，写到一半没预算了，
   整个文档因此不闭合，`json.loads` 拒绝全部。抬到 1500 只是一半答案——
   真正管用的是 `_salvage_regions()`：扫描所有括号配平的 `{...}` 片段，
   救回它写完的那些，丢弃没写完的那个。第一版用单层 depth 计数，结果内层
   region 被外层包裹器吞掉；第二版改成递归，又无限递归。最后用开放括号栈 +
   每遇 `}` 尝试解析，才同时拿到外层和内层。
2. **`area_basis` 硬编码 `visible_leaf_area`**。herb 模式观察的不是叶片，
   这个常量让每一次药材观察都校验失败。改成枚举，两种依据都合法。
3. **600 秒超时不够**。同一份代码，快时 12 秒，慢时 167 秒——节点上还有别人
   的进程在抢 GPU。huangqi_04 在批量里失败、单独跑却成功（19 个观察），
   说明是间歇性超时而非代码缺陷。超时提到 900 秒，并在注释里写明这不是保守
   而是最慢那张真实图像需要的。

### 诚实的保留

- **没有地面真值**。57 个观察是模型的描述，没有人标注过这些图究竟该有什么性状。
  所以这一轮证明的是"链路能跑通、能结构化、能审计"，**不是**"识别准确"。
- **7/15 张没有产出观察**。模型判 `image_usable: true` 但没找到可报告区域。
  空结果是合法输出（schema 已允许 `observations` 为空），但覆盖率只有 47%，
  这个数字不掩盖。
- **延迟方差 14 倍**。受节点争用影响，不能当性能指标。

## 八、第二组真实数据：15 张植株叶片

又拿到一组图（`黄芪植株叶片合集.zip`）。仍然是**先普查再动手**，这次的结果与上一轮相反：

> **15 张全部是活的黄芪植株**——羽状复叶、花、荚果。第 7 张模型直接认出
> *"a flowering legume (likely a species of Astragalus)"*。

所以 `live` 模式（叶片表型）终于有了对得上的输入。两个模式现在都有真实数据：

| 模式 | 图集 | 全链路结果 |
| --- | --- | --- |
| `herb` | 15 张药材切片 | 76 个观察，106 supported claims |
| `live` | 15 张植株叶片 | 3 个观察，33 supported claims |

`live` 只产出 3 个观察，**这不是失败**。用开放式 prompt 追问那些零观察的图，模型的回答是：

> The image shows a close-up of a green plant with compound leaves, each
> consisting of multiple small, oval-shaped leaflets arranged along a central
> stem.

叶片是健康的，没有可见病变。模型报告"没有可报告区域"是**正确行为**——
`schema` 早已允许 `observations` 为空，强制非空等于强制编造。

### 这一轮修掉的两个真问题

1. **全图覆盖框被当成了区域观察。** 换用一个更直白的区域 prompt 时，模型返回
   `bbox: [0, 0, 1000, 1000]`——像素版被我的越界检查拒绝，但归一化的
   `[0, 0, 1, 1]` 会通过。一个覆盖整张图的框不是区域定位，而是模型在说
   "我不想定位"。放它过去等于把"我看到了这张图"变成一条有据可查的结论。
   现在两个解析器都拒绝宽高均 ≥ 0.98 的框。
2. **两次普查互相覆盖。** `dgx/survey.py` 把结果写死到 `dataset-survey.json`，
   叶片普查静默覆盖了药材普查的记录。改为按目录名派生文件名，并把 `directory`
   记进 JSON。数据丢失这种错误不该靠"我记得跑过什么"来发现。

### 仍然没有的

**地面真值。** 两组共 30 个观察全部是模型的描述，没有人标注过这些图究竟该有什么
性状。这一轮证明的是：两个模式都能跑通、能结构化、能审计、能在证据不足时
如实报告空结果。它**不证明**识别准确率——那需要人工标注，只有项目所有者能做。
