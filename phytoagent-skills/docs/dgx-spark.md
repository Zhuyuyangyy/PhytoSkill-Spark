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
