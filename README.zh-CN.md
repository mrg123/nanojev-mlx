# nanojev-mlx

**NanoJev 的 Apple Silicon 原生移植 —— 用 MLX 在 Mac 上跑 0.6B 并行决策模型。**
输入状态和问题，直接输出完整概率分布，零输出 token 解码。

原版 NanoJev 硬性要求 CUDA。本移植**不改动模型本身**，只把决策头用 MLX 重新实现，
于是它能在任何 Apple 芯片的 Mac 上原生跑在 Metal GPU 上——不需要 CUDA、不需要
PyTorch、不需要联网。

> 非官方社区移植，与 NanoJev 及 TypeSafe 作者无关。

---

## 实测对齐，不是口头保证

移植版与原 PyTorch 实现逐概率比对。`tests/reference/expected_pytorch.json` 是**原仓库**
`scripts/predict_toy_decisions.py` 在 CPU 上以 float32 跑出的基准答案。

| 检查项 | 结果 |
|---|---|
| 与 PyTorch（fp32）最大概率偏差 | **1.6e-07** |
| backbone 末层 hidden state 最大绝对误差 | 1.0e-04（数值量级达 67.6） |
| 概率分布归一化 | 全部通过 |
| 测试套件 | **19 项全部通过** |

语义在基准用例上全部一致：

| 问题 | PyTorch 参考 | nanojev-mlx |
|---|---|---|
| 重复扣款 → 哪个团队 | `billing` @ 0.9999974 | `billing` @ 0.999997 |
| 「退款是否已到账」（文中说尚未到账） | `p_true` 0.017378978 | 0.017379 |
| 软件报错 → 哪个团队 | `technical` @ 0.9994036 | `technical` @ 0.999404 |
| 严重度（4 档有序） | `score` 1.9999634 | 2.0000 |

自己复现：

```bash
NANOJEV_CHECKPOINT=/path/to/checkpoints/NanoJev python -m unittest discover -s tests -v
```

## 性能实测

Apple M4 / 16GB 统一内存 / macOS 26.5.1 / Python 3.14.6 / MLX 0.32.2。
负载：参考用例 —— 2 个 state、6 个问题、15 条候选路径、1 次 backbone 前向。

| 指标 | PyTorch + MPS | **nanojev-mlx** | 变化 |
|---|---:|---:|---:|
| 稳态单次耗时 | 470 ms | **352 ms** | **快 1.3×** |
| 峰值内存 | 4.91 GB | **2.75 GB** | **−44%** |
| 冷启动到首次出结果 | 9.9 s | **3.8 s** | **快 2.6×** |
| 依赖体积 | PyTorch（约 2.5 GB） | MLX（约 65 MB） | — |

**关于加速比，说实话。** 1.3× 是真实数字但不算惊艳，原因是结构性的：这个模型
**为每一条候选路径完整跑一遍 28 层 backbone**，所以算力瓶颈在 backbone，而不是本移植
重写的那 20 万参数决策头。MLX 在内存和启动上优势明显；纯延迟的收益受限于模型把
FLOPs 花在哪里。详见[已知限制](#已知限制)。

## 安装

需要 Apple 芯片 Mac（M1 及以上）与 Python ≥ 3.10。

```bash
git clone https://github.com/mrg123/nanojev-mlx.git
cd nanojev-mlx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

从 Hugging Face 取权重（约 2.4 GB）到 `checkpoints/NanoJev`：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="C-Tianyu/NanoJev",
    local_dir="checkpoints/NanoJev",
    allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"],
)
```

**没有格式转换步骤。** 本移植直接读取 `best.safetensors`，在加载时映射 PyTorch 的权重
布局，因此不会在磁盘上多出一份第二格式的权重副本。

## 用法

```bash
# 命令行
python -m nanojev_mlx.predict --checkpoint-dir checkpoints/NanoJev --input request.json

# 本地 HTTP 服务（与原版 POST /api/evaluate 协议兼容）
python -m nanojev_mlx.serve --checkpoint-dir checkpoints/NanoJev --port 8765
```

```python
from nanojev_mlx import load_model, run_prediction

model = load_model("checkpoints/NanoJev")
result = run_prediction(model, {
    "states": [{
        "id": "my_state",
        "state": "订单尚未退款。",
        "questions": {
            "already_refunded": {"type": "boolean", "instructions": "订单是否已经完成退款？"},
        },
    }],
})
# p_true = 0.0145899..., value = False
```

## 原理

模型 = 标准 **Qwen3-0.6B** backbone（占 99.97% 参数）+ 一个极小的决策头
（200,578 参数，占 0.034%）：

```
分组后的 hidden state [题数, 候选数, 1024]
  └─ LayerNorm
  └─ scalar         Linear(1024 → 1)      每条候选的基础 logit
  └─ set_project    Linear(1025 → 128)    hidden + log(候选数)
  └─ set_attention  4 头自注意力，在候选集合内做相对修正
  └─ set_output     Linear(128 → 1)       tanh 残差 → 每条候选的修正量
```

三种题型由同一次前向导出：

- **boolean** —— 单条语义路径，logits 为 `[0, z]`，故 `p_true = sigmoid(z)`
- **choice** —— 对修正后的候选 logits 做 softmax
- **score** —— 对有序等级描述做 softmax，期望值即分数

所有候选路径右填充成一个矩阵，**一次** backbone 前向算完。由于 padding 永远在真实
token 右侧，纯因果掩码产生的 hidden state 与原实现「因果 + padding 双重掩码」完全
等价——这正是本移植不需要自定义注意力算子的原因。

### 到底改了什么

只改运行时。模型、权重、分词器、提示构造、概率语义全部未动：

| 组件 | 原版 | 本移植 |
|---|---|---|
| backbone | HF `Qwen3Model`（PyTorch） | `mlx_lm` `Qwen3Model` |
| 决策头 | `nn.MultiheadAttention` | MLX 显式注意力 + 加性掩码 |
| 融合的 `in_proj_weight` | 保持融合 | 拆成 `q_proj` / `k_proj` / `v_proj` |
| 设备门禁 | 硬性要求 CUDA | 走 MLX 的 Metal GPU |

`prepare_examples`（把状态、问题、候选集变成 token 路径的那段）是**逐行**移植的——
只要差一个 token，下游所有概率都会变。

## 已知限制

- **只做推理。** 原仓库的训练、评测、benchmark 流水线没有移植。本仓库负责把模型跑
  起来，不负责复现论文的训练数字。
- **未做前缀共享。** 每条候选路径都从头跑一遍 backbone，与原版一致
  （`prefix_sharing: false`）。同一问题下的候选共享很长一段前缀，缓存它是显而易见
  的下一步优化，能实质减少计算量。
- **尚未量化。** 权重以 float32 加载以保证数值保真。MLX 支持 4bit/8bit 量化，可将内存
  降到约 1/4 并很可能提速——但它会改变数值，需要单独一轮验证才能采信。
- **不含权重。** 请按上面的方式从 Hugging Face 下载。
- **仅 float32。** 上游是在 A100 上用 bf16 autocast 训练的。本移植跑 fp32，这是一条
  **不同**的数值路径——它更接近用于等价性验证的 CPU fp32 基准，而不是论文公布的指标。

## 致谢与许可

- 原始模型、训练流水线与参考实现：[TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) —— MIT
- backbone：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- MLX 移植：MIT，见 [LICENSE](LICENSE)，保留原始版权声明

本项目与原项目均为独立研究工作，与 TypeSafe 或 Jev 无关。
