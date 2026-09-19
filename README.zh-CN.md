# nanojev-mlx

**简体中文** | [English](README.md)

[![tests](https://github.com/mrg123/nanojev-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/mrg123/nanojev-mlx/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**NanoJev 的 Apple Silicon 原生移植 —— 用 MLX 在 Mac 上跑 0.6B 并行决策模型。**
输入状态和问题，直接输出完整概率分布，零输出 token 解码。

原版 NanoJev **以 CUDA 为先**：它的推理脚本没有 CUDA 会直接报错退出，记录的训练环境也
面向 A100、bf16 精度。为了做**同口径**对比，下面的基准只移除那一处设备门禁——仅此一处，
原仓库其余代码一行未动——让**同一份**脚本能跑在 PyTorch MPS 上；代价是必须拖上 PyTorch，
内存和启动时间也明显更高。本移植**不改动模型本身**，只把决策头改用 MLX 重新实现，于是
同一份权重可以原生跑在 Metal GPU 上，且完全不依赖 PyTorch。

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
| 测试套件 | **36 项全部通过**（其中 4 项无权重时跳过） |

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
| 稳态单次耗时 | 456 ms | **344 ms** | **快 1.3×** |
| 峰值内存 | 4.91 GB | **2.75 GB** | **−44%** |
| 冷启动到首次出结果 | 8.2 s | **2.2 s** | **快 3.8×** |
| 运行时依赖占用 | PyTorch 约 590 MB | **MLX 约 210 MB** | **小约 2.8×** |

在你自己的机器上复现全部数字：

```bash
python benchmarks/benchmark.py --checkpoint-dir checkpoints/NanoJev \
    --backend both --nanojev-repo ../NanoJev
```

`--backend both` 会在各自独立的子进程里测量，避免两套栈共存污染内存读数。只测 MLX 移植版
的话，去掉 `--nanojev-repo` 和 `--backend both` 即可。

**关于加速比，说实话。** 1.3× 是真实数字但不算惊艳，原因是结构性的：这个模型
**为每一条候选路径完整跑一遍 28 层 backbone**，所以算力瓶颈在 backbone，而不是本移植
重写的那 20 万参数决策头。MLX 在内存和启动上优势明显；纯延迟的收益受限于模型把
FLOPs 花在哪里。详见[已知限制](#已知限制)。

**这些数字是怎么测的。** 同一台机器、同一输入（参考用例）、双方均为 fp32：延迟取 6 轮
预热后的**中位数**，内存取峰值 RSS，冷启动取「导入运行时 + 加载模型 + 首次出结果」——
也就是用户跑命令行时真正要等的全部时间。依赖占用按 Apple 芯片上安装后的 `site-packages`
体积统计——`torch` 对比 `mlx` + `mlx-lm`；两套方案都必须安装的 `transformers`、`numpy`、
`safetensors` 相互抵消，不计入对比。

冷启动包含从磁盘读取 2.4 GB 权重，因此在一台机器上**头一次**运行（文件还不在系统页缓存
里）双方各自会慢约 1.5 秒，比值也随之收窄。上表用的是稳定可重复的那组数字。

## 安装

**需要 Apple 芯片 Mac（M1 及以上）、macOS 14（Sonoma）或更高版本**，以及 Python ≥ 3.10。
MLX 只为 macOS 14 / 15 / 26 提供 wheel——macOS 13 及更早版本装不上，Intel Mac 完全无法运行。

```bash
git clone https://github.com/mrg123/nanojev-mlx.git
cd nanojev-mlx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

可选：把包本身也装上，会额外得到 `nanojev-mlx` 与 `nanojev-mlx-serve` 两个命令：

```bash
pip install -e .
```

从 Hugging Face 取权重（约 2.4 GB）到 `checkpoints/NanoJev`。`huggingface_hub` 已作为
`transformers` 的依赖存在：

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

[`examples/request.json`](examples/request.json) 是一个可直接运行的请求，两个 state、覆盖全部
三种题型（choice / boolean / score），建议从这里开始：

```bash
python -m nanojev_mlx.predict --checkpoint-dir checkpoints/NanoJev --input examples/request.json
```

它打印完整的结果 JSON（每条候选的概率、实际生效的 temperature，以及标明设备与
零解码步数的 `execution` 段）。把上面那次运行的答案按每题一行整理出来是：

| State | 问题 | 题型 | 答案 |
|---|---|---|---|
| `refund` | `team` | choice | `billing` @ 1.0000 |
| `refund` | `arrived` | boolean | `p_true` 0.0174 → `False` |
| `refund` | `severity` | score | 1.464（4 档中的第 2 档） |
| `button_error` | `team` | choice | `technical` @ 0.9847 |
| `button_error` | `blocking` | boolean | `p_true` 0.2363 → `False` |

```bash
# 本地 HTTP 服务（与原版 POST /api/evaluate 协议兼容）
python -m nanojev_mlx.serve --checkpoint-dir checkpoints/NanoJev --port 8765

curl -X POST http://127.0.0.1:8765/api/evaluate \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
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

## 测试

分两层，按所需条件划分：

| 套件 | 需要什么 | 能跑在哪 |
|---|---|---|
| `tests/test_contract.py` —— 请求校验与答案组装 | 只要 Python | 任何平台；不需要 MLX、GPU、权重 |
| `tests/test_equivalence.py` —— 与 PyTorch 参考逐概率比对 | MLX + 2.4 GB checkpoint | Apple 芯片 |

```bash
# 只跑协议层 —— 不需要 MLX、GPU、权重。CI 跑的就是这一条
python -m unittest discover -s tests -p "test_contract.py" -v

# 全部，含数值等价性验证
NANOJEV_CHECKPOINT=/path/to/checkpoints/NanoJev python -m unittest discover -s tests -v
```

共 **36 项测试**：15 项协议层、17 项 demo、4 项等价性。没有 checkpoint 时那 4 项会**跳过而不是
失败**，所以刚克隆下来也能跑其中的 **32 项**。

CI 在 Linux 上覆盖协议层。等价性测试刻意不进 CI：MLX 在无头 / 无 GPU 环境中**导入即崩溃**
（[ml-explore/mlx#3148](https://github.com/ml-explore/mlx/issues/3148)），而 GitHub 托管的
macOS runner 不提供 Metal 设备。这也正是把输出路径中纯逻辑的那一半拆到
`nanojev_mlx.answers`、不引入 MLX 的原因。

## 演示：看它玩贪吃蛇

本仓库其他地方，模型的输出都是 JSON——对程序是正确的输出，对人却是糟糕的第一印象。
演示脚本在你本机 GPU 上跑完整一局，写出**一个自包含的 HTML 页面**：不需要服务器、不需要
联网、不依赖任何 CDN：

```bash
python -m demo.play --checkpoint-dir checkpoints/games/variants/games_gold_seed17 --open
```

页面会逐步回放这一局，并在**每一步**展示模型给**每个**候选走法分配的概率——这正是逐
token 生成的模型在结构上无法展示的东西。

它需要的是贪吃蛇专用的 checkpoint，不是根发布版——下载方式见
[`demo/README.md`](demo/README.md)。注意页面会把这个架构暴露出来：代码规划器先过滤掉会
碰撞的走法、保留到食物静态最短路径上的那些；**只有在剩下两个及以上候选时**才问模型。
只剩一个时候选时，走法是被强制的，**根本不会调用模型**。

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
| 设备门禁 | 没有 CUDA 直接报错退出（基准仅移除这一处门禁） | 走 MLX 的 Metal GPU，不依赖 PyTorch |

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

## 故障排查

**`pip install` 找不到 `mlx`，或解析到非常旧的版本。**
MLX 只为 Apple 芯片的 macOS 14 / 15 / 26 提供 wheel。macOS 13 及更早、或 Intel Mac 上没有
wheel，本移植无法在这些环境运行。

**`ValueError: checkpoint 缺少 best.safetensors`（或 `backbone_config`、`tokenizer`）。**
`--checkpoint-dir` 必须指向**直接包含**这些条目的目录。在 Hugging Face 仓库上，这要么是仓库
根目录，要么是 `variants/<name>/` 目录——两者布局完全相同：

```
best.safetensors
config.json
backbone_config/config.json
tokenizer/{tokenizer.json, tokenizer_config.json, chat_template.jinja}
```

所以 `variants/local_atomic_seed17` 也能用，只要指向 variant 目录本身而不是它的上级。注意
variant 是在不同数据（迷宫、Snake）上训练的，不会复现本 README 的数字——那组数字用的是根
发布版。

**导入时报 `OSError` / `NSRangeException`，或还没输出任何东西就崩溃。**
MLX 需要 Metal GPU。这在无头虚拟机、CI 容器以及部分远程 / SSH 会话中会发生。这不是本移植
能绕过的——请在正常的 macOS 桌面会话里运行。

**16 GB 机器内存吃紧。**
本移植峰值约 2.75 GB。同时再跑 PyTorch 版会再加约 4.9 GB，两者并行会让 16 GB 机器吃紧。
本仓库其余部分都不占内存。

## 致谢与许可

- 原始模型、训练流水线与参考实现：[TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) —— MIT
- backbone：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- MLX 移植：MIT，见 [LICENSE](LICENSE)，保留原始版权声明

本项目与原项目均为独立研究工作，与 TypeSafe 或 Jev 无关。
