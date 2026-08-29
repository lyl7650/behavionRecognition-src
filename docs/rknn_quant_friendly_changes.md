# RKNN 量化友好性改造文档

针对板端转换日志 `convert_cnn(1).log` 暴露的两个问题，对学生模型（RepViT-M1 + GRU）做两项改造。

- **问题 1**：12 个卷积权重存在离群值，int8 量化刻度被拉大，正常权重被压平 → 掉点。
- **问题 2**：CNN 特征输出（GAP+Flatten）被工具链默认降成 int8，只有 256 级分辨率就送进 GRU，误差被时序累积放大。

两项改造分别对症：**改造 A（部署形态训练）** 治问题 1，**改造 B（特征归一化）** 治问题 2。

---

## 改造 A：先重参数化，再微调（治离群权重）

### 为什么

RepViT 训练时每个 block 是「多分支卷积 + BatchNorm」，导出时才把 BN 的缩放系数折叠进卷积权重。个别 BN 通道的 `gamma / sqrt(running_var)` 很大，折叠瞬间就造出巨大的权重值。整个训练过程对此毫无感知——权重在训练时看起来很正常，是"折叠"这一步凭空放大了动态范围。

解法是把折叠提前到训练结束前：折叠完成后再用小学习率微调几个 epoch，让优化过程直接看到并抑制这些极端值。

### 怎么改

timm 提供了现成工具 `timm.utils.model.reparameterize_model`，它会递归调用每个子模块的 `fuse()`（RepViT 的 `ConvNorm` / `RepVggDw` 都实现了）。**注意 `features_only` 包装后的模型自身没有 `fuse` 方法，必须用这个工具函数递归处理，不要去找 `model.fuse()`。**

在 `models/timm_gru_student.py` 的 `TimmGRUStudent` 上加一个方法：

```python
@torch.no_grad()
def reparameterize(self):
    """把 backbone 的 Conv+BN 折叠成单个卷积（部署形态）。
    折叠不可逆，调用后不能再回到训练形态。"""
    from timm.utils.model import reparameterize_model
    self.backbone = reparameterize_model(self.backbone, inplace=True)
    return self
```

### 训练流程怎么接

在蒸馏脚本 `distill/distill_normal_3.py` 里，把原本"训练到收敛就导出"改成三段：

1. 正常蒸馏训练到收敛（不变）。
2. 调用 `student.reparameterize()`。
3. 用**原学习率的 1/10 ~ 1/20**、开着同样的蒸馏损失，再跑 3~5 个 epoch。

第 3 步的几个要点：

- 折叠后模型里已经没有 BatchNorm 了，所以**不要再依赖 BN 的 running stats 更新**，也别指望 BN 重新校准。
- EMA 权重要在折叠后重新初始化一次，否则 EMA 里存的是折叠前的参数形态，shape 和 key 都对不上。
- 折叠前后先做一次数值对齐验证（见下方验证清单），确认折叠本身没引入误差，再开始微调。

### 配套：让 weight decay 覆盖 BN 的 gamma

离群融合权重的来源是个别 BN 通道 gamma 过大。检查 `distill/distill_normal_3.py` 和 `train/train_finetune_normal_3.py` 的优化器参数分组：如果 BN 参数（`weight` / `bias`，一维参数）被排除在 weight decay 之外（这是很常见的默认做法），就是在放任 gamma 长大。

改法是给 BN 的 `weight`（即 gamma）单独一组，施加一个**适度**的 weight decay（建议 1e-5 ~ 1e-4，不要用主干那么大的值），BN 的 `bias` 仍然不加。这是预防措施，对已经训好的模型无效，只影响下一轮重训。

---

## 改造 B：GAP 之后加特征归一化（治 int8 特征输出）

### 为什么

现在 `forward` 里 GAP 出来的 384 维特征直接进 GRU，这个向量的动态范围完全没有约束，不同样本、不同通道的尺度可能差很多倍。RKNN 把它量化成 int8 时只能用一个统一刻度，尺度小的维度就被抹掉了。加一层归一化让特征分布稳定、有界，同一个 int8 刻度能表达的有效信息就多得多。

### 怎么改

在 `models/timm_gru_student.py` 的 `TimmGRUStudent.__init__` 里，`self.flatten` 之后、`self.gru` 之前插入：

```python
self.feat_norm = nn.LayerNorm(in_channels)
```

然后在 **`forward` 和 `forward_seq` 两个方法里都要加上这一层**（这点容易漏——两条路径必须完全一致，否则训练用的是 `forward_seq`、导出用的是 `forward`，行为会不一致）：

```python
# forward 里：
x = self.flatten(x)
x = self.feat_norm(x)          # 新增
x = x.unsqueeze(1)

# forward_seq 里：
x = self.flatten(x)
x = self.feat_norm(x)          # 新增
x = x.view(b, t, -1)
```

选 LayerNorm 而不是 BatchNorm1d 的原因：LayerNorm 是逐样本归一化，端侧单帧流式推理（batch=1）时行为和训练时完全一致；BatchNorm1d 依赖 batch 统计量，单帧推理要靠 running stats，多一层不确定性。

### 代价

这一层改变了模型结构，**必须重新蒸馏学生模型**，不能在现有 checkpoint 上热改（会 shape 不匹配）。所以建议和改造 A 合并成一次重训。

### 导出侧配合（这条要转告部署同事）

结构上加了归一化只是让量化"更容易"，**还要在转换时显式把 CNN 的特征输出设为 float16，不要接受默认的 int8**。日志里那条 `The default output dtype of '/Flatten_output_0' is changed from 'float32' to 'int8'` 就是需要覆盖掉的默认行为。这是最后一层的输出，改成 fp16 对整体性能影响很小。

---

## 建议的执行顺序

1. 先做改造 B 的代码修改（加 `feat_norm`，两条 forward 路径都改）。
2. 加上改造 A 的 `reparameterize()` 方法和优化器 BN weight decay 分组。
3. 重新蒸馏一轮：正常训练收敛 → 折叠 → 小学习率微调 3~5 epoch。
4. 导出 ONNX，交给部署同事按 fp16 特征输出 + 混合量化清单（见 `docs/rknn_hybrid_quant_layers.md`）转换。

## 验证清单

每一步都要能验证，否则问题会叠加到最后才发现：

- **折叠正确性**：折叠前后用同一批输入对比 **backbone 特征输出的相对误差**（`最大绝对差 / 输出幅值`），应低于 1e-3；同时 logits 的 argmax 必须完全不变。
  - 不要用「logits 的绝对误差 < 1e-4」这类判据。Conv+BN 折叠是精确的恒等变换，残差只来自 float32 运算顺序改变，而这个噪声**随折叠后权重的量级成比例放大**——正是那些 abs_max 达到 60 的离群层会把它推高。实测：未训练权重的 logits 绝对误差约 5e-7，把 BN `running_var` 调小模拟训练后状态（折叠权重变大）后就涨到 3e-5，真实训练模型可以到 8e-4 量级，但 argmax 始终不变、折叠本身完全正确。相对误差对权重量级不敏感，才是折叠正确性的有效指标。
- **离群值是否真的降下来了**：导出新 ONNX 后，用下面这段脚本重新统计各卷积权重的 `abs_max / abs_mean` 比值，和当前基线对比。目标是把 `ratio > 30x` 的层清掉。
- **微调没有掉点**：折叠+微调后的验证集准确率，不低于折叠前的收敛精度。
- **端到端量化精度**：板端 int8 模型 vs PyTorch fp32 在同一验证集上的准确率差值——这是最终唯一重要的指标。

统计离群值的脚本（对着导出的 ONNX 跑）：

```python
import onnx, numpy as np
from onnx import numpy_helper

model = onnx.load("repvit_backbone_sim.onnx")
init = {i.name: i for i in model.graph.initializer}
producer = {}
for node in model.graph.node:
    for inp in node.input:
        producer.setdefault(inp, []).append(node)

for name, tensor in init.items():
    arr = numpy_helper.to_array(tensor).astype(np.float32)
    if arr.ndim != 4:          # 只看卷积权重
        continue
    per_ch = np.abs(arr).reshape(arr.shape[0], -1).max(axis=1)
    ratio = per_ch.max() / max(np.median(per_ch), 1e-8)
    if ratio > 10:
        layer = producer.get(name, [None])[0]
        print(f"{ratio:6.1f}x  {layer.name if layer else '?'}")
```
