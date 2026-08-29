# RKNN 混合量化层清单（RepViT-M1 backbone）

依据：转换日志 `convert_cnn(1).log` 的 `found outlier value` 告警 + `repvit_backbone_sim.onnx` 权重实测。

日志告警里列了 13 行，其中 `onnx::Conv_1377` 和 `onnx::Conv_1557` 各报了两个离群值（占两行），**实际是 12 个卷积层**。

## 12 个离群层的层名对照表

`onnx::Conv_xxxx` 是 ONNX 里的权重张量名（初始化器名），做混合量化配置时要用的是右边的**节点名**。

| 权重张量名 | ONNX 节点名 | 权重 shape | abs_mean | abs_max | 离群倍率 |
|---|---|---|---|---|---|
| onnx::Conv_1296 | `/backbone/stages_1/blocks/blocks.1/token_mixer/conv/c/Conv` | [48, 1, 3, 3] | 0.46 | 53.64 | **301x** |
| onnx::Conv_1512 | `/backbone/stages_2/blocks/blocks.11/token_mixer/conv/c/Conv` | [192, 1, 3, 3] | 0.17 | 13.66 | **55x** |
| onnx::Conv_1377 | `/backbone/stages_2/downsample/ffn/conv2/c/Conv` | [192, 384, 1, 1] | 0.17 | 19.46 | **31x** |
| onnx::Conv_1365 | `/backbone/stages_2/downsample/pre_block/channel_mixer/conv2/c/Conv` | [96, 192, 1, 1] | 1.60 | 60.92 | 13x |
| onnx::Conv_1401 | `/backbone/stages_2/blocks/blocks.1/channel_mixer/conv2/c/Conv` | [192, 384, 1, 1] | 0.28 | 13.57 | 10x |
| onnx::Conv_1485 | `/backbone/stages_2/blocks/blocks.8/channel_mixer/conv2/c/Conv` | [192, 384, 1, 1] | 0.29 | 11.12 | 10x |
| onnx::Conv_1533 | `/backbone/stages_2/blocks/blocks.12/channel_mixer/conv2/c/Conv` | [192, 384, 1, 1] | 0.60 | 22.21 | 10x |
| onnx::Conv_1557 | `/backbone/stages_3/downsample/pre_block/channel_mixer/conv2/c/Conv` | [192, 384, 1, 1] | 0.78 | 21.49 | 8x |
| onnx::Conv_1593 | `/backbone/stages_3/blocks/blocks.1/channel_mixer/conv2/c/Conv` | [384, 768, 1, 1] | 0.78 | 20.88 | 7x |
| onnx::Conv_1425 | `/backbone/stages_2/blocks/blocks.3/channel_mixer/conv2/c/Conv` | [192, 384, 1, 1] | 0.47 | 15.64 | 7x |
| onnx::Conv_1290 | `/backbone/stages_0/blocks/blocks.0/channel_mixer/conv1/c/Conv` | [96, 48, 1, 1] | 0.41 | 13.19 | 6x |
| onnx::Conv_1497 | `/backbone/stages_2/blocks/blocks.9/channel_mixer/conv2/c/Conv` | [192, 384, 1, 1] | 0.49 | 11.09 | 6x |

「离群倍率」= 最大通道的权重峰值 / 各通道峰值的中位数。倍率越高，per-channel 量化越救不回来。

## 关键发现：先确认是否已启用 per-channel 权重量化

统计每个输出通道各自的权重峰值后可以看到，**离群值高度集中在少数几个输出通道**——例如 `Conv_1296` 的 48 个通道里，通道峰值的中位数只有 0.18，而最大的那个通道达到 53.64。

这意味着：

- 如果当前转换用的是 **per-layer**（整层一个刻度）权重量化，那么**光是切到 per-channel（逐输出通道一个刻度）就能解决大部分问题**，几乎不损失性能。这应该是第一个动作。日志里的 `--> Building RKNN` 参数行没有显示量化粒度，需要查转换脚本里 `rknn.config()` 的量化配置确认。
- 即便开了 per-channel，倍率最高的三层仍然可能救不回来——因为离群值不只是"某几个通道整体偏大"，而是通道内部也存在极端点。

## 分两档处理

**第一档：必须保留高精度的 3 层**（倍率 ≥ 30x，per-channel 也难救）

```
/backbone/stages_1/blocks/blocks.1/token_mixer/conv/c/Conv
/backbone/stages_2/blocks/blocks.11/token_mixer/conv/c/Conv
/backbone/stages_2/downsample/ffn/conv2/c/Conv
```

这三层建议在混合量化配置里指定为 float16。前两层是 depthwise 3x3 卷积（token_mixer），权重量很小（48×9 和 192×9），保留 fp16 对模型体积和速度的影响可以忽略。

**第二档：观察，按需追加的 9 层**（倍率 6~13x）

表格里剩下的 9 层，先只开 per-channel 试一轮。如果精度已经达标就不用动；如果还差，按倍率从高到低（1365 → 1401 → 1485 → 1533 → 1557 → 1593 → 1425 → 1290 → 1497）逐步加进 fp16 清单，每加一批测一次精度，避免一次全设 fp16 让性能白掉。

第二档里 `conv2/c/Conv` 这类 1x1 卷积权重较大（192×384、384×768），设成 fp16 对体积和带宽有实际影响，所以要按需追加，别一次性全开。

## 另外两项转换侧配置

1. **特征输出改回 fp16**。日志里 `The default output dtype of '/Flatten_output_0' is changed from 'float32' to 'int8'` 需要覆盖——这是要送进 GRU 继续计算的特征向量，压到 int8 会被时序放大。最后一层输出改 fp16 对性能影响很小。
2. **量化校准集要覆盖真实分布**。离群值的实际影响取决于激活值分布，校准集要用真实场景的猫狗视频帧，而且要覆盖各个类别，别只用几张图。

## 长期解法

这份清单是转换侧的兜底。离群值的根因是导出时 Conv+BN 折叠，训练端的处理见 `docs/rknn_quant_friendly_changes.md`——重训一轮后这份清单大概率可以缩短甚至清空，届时重新跑一遍该文档里的统计脚本更新即可。
