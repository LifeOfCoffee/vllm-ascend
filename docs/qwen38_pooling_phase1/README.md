# 阶段 1：Qwen3.8 Next AscendStore 初始化验证补丁

本目录保存阶段 1 的原始 Git 补丁与验证说明。代码审阅分支基于本仓库 main，
仅增加这两个交付文件，**没有把补丁应用到 main 的运行源码**。
当前 main 缺少补丁依赖的 QSA unified cache、page-strided view 和相关布局代码。
因此本分支 用于保存与审阅阶段 1，不代表 main 已具备可独立运行的模型适配。

## 范围与基线

- 补丁：[0001-initialize-qsa-caches.patch](0001-initialize-qsa-caches.patch)
- 原始阶段 1 提交：`b01d71125322bf4338d1969a87c2f4340166dfec`
- 原始 Ascend 镜像基线：`2a450bd527f8a51be456de623369f7062a107427`
- 原始 vLLM 镜像基线：`c4df37dfcc7f72ca41cd43801a468526ecf99a2b`
- 补丁包含 5 个文件，411 行增加、23 行删除：model runner、AscendStore connector、
  两个单测文件与阶段 1 文档。
- 镜像既有模型实现及 QSA 依赖不在本分支 中；GBSA 实现、文档及构建改动不在本分支 中。
  本分支 不引入镜像基线提交或其历史。

## 阶段 1 实际改动

1. 按真实 cache spec 构造 view，保留 page stride、padding 和 allocation slice offset，
   检查 dtype 对齐、每页 payload 与 slice 边界。
2. 主 KV 使用每页连续 K 平面与 V 平面，返回共享 allocation 的 K/V view；
   compressed/raw indexer 按各自 state 数与宽度构造 view。
3. 增加严格布尔配置 `initialization_only=true`，要求 eager、
   `use_layerwise=false`、`load_async=false`。
4. 实际创建 memcache 后端并注册 HBM，关闭外部 lookup/load/save；
   在 raw-cache snapshot restore 未完成前，拒绝 QSA 正常外部复用。

阶段 1 不实现外部缓存搬运、raw indexer 边界快照、主 KV/indexer/GDN/PLE
在同一提交边界的一致性恢复；connector 下的 shared-backing/six-region 支持也仍待后续阶段。

## 在正确基线上应用

在包含上述镜像基线的独立代码目录中执行，先检查基线：

```bash
git rev-parse HEAD
git apply --check /path/to/0001-initialize-qsa-caches.patch
git am /path/to/0001-initialize-qsa-caches.patch
```

补丁保留原始 author、Signed-off-by 与提交说明。不要直接在当前 main 上应用并
假设依赖已存在；依赖基线就绪后，才可将补丁应用到运行源码。

## 已完成验证

以下结果对应“原始完整镜像基线 + 阶段 1”，不是本分支 的 main 独立构建结果。

- 定向单测：193 passed，12 skipped；worker 测试另有 73 deselected。
- 原始基线也失败的 CPU/NPU custom-op 测试未纳入定向选择。
- Ruff 0.14.0：原始与修改后文件同样存在 7 条既有 import-order 诊断，没有新增。
- `git diff --check` 通过；`bash format.sh ci` 因镜像缺少 pre-commit 未完成，
  不声明完整格式检查或 CI 通过。
- TP8 / DP1，后 8 张卡，eager，MTP3，本地 prefix cache 与 Mamba align：
  8 个 rank 均完成真实 memcache 初始化与 HBM 注册。
- 每 rank：282 blocks、76 underlying storages、125 bound/transfer tensors；
  view 边界与 pointer/payload/page stride 检查通过，诊断请求前后指针稳定。
- 先前离线诊断：短生成 16 token 与基线 smoke token IDs 相同；
  768-token 输入的两次 16-token 输出 IDs 相同。
- HTTP 服务：4 次真实请求均 HTTP 200；8 个 worker 各记录 76 次
  RegisterMemory success，检查时日志没有 ERROR/Traceback。

### 中文聊天输入输出

请求：`/v1/chat/completions`，temperature=0，max_tokens=128，
`chat_template_kwargs.enable_thinking=false`。

输入：

> 请用一句话解释什么是大语言模型的 KV 缓存。

输出：

> KV 缓存是大语言模型在自回归生成过程中，通过存储已处理 token 的键（Key）和值（Value）向量，以避免重复计算历史注意力机制、从而显著加速推理过程的技术。

HTTP 200；23 个输入 token、45 个输出 token；finish_reason=stop；
单次端到端耗时约 8.34 秒。这是 smoke 结果，不是性能基准。

### 长输入重复请求的限制

同一 1618-token 输入、temperature=0、seed=0、max_tokens=32，顺序执行两次，
均 HTTP 200，耗时约 13.52 / 12.36 秒。两次生成文本不同，
且都因 token 上限结束；因此不能声明真实 HTTP 重复请求数值一致性通过。
原因尚未定位，当前没有对照试验来区分镜像基线、MTP、本地 prefix cache
或阶段 1 的影响。

所有这些请求均运行在 initialization-only 模式，不能证明 memcache 外部命中、
字节往返、异地 block 恢复、恢复后的续算正确性或推测解码 accept/reject 正确性。

## 当前交付状态

按要求仅上传独立分支，不创建 PR，不修改 main。QSA 依赖源码上传被自动审批拦截；本分支先提供阶段 1 的原始改动供审阅，没有上传被拒绝的依赖载荷。
