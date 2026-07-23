# Progress report — Round 1 + systematic confound control (2026-07)

Written to sync with the mentor. Companion to `HANDOFF.md` (full state) and
`flow-matching-notes.md` (deep dive). Numbers are on the round-1 set (3000 Suno + 3000 FMA
instrumental, 24k/mono/30s/LUFS), run on the Fudan GPU.

## 做了什么
把整条检测 pipeline 在复旦 GPU 上端到端跑通:提取 → 多编码器逐层特征 → linear probe → EER。
编码器:**MERT / wav2vec2 / MuQ / EnCodec**(XLS-R 因模型没预下载暂缺)。

## 结果(per-layer 最佳 test EER)
MuQ **0.00%** / MERT **0.67%** / wav2vec2 **1.78%** / EnCodec **3.67%**。音乐 SSL 碾压语音 SSL。

## 关键判断:这个近 0% 是混淆,不是 AI-detection(已严格验证)
- MuQ/MERT **每一层(含第 0 层)都近 0%** → 粗大混淆签名(几乎任何特征都能分开);
- wav2vec2 **早层最好、越深越差** → 信号在低层制作/保真,不在高层结构;
- 低层诊断:8 个可解释低层特征合起来只到 **24% EER**,而 SSL 到 0–2% → SSL 抓到了比 crude 制作
  统计丰富得多的东西(即不是廉价混淆)。

## 混淆控制(confound battery)—— 系统性排查每一类,而非只报数
- **已控住**:
  - *响度*:LUFS 归一化 → 诊断确认近瞎猜(rms EER 45.78%),排除;
  - *genre*:早前 genre-matched 实验 → 不是混淆源。
- **已定位为真实混淆、正在控**:
  - **era / production(录音时代 + 制作/母带差)** —— FMA 老/lo-fi(2008–2017)vs Suno 现代/干净。
    诊断显示 **Suno 更亮、高频更多(hf>10kHz AUC 0.744)、带宽更满**。控制手段:
    - **① 带宽/频谱匹配**:两边低通到同一截止,削掉 Suno 多出的高频/亮度后重跑 → 看 EER 还剩多少
      (直接在信号层把 era/production 代差控住);
    - **② 重建式 de-confounder**(借 Afchar 思路):"每首 vs 它自己的重建",内容/制作/**时代**/码率
      全锁死,只剩生成痕迹 → 从构造上消掉 era 混淆;
    - **③ era 对齐的人类对照**(备选,天花板有限,优先级低于 ①②)。
  - **cross-generator(AI 侧控制)** —— 用 **ACE-Step** 生成一批测 Suno-训练的检测器,分"通用 AI
    痕迹 vs Suno 专属"。
  - **判据**:只有 EER **在上述所有控制下都存活**,才敢称"检测到 AI",否则就是"检测数据集"。

## 结论 + 下一步
round-1 分离主要由 era/production 混淆驱动;pipeline 已验证,但编码器/层选型要等去混淆后才定。
近期先做 **①带宽匹配**(便宜)+ **cross-generator/ACE-Step**(决定性)。
**分类器(AASIST/SpecTTTra)与对抗留到 de-confound 之后**再投入。

## Flow-Matching / 重建法 —— 深挖(呼应师哥的核心思路)
沿着"**人类音频不在 AI 生成路径上**",把它落到一条很活的研究线:**用生成模型自身机制测一段音频
在不在生成流形上**(重建误差 / 似然),而非纯判别式分类。读了两篇代表作:
- **Afchar / Deezer,ICASSP 2025**:训分类器分"**原曲 vs 它自己过一遍神经 autoencoder 的重建版**"
  (EnCodec/DAC/GriffinMel/Musika),重建版存**同码率** → 内容、压缩锁死,只学到神经解码器痕迹。
  原版 vs 重建 99.8%,连没见过的 MusicGen 也 99.9%。
- **Diffusion-Reconstruction ADD(2026)**:潜扩散 codec 重建当难样本,后端 **XLS-R → AASIST** +
  对比学习;跨生成器 EER 36.8% → 20.2%(有改善,未解决)。

**对我们最有价值的一点**:"每首 vs 它自己的重建"是个**原则性 de-confounder**(同内容/同制作/同码率,
唯一差异是生成痕迹)—— 正好戳中 Suno-vs-FMA 内容/制作/时代全不同的痛点,可**从构造上绕开混淆**。
**诚实判断**:重建法**跨生成器泛化也不是免死金牌**(Afchar 自承跨解码器家族会崩),得实测跨
Suno→ACE-Step。

**落地(exploratory 旁支,不占 AASIST 主线)**:
- **PoC-A(最稳,料在手)**:用已有的 **EnCodec** 把每首重建一遍,训"原版 vs 自身重建"分类器,再测
  Suno / ACE-Step 带不带神经解码器痕迹;
- **PoC-B**:用 flow-matching/diffusion 模型做**往返重建(加噪→去噪)**,以重建误差当分数。

## 并行预备(代码)
帧级特征提取(`encode_frames`)+ 时序分类头脚手架(`classifiers/temporal.py`)已搭好并本地验证;
AASIST/SpecTTTra 到训练阶段从官方 repo 接入(不从零手写,避免细微 bug)。
