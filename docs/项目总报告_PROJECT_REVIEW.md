# BBBB 项目总报告：用自回归大模型生成 PDE 的「全部共存解」

> 本文档对 `BBBB_qwen_pde_branch` 整个项目做一次完整梳理与归纳,覆盖研究主线、统一方法框架、各子项目、以及主力工作 **2D Gray–Scott 多解生成器** 的全部细节:物理与数据生成、模型架构、token 序列(squeue)构造、训练流程与多阶段策略、所有关键 trick、设计演进与失败诊断、评估方法、L=1/L=2 结果,以及后期的「救援 / 内插 / 外推」研究。
>
> 约定:**Setup A = L=1**,**Setup B = L=2**,两者指标必须**分开汇报**(难度不同,指标方向可能相反,绝不取全局平均)。

---

## 0. 一句话总览

整个项目的统一命题:**把一个微分方程的参数 `p` 映射到该参数下「所有共存定态解」的集合**,用一个自回归语言模型(Qwen2.5-7B + LoRA)+ 冻结的卷积/MLP 自编码器(AE)来实现。模型一次性、确定性地把参数「翻译」成一串解,再用传统数值求解器(Newton / 拟 Newton)做精修,从而以远低于传统「多初值穷举 + 逐个 Newton」的代价,发现 PDE 的多解结构。

研究主线是由易到难的三级阶梯,最终落到 Gray–Scott:

```
1D 单参数 BVP  →  1D 双参数 BVP  →  2D 标量椭圆 PDE  →  2D Gray–Scott 反应扩散系统(主力)
 (1D_p)          (1D_a2a4)         (2D_multisolution)      (2d_GrayScott: qwen / qwen_ord / *_L2)
```

> **⚠ 关键区别(贯穿全项目的设计分水岭):前三个问题「解的数量是已知/确定的」,而 Gray–Scott「解的数量是未知/开放的」—— 这是后面一切设计调整的根本原因。**
>
> - **1D_p(≤7 支)、1D_a2a4(≤6 支)、2D_multisolution(10 支→D4 约简到 4)**:给定参数下,解支数量由**参数延拓穷举得到、有限且确定**,数据集可视为「**完备**」。因此模型可以**显式学习并预测解的个数**:用 STOP 头 teacher-force 到一个完整、固定的集合即可,生成到 STOP 就停。
> - **Gray–Scott**:给定 (ρ,μ) 下究竟有多少共存定态,**既无理论上界、GT 枚举本身也不完备**(模型确实能找到 GT 之外的新解,L=1 约 70% 参数有 GT 外新解)。于是「学一个准确的计数」**既不现实**(STOP 会过早触发→计数塌缩、欠生成)**也不是目标**(我们恰恰想超越 GT)。
> - **所以 Gray–Scott 做了根本调整**:① **放弃 STOP / 放弃预测个数**,改为**固定 K=24 槽 + 无 STOP + 过生成后 D4 去重**(K 远大于已知最大解数;过生成安全、欠生成才致命);② 把真解数 K_t 之后的「**tail 槽**」交给**物理残差 loss 去自由 free-run 发现**超越 GT 的解。详见 §3.3(序列构造)、§3.6(失败诊断)、§4(物理 loss 的演进)。

---

## 1. 统一方法框架(贯穿所有子项目)

所有子项目共享同一套「LLM 多解生成器」范式,组件几乎一致,只在 AE、参数维度、最大解数上有别:

1. **冻结自编码器(AE)**:把一条解(1D 向量 / 2D 场)压成 256 维 latent `z`,并能解码回物理空间。训练时 AE 先单独预训练,之后全程冻结。
2. **特殊 token 嵌入**:5 个可学习 token —— `P_START`(参数块起点)、`SOL`(解槽)、`STOP`(解集结束)、`VECTOR`(路由信号:此处要回归一个连续 latent)、`PAD`(零向量,不学习)。
3. **输入投影器 InputProjector**:把 `[z(256) ; p_norm(参数维)]` 经 2 层 GELU MLP 投到 Qwen 隐藏维(3584),作为「软提示」喂进 transformer。
4. **Qwen2.5-7B-Instruct + LoRA**:作为自回归骨干。LoRA `r=16, alpha=32, dropout=0.05`,只加在注意力的 `q_proj,k_proj,v_proj,o_proj`。
5. **双头 DualHead**:从 Qwen 最后隐状态出两路 —— 分类头(预测下一个 token 类型,含 STOP)+ 回归头(预测下一个解的 256 维 latent)。
6. **In-Context Learning(ICL)上下文**:训练序列前面可放若干「上下文参数块」(邻近参数的已知解),让模型 few-shot 学映射;最终模型再用「无上下文」微调以支持 zero-shot 推理。
7. **规范化排序(canonical order)**:同一参数的多个解按某个不变量排序(1D 用 L1 积分;2D Gray–Scott 用 mean(A)),得到**确定、唯一**的 teacher-forcing 目标序列 —— 这是让「集合预测」变得可学的关键(见 §3.6)。
8. **后置数值精修**:模型给的是「足够好的初值」,再用 Newton / 拟 Newton 收敛到机器精度;最后做对称去重得到不同解的计数。

**核心创新点**:用「**有序的、位置对齐的**」自回归目标,把「无序集合预测」这一本质难题转化为可 teacher-force 的序列问题;并配合 scheduled sampling 抗 exposure bias、物理残差 loss 做无 GT 区的发现。

---

## 2. 兄弟子项目速览(研究铺垫)

### 2.1 `1D_p_multisolution` —— 单参数非线性 BVP
- **方程**:`-u''(x) + u²(x)(u²(x) − p) = 0`,`x∈[0,1]`,`u'(0)=0, u(1)=0`,参数 `p∈[0,18]`。
- **多解**:高 p 时最多 **7 条解支**共存,解支数随 p 变化。
- **数据生成**:从 `p=18` 用多重网格 Newton(N: 2→4→…→1024)找全 7 支,再**反向参数延拓** `p:18→0`(步长 −1),每步 Newton 精修(tol 1e-8),L2<1e-4 去重。共 **18001 个 p、54629 个解**,均值 k≈3.03,max 7。N=1024 离散。
- **模型**:`V2PDEModel` = UNet1d(1024→256 latent,冻结)+ 5 特殊 token + InputProjector + Qwen+LoRA + OutputProjector + DualHead。
- **三阶段训练**:Phase1 UNet 预训练(200 ep,val MSE 1.31e-4)→ Phase2 全模型 + 动态上下文(MSE+CE+PDE,100 ep)→ Phase3 无上下文微调(60 ep)。loss 权重 λ_MSE=1.0, λ_CE=1.0, λ_PDE=0.05(warmup 5–15 ep),CE-STOP 权重 3.0。8 卡 DDP。

### 2.2 `1D_a2a4_multisolution` —— 双参数 BVP(三次–五次非线性)
- **方程**:`-u''(x) + a₄·u⁴ + a₂·u² = 0`,参数 `(a₄,a₂)∈ℝ²`,最多 **6 支**,解支在 (a₄,a₂) 平面形成「fold 曲线」。
- **数据生成**:GPU 全空间生成(`generate_fullspace_gpu.py`),结构化种子 + 随机种子,Newton(tol 1e-9)精修去重。
- **模型**:与 1D_p 完全相同,只是 `param_dim=2, max_k=6`,并用**加权采样器**处理 k 的类别不平衡。
- **意义**:验证方法可推广到**高维参数空间**。

### 2.3 `2D_multisolution` —— 2D 标量椭圆 PDE(D₄ 对称)
- **方程(文献 Eq.61)**:`−Δu − u² = −s·sin(πx)sin(πy)`,`[0,1]²`,Dirichlet `u=0`,参数 `s∈[−138,1600]`。
- **多解**:最多 10 支(2 对称 + 4+4 非对称轨道),用 **D₄(二面体)对称**约简到 4 个代表。
- **数据生成**:FEM 网格(ell=3,145 节点),从 s=1600 用伪弧长延拓找全解 + D₄ 增广,Newton(maxiter 50)精修,正反向延拓追踪各支。
- **模型**:`V3PDEModel`,把 UNet 换成 **MLP 型 2D 自编码器**(145→512→256);**关闭 PDE loss**(λ_PDE=0,多支 MSE 与物理残差冲突);训练 1000 ep。
- **意义**:在 2D 标量 PDE 上验证整套机器,是 Gray–Scott(耦合系统)的前置 PoC。

### 2.4 顶层 `slides/`(速度结果)
34 页 Beamer,讲 1D_p / 1D_a2a4 / 2D 三部分。核心卖点是**相对传统方法的加速比**(Qwen + 插值 + Newton vs 传统多重网格/种子穷举):

| 问题 | 规模 | Qwen+Newton | 传统 | 加速 |
|---|---|---|---|---|
| 1D-p (N=1024) | 8 解 | 2.95 s | 32.5 s | 11× |
| 1D-p (N=2048) | 8 解 | 9.82 s | 203.8 s | 21× |
| 1D-p (N=4096) | 8 解 | 21.2 s | 1216 s | 58× |
| 2D (ell=3) | 4 解 | 0.78 s | 13.1 s | 17× |
| 2D (ell=4) | ~4 解 | 0.75 s | 49.9 s | 67× |
| 2D (ell=5,2113 节点) | ~4 解 | 1.34 s | 136.9 s | 115× |

加速来源:Qwen 前向近似常数时间,Newton 精修 O(n);传统种子穷举随网格变细爆炸增长。网格越细,优势越大。

---

## 3. 主力项目:2D Gray–Scott 多解生成器

目录结构:
- `fdm/` —— 数据生成与数值求解器(拟 Newton FDM、算子矩阵、L2_gen/)
- `Quasi-Newton-Method/` —— 拟 Newton 方法的参考实现与理论
- `qwen/` —— **第一代**:带 STOP 头的多解生成器(Setup A 父模型)
- `qwen_L2/` —— 带 STOP 头的 L=2 父模型
- `qwen_ord/` —— **最终交付(L=1)**:有序、无 STOP、固定 K=24
- `qwen_ord_L2/` —— **最终交付(L=2)**
- `slides_ord_combined/` —— L=1 + L=2 合并讲义(32 页)

### 3.1 物理问题与数据生成(`fdm/`)

#### 3.1.1 Gray–Scott 定态方程
```
DA·Lap(A) − S·A² + (mu+rho)·A = 0      (激活物 A)
DS·Lap(S) + S·A² − rho·(1−S)   = 0      (底物 S)
```
- 参数:`rho`(ρ,进料,网格里也记作 F)、`mu`(μ,消耗,记作 k)。
- 扩散系数:L=1 用算子 `op_n128.mat` 内置的 DA、DS;**L=2 = 在 L=1 算子上将 DA、DS 各除以 4**(代码里 `--L2` 即 `op.DA/=4; op.DS/=4`),物理上等价于域放大、图案更细密(条纹/迷宫/点阵)。
- 域:单位正方形,**Neumann 边界**;网格 **n=128**。

#### 3.1.2 算子矩阵 `op_n128.mat`(张量积 Neumann 拉普拉斯)
存了 `Tx, TxInv`(1D Neumann 拉普拉斯的特征向量与逆)、`eigx`(1D 特征值)、`DA, DS`。2D 特征值 `Lam[i,j]=eigx[i]+eigx[j]`(128×128)。在特征基里:
- 变换到特征基:`X_eig = TxInv @ X @ TxInv^T`
- 反变换:`X = Tx @ X_eig @ Tx^T`
- 拉普拉斯 = 在特征基里逐元素乘 `Lam` 再反变换;
- **平移求逆** `inv_shift(R, scale) = from_eig( to_eig(R) / scale )`(特征基里逐元素除)。

这套算子在**数据生成求解器**和**模型物理残差 loss**里完全一致,保证「真解的残差≈0」。

#### 3.1.3 拟 Newton + FDM 求解器(`fdm/gs_torch.py: solve_batch`)
不在每步求全 Jacobian 的逆,而是用「**特征分解 + 平移**」近似:
```
[A_h + βI]^{-1} ≈ [A_h + N_u]^{-1}    →   特征基里逐元素 1/(λ+β)
```
每步迭代(批量、float64、GPU):
1. 残差 `rA = DA·lap(A) − S·A² + (mu+rho)·A`,`rS = DS·lap(S) + S·A² − rho·(1−S)`;`crit = max(|rA|∞, |rS|∞)`。
2. 收敛 `crit<tol(1e-9)` 标记完成;`crit>maxtol(1000)` 或 NaN 标记发散;`early_iter` 处把 `crit>early_tol` 的种子剔除(加速)。
3. **β 平移**:在每个像素算 2×2 反应 Jacobian 的特征值范围,取 `β=(max特征值+min特征值)/2`(逐 candidate 一个标量),保证平移后可逆。
4. **更新**:`dA = inv_shift(rA, DA·Lam+β)`,`dS = inv_shift(rS, DS·Lam+β)`,`A ← A − stepsize·dA`(stepsize=0.1),仅对活跃 candidate。

(MATLAB 参考实现 `gs_fdm_solve.m` 与 PyTorch 版本校验一致,误差 <1e-7。)

#### 3.1.4 多解的发现:丰富种子库 + 批量求解 + D4 去重
GT 多解的关键是「用大量多样初值同时收敛」。种子库(`seed_bank`,以 L=2 为例 356–436 个):
- 8 个 MATLAB 连续化得到的基础点状解;
- **合成条纹** 96 个(频率 2–13 × 2 相位 × 4 朝向 `sin²(fπX)` 等);
- **棋盘/洞点阵** 18 个(9 波长 × 2 极性);
- **迷宫型乘积** 14 个;
- **带限随机场**(L=2 dense 用 nrand=300):随机场经 FFT 频率截断,只保留低频。

每个 (ρ,μ) 把整个种子池批量喂进 `solve_batch`(batch 3000–4000),保留收敛解 → 滤掉平凡解(`range(A)<0.05`)→ **D4 对称去重**:单位正方形 Neumann 有 8 元二面体对称(4 旋转 + 4 反射),两解若在 8 个 D4 位姿下最小**相对 L2 距离 ≤ 0.15** 即视为同一解;按振幅降序贪心保留,每参数上限 kmax(L=1:15,L=2:20)。

#### 3.1.5 参数网格 / 解带 / L1 vs L2
- **L=1**:ρ∈[0.030,0.110](81点)、μ∈[0.045,0.075](31点)= 2511 格,中心点 (0.05,0.065) 种子。解只存在于一条**对角「解带」**(crescent),带外只有平凡解。用 BFS 参数延拓(`gs_continuation.py`)从点状解铺开,再用 `gs_expand_gen.py` 补低 μ 的条纹/迷宫区。
- **L=2**:DA、DS 各 /4。基础网格 ρ∈[0.038,0.104]、μ∈[0.052,0.072](34×41),后加密到 67×81(`gen_L2_dense.py`,seed=7,8 GPU 分片),只算解带邻域。

#### 3.1.6 数据打包(`build_*.py`)
载入所有 `cell_*.npz` → 滤平凡 → 按 (ρ,μ) 分组 → **D4 约简 + 规范位姿**(旋转/反射到最大化与一个 NE-ramp 模板的内积,再按 mean(A)+1e-6·energy 排序)→ 生成查找表:
- `gs_lookup_*.pt`:`{p_values:[P,2], solutions_by_p: list[P] of [K_p,2,128,128]}`
- `{train,val,test}_p_idx_*.pt`:**按参数 70/15/15 划分**,固定 `SEED=42`。
- `norm_stats_*.pt`:仅用训练集算的解通道 mean/std 与参数 mean/std。

**关键数字**:L=1 共 1083 参数;L=2 原始 471 参数(72 测试),扩展后 **1930 参数**(292 测试)。

### 3.2 模型架构(`qwen_ord/model/`)

#### 3.2.1 冻结卷积自编码器 `autoencoder2d.py`
- 输入 `[B,2,128,128]`(A,S 两通道)。编码器 5 个 stride-2 下采样块:128→64→32→16→8→4,通道 [32,64,128,256,256];瓶颈 256×4×4=4096 → `Linear(4096,256)` + `LayerNorm` → **256 维 latent**。
- 解码器对称上采样,末 `Conv2d(32,2)` 出 `[B,2,128,128]`。
- 每块:`Conv-GroupNorm-GELU ×2`。
- 归一化 buffer(训练集统计):编码 `(x−mean)/std`,解码 `x·std+mean` 回物理量。
- **全程冻结**(`eval()` + `requires_grad=False`)。

#### 3.2.2 特殊 token / 输入投影 / 双头
- `special_tokens.py`:`Embedding(4, 3584)`(P_START=0,SOL=1,STOP=2,VECTOR=3 可学;PAD=4 为零向量)。
- `input_projector.py`:`in=258(=256+2) → 1921 → 3584`,GELU。参数先 `(p−p_mean)/p_std` 归一化。
- `dual_head.py`:分类头 `Linear(3584,5)` + 回归头 `Linear(3584,256)`;`route()` = argmax 分类 + 取回归 latent。**`qwen_ord` 把分类/STOP 头的 loss 关掉**(`lambda_ce=0`),只用回归头。
- Qwen2.5-7B-Instruct,LoRA(r16/α32/dropout0.05,q/k/v/o),bf16。

#### 3.2.3 物理残差 loss `pde_residual.py`(关键 trick)
计算 `res2 = rA² + rS²`(用与数据生成同一张量积 Neumann 拉普拉斯)。**地板 hinge**:
```
hinged = relu(res2 − floor).mean(),  floor=1e-3
```
原因(文档明确写出):原始残差的**全局最小是平凡态**(A≈0,S≈1,残差~1e-12),比真解经过有损 AE 后的残差(~1.7e-4–4.8e-4)还低约 8 个量级;直接惩罚原始残差会把模型拉向「漂白的平凡解」,且拉普拉斯会放大 AE 的高频误差。**地板设在 1e-3(≈AE 往返残差上限的 2 倍)**,使真解和平凡解都低于地板→0 惩罚,只惩罚明显模糊/非物理的预测(把物理项变成「安全的抛光」,而非硬约束)。

### 3.3 token 序列构造(squeue,`data/sequence_builder.py`)

#### 3.3.1 块结构
每个参数(上下文或目标)是一个块:
```
[P_START] [p_proj] [SOL] [sol_proj_1] [SOL] [sol_proj_2] ... [SOL] [sol_proj_K] [STOP?]
```
- `[p_proj] = InputProjector(z=0, p)`(参数锚点);`[sol_proj_k] = InputProjector(z_k, p)`(第 k 个解的嵌入)。
- **自回归对齐**:`hidden[i]` 预测位置 `i+1`;回归目标 `z_k` 放在该解前的 `[SOL]` 位置。
- **上下文块**(`is_target=False`):放邻近参数的 K_t 个真解 + `[STOP]`,**不计 loss**。
- **目标块**(`is_target=True`):**固定 fixed_k=24 个解槽,无 STOP**(`qwen_ord` 的核心)。

#### 3.3.2 固定 K=24 的 head / tail 切分
设该参数真解数 `K_t = min(真实K, 24)`:
- **head(k < K_t)**:放真解嵌入。`loss_mask=True`(算有序 MSE)、`cls=VECTOR_ID`(供 scheduled sampling 标记)、`reg_target=z_k`(AE 编码的真 latent)。
- **tail(k ≥ K_t)**:放零 latent 占位符。`loss_mask=False`、`tail_mask=True`(只算物理残差)、`reg_target=None`;前向 pass-2 会用**模型自己的预测**覆盖这个占位输入(自由 free-run)。
- 因为 `fixed_k=24 ≥ 全局 max K_t`,所有真解都能放下,**不需要 STOP** 来表示「结束」。

序列拼接:多个上下文块 + 1 个目标块,超过 `max_seq_len=768` 从左截断。collate 把各样本 pad 到 batch,并附 `target_sols_padded、target_k_counts、target_p_vals、target_block_offsets` 供前向重建目标块嵌入。因果 mask:`j≤i 且 j<lens`。

#### 3.3.3 规范化排序 `canonicalize.py`
同一参数的解**按 mean(A) 升序**排序,得到确定顺序 → head/tail 切分稳定、teacher-forcing 目标唯一。(注:当前数据未做 D4 约简到序列层,mean 相同会平局,是已知小瑕疵。)

#### 3.3.4 生产推理 `generate_fixed_k(p, K, noise_std=0)`
忽略 STOP,强制生成 K 个解的自回归循环(带 KV cache):
1. 序列起点 `[P_START] + [p_proj(z=0)]`,接 `[SOL]`;
2. 每步:Qwen 前向取最后位 → 回归头出 `reg`(256);
3. **可选噪声** `reg ← reg + noise_std·ε`(ε~N(0,I));
4. AE 解码成场 `sol`(存下);
5. **解码→再编码→回喂**:`z = AE.encode(sol)`,`nxt = InputProjector(z, p)` 作为下一步输入(让模型始终「看到」物理真实,防 latent 漂移累积);
6. 循环 K 次,返回 `[K,2,128,128]`。

### 3.4 训练流程与多阶段策略

#### 3.4.1 多阶段:STOP 父模型 → 有序无 STOP 微调
- **Phase 2(带上下文)** [`qwen/`]:800 ep,`no_context_prob=0`,`context_mode=local`(取 (ρ,μ) 最近的 10 个参数做 few-shot 上下文,比随机强约 4×),含 CE+MSE+latent+PDE。学映射。
- **Phase 3(无上下文)** [`qwen/`]:120 ep,`no_context_prob=1.0`,开 scheduled sampling(ss_prob 0.30,warmup 3→30),lr 降 3×。逼模型只靠参数本身、并适应推理分布。
- **有序无 STOP 微调** [`qwen_ord/`]:**从 STOP Phase3 的 best_model 出发**,只微调 **60 ep** 即可。

#### 3.4.2 `train.py` 训练循环
- **两组优化器**:`opt_proj`(special_tok+input_proj+dual_head)与 `opt_lora`(LoRA 参数),各自 AdamW + `ReduceLROnPlateau(factor=0.5, patience=25, min_lr=0.1×init)`,监控 **val MSE**。
- **梯度累积** `grad_accum`,DDP 下手动 `all_reduce` 梯度求平均,`clip_grad_norm=1.0`。
- **EpochShuffleSampler**:`target_repeat_per_epoch=10`(每个样本一轮重复 10 次,提高梯度密度)。
- 存 `best_model`(val 最优)与 `epoch_N`(每 10 ep)。`--smoke N` 跑 N 步即退,用于调试。
- **前向两遍**:pass-1 全序列前向;pass-2 把 tail(总是)+ head(以 ss_prob 概率)的下一步输入换成模型 pass-1 的(detached)预测,再前向一次 → 抗 exposure bias。

#### 3.4.3 生产 slurm(L=1,`scripts/run_train_L1.slurm`)
```bash
torchrun --nproc_per_node=4 --master_port=29531 train.py \
  --batch_size 8 --grad_accum 4 --num_epochs 60 \
  --lr_projector 3e-5 --lr_lora 1e-5 \
  --no_context_prob 1.0 --val_no_context_prob 1.0 --context_mode local \
  --resume_from .../qwen/phase3_big_local/best_model \
  --output_dir ./checkpoints_final
```
4 卡 DDP、bf16、conda 环境 `torch124`、`NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1`。

#### 3.4.4 关键超参数(config.py 默认值)
| 类别 | 参数 | 值 |
|---|---|---|
| 维度 | latent_dim / img / in_ch | 256 / 128 / 2 |
| 序列 | fixed_k / max_solutions_per_p / max_seq_len / max_context_p | 24 / 24 / 768 / 10 |
| LoRA | r / alpha / dropout / 模块 | 16 / 32 / 0.05 / q,k,v,o |
| loss | λ_mse / λ_latent / λ_ce | 0.5 / 0.5 / 0(无 STOP) |
| loss | λ_pde(head) / λ_pde_tail | 0.02 / 0.01 |
| 物理 | pde_floor / pde_warmup | 1e-3 / [0,10] 线性 |
| 采样 | ss_prob / warmup | 0.25 / [0,15] |
| 训练 | batch / grad_accum / epochs | 8 / 4 / 60 |
| lr | lr_projector / lr_lora(生产值) | 3e-5 / 1e-5 |
| 非平凡(外推用,默认关) | λ_nontrivial / range_floor / pde_tail_floor | 0 / 0.30 / −1 |

### 3.5 关键 tricks 汇总
1. **有序、位置对齐的目标(uncopyable)**:把无序集合预测变成可 teacher-force 的序列,杜绝「抄上一个解」的退化(见 §3.6)。
2. **固定 K=24、无 STOP**:K 远大于最大真解数,过生成靠去重处理(安全),欠生成才致命 → 不再依赖会过早触发的 STOP。
3. **解码→再编码→回喂**:自回归每步把生成解投回物理空间再编码,防 latent 漂移累积。
4. **物理残差地板 hinge(floor=1e-3)**:让真解与平凡解都 0 惩罚,只压模糊预测,避免被平凡态吸引。
5. **Scheduled sampling**:训练时按概率用模型自身预测做历史,抗 exposure bias;微调从 ep0 即开、ramp 更快。
6. **物理 warmup**:λ_pde 从 0 线性升到目标,先成形再抛光。
7. **tail 物理-only**:K_t 之后的槽无 GT,只用物理残差做「超越 GT 的发现」。
8. **非平凡 hinge(外推用)**:`relu(0.30 − range(A))`,防无 GT 区塌成平坦平凡解。
9. **噪声多次前向取并集**:确定性单次已足够多样;若要更多,跑 N 次独立带噪前向取并集(见 §5)。

### 3.6 设计演进与失败诊断(`docs/`)
- **集合损失(Chamfer)为何失败**(`SET_FAILURE_DIAGNOSIS.md`):排列不变的 Chamfer + teacher forcing 下,最省力的下降是 `pred_k := gt_{k-1}`(把注意力范围内的上一个真解抄过来,前后向都匹配上某个真解)→ 模型学成「抄上一个 token」而非学映射,free-run 时漂移。**修复 = 有序、位置对齐 MSE**(`pred_k` 必须等于排好序的第 k 个解,抄上一个变贵)+ scheduled sampling。
- **STOP 头为何欠生成**:无上下文推理时 STOP 被模型当「不确定就喊停」的逃生按钮,过早触发 → 计数塌缩。**修复 = 去掉 STOP、固定 K=24**。

### 3.7 评估方法(`eval_ord.py`)
对每个测试参数:`generate_fixed_k(24)` → 测原始残差 → **FDM 拟 Newton 精修** `solve_track` → **D4 去重(rel-L2>0.15)** → 数「不同的、收敛的」解 + GT 覆盖率。
- 指标:`det_distinct`(平均每参数不同解数,越高越好)、`coverage`(覆盖 GT 的比例)、`Ktrue`、`init_residual`。
- **后处理 L1 vs L2**:L=1 用 `maxiter 8000, early_iter 3000`;**L=2 收敛更慢,用 `maxiter 30000, early_iter 0`(不早停)**。

### 3.8 L=1 结果(Setup A,163 测试参数)
| 模型 | det_distinct | 说明 |
|---|---|---|
| STOP(确定性) | 2.53 | greedy STOP,欠生成 |
| **ord-K(确定性,单次)** | **5.18** | 有序+SS+物理,无需噪声 |
| ord-K + 噪声(多次并集) | 9.16 | 论文的噪声旋钮:每多一次=一次前向,增量多为 GT 外新解 |

ord-K **超过 STOP-det(2.53)**;覆盖率 0.336,Ktrue 9.40。**约 70% 参数能生成 ≥1 个超越 GT 的新解**。交付 ckpt:`qwen_ord/checkpoints_final/epoch_60`(比 best_model 更强)。

### 3.9 L=2 collapse 诊断与修复(Setup B)
- **现象**:朴素 L=2 ord 模型 `det_distinct=1.00`(严格后处理),0 解参数 50%。
- **诊断**:**不是 AE**(L=2 GT 重构误差仅 1.1%),**不是缺 GT 多样性**(Kt≈7,74% 参数多解),而是**回归器没学会 L=2 映射** —— 吐出「在流形外」的场,FDM 无法精修。根因:数据约少 3×(原始 471 参数)+ 从已塌缩的 STOP 基座微调。
- **修复三步**:① 放宽后处理(1.00→2.14,确认瓶颈在初值);② **扩数据 471→1930**;③ 从**混合初值**(L=1-ord 的 LoRA/special_tokens 先验 × L=2 的 AE/heads)重训。
- **结果(72 原始测试参数)**:`det_distinct=2.82`、coverage 0.066(原 0.045)、0 解参数 50%→29%。消融:3a(混合初值)2.82 vs 3b(仅扩数据)2.79 → **扩数据是主因**,混合初值增益小。交付:`qwen_ord_L2/checkpoints_final/epoch_60`。
- **L=1 与 L=2 必须分开汇报**:5.18 vs 2.82 不可平均;L=2 内在更难(细密图案、扩散 /4、收敛慢)。

---

## 4. 物理 loss 的处理:从「已知解数」到「未知解数」的演进(核心创新)

物理(PDE 残差)loss 在**所有**子项目里都出现,但**用法随问题难度发生了本质演进** —— 这是本项目最核心的创新之一,且与上面「解数已知 vs 未知」的分水岭紧密相扣。

### 4.1 跨项目对比
| 项目 | 解数 | 物理 loss 用法 | 权重 |
|---|---|---|---|
| 1D_p | 已知 ≤7 | 普通 PDE 残差,小权重,warmup ep5–15 | λ_PDE = 0.05 |
| 1D_a2a4 | 已知 ≤6 | 同上 | λ_PDE = 0.05 |
| 2D_multisolution | 已知 ≤10→4 | **直接关闭**(与多支 MSE 冲突) | λ_PDE = 0.0 |
| **Gray–Scott** | **未知 / 开放** | **地板 hinge 残差 + tail 物理-only 发现 + 非平凡护栏** | head 0.02 / tail 0.01 |

### 4.2 三阶段演进逻辑
1. **1D(简单残差,锦上添花)**:解之间分得开、数据完备,物理残差只是小权重的「抛光」,把模型给的近似初值再往真解推一点;warmup 让先成形再上物理。
2. **2D(被迫关闭)**:**朴素物理残差与「有序多支 MSE」冲突** —— 残差对每个预测都想推向「某个真解」,而位置对齐 MSE 要它精确等于「特定的第 k 支」,两者打架,于是 `λ_PDE` 直接设 0。这暴露了朴素物理 loss 在多解问题上会失效。
3. **Gray–Scott(重新启用并升级为「发现引擎」)** —— 核心创新:因为 GS 解数未知、GT 不完备,物理 loss 被从「抛光」**重新定位为「在无标注区生成合法新解」的主力**,但必须先解决两个致命问题:
   - **平凡解吸引子**:原始残差的全局最小是平凡态(残差比真解经 AE 后还低约 8 个量级),直接惩罚会把模型「漂白」成平坦解。→ **地板 hinge `relu(res² − floor)`,floor = 1e-3**:真解与平凡解都低于地板→0 惩罚,只压明显非物理的模糊预测。把物理项从「硬约束」变成「**只在该起作用时才起作用**的安全抛光」。
   - **无 GT 的槽怎么学**:固定 K=24 里,真解数 K_t 之后的 tail 槽没有真解可监督。→ **tail 物理-only loss(λ_pde_tail = 0.01)**:这些槽用模型自身预测自由 free-run,只受物理残差约束,去 best-effort **发现** GT 之外的解。
   - (外推研究里进一步加 **非平凡 hinge `relu(0.30 − range(A))`** 防 tail 塌平凡;并实测发现**冻结 AE 流形几乎不含平坦场,天然抗平凡塌缩**。)

### 4.3 为什么这是核心创新
- 把「**无序、数量未知、GT 不完备**」的多解问题,用一组件协同解决:**有序固定 K(让序列可 teacher-force)+ 物理残差(无 GT 也能约束合法性)+ 地板 hinge(避免平凡塌缩)+ tail 物理-only(发现新解)**。
- 物理 loss 由「可有可无的正则」升级为「**无标注区的解发现引擎**」,且地板设计保证它不会反噬(不被平凡态拖走)。
- 直接成果:L=1 确定性单次超越 STOP-det(2.53),达 **5.18**,并能发现 ~70% 参数的 GT 外新解;后续外推研究也正是建立在「tail 物理-only + 非平凡 + 地板」这套机制上(见 §5)。

---

## 5. 后期研究:救援 / 内插 / 外推 + 训练式外推尝试

在 L=1、L=2 两套交付模型之上,系统研究了「能不能让更多参数(含无 GT 的参数)出解」。三类(均**纯推理,不训练**,除特别说明):

- **第一类 救援(rescue)**:数据集里**本来有 GT、但模型原本 0 解**的参数,用噪声多次前向 + 放宽残差门限(1e-9→1e-6)救回。
- **第二类 内插(interpolation)**:训练参数包围盒**内部**、训练网格点**之间**新采样的 (ρ,μ)(无 GT)。
- **第三类 外推(extrapolation)**:包围盒**外**(每边外扩 25%)新采样的 (ρ,μ)(无 GT)。

### 4.1 结果(L=1 ↔ L=2,完全一致)
| 实验 | L=1 | L=2 |
|---|---|---|
| 救援(0 解参数救回) | 92/123 = **75%** | 321/435 = **74%** |
| 内插(域内新点) | 36/81 = **44%** | 39/81 = **48%** |
| 外推(域外) | 5/144 = **3.5%** | 9/144 = **6.2%** |
| 外推位置 | 解带右下延长(高 ρ、低 μ) | **同样**右下延长 |
| 外推解数 | 5 组合 / 49 解 | 9 组合 / 43 解 |

**外推只在解带的右下(高 ρ、低 μ)延长方向有限可行**;左上 / 上 / 整个带外角落均 0 解 —— 那里物理上本就没有非平凡稳态。

### 4.2 训练式外推 finetune(L=1,两版,均负结果)
为推广「无 GT 槽用物理 loss」到外推参数,新增了 `lambda_nontrivial / nontrivial_range_floor / pde_tail_floor` 开关、forward 的非平凡项、`build_extrap_data{,2}.py`、CLI、slurm:
- **v1**(整边缘采样,floor=1e-3):外推 5/144,**无提升**;训练 `tail_pde` 全程不降。
- **v2**(只在解带延长方向采样 + tail floor=0 纯残差下降 + 强非平凡护栏):外推仍 5/144,`tail_pde` 仅降 ~5%,**无提升**。

结论:小学习率 LoRA 的物理梯度搬不动 latent 去到真解 basin;带外本就无解可造。**L=2 的等价 finetune 因负结果可预见而未做**。

### 4.3 测试时 latent 优化(不训练,从基线 ckpt 出发)
对每个解带延长参数,拿模型 24 个猜测作初值,在 256 维 latent 上做 Adam 梯度下降最小化物理残差(纯残差 vs +非平凡护栏),解码→FDM→数解:
- **外推扩展不了**:出 box 即断崖归零,latent 优化 ≈ model-only(L=1);L=2 在已有解的点能多挖些解(总 distinct 升),但**不增加新的可外推参数**。
- **不会塌成平凡解**:所有模式 A-range 稳在 ~0.45–0.53,从不接近平凡阈值 0.30 —— 因为**冻结 AE 流形几乎不含平坦场**,天然抗平凡塌缩。

### 4.4 外推研究的总结论
1. **救援(~75%)和内插(~45%)都有效**;
2. **外推严格受限**于解带右下延长方向那一窄条;
3. **物理 finetune 与测试时 latent 优化都无法突破带边界**;
4. **物理优化不会塌成平凡解**(AE 瓶颈的天然保护)。
合并讲义 `slides_ord_combined/`:第 17 页(L=1 atlas+三类表)、第 28 页(L=2 atlas+三类表)、第 29 页(L=1 vs L=2 总览),共 32 页。

---

## 6. 关键文件清单

| 用途 | 路径 |
|---|---|
| GS 求解器(拟 Newton FDM) | `2d_GrayScott/fdm/gs_torch.py` |
| 算子矩阵 | `2d_GrayScott/fdm/op_n128.mat` |
| L=1 数据生成 | `fdm/gs_continuation.py`、`fdm/gs_expand_gen.py`、`fdm/build_big.py` |
| L=2 数据生成 | `fdm/L2_gen/gen_L2.py`、`gen_L2_dense.py`、`build_L2_dense.py` |
| D4 去重 | `fdm/filter_d4.py` |
| **L=1 模型代码** | `qwen_ord/model/{v3_model,autoencoder2d,dual_head,input_projector,special_tokens,pde_residual,canonicalize}.py` |
| 序列构造 / 数据 | `qwen_ord/data/{sequence_builder,dataset}.py` |
| 训练 / 评估 / 配置 | `qwen_ord/{train.py,eval_ord.py,config.py}` |
| **L=1 交付 ckpt** | `qwen_ord/checkpoints_final/epoch_60`(+ `epoch_60_BASELINE` 备份) |
| **L=2 交付 ckpt** | `qwen_ord_L2/checkpoints_final/epoch_60` |
| 外推实验(L=1) | `qwen_ord/experiments/{extrapolate,gen_extrap_fields,latent_opt_extrap,plot_corners}.py` |
| 外推实验(L=2) | `qwen_ord_L2/experiments/{extrapolate,gen_extrap_fields,latent_opt_extrap}.py` |
| STOP 父模型 | `qwen/`(L=1)、`qwen_L2/`(L=2) |
| 设计文档 | `qwen_ord/docs/{REPORT.md, SET_FAILURE_DIAGNOSIS.md}` |
| 合并讲义 | `slides_ord_combined/slides.pdf`(32 页) |
| 兄弟项目 | `1D_p_multisolution/`、`1D_a2a4_multisolution/`、`2D_multisolution/`、顶层 `slides/` |

## 7. 复现关键命令
```bash
# 环境
conda activate torch124   # torch 2.4 + CUDA 12.4 + peft/transformers

# L=1 训练(从 STOP phase3 best 微调)
cd 2d_GrayScott/qwen_ord && sbatch scripts/run_train_L1.slurm

# L=1 评估
python eval_ord.py --ckpt ./checkpoints_final/epoch_60 --out ./eval/ord_eval.json

# L=2 评估(放宽后处理)
cd ../qwen_ord_L2
python eval_ord.py --L2 --ckpt ./checkpoints_final/epoch_60 \
    --maxiter 30000 --early_iter 0 --out ./eval/ord_eval.json

# 外推研究(纯推理)
python experiments/extrapolate.py [--L2] --mode rescue   # 救援
python experiments/extrapolate.py [--L2] --mode grid     # 内插/外推
```

---

*本报告由对 `BBBB_qwen_pde_branch` 全项目的代码审阅综合而成,涵盖数据生成、模型、序列构造、训练、评估、各子项目与外推研究的全部关键细节。具体数值与实现以对应源码为准。*
