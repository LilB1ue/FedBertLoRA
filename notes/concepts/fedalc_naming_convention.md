# FedALC 方法命名規則

> 最後更新：2026-04-16

## 命名規則

`FedALC-{clustering_algorithm}[-{variant}]`

- `{clustering_algorithm}`：使用的 clustering 演算法 (AP / Spectral / Agglo 等)
- `{variant}` (可選)：在該演算法 baseline 上的額外 component
  - `LWC` = Metric B layer selection + freeze（無 warm-up；ablation baseline）
  - `Multi` = 內建 layer selection + Hopkins adaptive trigger + cumulative ΔB + freeze (主方法，targets multi-task)

所有 FedALC-* 變體**共享相同的核心設計**（A global / B per-cluster / Others local），differentiation 在 clustering 演算法 + 額外 component。

## 核心設計（所有 FedALC-* 共享）

```
基礎架構 (基本 FedALC-{algorithm}, e.g. FedALC-AP):
  - A: global FedAvg（shared subspace basis）
  - B: per-cluster FedAvg（用對應 clustering 演算法）
  - Others: per-client local
```

Multi variant（例：FedALC-AP-Multi）在上述基礎再加：

```
Phase 0 (adaptive warm-up):
  - FedSA mode (A global, B local)
  - 累積 ΔB = B_current - B_init   (標準 LoRA B_init = 0)
  - [內建 LWC 前處理] Metric B 選 top-K layers   ← 降維讓 Hopkins 有意義
  - Hopkins statistic on top-K layer ΔB
  - H > 0.75 或 max_rounds → 進 Phase 1

Phase 1 (clustering):
  - 沿用基礎架構 (A global, B per-cluster, Others local)
  - feature space 是 top-K ΔB (跟 Phase 0 一致)

Phase 2 (frozen):
  - Silhouette > 0.9 或 cluster 連續 N 輪不變 → freeze
  - 沿用固定 cluster assignment
```

> **為什麼 layer selection 是 Multi variant 的內建前處理，不是獨立 variant？**
> Hopkins 在 D > 50 後因 curse of dimensionality 失效（所有 pairwise distance 趨近相等 → H ≈ 0.5），加上 `u_dist**d` 在 D≈50K 會數值溢位。用 Metric B 挑 top-K 層把 D 壓到 ~10K 以下，Hopkins 才真的能 trigger adaptive warm-up。因此 layer selection 邏輯是 Multi variant 的**前提**，不是外掛選項。

## 變體列表

### FedALC-AP（basic，原版簡單 baseline）

- **File**: `bert/fedalc_ap_strategy.py`
- **Class**: `FedALCAPStrategy`
- **Clustering algorithm**: Affinity Propagation（自動決定 K）on full B
- **Config mode**: `aggregation-mode = "fedalc-ap"`
- **Run script**: `bash run_fedalc_all.sh` / `bash run_fedalc_alpha03.sh`
- **說明**：最簡單版本，AP on full B，R1 就 clustering，無 warm-up 無 freeze。
- **保留目的**：作為 historical baseline、最簡 form。

### FedALC-AP-LWC（ablation baseline）

- **File**: `bert/fedalc_ap_lwc_strategy.py`
- **Class**: `FedALCAPLWCStrategy`
- **Clustering algorithm**: AP on top-K B (Metric B layer selection)
- **Trigger**: 無 warm-up（R1 起直接 cluster）；freeze 由 silhouette ≥ `freeze_sil_threshold` 或 cluster 連 N 輪不變觸發
- **Config mode**: `aggregation-mode = "fedalc-ap-lwc"`
- **Run script**: `bash run_fedalc_ap_lwc.sh`
- **定位**：「LWC without adaptive Hopkins trigger」的 ablation baseline，證明 Hopkins + cumulative ΔB 的貢獻。
- **不要再加新功能**：layer selection 的演進改在 `fedalc_ap_multi_strategy.py`。

### FedALC-AP-Multi（**主方法**）

- **File**: `bert/fedalc_ap_multi_strategy.py`
- **Class**: `FedALCAPMultiStrategy`
- **Clustering algorithm**: AP on top-K cumulative ΔB
- **Trigger**: Hopkins adaptive + max_rounds fallback
- **內建 LWC**: Metric B layer selection 作為 Hopkins 降維前處理
- **Config mode**: `aggregation-mode = "fedalc-ap-multi"`
- **Run script**: `bash run_fedalc_ap_multi.sh` (smoke: `run_fedalc_ap_multi_smoke.sh`)
- **定位**：主方法。結合 warm-up + adaptive trigger + layer selection + cumulative signal + freeze，targets multi-task FL 場景。
- **對標 FedLEASE**：Hopkins adaptive vs fixed E；shared A + clustered B vs per-cluster full LoRA + MoE。

### FedALC-Spectral / FedALC-Agglo（規劃中）

- 檔案：尚未實作（`bert/fedalc_spectral_strategy.py` / `bert/fedalc_agglo_strategy.py`）
- Clustering algorithm 不同，其他核心設計可以複用（透過 refactor 共享 base class 或 helper）
- 之後如果 Multi variant 需要跨演算法 ablation，會實作 `FedALC-Spectral-Multi` / `FedALC-Agglo-Multi`。

## Ablation 設計

### Clustering 演算法比較（固定其他設計為 Multi）

| 方法 | Algorithm | 選 K | 預期穩定性 |
|---|---|---|---|
| FedALC-AP-Multi | AP | Auto | 後期不穩定（需 freeze） |
| FedALC-Spectral-Multi（future）| Spectral | Eigengap | 穩定 |
| FedALC-Agglo-Multi（future）| Agglomerative | Silhouette scan | 穩定 |

### Core design 的 ablation（固定 AP）

| 方法 | Warm-up | Layer selection | Clustering feature | Freeze |
|---|---|---|---|---|
| FedALC-AP (baseline) | 無 | 無（full B） | Single B | 無 |
| FedALC-AP-LWC | 無 ⚠ | Metric B top-K | Single B (top-K) | Yes |
| **FedALC-AP-Multi (主方法)** | **Adaptive (Hopkins)** | **Metric B top-K（內建）** | **Cumulative ΔB (top-K)** | Yes |
| FedALC-AP-Multi-no-layer-sel | Adaptive (Hopkins) | **關閉** (full B) | Cumulative ΔB (full) | Yes |
| FedALC-AP-Multi-no-warmup | 無（R1 直接 cluster）| Metric B top-K | Single B (top-K) | Yes |
| FedALC-AP-Multi-fixed-E | Fixed E rounds | Metric B top-K | Cumulative ΔB (top-K) | Yes |
| FedALC-AP-Multi-current-B | Adaptive (Hopkins) | Metric B on current B | Cumulative ΔB (top-K) | Yes |

> `FedALC-AP-Multi-no-layer-sel` 預期 Hopkins 會數值溢位 / 落在 0.5 區間，證明 layer selection 的必要性。
>
> ⚠ **目前 LWC 實作無 warm-up**（code 直接從 Phase 1 開始）。若要做純 Hopkins ablation（LWC = sil-based warm-up vs Multi = Hopkins），需先把 silhouette-based warm-up 補回 LWC，否則 LWC vs Multi 的差異是「warm-up + cumulative ΔB」兩個 component 捆綁，不是純 trigger 機制的差異。
> `FedALC-AP-Multi-current-B` 用 `layer-score-feature = "current_b"` 切換，測試 layer scoring feature 的影響。

## Paper 組織建議

### Main method: FedALC-AP-Multi

Paper 的主方法就是 FedALC-AP-Multi，描述完整的三階段設計，場景 target multi-task FL。

### Ablation tables

**Table 1**: core design ablation（固定 AP）
- 去掉 warm-up / 去掉 cumulative / 去掉 freeze / 去掉 layer selection / fixed E 等
- 目的：證明每個 component 的貢獻

**Table 2 (future)**: clustering algorithm variants
- FedALC-AP-Multi vs FedALC-Spectral-Multi vs FedALC-Agglo-Multi
- 目的：證明方法論不綁定特定 clustering 演算法

**Table 3**: head-to-head comparison with baselines
- FedAvg / FedSA / FFA / FedLEASE / HiLoRA / FedADC vs FedALC-AP-Multi
- 在 single-task 和 multi-task 兩個 setup 下跑

### Method section 寫法

> "We propose FedALC-AP-Multi, a federated LoRA clustering framework for multi-task scenarios. It extends the basic FedALC-AP baseline with four components: (1) adaptive warm-up via Hopkins statistic, (2) built-in Metric B layer selection as dimensionality reduction (so Hopkins remains meaningful), (3) cumulative ΔB as clustering feature, (4) clustering freeze via silhouette threshold or stability. In contrast to FedLEASE's fixed warm-up E and per-cluster full LoRA + MoE, our design is data-driven and lightweight (shared A + clustered B only)."

## Naming 相容性

- **Config mode** 與 class name 對應：
  - `"fedalc-ap"` → `FedALCAPStrategy`
  - `"fedalc-ap-lwc"` → `FedALCAPLWCStrategy`
  - `"fedalc-ap-multi"` → `FedALCAPMultiStrategy`
- **File name** 與 class name 對應：
  - `fedalc_ap_strategy.py` → `FedALCAPStrategy`
  - `fedalc_ap_lwc_strategy.py` → `FedALCAPLWCStrategy`
  - `fedalc_ap_multi_strategy.py` → `FedALCAPMultiStrategy`
- **Log directory** 沿用 config mode：
  - `logs/{timestamp}/{task}_fedalc-ap/`
  - `logs/{timestamp}/{task}_fedalc-ap-lwc/`
  - `logs/{timestamp}/{task}_fedalc-ap-multi/`
