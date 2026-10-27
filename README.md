# Robust Model Calibration with ISP and MaxPro

使用 Python 實作模型校準的實驗設計，探討「參數估計準確」與「整體預測準確」的差別。
這是一個含中文逐步說明的學習專案，使用 NumPy 與 Matplotlib，無須 SciPy。

## 方法來源與專案範圍

本專案參考 V. Roshan Joseph，*Experimental Design for Data Science and Engineering*，
Chapter 9.2，DOI：[10.1201/9781003661535-9](https://doi.org/10.1201/9781003661535-9)。

沿用教材的指數模型（式 9.12）、parameter uncertainty、以 importance support points
（ISP，式 5.17）整合 local designs，以及 sequential MaxPro（式 4.27）增加設計點的策略。
程式為自行實作的數值近似，**不是作者官方實作，也不宣稱完整重現書中結果**。

真實參數、discrepancy 函數、noise 與數值搜尋設定均為本專案的延伸模擬。
程式與說明由作者在 AI 協助下整理；教材 PDF 未包含在 repository 中。

## 如何執行

建議使用 Python 3.10 或以上，在 repository 根目錄執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab simulation.ipynb
```

Windows 啟動虛擬環境請改用 `.venv\Scripts\activate`。
在 Notebook 由上至下按 Shift + Enter；Step 3 的 ISP 最佳化需要一些計算時間。
重新執行時先重跑 Step 0，避免亂數產生器狀態改變。

或直接執行腳本並將圖檔、數據寫入 `results/`：

```bash
MPLBACKEND=Agg python simulation.py
```

Notebook 的輸出是刻意保留空白，供讀者自行逐步執行；`results/` 提供已驗證的一次執行結果。
數值最佳化在不同平台或套件版本下可能有小幅差異。

## 模擬設定

模型為 $h(x;\eta)=\eta_0+\exp(-\eta_1x_1)+\exp(-\eta_2x_2)$，$x\in[0,1]^2$。

| 項目 | 設定 |
|---|---|
| 初始 parameters | $(0,2,2)'$ |
| 真實 parameters | $(0.30,2.70,1.40)'$ |
| Parameter uncertainty | 獨立的 $\eta_1\sim U(1,3)$、$\eta_2\sim N(2,0.5^2)$ |
| Parameter samples | 500 組，整合 2000 個 local-design points |
| ISP 最佳化 | 投影 Adam、6 組起點、每組 1800 次迭代，平滑距離使用 $10^{-10}$ |
| MaxPro | 從 50,000 個均勻候選位置中逐點選出 4 個點 |
| Measurement noise | 獨立 $N(0,0.03^2)$，標準差當成已知 |
| 設計建構 seed | 20261013 |
| 三組 observation seeds | 101、102、103 |

非正的 $\eta_2$ 重新抽樣，因此實際採用正值截斷常態。
四點解析設計的座標若超出 $[0,1]$，程式以邊界截斷近似處理；未對這些情況另外證明 constrained D-optimality。

資料生成式：

$$y_i=h(x_i;\eta^*)+\delta(x_i)+\epsilon_i,$$

$$\delta(x)=0.35\exp\left[-\frac{(x_1-0.82)^2+(x_2-0.82)^2}{2(0.18)^2}\right].$$

此局部凸起放在 local design 較少探索的區域，用來展示 space-filling 的作用；
並非根據真實物理資料推導，也不是教材給定的 discrepancy。

## 三種設計與分析

| 設計 | 預算分配 |
|---|---|
| Local D-optimal | 4 個位置，各重複 2 次 |
| Parameter robust | 8 個 ISP points，允許位置重複 |
| Combined robust | 4 個 ISP points + 4 個 sequential MaxPro points |

![Experimental designs](results/design_comparison.png)

三種設計各自生成 8 個 physical observations，再以相同 two-stage 方法分析：

1. 以帶阻尼、回溯步長與參數邊界的 Gauss–Newton 最小化平方誤差，估計 $\hat\eta$。
2. 對 $e_i=y_i-h(x_i;\hat\eta)$ 擬合 zero-mean GP。
3. 最終預測 $\hat\mu(x)=h(x;\hat\eta)+\hat\delta(x)$。

GP 使用 squared-exponential covariance，length scales 各搜尋 $[0.15,2]$ 中的
12 個等比候選值，process 標準差搜尋 $[0.02,0.60]$ 中的 10 個候選值。
以最大概似選擇 1440 組候選中的最佳組合；covariance 加 $10^{-8}$ 數值穩定項。

## 評估與一次執行結果

以不參與擬合的 $61\times61=3721$ 個規則網格位置，比較預測與不含 measurement noise 的真實平均曲面。
Parameter error 為 $\|\hat\eta-\eta^*\|_2$；prediction RMSE 為
$\sqrt{M^{-1}\sum_j[\hat\mu(x_j)-\mu(x_j)]^2}$。

| 設計 | Parameter error | Prediction RMSE |
|---|---:|---:|
| Local | 0.4403 | 0.0870 |
| Parameter robust | 0.0794 | 0.0849 |
| Combined robust | 0.2409 | 0.0321 |

![Prediction surfaces](results/prediction_comparison.png)

本次 parameter-robust design 的參數誤差較小；combined design 能在右上方取得資料，
因此整體預測誤差較小。色階表示 response，非誤差大小。
完整實驗位置見 `results/design_points.csv`；參數估計與誤差見 `results/summary.csv`。

## 限制與後續工作

- 僅一次固定曲面模擬，且三種設計 noise realizations 不同，不能推論普遍排名。
- 尚未做多次模擬、不同 discrepancy 情境與搜尋超參數的敏感度比較。
- ISP 使用局部數值最佳化；MaxPro 使用有限候選集，皆不保證全域最佳。
- GP 超參數可能落在搜尋邊界，反映結果依賴搜尋範圍。
- Two-stage fitting 忽略 parameter uncertainty；discrepancy 也可能混入參數估計誤差。
- 更準的預測不代表估計到更準的真實物理參數。

可延伸為多 seed 重複模擬、改變凸起位置與幅度、增加第二輪實驗，並報告誤差分布。
