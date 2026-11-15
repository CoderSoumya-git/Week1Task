# Last-Mile Delivery & Replenishment Analytics (Prototype)

Python prototype accompanying the strategic planning report *Data-Driven Last-Mile
Delivery and Replenishment Planning*. It runs end to end on **synthetic data**:

1. Delivery-zone clustering (K-means + silhouette score)
2. Daily demand forecasting (gradient boosting vs. seasonal naive, WAPE)
3. Trip-time (ETA) regression (linear vs. gradient boosting, MAE)
4. Safety stock and reorder point from forecast error
5. Capacitated vehicle routing with time windows (Google OR-Tools) vs. an angular-sweep baseline

## Run
```bash
pip install -r requirements.txt
python logistics_pipeline.py
```
Outputs (metrics JSON and figures) are written to `outputs/`. Runtime is about 30 seconds.

## Notes
- Results are from simulated data and a deliberately simple baseline; they demonstrate the
  method, not real-world savings.
- To use real data, swap the generators for loaders on public datasets such as the Olist
  Brazilian E-Commerce dataset or DataCo Smart Supply Chain (both on Kaggle).

## Reproducibility
Routing results are deterministic (573.1 km → 402.9 km, 48% → 100% on time). The gradient-boosting
metrics (forecast WAPE about 4.5%, ETA MAE about 3.2 min) can differ in the second decimal
between scikit-learn versions.
