"""
Illustrative last-mile logistics analytics pipeline (synthetic data).
Stages: data generation -> zoning (clustering) -> demand forecasting (regression)
        -> ETA model (regression) -> inventory policy -> route optimisation (OR-Tools).
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

os.makedirs("outputs", exist_ok=True)
RNG = np.random.default_rng(42)
SPEED_KMPH, DETOUR, SERVICE_MIN = 25.0, 1.3, 5
CAPACITY, N_VEHICLES = 50, 12
COST_PER_KM, COST_PER_ROUTE = 12.0, 450.0          # INR, illustrative
OUT = "outputs/fig_"
metrics = {}

# ---------------------------------------------------------------- 1. DATA
def make_customers(n=100):
    centres = np.array([[-8, 6], [7, 8], [-5, -9], [9, -6], [0, 1]])
    pts = np.vstack([RNG.normal(c, 2.2, size=(n // 5, 2)) for c in centres])
    slot = RNG.choice([0, 1, 2], size=len(pts), p=[.35, .35, .30])
    tw_start = np.array([0, 180, 360])[slot]                 # minutes after 09:00
    df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1],
                       "demand": RNG.integers(1, 8, len(pts)),
                       "tw_start": tw_start, "tw_end": tw_start + 180})
    return df

cust = make_customers()
depot = np.array([0.0, 0.0])
coords = np.vstack([depot, cust[["x", "y"]].to_numpy()])
dist_km = np.linalg.norm(coords[:, None] - coords[None], axis=2) * DETOUR
travel_min = (dist_km / SPEED_KMPH * 60).round().astype(int)

# ---------------------------------------------------------------- 2. CLUSTERING
best = max(range(3, 9), key=lambda k: silhouette_score(
    cust[["x", "y"]], KMeans(k, n_init=10, random_state=0).fit_predict(cust[["x", "y"]])))
cust["zone"] = KMeans(best, n_init=10, random_state=0).fit_predict(cust[["x", "y"]])
metrics["best_k_zones"] = int(best)

# ---------------------------------------------------------------- 3. DEMAND FORECAST
def make_daily_demand(days=730):
    t = np.arange(days)
    dow = t % 7
    promo = (RNG.random(days) < 0.08).astype(int)
    rain = (RNG.random(days) < 0.20).astype(int)
    base = 400 + 0.15 * t + 60 * np.isin(dow, [5, 6]) + 90 * promo + 35 * rain
    return pd.DataFrame({"t": t, "dow": dow, "promo": promo, "rain": rain,
                         "orders": RNG.poisson(base)})

d = make_daily_demand()
d["lag7"] = d["orders"].shift(7)
d["roll7"] = d["orders"].shift(1).rolling(7).mean()
d = d.dropna().reset_index(drop=True)
split = len(d) - 90                                        # last 90 days = test
feat = ["t", "dow", "promo", "rain", "lag7", "roll7"]
train, test = d.iloc[:split], d.iloc[split:]
wape = lambda y, p: np.abs(y - p).sum() / y.sum() * 100
naive = test["lag7"].to_numpy()
gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
gbm.fit(train[feat], train["orders"])
pred = gbm.predict(test[feat])
metrics["forecast_wape_naive_pct"] = round(wape(test["orders"], naive), 2)
metrics["forecast_wape_gbm_pct"] = round(wape(test["orders"], pred), 2)

# safety stock & reorder point from forecast error (lead time L = 2 days, 95% service)
sigma = (train["orders"] - gbm.predict(train[feat])).std()
z, L = 1.645, 2
safety = z * sigma * np.sqrt(L)
metrics["safety_stock_units"] = int(round(safety))
metrics["reorder_point_units"] = int(round(pred.mean() * L + safety))

# ---------------------------------------------------------------- 4. ETA MODEL
def make_trips(n=6000):
    km = RNG.gamma(3, 2.0, n)
    hour = RNG.integers(9, 20, n)
    rain = RNG.integers(0, 2, n)
    peak = np.isin(hour, [10, 11, 17, 18, 19]).astype(int)
    speed = 28 - 7 * peak - 4 * rain + RNG.normal(0, 2, n)
    minutes = km * DETOUR / np.clip(speed, 8, None) * 60 + RNG.normal(0, 3, n)
    return pd.DataFrame({"km": km, "hour": hour, "rain": rain, "peak": peak,
                         "minutes": minutes.clip(2)})

trips = make_trips()
Xf = ["km", "hour", "rain", "peak"]
tr, te = trips.iloc[:4800], trips.iloc[4800:]
lin = LinearRegression().fit(tr[Xf], tr["minutes"])
hgb = HistGradientBoostingRegressor(random_state=0).fit(tr[Xf], tr["minutes"])
metrics["eta_mae_min_linear"] = round(mean_absolute_error(te["minutes"], lin.predict(te[Xf])), 2)
metrics["eta_mae_min_gbm"] = round(mean_absolute_error(te["minutes"], hgb.predict(te[Xf])), 2)

# ---------------------------------------------------------------- 5. ROUTING
demand = [0] + cust["demand"].tolist()
tw = [(0, 600)] + list(zip(cust["tw_start"], cust["tw_end"]))

def evaluate(routes):
    """Simulate a set of routes: distance, on-time rate, cost."""
    total_km, on_time, served = 0.0, 0, 0
    for r in routes:
        t, prev = 0, 0
        for n in r:
            t += travel_min[prev, n]
            t = max(t, tw[n][0])                           # wait if early
            on_time += int(t <= tw[n][1]); served += 1
            t += SERVICE_MIN; total_km += dist_km[prev, n]; prev = n
        total_km += dist_km[prev, 0]
    n_routes = sum(1 for r in routes if r)
    cost = total_km * COST_PER_KM + n_routes * COST_PER_ROUTE
    return dict(km=round(total_km, 1), routes=n_routes, served=served,
                on_time_pct=round(on_time / served * 100, 1), cost_inr=round(cost))

def sweep_baseline():
    """Manual-dispatch proxy: sort by polar angle, fill vehicles to capacity."""
    ang = np.arctan2(coords[1:, 1], coords[1:, 0])
    order = (np.argsort(ang) + 1).tolist()
    routes, cur, load = [], [], 0
    for n in order:
        if load + demand[n] > CAPACITY:
            routes.append(cur); cur, load = [], 0
        cur.append(n); load += demand[n]
    routes.append(cur)
    return routes

def solve_vrptw(time_limit_s=20):
    m = pywrapcp.RoutingIndexManager(len(coords), N_VEHICLES, 0)
    r = pywrapcp.RoutingModel(m)
    dcb = r.RegisterTransitCallback(lambda i, j: int(dist_km[m.IndexToNode(i), m.IndexToNode(j)] * 100))
    r.SetArcCostEvaluatorOfAllVehicles(dcb)
    r.SetFixedCostOfAllVehicles(int(COST_PER_ROUTE / COST_PER_KM * 100))   # km-equivalent of a route
    qcb = r.RegisterUnaryTransitCallback(lambda i: demand[m.IndexToNode(i)])
    r.AddDimensionWithVehicleCapacity(qcb, 0, [CAPACITY] * N_VEHICLES, True, "Load")
    tcb = r.RegisterTransitCallback(
        lambda i, j: int(travel_min[m.IndexToNode(i), m.IndexToNode(j)] + (SERVICE_MIN if m.IndexToNode(i) else 0)))
    r.AddDimension(tcb, 240, 900, False, "Time")           # up to 240 min waiting slack
    td = r.GetDimensionOrDie("Time")
    for node in range(1, len(coords)):
        td.CumulVar(m.NodeToIndex(node)).SetRange(int(tw[node][0]), int(tw[node][1]))
        r.AddDisjunction([m.NodeToIndex(node)], 100000)    # drop only if infeasible
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromSeconds(time_limit_s)
    sol = r.SolveWithParameters(p)
    routes = []
    for v in range(N_VEHICLES):
        i, route = r.Start(v), []
        while not r.IsEnd(i):
            node = m.IndexToNode(i)
            if node: route.append(node)
            i = sol.Value(r.NextVar(i))
        if route: routes.append(route)
    return routes

base_routes = sweep_baseline()
opt_routes = solve_vrptw()
metrics["baseline"] = evaluate(base_routes)
metrics["optimised"] = evaluate(opt_routes)
b, o = metrics["baseline"], metrics["optimised"]
metrics["km_saving_pct"] = round((1 - o["km"] / b["km"]) * 100, 1)
metrics["cost_saving_pct"] = round((1 - o["cost_inr"] / b["cost_inr"]) * 100, 1)
metrics["cost_per_drop_baseline"] = round(b["cost_inr"] / b["served"], 1)
metrics["cost_per_drop_optimised"] = round(o["cost_inr"] / o["served"], 1)

# ---------------------------------------------------------------- 6. FIGURES
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True, sharey=True)
for a, routes, title, m_ in zip(ax, [base_routes, opt_routes],
                                ["A. Baseline: angular sweep", "B. Optimised: CVRPTW (OR-Tools)"],
                                [b, o]):
    cmap = plt.get_cmap("tab20")
    for k, r in enumerate(routes):
        pts = np.vstack([depot, coords[r], depot])
        a.plot(pts[:, 0], pts[:, 1], "-", lw=1, color=cmap(k % 20), alpha=.85)
        a.scatter(coords[r, 0], coords[r, 1], s=10, color=cmap(k % 20))
    a.scatter(*depot, marker="s", s=70, color="black", zorder=5)
    a.set_title(f"{title}\n{m_['km']} km · {m_['routes']} routes · {m_['on_time_pct']}% on time", fontsize=9)
    a.set_xlabel("x (km)")
ax[0].set_ylabel("y (km)")
plt.tight_layout(); plt.savefig(OUT + "routes.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(test["t"].to_numpy(), test["orders"].to_numpy(), color="#444", lw=1.2, label="Actual")
ax.plot(test["t"].to_numpy(), naive, color="#bbb", lw=1, label=f"Seasonal naive (WAPE {metrics['forecast_wape_naive_pct']}%)")
ax.plot(test["t"].to_numpy(), pred, color="#1f6feb", lw=1.4, label=f"Gradient boosting (WAPE {metrics['forecast_wape_gbm_pct']}%)")
ax.set_xlabel("Day index (held-out 90 days)"); ax.set_ylabel("Daily orders"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "forecast.png", dpi=200); plt.close()

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
