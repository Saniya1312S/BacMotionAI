"""
Patch script for BacMotionAI_v2_AntiOverfit.ipynb
Fixes 8 bugs: scaler leakage, SHAP indexing, speed calcs, XGBoost deprecations,
dead code, tumble rate mismatch, unused optuna.
"""
import json, copy, sys, os

INPUT  = r"C:\Users\SANIYA\Downloads\BacMotionAI_v2_AntiOverfit.ipynb"
OUTPUT = r"C:\Users\SANIYA\OneDrive\Desktop\Test 2\BacMotionAI_v2_AntiOverfit_Fixed.ipynb"

with open(INPUT, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Helper: find a cell by its metadata id
def find_cell(cell_id):
    for i, c in enumerate(cells):
        if c.get("metadata", {}).get("id") == cell_id:
            return i
    raise ValueError(f"Cell '{cell_id}' not found")

# ============================================================
# FIX 1 — Remove unused optuna from install (Cell: "install")
# ============================================================
idx = find_cell("install")
cells[idx]["source"] = [
    "!pip install -q xgboost imbalanced-learn shap"
]
print("[Fix 1] Removed unused optuna from install cell")

# ============================================================
# FIX 7 — Tumble rate mismatch + Fix 3 — Bogus speed (Cell: "obj1_sim")
# ============================================================
idx = find_cell("obj1_sim")
cells[idx]["source"] = [
    "# ================================================================\n",
    "# CELL 3 — Objective 1: Trajectory Visualization (Dry vs Wet)\n",
    "# ================================================================\n",
    "\n",
    "# Per-chemical tumble rates (matches full simulation engine)\n",
    "TUMBLE_RATES = {'none': 0.005, 'attractant': 0.003,\n",
    "                'repellent': 0.005, 'antibiotic': 0.01}\n",
    "\n",
    "def get_trajectory(W, hydro, chemical='none', seed=0):\n",
    "    np.random.seed(seed)\n",
    "    cs = CHEM_STATS[chemical]\n",
    "    v_scale = cs['v_mu']; Dr_eff = cs['Dr_mu'] * Dr; drift = cs['drift_mu']\n",
    "    p_tumble = TUMBLE_RATES.get(chemical, 0.005)\n",
    "    x = np.random.rand()*L; y = np.random.rand()*W; theta = np.random.rand()*2*np.pi\n",
    "    xs, ys, ths = [], [], []\n",
    "    for _ in range(steps):\n",
    "        if np.random.rand() < p_tumble: theta = np.random.rand()*2*np.pi\n",
    "        if hydro:\n",
    "            h = min(y, W-y); h_eff = max(h, a)\n",
    "            omega = -kappa*(a/h_eff)**2*np.sin(2*theta)\n",
    "            v_eff = v0*v_scale*(1+eps*(a/h_eff))\n",
    "        else:\n",
    "            omega = 0.0; v_eff = v0*v_scale\n",
    "        theta += omega*dt + drift*dt + np.sqrt(2*Dr_eff*dt)*np.random.randn()\n",
    "        x += v_eff*np.cos(theta)*dt; y += v_eff*np.sin(theta)*dt\n",
    "        x %= L\n",
    "        if y < 0: y=-y; theta=-theta\n",
    "        elif y > W: y=2*W-y; theta=-theta\n",
    "        xs.append(x); ys.append(y); ths.append(theta)\n",
    "    return xs, ys, ths\n",
    "\n",
    "cases = [\n",
    "    {'W':10,  'hydro':False, 'label':'10µm | Dry',  'color':'#2196F3'},\n",
    "    {'W':50,  'hydro':False, 'label':'50µm | Dry',  'color':'#4CAF50'},\n",
    "    {'W':10,  'hydro':True,  'label':'10µm | Wet',  'color':'#F44336'},\n",
    "    {'W':50,  'hydro':True,  'label':'50µm | Wet',  'color':'#FF9800'},\n",
    "]\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(16, 7))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for ax, case in zip(axes, cases):\n",
    "    vmeans = []\n",
    "    for i in range(12):\n",
    "        xs, ys, ths = get_trajectory(case['W'], case['hydro'], seed=i)\n",
    "        # Fix 3: compute actual speed from displacements instead of trig identity\n",
    "        dxs = np.diff(xs); dys = np.diff(ys)\n",
    "        spd = np.mean(np.sqrt(np.array(dxs)**2 + np.array(dys)**2) / dt)\n",
    "        vmeans.append(spd)\n",
    "        t_arr = np.linspace(0, 1, len(xs))\n",
    "        for j in range(len(xs)-1):\n",
    "            ax.plot(xs[j:j+2], ys[j:j+2], color=case['color'],\n",
    "                    alpha=0.3 + 0.5*t_arr[j], lw=0.8)\n",
    "        for x,y,t in zip(xs[::50], ys[::50], ths[::50]):\n",
    "            dx=1.2*np.cos(t); dy=1.2*np.sin(t)\n",
    "            ax.plot([x-dx,x+dx],[y-dy,y+dy], 'k-', lw=1.2, alpha=0.4)\n",
    "    ax.axhline(0, color='k', lw=3); ax.axhline(case['W'], color='k', lw=3)\n",
    "    ax.set_xlim(0, L); ax.set_ylim(-3, case['W']+3)\n",
    "    ax.set_title(f\"{case['label']}  |  ⟨v⟩={np.mean(vmeans):.2f} µm/s\",\n",
    "                 fontweight='bold', fontsize=11)\n",
    "    ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')\n",
    "    ax.set_facecolor('#f5f5f5')\n",
    "\n",
    "fig.suptitle('Objective 1 — E. coli Trajectories: Dry vs Wet × Channel Width', fontsize=13, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('obj1_trajectories.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]
print("[Fix 3] Fixed bogus speed calculation (was always v0 due to trig identity)")
print("[Fix 7] Fixed tumble rate mismatch in get_trajectory()")

# ============================================================
# FIX 4 — Flat speed distribution in Obj 2 (Cell: "obj2_chem")
# ============================================================
idx = find_cell("obj2_chem")
cells[idx]["source"] = [
    "# ================================================================\n",
    "# CELL 4 — Objective 2: Chemical Stimuli Analysis\n",
    "# ================================================================\n",
    "\n",
    "chemicals = ['none', 'attractant', 'repellent', 'antibiotic']\n",
    "W_chem = 20\n",
    "\n",
    "fig, axes = plt.subplots(2, 4, figsize=(18, 9))\n",
    "chem_stats_collected = {}\n",
    "\n",
    "for col_idx, chem in enumerate(chemicals):\n",
    "    ax_traj = axes[0, col_idx]\n",
    "    ax_spd  = axes[1, col_idx]\n",
    "    c = PALETTE[chem]\n",
    "\n",
    "    all_spds, all_cos = [], []\n",
    "    for i in range(15):\n",
    "        xs, ys, ths = get_trajectory(W_chem, hydro=True, chemical=chem, seed=i+50)\n",
    "        # Fix 4: compute per-step speed from actual displacements\n",
    "        dxs_i = np.diff(xs); dys_i = np.diff(ys)\n",
    "        spds_i = np.sqrt(np.array(dxs_i)**2 + np.array(dys_i)**2) / dt\n",
    "        all_spds.extend(spds_i.tolist())\n",
    "        all_cos.extend([abs(np.cos(t)) for t in ths])\n",
    "        ax_traj.plot(xs, ys, color=c, lw=0.6, alpha=0.5)\n",
    "        for x,y,t in zip(xs[::50], ys[::50], ths[::50]):\n",
    "            dx=1.0*np.cos(t); dy=1.0*np.sin(t)\n",
    "            ax_traj.plot([x-dx,x+dx],[y-dy,y+dy],'k-',lw=1,alpha=0.4)\n",
    "\n",
    "    ax_traj.axhline(0, color='k', lw=2.5); ax_traj.axhline(W_chem, color='k', lw=2.5)\n",
    "    ax_traj.set_xlim(0, L); ax_traj.set_ylim(-2, W_chem+2)\n",
    "    ax_traj.set_title(f\"{chem.capitalize()}\\n⟨v⟩={np.mean(all_spds):.2f}  ⟨align⟩={np.mean(all_cos):.3f}\",\n",
    "                      fontweight='bold', fontsize=9, color=c)\n",
    "    ax_traj.set_xlabel('x (µm)'); ax_traj.set_ylabel('y (µm)')\n",
    "    ax_traj.set_facecolor('#fafafa')\n",
    "\n",
    "    ax_spd.hist(all_spds,\n",
    "                bins=35, color=c, alpha=0.75, edgecolor='none', density=True)\n",
    "    ax_spd.axvline(np.mean(all_spds), color='black', lw=2, linestyle='--')\n",
    "    ax_spd.set_xlabel('Speed (µm/s)'); ax_spd.set_ylabel('Density')\n",
    "    ax_spd.set_title('Speed Distribution', fontsize=9)\n",
    "    ax_spd.set_facecolor('#fafafa')\n",
    "\n",
    "    chem_stats_collected[chem] = {'vmean': np.mean(all_spds), 'align': np.mean(all_cos)}\n",
    "\n",
    "fig.suptitle('Objective 2 — Chemical Stimuli: Trajectories + Speed Distributions', fontsize=13, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('obj2_chemical.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# Summary bar chart\n",
    "fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))\n",
    "labels_c = [c.capitalize() for c in chemicals]\n",
    "cols_c   = [PALETTE[c] for c in chemicals]\n",
    "\n",
    "axes2[0].bar(labels_c, [chem_stats_collected[c]['vmean'] for c in chemicals],\n",
    "             color=cols_c, edgecolor='black', linewidth=0.8, alpha=0.85)\n",
    "axes2[0].set_title('Mean Speed per Chemical Condition', fontweight='bold')\n",
    "axes2[0].set_ylabel('Speed (µm/s)')\n",
    "\n",
    "axes2[1].bar(labels_c, [chem_stats_collected[c]['align'] for c in chemicals],\n",
    "             color=cols_c, edgecolor='black', linewidth=0.8, alpha=0.85)\n",
    "axes2[1].set_title('Mean Alignment |cos θ| per Chemical', fontweight='bold')\n",
    "axes2[1].set_ylabel('⟨|cos θ|⟩')\n",
    "\n",
    "for ax in axes2: ax.set_facecolor('#f8f9fa')\n",
    "fig2.suptitle('Objective 2 — Chemical Summary', fontsize=13, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('obj2_summary.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]
print("[Fix 4] Fixed flat speed distribution in Obj 2 (now uses displacements)")

# ============================================================
# FIX 1 — Scaler data leakage in train/test split (Cell: "train_test_split_cell")
# ============================================================
idx = find_cell("train_test_split_cell")
cells[idx]["source"] = [
    "# ================================================================\n",
    "# CELL 8 — Proper Train/Test Split (SMOTE only on train)\n",
    "# ================================================================\n",
    "\n",
    "le = LabelEncoder()\n",
    "df['regime_enc'] = le.fit_transform(df['regime'])\n",
    "\n",
    "X_raw = df[FEATURE_COLS].values\n",
    "y_raw = df['regime_enc'].values\n",
    "y_spd = df['mean_speed'].values\n",
    "\n",
    "# ── Step 1: Hold out 20% test set BEFORE any balancing ──\n",
    "X_trainval, X_test, y_trainval, y_test = train_test_split(\n",
    "    X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw\n",
    ")\n",
    "\n",
    "# ── Step 2: Scale (fit on trainval only) ──\n",
    "scaler = StandardScaler()\n",
    "X_trainval_sc = scaler.fit_transform(X_trainval)\n",
    "X_test_sc     = scaler.transform(X_test)   # no fitting on test!\n",
    "\n",
    "# ── Step 3: SMOTE only on trainval ──\n",
    "smote = SMOTE(random_state=42, k_neighbors=5)\n",
    "X_tr_sm, y_tr_sm = smote.fit_resample(X_trainval_sc, y_trainval)\n",
    "\n",
    "# ── Step 4: Regression split (separate scaler to avoid leakage) ──\n",
    "X_raw_tr, X_raw_te, y_tr_r, y_te_r = train_test_split(\n",
    "    X_raw, y_spd, test_size=0.2, random_state=42\n",
    ")\n",
    "scaler_reg = StandardScaler()\n",
    "X_tr_r = scaler_reg.fit_transform(X_raw_tr)\n",
    "X_te_r = scaler_reg.transform(X_raw_te)\n",
    "\n",
    "print('Train (after SMOTE):', X_tr_sm.shape)\n",
    "print('Test  (clean):      ', X_test_sc.shape)\n",
    "print('\\nTest class distribution:')\n",
    "for cls, cnt in zip(*np.unique(y_test, return_counts=True)):\n",
    "    print(f'  {le.classes_[cls]}: {cnt}')\n",
    "print('\\nTrain (SMOTE) distribution:')\n",
    "for cls, cnt in zip(*np.unique(y_tr_sm, return_counts=True)):\n",
    "    print(f'  {le.classes_[cls]}: {cnt}')"
]
print("[Fix 1] Fixed scaler data leakage — regression now uses separate scaler_reg")

# ============================================================
# FIX 5 — XGBoost deprecated params (Cell: "xgb_model")
# ============================================================
idx = find_cell("xgb_model")
cells[idx]["source"] = [
    "# ================================================================\n",
    "# CELL 12 — Model 3: XGBoost with Early Stopping\n",
    "# ================================================================\n",
    "\n",
    "# Internal validation split for early stopping\n",
    "X_xgb_tr, X_xgb_val, y_xgb_tr, y_xgb_val = train_test_split(\n",
    "    X_tr_sm, y_tr_sm, test_size=0.15, random_state=42, stratify=y_tr_sm\n",
    ")\n",
    "\n",
    "xgb_clf = xgb.XGBClassifier(\n",
    "    n_estimators=500, max_depth=4, learning_rate=0.05,\n",
    "    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,\n",
    "    eval_metric='mlogloss',\n",
    "    random_state=42, n_jobs=-1\n",
    ")\n",
    "xgb_clf.fit(\n",
    "    X_xgb_tr, y_xgb_tr,\n",
    "    eval_set=[(X_xgb_val, y_xgb_val)],\n",
    "    verbose=False,\n",
    "    early_stopping_rounds=30\n",
    ")\n",
    "y_pred_xgb  = xgb_clf.predict(X_test_sc)\n",
    "y_proba_xgb = xgb_clf.predict_proba(X_test_sc)\n",
    "\n",
    "acc_xgb  = accuracy_score(y_test, y_pred_xgb)\n",
    "ll_xgb   = log_loss(y_test, y_proba_xgb)\n",
    "auc_xgb  = roc_auc_score(y_test, y_proba_xgb, multi_class='ovr', average='macro')\n",
    "\n",
    "print(f'⚡ XGBoost — Acc: {acc_xgb:.4f}  LogLoss: {ll_xgb:.4f}  ROC-AUC: {auc_xgb:.4f}')\n",
    "print(f'Best iteration: {xgb_clf.best_iteration}')\n",
    "print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))\n",
    "\n",
    "cv_xgb = cross_val_score(xgb_clf, X_tr_sm, y_tr_sm, cv=5, scoring='accuracy', n_jobs=-1)\n",
    "print(f'5-Fold CV: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}')\n",
    "print(f'Generalization gap: {cv_xgb.mean()-acc_xgb:.4f}')\n",
    "\n",
    "# Training history\n",
    "results = xgb_clf.evals_result()\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "axes[0].plot(results['validation_0']['mlogloss'], color='#F44336', lw=2)\n",
    "axes[0].axvline(xgb_clf.best_iteration, color='black', ls='--', lw=1.5, label='Best iter')\n",
    "axes[0].set_xlabel('Boosting Round'); axes[0].set_ylabel('Log Loss')\n",
    "axes[0].set_title('XGBoost Training — Early Stopping', fontweight='bold')\n",
    "axes[0].legend(); axes[0].set_facecolor('#f8f9fa')\n",
    "\n",
    "ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_xgb), display_labels=le.classes_).plot(\n",
    "    ax=axes[1], cmap='Purples', colorbar=False)\n",
    "axes[1].set_title('XGBoost — Confusion Matrix', fontweight='bold')\n",
    "axes[1].tick_params(axis='x', rotation=30)\n",
    "plt.tight_layout()\n",
    "plt.savefig('xgb_eval.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]
print("[Fix 5] Removed deprecated XGBoost params (use_label_encoder, early_stopping_rounds in constructor)")

# ============================================================
# FIX 2 — SHAP values indexing (Cell: "shap_cell")
# ============================================================
idx = find_cell("shap_cell")
cells[idx]["source"] = [
    "# ================================================================\n",
    "# CELL 17 — SHAP Explainability (XGBoost)\n",
    "# ================================================================\n",
    "\n",
    "explainer   = shap.TreeExplainer(xgb_clf)\n",
    "shap_values = explainer.shap_values(X_test_sc[:200])  # subset for speed\n",
    "\n",
    "# Mean absolute SHAP per class\n",
    "fig, axes = plt.subplots(1, min(3, len(le.classes_)), figsize=(18, 5))\n",
    "for i, (ax, cls) in enumerate(zip(axes, le.classes_[:3])):\n",
    "    # Fix 2: handle both list (old SHAP) and 3D array (new SHAP) formats\n",
    "    if isinstance(shap_values, list):\n",
    "        sv = shap_values[i]\n",
    "    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:\n",
    "        sv = shap_values[:, :, i]\n",
    "    else:\n",
    "        sv = shap_values\n",
    "    mean_abs = np.abs(sv).mean(axis=0)\n",
    "    idx_sorted = np.argsort(mean_abs)[::-1]\n",
    "    ax.barh([FEATURE_COLS[j] for j in idx_sorted[:10]], mean_abs[idx_sorted[:10]],\n",
    "            color='#FF6F00', edgecolor='black', linewidth=0.5)\n",
    "    ax.set_title(f'SHAP — {cls}', fontweight='bold')\n",
    "    ax.set_xlabel('Mean |SHAP value|')\n",
    "    ax.invert_yaxis()\n",
    "    ax.set_facecolor('#f8f9fa')\n",
    "\n",
    "fig.suptitle('SHAP Feature Importances per Regime (XGBoost)', fontsize=13, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]
print("[Fix 2] Fixed SHAP values indexing for 3D numpy arrays (newer SHAP versions)")

# ============================================================
# FIX 6 — Dead code in widget (Cell: "widget")
# ============================================================
idx = find_cell("widget")
cells[idx]["source"] = [
    "# ================================================================\n",
    "# CELL 19 — Interactive Widget\n",
    "# User selects conditions → Simulation → ML predicts regime\n",
    "# ================================================================\n",
    "\n",
    "best_clf_map = {'Random Forest': rf, 'XGBoost': xgb_clf, 'MLP': mlp, 'Logistic Reg.': lr}\n",
    "best_auto    = max({'Random Forest': auc_rf, 'XGBoost': auc_xgb, 'MLP': auc_mlp, 'Logistic Reg.': auc_lr},\n",
    "                   key=lambda k: {'Random Forest': auc_rf, 'XGBoost': auc_xgb, 'MLP': auc_mlp, 'Logistic Reg.': auc_lr}[k])\n",
    "\n",
    "w_width = widgets.IntSlider(value=20, min=5, max=100, step=5, description='Width (µm):',\n",
    "                             style={'description_width':'130px'}, layout=widgets.Layout(width='420px'))\n",
    "w_chem  = widgets.Dropdown(options=['none','attractant','repellent','antibiotic'],\n",
    "                            value='none', description='Chemical:',\n",
    "                            style={'description_width':'130px'}, layout=widgets.Layout(width='300px'))\n",
    "w_hydro = widgets.ToggleButtons(\n",
    "    options=[('🌵 Dry (No Hydro)', False), ('💧 Wet (Hydrodynamic)', True)],\n",
    "    value=False, description='Condition:', style={'description_width':'130px'})\n",
    "w_model = widgets.Dropdown(\n",
    "    options=['Random Forest','XGBoost','MLP','Logistic Reg.'],\n",
    "    value=best_auto, description='ML Model:',\n",
    "    style={'description_width':'130px'}, layout=widgets.Layout(width='300px'))\n",
    "w_btn   = widgets.Button(description='▶  Run Simulation + Predict',\n",
    "                          button_style='success',\n",
    "                          layout=widgets.Layout(width='250px', height='38px'))\n",
    "w_out   = widgets.Output()\n",
    "\n",
    "def on_run(b):\n",
    "    with w_out:\n",
    "        clear_output(wait=True)\n",
    "        W      = w_width.value\n",
    "        chem   = w_chem.value\n",
    "        hydro  = w_hydro.value\n",
    "        mname  = w_model.value\n",
    "        clf    = best_clf_map[mname]\n",
    "\n",
    "        # Run simulation → get feature record\n",
    "        recs = run_simulation(W, hydro, chem, N=20, seed=77)\n",
    "        rec  = recs[0]  # representative bacterium\n",
    "\n",
    "        feat_vec = np.array([[rec[f] for f in FEATURE_COLS]])\n",
    "        feat_sc  = scaler.transform(feat_vec)\n",
    "\n",
    "        pred_enc   = clf.predict(feat_sc)[0]\n",
    "        pred_proba = clf.predict_proba(feat_sc)[0]\n",
    "        pred_label = le.classes_[pred_enc]\n",
    "        pred_speed = rf_reg.predict(feat_sc)[0]\n",
    "        confidence = pred_proba.max()\n",
    "\n",
    "        # Get trajectory for display\n",
    "        xs, ys, ths = get_trajectory(W, hydro, chem, seed=77)\n",
    "\n",
    "        # --- Plot ---\n",
    "        fig = plt.figure(figsize=(18, 10))\n",
    "        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.4)\n",
    "\n",
    "        col_c = PALETTE.get(chem, '#607D8B')\n",
    "        col_r = REGIME_PALETTE.get(pred_label, '#607D8B')\n",
    "\n",
    "        # 1. Trajectory\n",
    "        ax1 = fig.add_subplot(gs[0, :2])\n",
    "        t_arr = np.linspace(0, 1, len(xs))\n",
    "        for j in range(0, len(xs)-1, 2):\n",
    "            ax1.plot(xs[j:j+2], ys[j:j+2], color=col_c, alpha=0.3+0.4*t_arr[j], lw=1.0)\n",
    "        for x, y, t in zip(xs[::40], ys[::40], ths[::40]):\n",
    "            dx=1.3*np.cos(t); dy=1.3*np.sin(t)\n",
    "            ax1.plot([x-dx,x+dx],[y-dy,y+dy],'k-',lw=1.5,alpha=0.45)\n",
    "        ax1.axhline(0, color='k', lw=3); ax1.axhline(W, color='k', lw=3)\n",
    "        ax1.set_xlim(0, L); ax1.set_ylim(-3, W+3)\n",
    "        cond_lbl = '💧 Wet' if hydro else '🌵 Dry'\n",
    "        ax1.set_title(\n",
    "            f'Simulated Trajectory  |  W={W}µm  |  {cond_lbl}  |  {chem.capitalize()}',\n",
    "            fontweight='bold', fontsize=11)\n",
    "        ax1.set_xlabel('x (µm)'); ax1.set_ylabel('y (µm)')\n",
    "        ax1.set_facecolor('#f0f4f8')\n",
    "\n",
    "        # 2. Probability bar\n",
    "        ax2 = fig.add_subplot(gs[0, 2])\n",
    "        bars = ax2.barh(le.classes_, pred_proba,\n",
    "                        color=[REGIME_PALETTE.get(c,'gray') for c in le.classes_],\n",
    "                        edgecolor='black', linewidth=0.7, alpha=0.85)\n",
    "        for bar, p in zip(bars, pred_proba):\n",
    "            ax2.text(p+0.01, bar.get_y()+bar.get_height()/2,\n",
    "                     f'{p:.2f}', va='center', fontsize=9, fontweight='bold')\n",
    "        ax2.set_xlim(0, 1.15)\n",
    "        ax2.set_title(f'ML Prediction\\n→ {pred_label} ({confidence:.1%})',\n",
    "                      fontweight='bold', fontsize=10, color=col_r)\n",
    "        ax2.set_xlabel('Probability')\n",
    "        ax2.set_facecolor('#f8f9fa')\n",
    "\n",
    "        # 3. Speed distribution across N bacteria\n",
    "        ax3 = fig.add_subplot(gs[0, 3])\n",
    "        all_mean_spds = [r['mean_speed'] for r in recs]\n",
    "        ax3.hist(all_mean_spds, bins=15, color=col_c, alpha=0.8, edgecolor='black', linewidth=0.5)\n",
    "        ax3.axvline(np.mean(all_mean_spds), color='k', lw=2, ls='--', label=f'Sim={np.mean(all_mean_spds):.2f}')\n",
    "        ax3.axvline(pred_speed, color='red', lw=2, ls=':', label=f'ML={pred_speed:.2f}')\n",
    "        ax3.set_xlabel('Speed (µm/s)'); ax3.set_ylabel('Count')\n",
    "        ax3.set_title('Speed Distribution\\n(N=20 bacteria)', fontweight='bold')\n",
    "        ax3.legend(fontsize=8)\n",
    "        ax3.set_facecolor('#f8f9fa')\n",
    "\n",
    "        # 4. Near-wall fraction by regime\n",
    "        ax4 = fig.add_subplot(gs[1, 0])\n",
    "        ax4.hist(ys, bins=20, color='#FF9800', alpha=0.8, orientation='horizontal',\n",
    "                 density=True, edgecolor='none')\n",
    "        ax4.set_ylabel('y position (µm)'); ax4.set_xlabel('Density')\n",
    "        ax4.set_title('y-Distribution\\n(Wall Accumulation)', fontweight='bold')\n",
    "        ax4.set_ylim(0, W); ax4.set_facecolor('#f8f9fa')\n",
    "\n",
    "        # 5. Feature radar-style bar\n",
    "        ax5 = fig.add_subplot(gs[1, 1])\n",
    "        feat_disp = ['mean_speed','cv_speed','mean_alignment','near_wall_frac','mean_run_len']\n",
    "        feat_vals = [rec[f] for f in feat_disp]\n",
    "        # Normalize 0-1\n",
    "        feat_norms = [min(v/25, 1) if 'speed' in f and 'cv' not in f\n",
    "                      else min(v, 1) for f, v in zip(feat_disp, feat_vals)]\n",
    "        feat_norms[4] = min(rec['mean_run_len'] / 300, 1)\n",
    "        ax5.barh(feat_disp, feat_norms, color=col_r, edgecolor='black', linewidth=0.5, alpha=0.8)\n",
    "        ax5.set_xlim(0, 1.1)\n",
    "        ax5.set_title('Key Feature Values\\n(Normalized)', fontweight='bold')\n",
    "        ax5.set_xlabel('Normalized Value')\n",
    "        ax5.set_facecolor('#f8f9fa')\n",
    "\n",
    "        # 6. Summary panel\n",
    "        ax6 = fig.add_subplot(gs[1, 2:])\n",
    "        ax6.axis('off')\n",
    "        summary = (\n",
    "            f\"SIMULATION + ML PREDICTION SUMMARY\\n\"\n",
    "            f\"{'─'*42}\\n\"\n",
    "            f\"  Inputs\\n\"\n",
    "            f\"    Channel Width  : {W} µm\\n\"\n",
    "            f\"    Condition      : {'Wet (Hydrodynamic)' if hydro else 'Dry (No Hydro)'}\\n\"\n",
    "            f\"    Chemical       : {chem.capitalize()}\\n\"\n",
    "            f\"    ML Model       : {mname}\\n\"\n",
    "            f\"{'─'*42}\\n\"\n",
    "            f\"  Simulation (N=20 bacteria)\\n\"\n",
    "            f\"    ⟨Speed⟩       : {np.mean([r['mean_speed'] for r in recs]):.3f} µm/s\\n\"\n",
    "            f\"    ⟨Alignment⟩   : {np.mean([r['mean_alignment'] for r in recs]):.3f}\\n\"\n",
    "            f\"    ⟨Run Length⟩  : {np.mean([r['mean_run_len'] for r in recs]):.1f} steps\\n\"\n",
    "            f\"    Near-Wall Frac : {np.mean([r['near_wall_frac'] for r in recs]):.3f}\\n\"\n",
    "            f\"{'─'*42}\\n\"\n",
    "            f\"  ML Prediction\\n\"\n",
    "            f\"    Regime         : {pred_label}\\n\"\n",
    "            f\"    Confidence     : {confidence:.1%}\\n\"\n",
    "            f\"    Pred Speed     : {pred_speed:.3f} µm/s\\n\"\n",
    "        )\n",
    "        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9.5,\n",
    "                 va='top', fontfamily='monospace',\n",
    "                 bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.95,\n",
    "                           edgecolor='#388E3C', linewidth=2))\n",
    "\n",
    "        fig.suptitle(f'Interactive Result — Predicted Regime: {pred_label}',\n",
    "                     fontsize=14, fontweight='bold', color=col_r)\n",
    "        plt.savefig('widget_result.png', dpi=150, bbox_inches='tight')\n",
    "        plt.show()\n",
    "\n",
    "w_btn.on_click(on_run)\n",
    "\n",
    "ui = widgets.VBox([\n",
    "    widgets.HTML('<h3 style=\"color:#1565C0;font-family:monospace\">🦠 Bacterial Motility Simulator + ML Predictor v2</h3>'),\n",
    "    widgets.HTML('<p style=\"color:#555\">Select conditions below and click Run to simulate + predict motility regime</p>'),\n",
    "    widgets.HTML('<hr>'),\n",
    "    w_width, w_chem, w_hydro, w_model, w_btn, w_out\n",
    "])\n",
    "display(ui)"
]
print("[Fix 6] Removed dead code (no-op loop) from widget cell")

# ============================================================
# Write fixed notebook
# ============================================================
nb["cells"] = cells

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"\n✅ Fixed notebook written to:\n   {OUTPUT}")
print("\nAll 8 fixes applied successfully:")
print("  [1] Scaler data leakage (Critical)")
print("  [2] SHAP values indexing (Critical)")
print("  [3] Bogus speed calc in Obj 1 (Critical)")
print("  [4] Flat speed distribution in Obj 2 (Medium)")
print("  [5] XGBoost deprecated params (Medium)")
print("  [6] Dead code in widget (Medium)")
print("  [7] Tumble rate mismatch (Medium)")
print("  [8] Unused optuna install (Minor)")
