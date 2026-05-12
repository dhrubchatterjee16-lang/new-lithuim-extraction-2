"""
DES Lithium Recovery Predictor — Gradio app for Hugging Face Spaces
Ensemble v6: Stacking meta-learner · VFT-aware · Physics-constrained · 35 features
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from model_data import load_all_models, ENCODERS, MODEL_RESULTS

# ── Load models once at startup ────────────────────────────────────────────────
models, le_hba, le_hbd = load_all_models()

ALL_FEATURES = ENCODERS['ALL_FEATURES']
DESC_CACHE   = ENCODERS['desc_cache']
MEAN_DESC    = ENCODERS['mean_desc']
DESC_KEYS    = ENCODERS['desc_keys']
LI_HBAS      = set(ENCODERS['li_hbas'])
LI_HBDS      = set(ENCODERS['li_hbds'])
W_IDX        = ALL_FEATURES.index('water_content_mol_fraction')

ALL_HBAS = sorted(le_hba.classes_.tolist())
ALL_HBDS = sorted(le_hbd.classes_.tolist())

BEST_LI_HBAS = [h for h in ['choline chloride','lactic acid','citric acid','betaine',
    'nicotinamide','glycine','dl-tartaric acid','malic acid',
    'guanidine hydrochloride','acetylcholine chloride'] if h in le_hba.classes_]
BEST_LI_HBDS = [h for h in ['ethylene glycol','oxalic acid','levulinic acid',
    'glycerol','lactic acid','malonic acid','glutaric acid','dl-malic acid',
    'citric acid','dl-tartaric acid','urea','phenol','acetic acid'] if h in le_hbd.classes_]

# ── Inference helpers ──────────────────────────────────────────────────────────
def get_desc(c):
    return (DESC_CACHE.get(c.lower().strip()) or
            DESC_CACHE.get(c.strip()) or
            DESC_CACHE.get(c) or MEAN_DESC)

def build_row(hba, hbd, ratio, temp_K, water):
    hba_enc   = le_hba.transform([hba])[0]
    hbd_enc   = le_hbd.transform([hbd])[0]
    is_li_hba = int(hba in LI_HBAS)
    is_li_hbd = int(hbd in LI_HBDS)
    water_reg = int(water >= 0.1)
    hba_d = get_desc(hba); hbd_d = get_desc(hbd)
    row = [hba_enc, hbd_enc, ratio, temp_K, water, is_li_hba, is_li_hbd, water_reg]
    for k in DESC_KEYS:
        row.append(hba_d[k] if isinstance(hba_d, dict) else MEAN_DESC[k])
    for k in DESC_KEYS:
        row.append(hbd_d[k] if isinstance(hbd_d, dict) else MEAN_DESC[k])
    T = temp_K; W = water
    row += [1/T, np.log(T), 1/(T-150), 1/(T-165), 1/(T-180), 1/(T-200),
            (T/300)**2, (1/T)*W, (1/(T-165))*W, W**2, (T/300)*W]
    return np.array([row])

def predict_all(hba, hbd, ratio, temp_K, water):
    X = build_row(hba, hbd, ratio, temp_K, water)
    preds = {}
    for target, (m_dry, m_wet, mlp_obj, meta_obj, log_t) in models.items():
        dry  = X[0, W_IDX] < 0.1
        p_xgb  = float(m_dry.predict(X)[0]) if dry else float(m_wet.predict(X)[0])
        p_alt  = float(m_wet.predict(X)[0]) if dry else float(m_dry.predict(X)[0])
        p_mlp  = float(mlp_obj['mlp'].predict(mlp_obj['scaler'].transform(X))[0])
        mX = np.array([[p_xgb, p_alt, p_mlp,
                        X[0, ALL_FEATURES.index('temperature_K')], X[0, W_IDX],
                        X[0, ALL_FEATURES.index('water_regime')],
                        X[0, ALL_FEATURES.index('vft_inv_T')],
                        X[0, ALL_FEATURES.index('vft_water_sq')]]])
        val = float(meta_obj['meta'].predict(meta_obj['meta_sc'].transform(mX))[0])
        val = float(np.expm1(val) if log_t else val)
        if target != 'reduction_potential_V_vs_SHE':
            val = max(0, val)
        preds[target] = round(val, 4)
    return preds

# ── Business logic helpers ─────────────────────────────────────────────────────
LI_SOURCES = {
    "LIB cathode (NMC/NCA)": {"bench_li":95,"bench_co":92,"bench_ni":90,"bench_mn":88},
    "LIB cathode (LFP)":     {"bench_li":97,"bench_co":0, "bench_ni":0, "bench_mn":0},
    "LIB cathode (LCO)":     {"bench_li":98,"bench_co":95,"bench_ni":0, "bench_mn":0},
    "LIB cathode (LMO)":     {"bench_li":93,"bench_co":0, "bench_ni":0, "bench_mn":91},
    "Mixed battery black mass":{"bench_li":88,"bench_co":82,"bench_ni":80,"bench_mn":78},
    "Li-ion pouch cell":     {"bench_li":91,"bench_co":85,"bench_ni":83,"bench_mn":81},
    "EV battery module":     {"bench_li":86,"bench_co":80,"bench_ni":78,"bench_mn":75},
}
OX_BOOST = {"None":0,"H2O2":9,"Citric acid":6,"Ascorbic acid":8,"FeSO4":5}
ASSIST_B = {"Conventional":0,"Microwave":8,"Ultrasound":6,"Microwave + Ultrasound":12}

def calc_li_recovery(preds, temp_K, source, oxidant, assist):
    visc   = preds['viscosity_cP']
    ph     = preds['pH_acidity']
    redox  = preds['reduction_potential_V_vs_SHE']
    li_sc  = preds.get('li_extraction_score', 60)
    bench  = LI_SOURCES[source]['bench_li']
    temp_C = temp_K - 273.15
    vf = max(0.70, 1 - max(0, visc - 50) / 3000)
    pf = 1.0 + max(0, (3 - ph) * 0.03) if ph < 5 else max(0.85, 1 - (ph-5)*0.04)
    rf = 1.0 + max(0, -redox - 0.9) * 0.04
    tf = 0.82 + min(0.18, (temp_C - 20) / 400)
    ox = OX_BOOST.get(oxidant, 0) / 100
    as_ = ASSIST_B.get(assist, 0) / 100
    eff = bench * vf * pf * rf * tf * (1+ox) * (1+as_)
    eff = 0.65 * eff + 0.35 * (li_sc/100 * bench)
    return round(min(99.5, max(20, eff)), 1)

def build_tips(preds, temp_K, source, oxidant, assist):
    tips = []
    visc=preds['viscosity_cP']; ph=preds['pH_acidity']
    redox=preds['reduction_potential_V_vs_SHE']; temp_C=temp_K-273.15
    if visc > 200:
        tips.append(f"High viscosity ({visc:.0f} cP) — add 10-20% water or raise temperature to improve Li+ mobility.")
    elif visc < 50:
        tips.append(f"Excellent viscosity ({visc:.0f} cP) — good Li+ mass transfer expected.")
    if ph > 4:
        tips.append(f"pH {ph:.1f} too high — switch to oxalic or malonic acid HBD for better Li oxide dissolution.")
    elif ph < 2.5:
        tips.append(f"Strongly acidic (pH {ph:.1f}) — will rapidly dissolve Li2CO3 and LiCoO2 phases.")
    if redox > -1.0:
        tips.append(f"Weak reduction potential ({redox:.2f} V) — add H2O2 or ascorbic acid to reduce Co3+ and release Li+.")
    elif redox < -1.3:
        tips.append(f"Strong reducing conditions ({redox:.2f} V) — excellent for Co3+ reduction and Li+ release.")
    if temp_C < 60:
        tips.append(f"Temperature {temp_C:.0f}C is low — raise to 60-90C for optimal Li leaching rate.")
    if oxidant == "None" and any(x in source for x in ["NMC","NCA","LCO"]):
        tips.append("Add H2O2 reductant — significantly boosts Li recovery from NMC/NCA/LCO by reducing transition metals.")
    if not tips:
        tips.append("Well-optimised conditions — parameters align with high-efficiency Li extraction benchmarks.")
    return tips[:4]

# ── Main prediction function (called by Gradio) ────────────────────────────────
def predict(hba, hbd, ratio, water, temp_c, source, oxidant, assist, show_all_hba, show_all_hbd):
    temp_K = temp_c + 273.15
    try:
        preds  = predict_all(hba, hbd, ratio, temp_K, water)
    except Exception as e:
        return (f"Error: {e}",)*10

    visc   = preds['viscosity_cP']
    dens   = preds['density_kg_m3']
    ph     = preds['pH_acidity']
    redox  = preds['reduction_potential_V_vs_SHE']
    li_sc  = preds.get('li_extraction_score', 0)
    li_rec = calc_li_recovery(preds, temp_K, source, oxidant, assist)
    bench  = LI_SOURCES[source]['bench_li']
    tips   = build_tips(preds, temp_K, source, oxidant, assist)

    is_li  = (hba in LI_HBAS and hbd in LI_HBDS)
    li_flag = "Li-optimised DES" if is_li else "Not Li-specific — consider ChCl:Oxalic or ChCl:Levulinic"
    grade  = "Excellent" if li_rec>=90 else ("Acceptable" if li_rec>=75 else "Needs improvement")

    # ── Property cards (HTML) ─────────────────────────────────────────────────
    v_col  = "#22c55e" if visc<100  else ("#f59e0b" if visc<300  else "#ef4444")
    p_col  = "#22c55e" if ph<3      else ("#f59e0b" if ph<6      else "#ef4444")
    r_col  = "#22c55e" if redox<-1.3 else ("#f59e0b" if redox<-1.0 else "#ef4444")
    l_col  = "#22c55e" if li_sc>=70  else ("#f59e0b" if li_sc>=50  else "#ef4444")
    g_col  = "#22c55e" if li_rec>=90 else ("#f59e0b" if li_rec>=75 else "#ef4444")
    delta  = li_rec - bench
    d_col  = "#22c55e" if delta>=0 else "#ef4444"
    d_sym  = "▲" if delta>=0 else "▼"

    cards_html = f"""
    <style>
      .des-grid {{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:8px 0}}
      .des-card {{background:#1e293b;border-radius:10px;padding:14px 16px;border:1px solid #334155}}
      .des-label {{font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px;text-transform:uppercase}}
      .des-value {{font-size:26px;font-weight:700}}
      .des-sub {{font-size:12px;color:#64748b;margin-top:3px}}
      .des-tip {{background:#0f172a;border-left:4px solid #4c6ef5;border-radius:8px;
                 padding:12px 16px;margin:6px 0;font-size:13px;color:#cbd5e1;line-height:1.7}}
      .des-banner {{background:#1e293b;border-radius:10px;padding:14px 20px;
                    display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
    </style>
    <div class="des-banner">
      <div>
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.06em">STATUS</div>
        <div style="font-size:15px;color:#f1f5f9;font-weight:600">{grade} &nbsp;·&nbsp; {li_flag}</div>
        <div style="font-size:12px;color:#64748b;margin-top:3px">Benchmark for {source}: {bench}%</div>
      </div>
      <div style="text-align:center;min-width:100px">
        <div style="font-size:38px;font-weight:700;color:{g_col}">{li_rec:.1f}%</div>
        <div style="font-size:12px;color:{d_col}">{d_sym} {abs(delta):.1f}% vs benchmark</div>
      </div>
    </div>
    <div class="des-grid">
      <div class="des-card">
        <div class="des-label">Li Suit. Score</div>
        <div class="des-value" style="color:{l_col}">{li_sc:.1f}<span style="font-size:14px;color:#64748b">/100</span></div>
        <div class="des-sub">{"Excellent" if li_sc>=70 else "Moderate" if li_sc>=50 else "Poor"}</div>
      </div>
      <div class="des-card">
        <div class="des-label">Viscosity</div>
        <div class="des-value" style="color:{v_col}">{visc:.0f}<span style="font-size:14px"> cP</span></div>
        <div class="des-sub">{"Low (good)" if visc<100 else "Moderate" if visc<300 else "High (limit mass transfer)"}</div>
      </div>
      <div class="des-card">
        <div class="des-label">Density</div>
        <div class="des-value" style="color:#f1f5f9">{dens:.1f}<span style="font-size:14px"> kg/m3</span></div>
        <div class="des-sub">R2 = 0.9994</div>
      </div>
      <div class="des-card">
        <div class="des-label">pH</div>
        <div class="des-value" style="color:{p_col}">{ph:.2f}</div>
        <div class="des-sub">{"Acidic (ideal)" if ph<3 else "Moderate" if ph<6 else "Basic (avoid)"}</div>
      </div>
      <div class="des-card">
        <div class="des-label">Redox Potential</div>
        <div class="des-value" style="color:{r_col}">{redox:.3f}<span style="font-size:14px"> V</span></div>
        <div class="des-sub">{"Strongly reducing" if redox<-1.3 else "Moderate" if redox<-1.0 else "Weak"}</div>
      </div>
      <div class="des-card">
        <div class="des-label">Model</div>
        <div style="font-size:13px;color:#60a5fa;font-weight:600;margin-top:4px">Ensemble v6</div>
        <div class="des-sub">Stacking · VFT · 35 features</div>
      </div>
    </div>
    <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.06em;
                margin:14px 0 6px">OPTIMISATION ADVICE</div>
    {''.join(f'<div class="des-tip">{t}</div>' for t in tips)}
    """

    # ── Gauge chart ───────────────────────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=li_rec,
        domain={'x':[0,1],'y':[0,1]},
        delta={'reference':bench,'valueformat':'.1f','suffix':'%',
               'font':{'size':16},'position':'bottom'},
        number={'suffix':'%','font':{'size':52,'color':'#f1f5f9','family':'Arial Black'}},
        title={'text':f"Li Recovery vs Benchmark ({bench}%)","font":{"size":13,"color":"#94a3b8"}},
        gauge={
            'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#64748b','tickfont':{'size':11}},
            'bar':{'color':'#4c6ef5','thickness':0.35},
            'bgcolor':'rgba(0,0,0,0)','borderwidth':0,
            'steps':[{'range':[0,60],'color':'#fee2e2'},
                     {'range':[60,75],'color':'#fef9c3'},
                     {'range':[75,100],'color':'#dcfce7'}],
            'threshold':{'line':{'color':'#ef4444','width':3},'thickness':0.8,'value':bench}
        }
    ))
    fig_gauge.update_layout(height=280,margin=dict(l=30,r=30,t=60,b=0),
                             paper_bgcolor='rgba(0,0,0,0)',font={'color':'#f1f5f9'})

    # ── Temperature sweep chart ───────────────────────────────────────────────
    temps_K = np.arange(278.15, 378.15, 5)
    visc_sw, ph_sw, li_sw = [], [], []
    for tK in temps_K:
        try:
            p2 = predict_all(hba, hbd, ratio, tK, water)
            visc_sw.append(p2['viscosity_cP'])
            ph_sw.append(p2['pH_acidity'])
            li_sw.append(p2.get('li_extraction_score', 0))
        except:
            visc_sw.append(0); ph_sw.append(0); li_sw.append(0)

    temps_C = temps_K - 273.15
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=temps_C, y=visc_sw, name='Viscosity (cP)',
        line=dict(color='#4c6ef5',width=2.5), yaxis='y'))
    fig_sweep.add_trace(go.Scatter(x=temps_C, y=li_sw, name='Li Score',
        line=dict(color='#37b24d',width=2.5), yaxis='y2'))
    fig_sweep.add_vline(x=temp_c, line_dash='dash', line_color='#ef4444',
                         annotation_text=f'{temp_c}C')
    fig_sweep.add_vrect(x0=60,x1=90,fillcolor="#22c55e",opacity=0.08,line_width=0,
                         annotation_text="Li optimal",annotation_position="top left")
    fig_sweep.update_layout(
        title="Property sweep vs Temperature",
        height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=60,t=40,b=10),
        xaxis_title="Temperature (C)",
        yaxis=dict(title="Viscosity (cP)", title_font=dict(color='#4c6ef5')),
        yaxis2=dict(title="Li Score", title_font=dict(color='#37b24d'),
                    overlaying='y', side='right', range=[0,100]),
        legend=dict(orientation='h', y=-0.2)
    )

    # ── Sensitivity bar chart ─────────────────────────────────────────────────
    def li_rec_for(**kw):
        h  = kw.get('hba',hba); hd=kw.get('hbd',hbd)
        r  = kw.get('ratio',ratio); tK=kw.get('temp_K',temp_K)
        w  = kw.get('water',water); ox=kw.get('ox',oxidant); a=kw.get('asst',assist)
        try:
            p2=predict_all(h,hd,r,tK,w)
            return calc_li_recovery(p2,tK,source,ox,a)
        except: return li_rec

    sens = {
        "Temp +20C":       li_rec_for(temp_K=temp_K+20) - li_rec,
        "Water +0.1":      li_rec_for(water=min(0.5,water+0.1)) - li_rec,
        "Add H2O2":        li_rec_for(ox="H2O2") - li_rec,
        "Add Microwave":   li_rec_for(asst="Microwave") - li_rec,
        "Ratio x2":        li_rec_for(ratio=ratio*2) - li_rec,
        "Temp -20C":       li_rec_for(temp_K=max(283.15,temp_K-20)) - li_rec,
        "Water -0.1":      li_rec_for(water=max(0,water-0.1)) - li_rec,
    }
    fig_sens = go.Figure(go.Bar(
        x=list(sens.values()), y=list(sens.keys()), orientation='h',
        marker_color=['#22c55e' if v>=0 else '#ef4444' for v in sens.values()],
        text=[f"{v:+.1f}%" for v in sens.values()], textposition='outside'
    ))
    fig_sens.update_layout(
        title="Sensitivity — Impact on Li Recovery",
        height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=80,t=40,b=10), xaxis_title="Delta Li Recovery (%)"
    )

    # ── Co-metal recovery bar chart ───────────────────────────────────────────
    src_d = LI_SOURCES[source]
    metals = {'Li':li_rec}
    if src_d['bench_co']>0: metals['Co']=round(src_d['bench_co']*0.95,1)
    if src_d['bench_ni']>0: metals['Ni']=round(src_d['bench_ni']*0.93,1)
    if src_d['bench_mn']>0: metals['Mn']=round(src_d['bench_mn']*0.90,1)
    fig_metals = go.Figure(go.Bar(
        x=list(metals.keys()), y=list(metals.values()),
        marker_color=['#4c6ef5','#37b24d','#f76707','#7950f2'][:len(metals)],
        text=[f"{v:.1f}%" for v in metals.values()], textposition='outside'
    ))
    fig_metals.update_layout(
        title=f"Co-metal Recovery Estimates — {source}",
        height=260, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=40,b=10),
        yaxis=dict(range=[0,110],ticksuffix='%')
    )

    return cards_html, fig_gauge, fig_sweep, fig_sens, fig_metals


# ── Gradio Interface ───────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #0f172a !important; color: #f1f5f9 !important; }
.gr-box, .gr-panel { background: #1e293b !important; border: 1px solid #334155 !important; }
.gr-button-primary { background: #4c6ef5 !important; border: none !important; }
.gr-button-secondary { background: #334155 !important; border: none !important; }
footer { display: none !important; }
.tab-nav button { font-size: 13px !important; }
h1, h2, h3, label { color: #f1f5f9 !important; }
"""

TITLE = """
<div style="text-align:center;padding:20px 0 10px">
  <div style="font-size:32px;font-weight:800;color:#f1f5f9">DES Lithium Recovery Predictor</div>
  <div style="font-size:14px;color:#94a3b8;margin-top:6px">
    Ensemble v6 &nbsp;·&nbsp; Stacking meta-learner &nbsp;·&nbsp;
    VFT temperature features &nbsp;·&nbsp; Physics-constrained &nbsp;·&nbsp;
    35 features &nbsp;·&nbsp; 5,790 experimental data points
  </div>
  <div style="margin-top:10px">
    <span style="background:#1e3a5f;color:#60a5fa;padding:4px 12px;border-radius:6px;font-size:12px;font-weight:600">
      XGBoost + MLP Stacking
    </span>
    &nbsp;
    <span style="background:#14532d;color:#4ade80;padding:4px 12px;border-radius:6px;font-size:12px;font-weight:600">
      Physics Constraints
    </span>
    &nbsp;
    <span style="background:#312e81;color:#a5b4fc;padding:4px 12px;border-radius:6px;font-size:12px;font-weight:600">
      VFT Thermal Model
    </span>
  </div>
</div>
"""

with gr.Blocks(css=CSS, title="DES Li Recovery Predictor") as demo:
    gr.HTML(TITLE)

    with gr.Tabs():

        # ── Tab 1: Predict ─────────────────────────────────────────────────────
        with gr.TabItem("Li Recovery Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### DES Composition")
                    show_all_hba = gr.Checkbox(label="Show all 144 HBAs", value=False)
                    show_all_hbd = gr.Checkbox(label="Show all 167 HBDs", value=False)
                    hba_dd = gr.Dropdown(choices=BEST_LI_HBAS, value='choline chloride',
                                         label="HBA (Hydrogen Bond Acceptor)")
                    hbd_dd = gr.Dropdown(choices=BEST_LI_HBDS, value='oxalic acid',
                                         label="HBD (Hydrogen Bond Donor)")
                    ratio_sl  = gr.Slider(0.05, 10.0, value=1.0, step=0.05, label="Molar ratio HBA:HBD")
                    water_sl  = gr.Slider(0.0, 0.50, value=0.0, step=0.01, label="Water content (mol fraction)")

                    gr.Markdown("### Process Conditions")
                    temp_sl   = gr.Slider(20, 120, value=70, step=5, label="Temperature (°C)")

                    gr.Markdown("### Li Recovery Parameters")
                    source_dd = gr.Dropdown(choices=list(LI_SOURCES.keys()),
                                            value="LIB cathode (NMC/NCA)", label="Battery source material")
                    oxidant_dd= gr.Dropdown(choices=list(OX_BOOST.keys()),
                                            value="None", label="Reductant / oxidant additive")
                    assist_dd = gr.Dropdown(choices=list(ASSIST_B.keys()),
                                            value="Conventional", label="Assist method")

                    run_btn   = gr.Button("Run Prediction", variant="primary", size="lg")

                with gr.Column(scale=2):
                    results_html = gr.HTML(label="Results")
                    gauge_plot   = gr.Plot(label="Li Recovery Gauge")
                    with gr.Row():
                        metals_plot = gr.Plot(label="Co-metal Recovery")
                        sens_plot   = gr.Plot(label="Sensitivity Analysis")

            sweep_plot = gr.Plot(label="Temperature Sweep")

            # Update HBA/HBD dropdowns when checkbox changes
            def toggle_hba(show_all):
                choices = ALL_HBAS if show_all else BEST_LI_HBAS
                return gr.Dropdown(choices=choices, value=choices[0] if choices else '')
            def toggle_hbd(show_all):
                choices = ALL_HBDS if show_all else BEST_LI_HBDS
                return gr.Dropdown(choices=choices, value=choices[0] if choices else '')

            show_all_hba.change(toggle_hba, inputs=show_all_hba, outputs=hba_dd)
            show_all_hbd.change(toggle_hbd, inputs=show_all_hbd, outputs=hbd_dd)

            inputs  = [hba_dd, hbd_dd, ratio_sl, water_sl, temp_sl,
                       source_dd, oxidant_dd, assist_dd, show_all_hba, show_all_hbd]
            outputs = [results_html, gauge_plot, sweep_plot, sens_plot, metals_plot]
            run_btn.click(predict, inputs=inputs, outputs=outputs)

        # ── Tab 2: Model Info ──────────────────────────────────────────────────
        with gr.TabItem("Model Info"):
            gr.Markdown("""
## Ensemble v6 Architecture

| Property | v2 R² | v5 R² | **v6 R²** | MAE v6 | Train N |
|---|---|---|---|---|---|
| Viscosity (cP) | 0.896 | 0.922 | **0.918** | 271 cP | 4,920 |
| Density (kg/m³) | 0.995 | 0.993 | **0.9994** | 0.32 kg/m³ | 1,264 |
| pH | 0.989 | 0.9998 | **0.9992** | 0.012 | 1,005 |
| Redox (V) | 0.973 | 0.9945 | **0.9946** | 0.005 V | 1,737 |
| Li score | 0.868 | 0.833 | **0.849** | 1.86% | 878 |

### Improvements across versions

**v6 — Stacking Ensemble with Meta-Learner:**
- 5-fold OOF: XGBoost_dry + XGBoost_wet + MLP predictions stacked
- Meta-learner (XGBoost or Ridge) learns optimal blend per situation
- Density R²: 0.993 → **0.9994** · MAE: 1.97 → **0.32 kg/m³** (−84%)
- Li score R²: 0.833 → **0.849** (+1.64%)

**v5 — VFT Temperature Features:**
- 11 features: 1/T, ln(T), 1/(T−T₀) for T₀ ∈ {150,165,180,200} K, plus interaction terms
- Encodes the Vogel-Fulcher-Tammann equation: ln(η) = A + B/(T−T₀)

**v4 — Physics-Constrained XGBoost:**
- Monotonicity constraints enforce: viscosity decreases with temperature & water;
  density decreases with temperature; pH increases with water; redox responds to temperature

**v3 — Molecular Descriptors:**
- 16 RDKit descriptors per compound: MW, logP, HBD, HBA, TPSA, rings, rotatable bonds, heavy atoms
- 24 total features (was 7)

### Dataset
- **Source:** Moradi & Bougie (2026), J. Mol. Liq. 443, 128903
- **5,790 rows** · 144 unique HBAs · 167 unique HBDs
- **Temperature range:** 278–378 K (5–105°C)
- **Water content:** 0.0–0.98 mol fraction
            """)

        # ── Tab 3: DES Recommendations ─────────────────────────────────────────
        with gr.TabItem("Li Extraction Guide"):
            gr.Markdown("""
## Recommended DES Systems for Li Extraction

### Top DES by Battery Chemistry

| Battery | Best DES | Temperature | Additive | Li Yield |
|---|---|---|---|---|
| NMC (LiNiMnCoO2) | ChCl : Oxalic acid (1:1) | 80°C | H2O2 2 vol% | 97% |
| NCA (LiNiCoAlO2) | ChCl : Levulinic acid (1:2) | 80°C | Ascorbic acid 5 wt% | 96% |
| LCO (LiCoO2) | ChCl : Malonic acid (1:1) | 70°C | Ascorbic acid 5 wt% | 98% |
| LFP (LiFePO4) | ChCl : Lactic acid (1:2) | 60°C | None required | 97% |
| LMO (LiMn2O4) | ChCl : Tartaric acid (1:1) | 80°C | H2O2 1 vol% | 93% |

### Optimal Operating Window

| Parameter | Optimal Range | Why it matters |
|---|---|---|
| Viscosity | < 200 cP | Li+ ion mobility |
| pH | 1.5 – 3.5 | Dissolves Li2CO3, LiCoO2 |
| Redox potential | < −1.0 V | Reduces Co3+→Co2+, releases Li+ |
| Temperature | 60 – 90°C | Kinetics vs DES stability |
| Water content | 0.0 – 0.2 | Preserves H-bond network |
| Li score | > 65 / 100 | Combined suitability index |

### Troubleshooting

**Recovery < 75%:** Check viscosity first (>300 cP = mass transfer bottleneck). If pH > 4,
switch to oxalic or malonic acid HBD. Ensure cathode is pre-treated (calcination 500°C removes PVDF binder).

**Recovery 75–88%:** Add H2O2 (2 vol%) — typically adds 5–8% yield by reducing Co3+.
Extend leaching time from 60 to 120 min (+3–5%). Try microwave assist.

**Maximising >95%:** Combine calcination + sieving (<75 µm) + microwave + H2O2.
Use solid:liquid ratio 1:20 g/mL.
            """)

    # Run prediction on page load with defaults
    demo.load(
        predict,
        inputs=[hba_dd, hbd_dd, ratio_sl, water_sl, temp_sl,
                source_dd, oxidant_dd, assist_dd, show_all_hba, show_all_hbd],
        outputs=[results_html, gauge_plot, sweep_plot, sens_plot, metals_plot]
    )

if __name__ == "__main__":
    demo.launch()
