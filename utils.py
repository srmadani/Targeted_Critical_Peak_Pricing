from home import Homes
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
import gurobipy as gp
from gurobipy import GRB
import seaborn as sns
from itertools import chain, combinations
import plotly.express as px
from itertools import product


def figs(tar = False, option = 'WCO', rate = 99, acts= {}, save = True, show_df = True):
    env = Homes(tar, option, rate)
    col = 'Day, Hour, Load, Returned_Load'.split(', ')
    df = pd.DataFrame(columns=col)
    # df['Returned_Load'] *= 10000/(1+rate)
    # df['Load'] *= 10000/(1+rate)
    done = False
    rws, lds, b40s, svs = [], [], [], []
    # _ = env.reset(mode = 'test', scale = False)
    _ = env.reset(mode = 'test', scale = False, dataset_mode = True, year = 2017, month = 2)
    year, month = env.year, env.month
    score_ = 0
    LL_ = []
    while not done:
        _, reward, done, _ = env.step(0)
        score_ += reward
        LL_.append(env.returned_load)
    # _ = env.reset(mode = 'test', scale = False)
    _ = env.reset(mode = 'test', scale = False, dataset_mode = True, year = 2017, month = 2)
    score, done = 0, False
    while not done:
        if env.day in acts:
            action = acts[env.day]
        else:
            action = 0
        _, reward, done, _ = env.step(action)
        score += reward
        df1 = pd.DataFrame(columns=col)
        df1['Day'] = env.day * np.ones(24, dtype=int)
        df1['Load'] = env.LL * env.rate + LL_[env.day-2]
        rws.append(reward)
        lds.append(env.returned_load)
        b40s.append(env.beyound40)
        svs.append(env.saving)
        df1['Hour'] = list(range(24))
        df1['Returned_Load'] = env.returned_load + env.rate * env.LL
        df = pd.concat([df,df1], ignore_index=True)
    if month == 12:
        date_rng = pd.date_range(start=f'{month}/1/{year}', end=f'{1}/1/{year+1}', freq='H')
    else:
        date_rng = pd.date_range(start=f'{month}/1/{year}', end=f'{month+1}/1/{year}', freq='H')
    df['DateHour'] = date_rng[:-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['DateHour'], y=df['Load'] * 10/(1+rate), name = 'Load (MWh)', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df['DateHour'], y=df['Returned_Load'] * 10/(1+rate), name = 'New Load (MWh)', line=dict(color='#d62728')))
    fig.add_trace(go.Scatter(x=df['DateHour'], y=[df['Load'].max() * 10/(1+rate)]*len(df['DateHour']), name = 'peak load (MW)', line=dict(dash='dash', color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df['DateHour'], y=[df['Returned_Load'].max() * 10/(1+rate)]*len(df['DateHour']), name = 'new peak (MW)', line=dict(dash='dash', color='#d62728')))
    
    # fig.update_layout(title = f'Prosumer % in market: {np.round(100/(1+rate),3)}%, option: {option}, targeted: {tar}',hovermode='x unified')
    for ii in acts:
        if acts[ii] == 1: #pink
            x0 = f'{year}-{month}-{ii} 6:00:00'
            x1 = f'{year}-{month}-{ii} 9:00:00'
            fig.add_vrect(x0=x0, x1=x1,fillcolor='#f0027f', opacity=0.2,layer="below", line_width=0)
        if acts[ii] == 2: #green
            x0 = f'{year}-{month}-{ii} 16:00:00'
            x1 = f'{year}-{month}-{ii} 20:00:00'
            fig.add_vrect(x0=x0, x1=x1,fillcolor='#2ca02c', opacity=0.2,layer="below", line_width=0)
        if acts[ii] == 3:
            x0 = f'{year}-{month}-{ii} 6:00:00'
            x1 = f'{year}-{month}-{ii} 9:00:00'
            fig.add_vrect(x0=x0, x1=x1,fillcolor='#f0027f', opacity=0.2,layer="below", line_width=0)
            x0 = f'{year}-{month}-{ii} 16:00:00'
            x1 = f'{year}-{month}-{ii} 20:00:00'
            fig.add_vrect(x0=x0, x1=x1,fillcolor='#2ca02c', opacity=0.2,layer="below", line_width=0)
    fig.update_layout(width=1200, height=600)
    fig.update_layout(
        legend=dict(
            x=0.8,  # x-position of the legend (change as needed)
            y=0.8,  # y-position of the legend (change as needed)
            xanchor='left',  # anchor point of the legend
            yanchor='top',
        )
    )
    # fig.show()
    fig.show("notebook")
    # print(f'rate: {np.round(100/(1+rate),3)}%')
    # print(f'SQ profit: {score_}')
    # print(f'new profit: {score}')
    # print(f'profit increase: {score - score_}')
    # print(f'acts: {acts}')
    # print(f'improve: {np.round(100 * (score-score_)/score_,2)}')
    data = {
        'Metric': ['rate', 'SQ profit', 'new profit', 'profit increase', 'acts', 'improve', 'peak load saving (kWh)'],
        'Value': [
            f'{np.round(100/(1+rate), 3)}%', score_, score, score - score_, acts,
            f'{np.round(100 * (score - score_)/np.abs(score_), 2)}%', (df['Load'].max()-df['Returned_Load'].max())  * 10000 / (1+rate)
        ]
    }

    df = pd.DataFrame(data)
    if save:
        fig.write_image(f"figures/{option}_{tar}_{np.round(100/(1+rate), 3)}.png", width=1200, height=600, scale=2)
    if show_df:
        print(df)


def prices():
    # Set the year and month
    year, month = 2017, 2
    # tar and option variables are part of your environment setup
    tar = False
    option = 'WCO'
    env = Homes(tar, option)

    # Querying the DataFrame for the specific year and month
    tmp = env.df.query(f'Year == {year} and Month == {month}')

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plotting the data
    ax.plot(tmp['DateTime'], 1.3*tmp['P'], linestyle='', marker='o', markersize=2, label='HQ Wholesale Price ($/kwh)')
    ax.axhline(y=env.d["PR"]['WCO'], linestyle='--', color='#ff7f0e', label='Winter Credit Option ($/kwh)')
    ax.axhline(y=env.d["PR"]['FXD'], linestyle='--', color='#d62728', label='Flex D ($/kwh)')
    ax.axhline(y=0.18035*1.3, linestyle='--', color='#7f7f7f', label='Basic Rate in Vermont ($/kwh)')

    # Set labels and title (if needed)
    ax.set_xlabel('DateTime')
    ax.set_ylabel('Price ($/kwh)')
    # ax.grid(which='major', linestyle='--', linewidth='0.5', color='grey')

    # ax.grid(True, linestyle='--', alpha=1, color='k')  # Grid lines are now black
    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
    fig.savefig("figures/prices_2017_Feb.png", dpi=300)


def model(L, R, E_max, E_min, A, V, B_max, B_min, D_max, d, sq, type, peaks, report):
    episodes = {'peak': peaks, 'off-peak': [i for i in range(24) if i not in peaks]}
    m = gp.Model('home_management')
    m.Params.LogToConsole = 0
    # Decision Vars
    x_GL   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^GL')
    x_GB   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^GB')
    x_GE   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^GE')
    x_RL   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^RL')
    x_RB   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^RB')
    x_RE   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^RE')
    x_RG   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^RG')
    x_BL   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^BL')
    x_EL   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^EL')
    x_BG   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^BG')
    x_EG   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^EG')
    x_D    = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^D')
    b_E    = m.addVars(d['T']+1, vtype=GRB.CONTINUOUS, name='b^E')
    b_B    = m.addVars(d['T']+1, vtype=GRB.CONTINUOUS, name='b^B')
    x_EG   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^EG')
    x_G    = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^G', lb=-GRB.INFINITY)
    x_SQ   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^SQ')
    x_SQ1   = m.addVars(d['T'], vtype=GRB.CONTINUOUS, name='x^SQ1', lb=-GRB.INFINITY)
    x_Lpos   = m.addVar(vtype=GRB.CONTINUOUS, name='x^Lpos')
    x_Lpos1   = m.addVar(vtype=GRB.CONTINUOUS, name='x^Lpos1', lb=-GRB.INFINITY)
    # Constraints
    m.addConstrs((x_GL[t] + x_GB[t] +  x_GE[t] - 0.999 * (x_RG[t] + d['eta_d'] * (x_BG[t] + x_EG[t])) == x_G[t] for t in range(d['T'])), name='x_G')
    m.addConstrs((x_GL[t] + d['eta_d'] * x_BL[t] + d['eta_d'] * x_EL[t] + x_RL[t] + x_D[t] == L[t] for t in range(d['T'])), name='c_load')
    m.addConstrs((x_D[t] <= D_max * L[t] for t in range(d['T'])), name='c_D_max')
    m.addConstrs((x_SQ1[t] == sq[t] - x_G[t] for t in range(d['T'])), name='c_x_SQ1')
    m.addConstrs((x_SQ[t] == gp.max_([x_SQ1[t],0.0])for t in range(d['T'])), name='c_x_SQ')
    m.addConstr((x_Lpos1 == gp.quicksum(x_G[t] for t in range(d['T']))-d['LL']), name='c_x_Lpos1')
    m.addConstr((x_Lpos == gp.max_([x_Lpos1,0.0])), name='c_Lpos')
    m.addConstr((b_E[0] == E_min ), name='c_init_EV')
    m.addConstr((b_B[0] == B_min ), name='c_init_Bat')
    m.addConstrs((x_BL[t] + x_BG[t] <= b_B[t] for t in range(d['T'])), name='c_bat_state')
    m.addConstrs((x_EL[t] + x_EG[t] <= b_E[t] * A[t] for t in range(d['T'])), name='c_ev_state')
    m.addConstrs((x_RE[t] + x_GE[t] <= E_max * A[t] for t in range(d['T'])), name='c_ev_state_2')
    m.addConstrs((x_BL[t] + x_BG[t] <= d['U_d'] for t in range(d['T'])), name='c_b_dch_max')
    m.addConstrs((x_EL[t] + x_EG[t] <= d['U_d'] for t in range(d['T'])), name='c_e_dch_max')
    m.addConstrs((x_GB[t] + x_RB[t] <= d['U_c'] for t in range(d['T'])), name='c_b_ch_max')
    m.addConstrs((x_GE[t] + x_RE[t] <= d['U_c'] for t in range(d['T'])), name='c_e_ch_max')
    m.addConstrs((b_B[t] <= B_max for t in range(d['T']+1)), name='c_b_B_max')
    m.addConstrs((b_B[t] >= B_min for t in range(d['T']+1)), name='c_b_B_min')
    m.addConstrs((b_E[t] <= E_max for t in range(d['T']+1)), name='c_b_E_max')
    m.addConstrs((b_E[t] >= E_min for t in range(d['T']+1)), name='c_b_E_min')
    m.addConstrs((x_RL[t] + x_RB[t] + x_RE[t] + x_RG[t] <= R[t] for t in range(d['T'])), name='c_gen_cap')
    m.addConstrs((b_B[t] + d['eta_c'] * (x_GB[t] + x_RB[t]) - (x_BL[t] + x_BG[t]) == b_B[t+1] for t in range(d['T'])), name='c_next_state_b_B')
    m.addConstrs((b_E[t] + d['eta_c'] * (x_GE[t] + x_RE[t]) - (x_EL[t] + x_EG[t]) - (V[t] / d['eta_d']) == b_E[t+1] for t in range(d['T'])), name='c_next_state_b_E')
    # Objective
    if type == 'WCO':
        m.setObjective(gp.quicksum(d["PR"]['WCO'] * x_G[t] + d['alpha'] * x_D[t] * x_D[t] for t in range(d['T'])) - gp.quicksum(d['PP'] * x_SQ[t] for t in episodes['peak']) - d["PR"]['WCO'] * (b_B[d['T']] + b_E[d['T']]) + (d["PH"]['WCO']-d["PR"]['WCO'])*x_Lpos , GRB.MINIMIZE)
    else:
        m.setObjective(gp.quicksum(d["PR"]['FXD'] * x_G[t] for t in episodes['off-peak']) + gp.quicksum(d['PP'] * x_G[t] for t in episodes['peak']) + gp.quicksum(d['alpha'] * x_D[t] * x_D[t] for t in range(d['T'])) - d["PR"]['FXD'] * (b_B[d['T']] + b_E[d['T']]) + (d["PH"]['FXD']-d["PR"]['FXD'])*(gp.quicksum(x_G[t] for t in range(d['T']))-d['LL']), GRB.MINIMIZE)

    # Optimize
    m.optimize()
    sol = [x_GL[i].x+x_GB[i].x+x_GE[i].x-x_RG[i].x-x_BG[i].x-x_EG[i].x for i in range(d['T'])]
    if report:
        clmn = 'x_GL,x_GB,x_GE,x_RL,x_RB,x_RE,x_RG,x_BL,x_EL,x_BG,x_EG,x_D,x_G,b_E,b_B,L,SQ,Peak Reduction,CPP,COST'.split(',')
        df = pd.DataFrame(data = np.zeros((25, len(clmn))), columns = clmn)
        df.loc[0:23,'x_GL'] = [x_GL[i].x for i in range(24)]
        df.loc[0:23,'x_GB'] = [x_GB[i].x for i in range(24)]
        df.loc[0:23,'x_GE'] = [x_GE[i].x for i in range(24)]
        df.loc[0:23,'x_RL'] = [x_RL[i].x for i in range(24)]
        df.loc[0:23,'x_RB'] = [x_RB[i].x for i in range(24)]
        df.loc[0:23,'x_RE'] = [x_RE[i].x for i in range(24)]
        df.loc[0:23,'x_RG'] = [x_RG[i].x for i in range(24)]
        df.loc[0:23,'x_BL'] = [x_BL[i].x for i in range(24)]
        df.loc[0:23,'x_EL'] = [x_EL[i].x for i in range(24)]
        df.loc[0:23,'x_BG'] = [x_BG[i].x for i in range(24)]
        df.loc[0:23,'x_EG'] = [x_EG[i].x for i in range(24)]
        df.loc[0:23,'x_D']  = [x_D[i].x for i in range(24)]
        df.loc[0:23,'x_G']  = [x_G[i].x for i in range(24)]
        df.loc[0:24,'b_B']  = [b_B[i].x for i in range(25)]
        df.loc[0:24,'b_E']  = [b_E[i].x for i in range(25)]
        df.loc[:23, 'L']    = L
        df.loc[:23, 'CPP'] = [sq[i] - x_G[i].x for i in range(d['T'])]
        df.loc[:23, 'SQ'] = sq
        df.loc[:23, 'Peak Reduction'] = [x_SQ[i].x for i in range(2)]
        return df, sol
    else:
        return sol

def rev_increase():
    # Data for 'improve' and 'profit increase'
    improve  = [0.10, 1.04, 2.08, 3.12, 5.20, 5.05, 4.84, 5.04, 3.12, 1.79, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    improve_ = [0.09, 0.93, 1.86, 2.80, 4.66, 5.17, 4.79, 4.40, 3.66, 1.49, -3.66, -6.09, -81.17, -99.85, -348.35, -2641.40]
    profit_increase  = [2331, 23313, 46626, 69939, 116565, 113143, 108434, 112987, 69891, 40199, 0, 0, 0, 0, 0, 0]
    profit_increase_ = [2089, 20886, 41772, 62657, 104429, 115797, 107261, 98725, 82180, 33473, -82141, -136902, -1834168, -2236646, -6365381, -12583495]
    rates = [0.01, 0.10, 0.20, 0.30, 0.50, 0.80, 0.90, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00, 20.00, 50.00, 100.00]

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(15, 8), dpi=150)  # Example values

    # Plotting 'improve' on the left y-axis
    ax1.plot(rates, improve, 'o--', color='blue', label='Profit Increase (%)')
    ax1.set_xlabel('PPP (%)')
    ax1.set_ylabel('Profit Increase (%)', color='blue')
    # ax1.set_yscale('symlog')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper right')

    # Adding a right y-axis for 'profit increase'
    ax2 = ax1.twinx()
    ax2.plot(rates, profit_increase, 's--', color='red', label='Profit Increase ($)')
    ax2.set_ylabel('Profit Increase ($)', color='red')
    ax2.set_yscale('symlog')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.95))

    # Adding a title and grid
    plt.title('Comparison of Profit Increase in Percentage and Dollars, Option: WCO, Targeted: False')
    plt.grid(True)

    # Highlighting the smaller changes (for the first few rates)
    # Creating an inset with a zoomed view
    ax_inset = fig.add_axes([0.5, 0.3, 0.35, 0.35])  # Inset position and size
    ax_inset.plot(rates[:10], improve[:10], 'o-', color='green')
    ax_inset.set_xlabel('PPP (%)')
    ax_inset.set_ylabel('Profit Increase (%)')
    ax_inset.set_title('Zoomed View for Smaller Rates')
    ax_inset.grid(True)

    # Adding a title and grid to the main plot
    plt.title('Profit Increase Percentage at Low Rates, Option: WCO, Targeted: False')
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig("figures/WCO_False_rev_increase.png", dpi=150)  # Example filename and DPI
    plt.show()

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(15, 8), dpi=150)  # Example values

    # Plotting 'improve' on the left y-axis
    ax1.plot(rates, improve_, 'o--', color='blue', label='Profit Increase (%)')
    ax1.set_xlabel('PPP (%)')
    ax1.set_ylabel('Profit Increase (%)', color='blue')
    ax1.set_yscale('symlog')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper right')

    # Adding a right y-axis for 'profit increase'
    ax2 = ax1.twinx()
    ax2.plot(rates, profit_increase_, 's--', color='red', label='Profit Increase ($)')
    ax2.set_ylabel('Profit Increase ($)', color='red')
    ax2.set_yscale('symlog')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.95))

    # Adding a title and grid
    plt.title('Comparison of Profit Increase in Percentage and Dollars, Option: FXD, Targeted: False')
    plt.grid(True)

    # Highlighting the smaller changes (for the first few rates)
    # Creating an inset with a zoomed view
    ax_inset = fig.add_axes([0.5, 0.3, 0.35, 0.35])  # Inset position and size
    ax_inset.plot(rates[:10], improve_[:10], 'o-', color='green')
    ax_inset.set_xlabel('PPP (%)')
    ax_inset.set_ylabel('Profit Increase (%)')
    ax_inset.set_title('Zoomed View for Smaller Rates')
    ax_inset.grid(True)

    # Adding a title and grid to the main plot
    plt.title('Profit Increase Percentage at Low Rates')
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig("figures/FXD_False_rev_increase.png", dpi=150)  # Example filename and DPI
    plt.show()

def pro_savings():

    tar, option, rate = False, 'WCO', 99
    acts = {10:3, 7:2, 1:1}
    
    env = Homes(tar, option, rate)
    _ = env.reset(mode = 'test', scale = False)
    product_combinations  = product([False, True], repeat=4)
    pro_wco = {(PV, EV, BAT, DR): 0 for PV, EV, BAT, DR in product_combinations }
    product_combinations  = product([False, True], repeat=4)
    pro_wco_sq = {(PV, EV, BAT, DR): 0 for PV, EV, BAT, DR in product_combinations }
    product_combinations  = product([False, True], repeat=4)
    pro_wco_diff = {(PV, EV, BAT, DR): 0 for PV, EV, BAT, DR in product_combinations }
    for PV,EV,BAT,DR in product([False, True], [False, True], [False, True], [False, True]):
        for day in range(1, 29):
            if day in acts:
                action = acts[day]
            else:
                action = 0
            q = f"Day == {day} and PV == {PV} and EV == {EV} and BAT == {BAT} and DR == {DR}"
            LL = env.data.query(q)[f"{'WCO'}{action}"].sum()
            LL_ = env.data.query(q)[f"{'WCO'}{0}"].sum()
            pro_wco[PV, EV, BAT, DR] += LL * env.d["PR"]['WCO'] + max(0, LL - 40) * env.d["PH"]['WCO']
            pro_wco_sq[PV, EV, BAT, DR] += LL_ * env.d["PR"]['WCO'] + max(0, LL_ - 40) * env.d["PH"]['WCO']
            if EV:
                pro_wco_sq[PV, EV, BAT, DR] -= 13.225 * env.d["PR"]['WCO'] 
                pro_wco[PV, EV, BAT, DR] -= 13.225 * env.d["PR"]['WCO'] 
            if action != 0:
                pro_wco[PV, EV, BAT, DR] -= env.data.query(q)[f"Saving{action}"].sum() * env.d["PP"]
        pro_wco_diff[PV, EV, BAT, DR] = pro_wco_sq[PV, EV, BAT, DR] - pro_wco[PV, EV, BAT, DR]
    
    env = Homes(tar= False, option= 'WCO', rate = 99)
    done, score, _, rews = False, 0, env.reset(mode='test', scale = False), []
    df = env.data.query('Day == 20')[['DateTime', 'PV', 'EV', 'BAT', 'DR', 'Saving1', 'Saving2', 'Saving3']].copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    savings = {}
    for PV_,EV_,BAT_,DR_ in product([False, True], [False, True], [False, True], [False, True]):
        savings[PV_, EV_, BAT_, DR_] = df.query(f'PV == {PV_} & EV == {EV_} & BAT == {BAT_} & DR == {DR_}')['Saving3'].max()
    savings  

    # Create adjusted labels and values for the pie chart
    adjusted_labels = []
    adjusted_values = []

    for key, value in pro_wco_diff.items():
        label = '&'.join([feat for feat, present in zip(['PV', 'EV', 'BAT', 'DR'], key) if present]) or 'None'
        adjusted_labels.append(label)
        adjusted_values.append(value)

    # Creating a Plotly pie chart with the adjusted data
    fig = px.pie(names=adjusted_labels, values=np.round(adjusted_values, 2), hole=0.3)
    # fig = px.pie(names=adjusted_labels, values=np.round(adjusted_values, 2), title='Saving ($) for each profile, Option: WCO, Targeted: False',hole=0.3)

    # Update the layout to put labels inside the chart and set the figure size
    fig.update_traces(textposition='inside', textinfo='percent+label+value')
    fig.update_layout(width=800, height=800)

    # Special handling for 'DR' label
    for d in fig.data:
        d['outsidetextfont'] = {'size': 14, 'color': 'black'}
        d['textposition'] = ['outside' if label in ['PV', 'PV&DR', 'DR'] else 'inside' for label in adjusted_labels]

    # Show the figure
    fig.show()
    fig.write_image("figures/WCO_False_prosumer_saving.png")

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(list(savings.items()), columns=['Combination', 'Saving'])
    df[['PV', 'EV', 'BAT', 'DR']] = pd.DataFrame(df['Combination'].tolist(), index=df.index)
    df = df.drop(columns=['Combination'])

    # Function to get all combinations of a list
    def all_combinations(lst):
        return list(chain(*map(lambda x: combinations(lst, x), range(1, len(lst)+1))))

    # Create interaction terms for all combinations of technologies
    for combo in all_combinations(['PV', 'EV', 'BAT', 'DR']):
        df['&'.join(combo)] = df[list(combo)].all(axis=1)

    # Melt the DataFrame for easier plotting
    df_melted = df.melt(id_vars=['Saving'], var_name='Technology', value_name='Present')

    # Create the combined bar plot
    plt.figure(figsize=(15, 6))
    sns.barplot(x='Technology', y='Saving', hue='Present', data=df_melted)
    # plt.title('Contribution of Technology Combinations on Peak Load Reduction, Option: WCO, Targeted: False')
    plt.ylabel('Average Saving (kWh)')
    plt.xlabel('Profiles')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure before showing it
    plt.savefig("figures/WCO_False_tech_comb.png", dpi=300)

    # Show the figure
    plt.show()



    tar, option, rate = False, 'FXD', 99
    acts = {10:3, 7:2, 1:1}
    
    env = Homes(tar, option, rate)
    _ = env.reset(mode = 'test', scale = False)
    product_combinations  = product([False, True], repeat=4)
    pro_fxd = {(PV, EV, BAT, DR): 0 for PV, EV, BAT, DR in product_combinations }
    for PV,EV,BAT,DR in product([False, True], [False, True], [False, True], [False, True]):
        for day in range(1, 29):
            if day in acts:
                action = acts[day]
            else:
                action = 0
            q = f"Day == {day} and PV == {PV} and EV == {EV} and BAT == {BAT} and DR == {DR}"
            LL = env.data.query(q)[f"{'FXD'}{action}"]
            LL = LL.reset_index(drop=True)
            mask = pd.Series([True] * len(LL))
            mask[6:9] = False
            mask[16:20] = False

            if all([not PV, not EV, not BAT, not DR]):
                pro_fxd[PV, EV, BAT, DR] += LL.sum() * env.d["PR"]['WCO'] + max(0, LL.sum() - 40) * env.d["PH"]['WCO']
            elif action != 0:
                pro_fxd[PV, EV, BAT, DR] += LL[mask].sum() * env.d["PR"]['FXD'] + np.max([0, LL[mask].sum() - 40]) * env.d["PH"]['FXD'] + (LL[6:9].sum() + LL[16:20].sum()) * env.d['PP']
            else:
                pro_fxd[PV, EV, BAT, DR] += LL.sum() * env.d["PR"]['FXD'] + np.max([0, LL.sum() - 40]) * env.d["PH"]['FXD']
            if EV:
                pro_fxd[PV, EV, BAT, DR] -= 13.225 * env.d["PR"]['FXD']


    # Feature names
    features = ["PV", "EV", "BAT", "DR"]

    # Function to create labels for x-axis
    def create_labels(combinations):
        labels = []
        for comb in combinations:
            label = "&".join([f for f, b in zip(features, comb) if b])
            labels.append(label if label else "NONE")
        return labels

    # Create labels
    x_labels = create_labels(pro_wco.keys())

    # Plotting
    plt.figure(figsize=(15, 5))

    # Line for pro_wco
    plt.plot(x_labels, list(pro_wco.values()), label='WCO', marker='o', color = '#FF7F0E')

    # Line for pro_fxd
    plt.plot(x_labels, list(pro_fxd.values()), label='FXD', marker='s', color = '#1F77B4')

    plt.plot(x_labels, [list(pro_fxd.values())[0]] * len(x_labels), label='SQ', linestyle='--', color = 'green', marker='x',)

    plt.xlabel('Profiles')
    plt.ylabel('Monthly bill ($)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("figures/pro_bills.png", dpi=300)
    plt.show()
    

def profits():

    # Rates and profits for WCO and FXD
    rates = [0.01, 0.10, 0.20, 0.30, 0.50, 0.80, 0.90, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00, 20.00, 50.00, 100.00]

    wco_sq_profit = [2239743, 2239923, 2240123, 2240323, 2240723, 2241323, 2241523, 2241723, 2242724, 2243724, 2245722, 2249726, 2259729, 2239928, 1827270, 476396]
    wco_new_profit = [2242074, 2263236, 2286749, 2310262, 2357288, 2354466, 2349957, 2354710, 2312614, 2283923, 2245722, 2249726, 2259729, 2239928, 1827270, 476396]

    # fxd_sq_profit = [2239448.698, 223698.1377, 111711.9955, 74383.27391, 44520.31011, 27722.38877, 24611.66011, 22123.08166, 14657.32392, 10924.46744, 7191.588565, 3985.840925, 1572.86808, 345.853738, -403.666516, -653.5066]
    fxd_new_profit = [2241831, 2260809, 2281895, 2302980, 2345152, 2357120, 2348784, 2340448, 2324904, 2277197, 2163581, 2112824, 425561, 3282, -4538111, -12107100]

    zoomed_rates = rates[:10]  # Selecting rates from 0.01% to 2%

    # Calculating the difference between FXD new profit and WCO new profit
    diff_new_profit = np.array(fxd_new_profit) - np.array(wco_new_profit)

    # Adjusting the inset figure to only show the difference
    fig, ax = plt.subplots(figsize=(12, 8))
    ax_inset = fig.add_axes([0.2, 0.4, 0.35, 0.35])  # Inset axes: [x, y, width, height] in figure coordinate

    # Plot data on main axes with SQ profits as dashed lines
    ax.plot(rates, wco_sq_profit, label='SQ Profit', marker='x', color = 'green')
    ax.plot(rates, wco_new_profit, label='WCO New Profit', marker='o', linestyle='--', color = '#FF7F0E')
    ax.plot(rates, fxd_new_profit, label='FXD New Profit', marker='s', linestyle='--', color = '#1F77B4')

    # Setup main axes
    ax.set_title('SQ Profit and New Profit for WCO and FXD')
    ax.set_xlabel('PPP (%)')
    ax.set_ylabel('Profit')
    ax.set_xscale('log')
    ax.set_yscale('symlog')
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{rate}" for rate in rates], rotation=45)
    ax.set_ylim([-1e8, 1e7])
    ax.grid(True)
    ax.legend()


    # # Plot the difference data on inset axes and highlight positive and negative values
    # for i in range(len(zoomed_rates) - 1):
    #     if diff_new_profit[i] >= 0:
    #         ax_inset.scatter(zoomed_rates[i], diff_new_profit[i], marker='o', color='cyan')
    #     else:
    #         ax_inset.scatter(zoomed_rates[i], diff_new_profit[i], marker='o', color='purple')
    
    ax_inset.plot(rates[:10], wco_new_profit[:10], marker='o', color = '#FF7F0E', linestyle='--')
    ax_inset.plot(rates[:10], fxd_new_profit[:10], marker='s', color = '#1F77B4', linestyle='--')
    ax_inset.plot(rates[:10], wco_sq_profit[:10], marker='x', color = 'green')

    # Setup inset axes
    # ax_inset.set_title('FXD-WCO Profit')
    ax_inset.set_xscale('log')
    ax_inset.set_yscale('symlog')
    ax_inset.set_xticks(zoomed_rates)
    ax_inset.set_xticklabels([f"{rate}" for rate in zoomed_rates], rotation=45)
    ax_inset.set_ylim([2.22e6, 2.38e6])
    ax_inset.grid(True)
    fig.savefig("figures/sq_new_proftis.png", dpi=300)
    plt.show()

import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from itertools import chain, combinations
import plotly.express as px
from itertools import product


def figs_tar(tar = True, option = 'WCO', rate = 99, acts= {10: {1:3, 5:3, 12:1, 6:3, 7:3}}, show_fig = True, save_fig = False, show_df = True):
    env = Homes(tar, option, rate)
    col = 'Day, Hour, Load, Returned_Load'.split(', ')
    df = pd.DataFrame(columns=col)
    done = False
    _ = env.reset(mode = 'test', scale = False)
    year, month = env.year, env.month
    score_ = 0
    LL_ = []
    while not done:
        _, reward, done, _ = env.step(0)
        LL_.append(env.returned_load)
        score_ += reward
    _ = env.reset(mode = 'test', scale = False)
    score, done, cc = 0, False, 0
    while not done:
        df1 = pd.DataFrame(0, index=np.arange(24), columns=col)
        df1['Day'] = env.day * np.ones(24, dtype=int)
        df1['Hour'] = list(range(24))
        for _ in range(16):
            if env.day in acts and env.n in acts[env.day]:
                    action = acts[env.day][env.n]
            else:
                action = 0
            _, reward, done, _ = env.step(action)
            score += reward
            df1.loc[:,'Load'] += env.LL * env.rate + LL_[cc]
            cc += 1
            df1.loc[:,'Returned_Load'] += env.returned_load + env.rate * env.LL
        df = pd.concat([df,df1], ignore_index=True)
    if show_fig:
        if month == 12:
            date_rng = pd.date_range(start=f'{month}/1/{year}', end=f'{1}/1/{year+1}', freq='H')
        else:
            date_rng = pd.date_range(start=f'{month}/1/{year}', end=f'{month+1}/1/{year}', freq='H')
        df['DateHour'] = date_rng[:-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['DateHour'], y=df['Load'] * 10/(1+rate), name = 'Load (MWh)', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=df['DateHour'], y=df['Returned_Load'] * 10/(1+rate), name = 'New Load (MWh)', line=dict(color='#d62728')))
        fig.add_trace(go.Scatter(x=df['DateHour'], y=[df['Load'].max() * 10/(1+rate)]*len(df['DateHour']), name = 'peak load (MW)', line=dict(dash='dash', color='#1f77b4')))
        fig.add_trace(go.Scatter(x=df['DateHour'], y=[df['Returned_Load'].max() * 10/(1+rate)]*len(df['DateHour']), name = 'new peak (MW)', line=dict(dash='dash', color='#d62728')))
        # fig.update_layout(title = f'Prosumer % in market: {np.round(100/(1+rate),3)}%, option: {option}, targeted: {tar}',hovermode='x unified')
        for ii in acts:
            for jj in acts[ii]:
                if acts[ii][jj]  == 1: #pink
                    x0 = f'{year}-{month}-{ii} 6:00:00'
                    x1 = f'{year}-{month}-{ii} 9:00:00'
                    fig.add_vrect(x0=x0, x1=x1,fillcolor='#f0027f', opacity=0.2,layer="below", line_width=0)
                if acts[ii][jj]  == 2: #green
                    x0 = f'{year}-{month}-{ii} 16:00:00'
                    x1 = f'{year}-{month}-{ii} 20:00:00'
                    fig.add_vrect(x0=x0, x1=x1,fillcolor='#2ca02c', opacity=0.2,layer="below", line_width=0)
                if acts[ii][jj] == 3:
                    x0 = f'{year}-{month}-{ii} 6:00:00'
                    x1 = f'{year}-{month}-{ii} 9:00:00'
                    fig.add_vrect(x0=x0, x1=x1,fillcolor='#f0027f', opacity=0.2,layer="below", line_width=0)
                    x0 = f'{year}-{month}-{ii} 16:00:00'
                    x1 = f'{year}-{month}-{ii} 20:00:00'
                    fig.add_vrect(x0=x0, x1=x1,fillcolor='#2ca02c', opacity=0.2,layer="below", line_width=0)
        fig.update_layout(width=1200, height=600)
        fig.update_layout(
            legend=dict(
                x=0.8,  # x-position of the legend (change as needed)
                y=0.8,  # y-position of the legend (change as needed)
                xanchor='left',  # anchor point of the legend
                yanchor='top',
            )
        )
        fig.show("notebook")
    if show_df:
        data = {
            'Metric': ['rate', 'SQ profit', 'new profit', 'revenue increase', 'acts', 'improve', 'peak load saving (MWh)'],
            'Value': [
                f'{np.round(100/(1+rate), 3)}%', score_, score, score - score_, acts,
                f'{np.round(100 * (score - score_)/np.abs(score_), 2)}%', (df['Load'].max()-df['Returned_Load'].max()) * 10000 / (1+rate)
            ]
        }

        df = pd.DataFrame(data)
        print(df)
    if save_fig:
        fig.write_image(f"figures/{option}_{tar}_{np.round(100/(1+rate), 3)}.png", width=1200, height=600, scale=2)
