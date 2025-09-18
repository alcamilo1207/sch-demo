import pandas as pd
import streamlit as st
from energy_module.energyp import EnergyPrices
from energy_module.energyp import Scheduler
from energy_module.energyp import PV
from schedule_plot import get_schedule_plot
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
#import locale
#locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

st.set_page_config(layout='wide')

region_col_name = "Germany/Luxembourg [€/MWh] Original resolutions"

def get_schedule(date_pick, pv_installed_capacity, color_option):
    success = True

    # Energy prices data
    prices_df = EnergyPrices().get_prices(date_pick)

    #Scheduler plot
    prices_profile, prices = EnergyPrices().prices_profile(prices_df)
    if prices is None:
        success = False

    schedule_list, fig1, _ = get_schedule_plot(prices_profile, color = color_option) # $$

    # power data
    schedule = Scheduler()._get_schedule(schedule_list, date_pick)
    pw_df = Scheduler()._get_power(schedule, date_pick)

    pv_power = PV().get_pv_power(pv_installed_capacity, date_pick)
    if pv_power is not None:
        savings_df = PV()._get_power_difference(pv_power, schedule, date_pick)
    else:
        savings_df = None

    # Generation Capacity data
    caps_df = PV().get_caps(date_pick)

    return success, fig1, schedule_list, prices, prices_df, pv_power, pw_df, savings_df, caps_df

def create_figures(_pw_df,savings_df, prices_df, date_pick):
    fig2 = go.Figure()

    for i in range(_pw_df.shape[1]):
        if i == _pw_df.shape[1]-1:
            name = f"Total power"
            fig2.add_trace(go.Scatter(
                name=name,
                x=_pw_df.index.values,
                y=_pw_df.iloc[:,i],
                line=dict(color='black', width=4, dash='dot')
            ))
        else:
            name = f"machine {i}"
            fig2.add_trace(go.Scatter(
                name=name,
                x=_pw_df.index.values,
                y=_pw_df.iloc[:,i],
                fill='tozeroy',
            ))

    fig2.update_layout(
        height=300,
        title="Factory total power",
        yaxis_title="Power [kW]",
        margin=dict(
            l=95,
            r=30,
            b=0,
            t=50,
            pad=4
        ),
        legend=dict(
            orientation='h',
            yanchor="top",
            y=-0.3,
            xanchor="left",
            x=0
        )
    )
    
    dates = pd.date_range(datetime.now().strftime("%Y-%m-%d"), periods=12, freq='2h')
    x_values3 = [prices_df['Start date'][i*2] for i in range(12)]
    prices_df["Germany/Luxembourg [€/MWh] Original resolutions"] = prices_df["Germany/Luxembourg [€/MWh] Original resolutions"].astype(float)
    fig3 = px.line(prices_df,x="Start date", y="Germany/Luxembourg [€/MWh] Original resolutions",line_shape='hv',title="Energy prices")
    fig3.update_xaxes(tickvals=x_values3, ticktext=[d.strftime('%H:00') for d in dates], range=[0, 24], title_text="") # working

    # PV data
    #print(savings_df)
    fig4 = px.line(savings_df, title="Day-ahead PV generation prediction (without energy storage system)")
    fig4.update_xaxes(range=[date_pick, date_pick+timedelta(hours=24)], title_text="") # working
    fig4.update_layout(
        yaxis_title="Energy [kWh]",
        xaxis_title="Time",
        legend=dict(
            orientation='h',
            yanchor="top",
            y=-0.3,
            xanchor="left",
            x=0
        ))
        
    return fig2, fig3, fig4

def main():
    # Sidebar
    #st.sidebar.title("User input data")
    date_pick = st.sidebar.date_input(label="Select date",value=datetime.now() + timedelta(days=0)) # timedelta is kept for debuging purposes
    color_option = st.sidebar.selectbox("Schedule color scale",("energy","size","actual_duration", "reward","assigned_to"))
    pv_installed_capacity = st.sidebar.slider("PV installed capacity in MW", 0.0, 0.1, 0.01, step=0.01)

    success, fig1, schedule_list, prices, prices_df, pv_power, _pw_df, savings_df, caps_df = get_schedule(date_pick, pv_installed_capacity, color_option)

    if not success:
        st.write("<----- Data unavailable. Please try a preivous date.")
    else:
        fig2, fig3, fig4 = create_figures(_pw_df, savings_df, prices_df, date_pick)
        # View in UI
        with st.container():
            # Your container content here
            #Schedule
            st.subheader("Day-ahead job schedule and energy prices forecast")

            # Metrics
            row1 = st.columns(2)
            row2 = st.columns(4)
            _total_cost, _total_energy, _total_production, _total_number_of_jobs = EnergyPrices()._metrics(schedule_list, prices, _pw_df)

            metrics = [
                {"label": "energy cost per ton", "value": f"{_total_cost/_total_production:.2f} €/ton"},
                {"label": "specific energy consumption", "value": f"{_total_energy/_total_production:.2f} kWh/ton"},
                {"label": "Total energy cost", "value": f"{_total_cost:.0f} €"},
                {"label": "Total energy usage", "value": f"{(_total_energy):.0f} kWh"},
                {"label": "Total yield", "value": f"{_total_production:.0f} Tons"},
                {"label": "Total job orders completed", "value": f"{_total_number_of_jobs} jobs"},
            ]
            for i,col in enumerate((row1 + row2)):
                cont = col.container(height=120)
                cont.metric(metrics[i]["label"],metrics[i]["value"],)

            st.plotly_chart(fig1,use_container_width=True)
            #st.plotly_chart(fig11)
            with st.expander("Machine parameters"):
                m_params = pd.DataFrame([[m.id,m.speed,m.energy_usage] for m in Scheduler().machines],columns=['Machine','Speed [kg/h]','Energy usage [kWh/h]'])
                m_params.set_index('Machine')
                st.dataframe(m_params,hide_index=True)

            # Plant power plots
            st.plotly_chart(fig2, use_container_width=True)

            # Energy prices Data
            st.plotly_chart(fig3, use_container_width=True)

            # PV energy plots
            if pv_power is not None:
                st.plotly_chart(fig4, use_container_width=True)

            total_cost, delta_cost, total_energy_usage, delta_energy, total_savings, pv_energy_generated = PV().metrics(prices, schedule_list, savings_df, _pw_df)
            PV_metrics = [
                {"label": "New total energy cost", "value": f"{total_cost:.0f} €"},
                {"label": "Savings", "value": f"{total_savings:.0f} €"},
                {"label": "New total energy usage", "value": f"{total_energy_usage:.0f} kWh"},
                {"label": "PV energy generated", "value": f"{pv_energy_generated:.0f} kWh"}
            ]
            # PV metrics
            row3 = st.columns(4)
            for i,col in enumerate(row3):
                cont = col.container(height=130)
                if i == 0:
                    cont.metric(PV_metrics[i]["label"], PV_metrics[i]["value"], delta=f"{delta_cost}%", delta_color="inverse")
                elif i == 2:
                    cont.metric(PV_metrics[i]["label"], PV_metrics[i]["value"], delta=f"{delta_energy}%", delta_color="inverse")
                else:
                    cont.metric(PV_metrics[i]["label"], PV_metrics[i]["value"])

            # PV data
            st.subheader("Dataframes")
            with st.expander("Energy generation installed capacity in Germany"):
                st.dataframe(caps_df)


            # PV data
            with st.expander("Energy market and forecast energy data"):
                st.dataframe(prices_df)
                st.write("Data source: Bundesnetzagentur | SMARD.de. More info: https://www.smard.de")


            # Show flowchart image
            st.subheader("Documentation")
            st.image("Flowchart.svg", caption="Flowchart of the proposed RL scheduler illustrating its integrations with the UPMSP environment.")


if __name__ == "__main__":
    main()