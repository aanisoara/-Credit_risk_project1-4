
#       #D4CACA
import streamlit as st
import altair as alt
from scipy.signal import welch
import warnings
import matplotlib
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from arch.unitroot import PhillipsPerron
from scipy.stats import kendalltau
warnings.filterwarnings("ignore")
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import List
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import levene
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_white
import itertools
from scipy.stats import kendalltau
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
import plotly.subplots as sp
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from IPython.display import display, HTML
import argparse
import flask
from flask import Flask
from streamlit.components.v1 import html
from yellowbrick.regressor import PredictionError, ResidualsPlot, CooksDistance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.stattools import durbin_watson


#app = Flask(__name__)
# __________________________  import bases ____________________________________________________________________

default_rate = pd.read_excel('Bases/default_rate_quarterly.xlsx', index_col='Date')
var_macro = pd.read_excel('Bases/variables_macroeconomiques.xlsx', index_col='Date')
var_macro_baseline = pd.read_excel('Bases/variables_macroeconomiques_baseline.xlsx',index_col='Date')
var_macro_adverse = pd.read_excel('Bases/variables_macroeconomiques_adverse.xlsx', index_col='Date')
 
# Définir une fonction pour formater les cellules
def color_row(row):
    """
    Fonction pour mettre en forme une ligne en rouge si la valeur de 'Stationnarité' est 'OUI'.
    """
    color = '#6C73E5' if row['STATIONNARITE'] == 'Non stationarity' else ''
    return ['background-color: {}'.format(color) for _ in row]
# ______________________________________________________________________________________________________________
#position: fixed;              background: #F8F3F4; background: #A5EFBC;

st.set_page_config(
    page_title="Application MoSEF Nexialog", 
    page_icon=":guardsman:", 
    layout="wide")

st.write("<h1 style='font-family:Lucida Caligraphy;font-size:20px;color:DarkSlateBlue;text-align: center;'> Members: Anisoara ABABII, Seçil COSKUN, Gaoussou DIAKITE,  Eunice KOFFI </h1>", unsafe_allow_html=True)


html_code = """
<!DOCTYPE html>
<html>
<head>
<style type="text/css">
body {
    margin: 0;
}
#titre {
    position: -webkit-sticky;
    position: sticky;
    top: 0;
    z-index: 1;
    width: 100%;
    text-align: center;
    padding-top: 1px;
    padding-bottom: 1px;
    height: 200px;
    font-size: 10px;
    font-weight: bold;
    color: #A6032A;
    text-transform: uppercase;
    letter-spacing: 1px;
    /* ajout des logos en tant que background-image */
    background-image: url(https://www.nexialog.com/wp-content/uploads/2021/06/logo.png), url(https://cdn.helloasso.com/img/logos/mosef-bb706d4e5be74fc3b9724e49c7f2c255.png);
    background-position: left center, right center;
    background-repeat: no-repeat;
    /* redimensionnement des images de fond */
    background-size: 25%, 15%;
    /* modification de la police de l'écriture et de la taille du texte */
    font-family: Lucida Caligraphy;
    font-size: 40px
}
</style>
</head>
<body>
    <div id="titre">Challenge <br> Nexialog / MoSEF <br> 2023 <br> <br> <br> <br> <br> <br> <br></div>
</body>
</html>
"""


# ____________________________________________________________________________________

st.markdown(html_code, unsafe_allow_html=True)
st.markdown("<br/>", unsafe_allow_html=True)
column1, column2, column3, column4, column5, button6, button7 = st.columns(7)
button_style = {"color": "black", "font-family": "Lucida"}

# Define the default button clicked
buttons = {
"button1": "Presentation and Objective of the Challenge",
"button2": "Default rate analysis",
"button3": "Exploratory analysis and preprocessing of macroeconomic variables",
"button4": "Study of Macro-economic Variables",
"button5": "Basic Models and Validation of Hypotheses ",
"button6": "Model Challenger and Validation of Hypotheses",
"button7": "Conclusion & Areas for Improvement"
}

# Create the sidebar
with st.sidebar:
    st.write("Table of Contents")
    current_button = st.radio("", list(buttons.values()))

# Update the current button if a button is clicked
for key, value in buttons.items():
    if current_button == value:
        st.session_state.current_button = key
        break

# _____________________________________________________ button 1 __________________________________________
if st.session_state.current_button == "button1":
    st.write("""
    <hr style="border-width:4px;border-color:#1B0586">
    <div style="text-align: center;"><span style="font-family: Lucida Caligraphy; font-size: 35px;">
        Project Objectives
        </span></div>
    <hr style="border-width:4px;border-color:#1B0586">""",  unsafe_allow_html=True)
    
    st.write("""
    <div align="left"><span style="font-family:Lucida Caligraphy;font-size:25px">
    Methodologies for Modeling Default Rate Projections\n
    </span></div>
    """,  unsafe_allow_html=True)

    image_1 = Image.open(r"image\steps.jpg")
    col1, col2, col3, col4, col5= st.columns(5)
    with col2:
        st.image(image_1, width=900, use_column_width=False)
    
# _____________________________________________________ button 2 __________________________________________
elif st.session_state.current_button =="button2":
    st.session_state.button2_clicked = True
    st.title("1. Default rate Analysis")

    # Create an array of values for b
    beta = np.arange(0.01, 1, 0.01)
    df_vide = pd.DataFrame()
    # Fit the linear regression model for each value of b
    for b in beta:
        ts = default_rate - b*default_rate.shift(1)
        ts = ts.rename(columns = {"DR":"{}".format(b)})
        df_vide = pd.concat([df_vide, ts], axis = 1)

    df_vide = df_vide.dropna()

    def stationnarite_pvalues(data, vars_list):
        res_df = pd.DataFrame(columns=['BETA', 'P-VALUE FOR ADF TEST',
                                        'P-VALUE FOR PP TEST',
                                    'P-VALUE FOR KPSS TEST'])
        loop = 1
        for x in vars_list:        
            adf_result = adfuller(data[x])
            pp_result = PhillipsPerron(data[x])
            kpss_result = kpss(data[x])         
            res_df = res_df.append({'BETA': x.upper(),
                                    'P-VALUE FOR ADF TEST': adf_result[1],
                                    'P-VALUE FOR PP TEST': pp_result.pvalue,
                                    'P-VALUE FOR KPSS TEST': kpss_result[1],
                                    '_i': loop}, ignore_index=True)
            loop += 1
        res_df = res_df.sort_values(by=['_i'])
        del res_df['_i']
        return res_df

    p_values = stationnarite_pvalues(df_vide, df_vide.columns)

    def plot_p_values(p_values, column):
        p_values = p_values.sort_values(by=column)
        fig = px.line(p_values, x="BETA", y=column)
        fig.add_shape(
            type="line",
            yref="y", y0=0.05, y1=0.05,
            xref="x", x0=p_values["BETA"].min(), x1=p_values["BETA"].max(),
            line=dict(color="red", dash="dash")
        )
        p_values["BETA"] = p_values["BETA"].astype(float)
        x_intercept = np.interp(0.05, p_values[column], p_values["BETA"])
        fig.add_shape(
            type="line",
            xref="x", x0=x_intercept, x1=x_intercept,
            yref="y", y0=0, y1=1,
            line=dict(color="green", dash="dash")
        )
        fig.update_layout(
            xaxis_title="BETA",
            yaxis_title=column,
            width=900,
            height=500,
            title={
                'text': "Stationarity Tests for Default Rate",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            }
        )
        st.plotly_chart(fig)
        #st.write(f"x-intercept: {x_intercept}")
        return x_intercept

    column = st.selectbox("Select a column", ["P-VALUE FOR ADF TEST", "P-VALUE FOR PP TEST", "P-VALUE FOR KPSS TEST"])
    x_intercept = plot_p_values(p_values, column)
    st.write("""
        <div style="text-align: justify;"><span style="font-family: Lucida Caligraphy; font-size: 20px;">
        Stationnarity Tests for default rate
        </span></div>
        x-intercept: {}
        """.format(x_intercept), unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    model = smf.ols(formula='default_rate ~ default_rate.shift(1)' , data=default_rate).fit()
    #st.write(model.summary().as_html(), unsafe_allow_html=True)


    # _______________________    plot interval de  beta  ____________________________________

    def plot_p_values(p_values):
        fig = go.Figure()

        for column in p_values.columns[1:]:
            p_values = p_values.sort_values(by=column)

            fig.add_trace(
                go.Scatter(x=p_values["BETA"], y=p_values[column], mode="lines", name=column)
            )

            fig.add_shape(
                type="line",
                yref="y", y0=0.05, y1=0.05,
                xref="x", x0=p_values["BETA"].min(), x1=p_values["BETA"].max(),
                line=dict(color="red", dash="dash")
            )

            p_values["BETA"] = p_values["BETA"].astype(float)
            x_intercept = np.interp(0.05, p_values[column], p_values["BETA"])

            fig.add_shape(
                type="line",
                xref="x", x0=x_intercept, x1=x_intercept,
                yref="y", y0=0, y1=1,
                line=dict(color="green", dash="dash")
            )

        fig.update_layout(
            xaxis_title="BETA",
            yaxis_title="p-values",
            height=600,
            width=1000,
            title={
                'text': "Stationarity Tests for Default Rate and Beta Interval",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            }
        )
        
        # Get the common interval of beta
        common_beta_min = p_values.iloc[:, 0].min()
        common_beta_max = p_values.iloc[:, 0].max()
        for column in p_values.columns[1:]:
            p_values = p_values.sort_values(by=column)
            common_beta_min = min(common_beta_max, p_values.loc[p_values[column] < 0.05, "BETA"].max())
            common_beta_max = max(0,1)

        st.markdown(f"Common interval of beta: [{common_beta_min:.3f}, {common_beta_max:.3f}[")

        st.plotly_chart(fig)
        return common_beta_min, common_beta_max

    st.write("""
        <div style="text-align: justify;"><span style="font-family: Lucida Caligraphy; font-size: 20px;">
        Interval of Beta  :
        </span></div>
        """, unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    common_beta_min, common_beta_max = plot_p_values(p_values)

    #___________________________ model summary  _______________________________________


    image2 = Image.open(r"image\giphy3.gif")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.image(image2,width=250, use_column_width ='false')

    with col2:
        st.write(model.summary().as_html(), unsafe_allow_html=True)

# _____________________________________________________ button 3 __________________________________________
elif st.session_state.current_button =="button3":
    st.session_state.button3_clicked = True
    #__________________________________  plot macro variables séries  _____________________________________

    st.title("1. Macro Variables Analysis")
    var_selectbox = st.selectbox('Select a variable : ', options=var_macro.columns)

    col_name = var_selectbox
    if col_name in var_macro.columns:
        fig = px.line(x=var_macro.index, y=var_macro[col_name], title=col_name)
        fig.update_layout(
            xaxis_title="Index",
            yaxis_title="Value",
            width=900, 
            height=500, 
            title={
                'text': "Analysis of Macro Variables " + col_name,
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            }
        )
        st.plotly_chart(fig)
     
   #____________________________________________  plot variance _____________________________________

    # Définir une fonction pour tracer la variance roulante
    st.title("2. Variance of Macro Variables ")
    def plot_rolling_var(var_macro, col_name):
        ts = var_macro[col_name]
        
        # Calculer la variance sur des fenêtres glissantes de taille 4
        rolling_var = ts.rolling(window=4).var()
        # Tracer la variance
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=ts.index, y=rolling_var, name=col_name))
        fig1.update_layout(
            xaxis_title="Index",
            yaxis_title="Value",
            width=900, 
            height=500, 
            title={
                'text': "Variance of Macro Variable " + col_name,
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            }
        )
        st.plotly_chart(fig1)

    # Tracer la variance de la colonne sélectionnée
    plot_rolling_var(var_macro, var_selectbox)


        # _____________________________________ Spectral Density ______________________________________

    def spectral_density(df, col, sampling_freq):
        """
        Calcule la densité spectrale d'une colonne d'un dataframe de séries temporelles.

        :param df: Le dataframe contenant la série temporelle.
        :param col: Le nom de la colonne à analyser.
        :param sampling_freq: La fréquence d'échantillonnage de la série temporelle.
        """
        time_series = df[col].values
        f, Pxx = welch(time_series, fs=sampling_freq, nperseg=len(time_series))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=Pxx))
        fig.update_xaxes(title_text='Frequency (Hz)')
        fig.update_yaxes(title_text='Power Spectral Density')
        fig.update_layout(
            width=900, 
            height=500, 
            title={
                'text': f"Spectral Density {col}",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 15}
            }
        )

        return fig

    st.title("3. Spectral Density")
    sampling_freq = 100
    fig = spectral_density(var_macro, var_selectbox, sampling_freq)
    st.plotly_chart(fig)
    
# _____________________________________________________ button 4 _____________________________________________

elif st.session_state.current_button =="button4":
    # ______________________________ Stationnarité ____________

    st.title("1. Macro Variables Analysis and Stationnarity Tests")
    
    def stationarity_test(data, vars_list):
        sig = 0.04
        stationnaires = []
        res_df = pd.DataFrame(columns=['VARS', 'ADF TEST', 'P-VALUE FOR ADF TEST',
                                    'PP TEST', 'P-VALUE FOR PP TEST',
                                    'KPSS TEST', 'P-VALUE FOR KPSS TEST', 'STATIONNARITE'])
        loop = 1
        for x in vars_list:
            adf_result = adfuller(data[x], regression='c')
            pp_result = PhillipsPerron(data[x])
            kpss_result = kpss(data[x],  regression='c')
            if (adf_result[1] < sig and pp_result.pvalue < sig) or (adf_result[1] < sig and  kpss_result[1] > sig) or (pp_result.pvalue < sig and  kpss_result[1] > sig):
                flg = "Stationnarity" 
                stationnaires.append(x)
            else:
                flg = "Non stationarity" 
                
            res_df = res_df.append({'VARS': x.upper(),
                                    'ADF TEST': adf_result[0],
                                    'P-VALUE FOR ADF TEST': adf_result[1],
                                    'PP TEST': pp_result.stat,
                                    'P-VALUE FOR PP TEST': pp_result.pvalue,
                                    'KPSS TEST': kpss_result[0],
                                    'P-VALUE FOR KPSS TEST': kpss_result[1],
                                    'STATIONNARITE': flg,
                                    '_i': loop}, ignore_index=True)
            loop += 1
        res_df = res_df.sort_values(by=['_i'])
        del res_df['_i']
        return res_df

    res_df = stationarity_test(var_macro, var_macro.columns)  

    # Appliquer la fonction pour chaque ligne du dataframe
    res_df = res_df.style.apply(color_row, axis=1)
                                  
    st.write("""
            <div style="text-align: justify;"><span style="font-family: Lucida Caligraphy; font-size: 40px;">
            Stationnarity Tests
            </span></div>
            <div style="overflow-x: auto; max-width: none; text-align: center;">
            <table style="margin: auto;">
            {}
        """.format(res_df.to_html(index=False, justify="center")), unsafe_allow_html=True)

    # _____________________________________________  LEVENE TEST ______________________________________ 

    st.title("2. Macro Variables Analysis and LEVENE Test")
    st.write("""
        <div style="text-align: justify;"><span style="font-family: Lucida Caligraphy; font-size: 30 px;">
        Levene's Test: it is a statistical test used to check whether the variance of two or more groups are equal. 
        If the variances are not equal, then the ANOVA test results may be invalid. The Levene's test calculates a test 
        statistic and p-value, with a p-value less than the significance level indicating that the variances are significantly different.
        <br/></span></div>""", unsafe_allow_html=True)


    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Variable', 'Statistique de test', 'Valeur p'])

    # Call the test_levene function and add the results to the DataFrame
    def test_levene(df):
        for column in df.columns:    
            ts = df[column]

            rolling_var = ts.rolling(window=4).var()
            rolling_var_without_na = rolling_var[3:]
            premiere_partie = rolling_var_without_na.iloc[0:22]
            deuxieme_partie = rolling_var_without_na.iloc[23:45]
            statistic, pvalue = levene(premiere_partie, deuxieme_partie)
            
            # Add the results to the DataFrame
            results_df.loc[len(results_df)] = [column, statistic, pvalue]
    test_levene(var_macro)

    image = Image.open(r"image\find.gif")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image,width=250, use_column_width ='false')

    with col2:
        st.write("""
            <br/><div style="text-align: justify;"><span style="font-family: Lucida Caligraphy; font-size: 40px;">
            Levene's Test
            </span></div>
            <div style="overflow-x: auto; max-width: none; text-align: center;">
            <table style="margin: auto;">
            {}
            </table>
            </div>
        """.format(results_df.to_html(index=False, justify="center")), unsafe_allow_html=True)
    
    #____________________________________________ Lissage  ________________________________________

    var_macro_model = var_macro.copy()
    st.title("3. Smoothing of  Macroeconomic Variables")
    st.write("""
        <div style="text-align: justify;"><span style="font-family: Lucida Caligraphy; font-size: 25px;">

        </span></div>""", unsafe_allow_html=True)
    
    for column in var_macro.columns:    
        ts = var_macro[column]
        # Calcule la variance sur des fenêtres glissantes de taille 4
        rolling_var = ts.rolling(window=4).var()

        rolling_var_without_na = rolling_var[3:]
        result = adfuller(rolling_var_without_na)


    def lissage_ewma(df, cols, chart_width=50, chart_height=300):
        for col in cols:
            ts = df[col]
            ts_lissee = ts.ewm(span=4).mean()
            df[f"{col}_lissee2"] = ts_lissee

            # create plot
            base = alt.Chart(df.reset_index()).encode(x="Date:T")
            orig = base.mark_line(color="blue").encode(y=col)
            lissee = base.mark_line(color="red").encode(y=f"{col}_lissee2")
            chart = (orig + lissee).properties(width=chart_width, height=chart_height)
            #st.altair_chart(chart, use_container_width=True)
        return df
    
    cols = ["UNR", "RREP", "HICP", "RGDP", "IRLT"]
    var_macro = lissage_ewma(var_macro, cols, chart_width=50, chart_height=300)
    
    var_macro["RREP_lissee"] = var_macro["RREP_lissee2"] 
    var_macro = var_macro.drop(columns = ["RREP_lissee2"], axis =1)

    var_macro["UNR_lissee"] = pd.concat([var_macro["UNR"][:"2019-04-30"], var_macro["UNR_lissee2"]["2019-07-31":]], axis = 0)
    var_macro = var_macro.drop(columns = ['UNR_lissee2'])
    
    var_macro["HICP_lissee"] = var_macro["HICP_lissee2"] 
    var_macro = var_macro.drop(columns = ["HICP_lissee2"], axis =1)

    var_macro["RGDP_lissee"] = var_macro["RGDP_lissee2"] 
    var_macro = var_macro.drop(columns = ["RGDP_lissee2"], axis =1)

    var_macro["IRLT_lissee"] = var_macro["IRLT_lissee2"] 
    var_macro = var_macro.drop(columns = ["IRLT_lissee2"], axis =1)


    def plot_timeseries(data):
        """
        Plots the original and smoothed time series from data using the specified column names and colors.
        """
        # Get the column names and colors from the user
        col_name = st.selectbox("Select the name of the original time series:", data.columns)
        smoothed_col_name = st.selectbox("Select the name of the smoothed time series:", data.columns)
        col_color = st.columns(2)
        orig_color = col_color[0].color_picker("Select a color for the original time series:", "#1f77b4")
        smoothed_color = col_color[1].color_picker("Select a color for the smoothed time series :", "#ff7f0e")

        # Create the plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data[col_name], name='Original time series', line=dict(color=orig_color)))
        fig.add_trace(go.Scatter(x=data.index, y=data[smoothed_col_name], name='Smoothed time series ', line=dict(color=smoothed_color)))
        fig.update_layout(
            width=900, 
            height=500, 
            title={
                'text': f"Time Series Plot: {col_name} and {smoothed_col_name}",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 15}
            }
        )
        st.plotly_chart(fig)

    plot_timeseries(var_macro)

    # __________________________ taux de croissance  ________________________

    def taux_croissance(df):
        series_en_volume = df.columns
        df = df.dropna()
        for col in df.columns :
            df[f'gr_{col}'] = (df[col] - df[col].shift(4)) / df[col].shift(4)
        df = df.dropna()
        df = df.drop(list(series_en_volume), axis = 1)
        return df

    #st.title("4. Growth rates transformation")
    var_macro = taux_croissance(var_macro)

    #st.write(var_macro)
    # verification de stationnarité 
    res_df2 = stationarity_test(var_macro, var_macro.columns)  
    # Appliquer la fonction pour chaque ligne du dataframe
    res_df2 = res_df2.style.apply(color_row, axis=1)  
    #st.write("""
            #<div style="text-align: justify;"><span style="font-family: Lucida Caligraphy; font-size: 20px;">
            #Stationnarity Tests after applying exponential smoothing and growth rates transformation :
            #</span></div>
            #<div style="overflow-x: auto; max-width: none; text-align: center;">
            #<table style="margin: auto;">
            #{}
        #""".format(res_df2.to_html(index=False, justify="center")), unsafe_allow_html=True)

    # ______________________________ introduction of lags t +4 _______________________________

    var_macro = pd.concat([var_macro, default_rate], axis = 1)

    def create_lags_diffs(data, num_lags=1):
        new_data = pd.DataFrame()
        for col in data.columns:
            for i in range(1, num_lags + 1):
                new_data[f"{col}_lag{i}"] = data[col].shift(i)

        new_data = pd.concat([data, new_data], axis=1)
        new_data = new_data.dropna()

        #st.write("Original data:")
        #st.write(data)
        #st.write("Data with lags and differences:")
        #st.write(new_data)
        return new_data
    
    var_macro_lags = create_lags_diffs(var_macro, num_lags = 4)

    # ______________________________  Taux de Kendal __________________________

    def kendall_tau(df1, df2, col_name):
        """
        Returns a DataFrame with Kendall's tau correlation coefficients between each column 
        """
        corr_avec_target = []
        y = df2[col_name]
        result_df = pd.DataFrame(index=df1.columns, columns=['kendall_tau', 'p_value'])
        for col in df1.columns:
            x = df1[col]
            x_subset = x[:len(y)]
            tau, p_value = kendalltau(x_subset, y)
            if abs(tau) > 0.20 : #and p_value < 0.05: #on regarde le taux >30 et pas la p value !
                corr_avec_target.append(col)
            else:
                pass
            result_df.loc[col] = [tau, p_value]
        
        #st.write("Kendall's tau correlation coefficients:")
        #st.write(result_df)

        #st.write("Columns with a correlation coefficient > 0.20:")
        #st.write(corr_avec_target)

    var_macro_lags = var_macro_lags.drop(["DR"], axis = 1)
    default_rate1 = default_rate[8:]
    result_df = kendall_tau(var_macro_lags, default_rate1, 'DR')

# _____________________________________________________ button 5 _____________________________________________

elif st.session_state.current_button =="button5":
    var_macro = pd.read_excel('Bases/variables_macroeconomiques.xlsx', index_col='Date')
    var_macro_baseline = pd.read_excel('Bases/variables_macroeconomiques_baseline.xlsx',index_col='Date')
    var_macro_adverse = pd.read_excel('Bases/variables_macroeconomiques_adverse.xlsx', index_col='Date')

    # ______________________________ Stationnarité ____________

    st.title("1. Models")
    def lissage_mm(var_macro, col, window_size=4):
        #Calculate the smoothed series MM
        var_macro_smoothed = var_macro[col].rolling(window=window_size).mean()
        var_macro['{}_lissee'.format(col)] = var_macro_smoothed

        trace1 = go.Scatter(x=var_macro.index, y=var_macro[col], mode='lines', name='Original time series')
        trace2 = go.Scatter(x=var_macro.index, y=var_macro_smoothed, mode='lines', name='Smoothed series')
        layout = go.Layout(
            title=f"Smoothed series {col}",
            xaxis=dict(title='Date'),
            yaxis=dict(title='Value')
        )
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        return st.plotly_chart(fig)
    
    cols = ["HICP"]
    selected_col = st.selectbox("Select a variable to plot", cols)
    fig = lissage_mm(var_macro, selected_col, window_size=4)

    var_macro = var_macro.drop(["HICP"], axis = 1)

    #st.title("2. Growth rates transformation")
    def taux_croissance(df):
        series_en_volume = df.columns
        df = df.dropna()
        for col in df.columns :
            df[f'gr_{col}'] = (df[col] - df[col].shift(4)) / df[col].shift(4)
        df = df.dropna()
        df = df.drop(list(series_en_volume), axis = 1)
        return df
    
    var_macro = taux_croissance(var_macro)

    st.title("2. Macro Variables Analysis and Stationnarity Tests")
    
    def stationnarite(data, vars_list):
        sig = 0.04
        stationnaires = []
        res_df = pd.DataFrame(columns=['VARS', 'ADF TEST', 'P-VALUE FOR ADF TEST',
                                    'PP TEST', 'P-VALUE FOR PP TEST',
                                    'KPSS TEST', 'P-VALUE FOR KPSS TEST', 'STATIONNARITE'])
        loop = 1
        for x in vars_list:
            adf_result = adfuller(data[x], regression='c')
            pp_result = PhillipsPerron(data[x])
            kpss_result = kpss(data[x],  regression='c')
            if (adf_result[1] < sig and pp_result.pvalue < sig) or (adf_result[1] < sig and  kpss_result[1] > sig) or (pp_result.pvalue < sig and  kpss_result[1] > sig):
                flg = "Stationnarity" 
                stationnaires.append(x)
            else:
                flg = "Non stationarity" 
                
            res_df = res_df.append({'VARS': x.upper(),
                                    'ADF TEST': adf_result[0],
                                    'P-VALUE FOR ADF TEST': adf_result[1],
                                    'PP TEST': pp_result.stat,
                                    'P-VALUE FOR PP TEST': pp_result.pvalue,
                                    'KPSS TEST': kpss_result[0],
                                    'P-VALUE FOR KPSS TEST': kpss_result[1],
                                    'STATIONNARITE': flg,
                                    '_i': loop}, ignore_index=True)
            loop += 1
        res_df = res_df.sort_values(by=['_i'])
        del res_df['_i']
        data = data[stationnaires]
        return stationnaires, res_df, data
    
    stationnaires, res_df, var_macro = stationnarite(var_macro, var_macro.columns)

    # Appliquer la fonction pour chaque ligne du dataframe
    res_df = res_df.style.apply(color_row, axis=1)
    st.write("""
        <div style="text-align: justify;"><span style="font-family: Lucida Caligraphy; font-size: 40px;">
        Stationnarity Tests
        </span></div>
        <div style="overflow-x: auto; max-width: none; text-align: center;">
        <table style="margin: auto;">
        {}
    """.format(res_df.to_html(index=False, justify="center")), unsafe_allow_html=True)

    # ______________________________ introduction of lags t +4 _______________________________

    var_macro = pd.concat([var_macro, default_rate], axis = 1)
    
    
    def create_lags_diffs(data, num_lags=1):
        new_data = pd.DataFrame()
        for col in data.columns:
            for i in range(1, num_lags + 1):
                new_data[f"{col}_lag{i}"] = data[col].shift(i)
        new_data = pd.concat([data, new_data], axis=1)
        new_data = new_data.dropna()
        return new_data
    
    var_macro_lags = create_lags_diffs(var_macro, num_lags = 4)
    var_macro_lags = var_macro_lags.drop(columns = ["DR_lag2", "DR_lag3", "DR_lag4"])
# ______________________________  Taux de Kendal __________________________

    def kendall_tau(df1, df2, col_name):
        """
        Returns a DataFrame with Kendall's tau correlation coefficients between each column 
        """
        corr_avec_target = []
        y = df2[col_name]
        result_df = pd.DataFrame(index=df1.columns, columns=['kendall_tau', 'p_value'])
        for col in df1.columns:
            x = df1[col]
            x_subset = x[:len(y)]
            tau, p_value = kendalltau(x_subset, y)
            if abs(tau) > 0.25 : #and p_value < 0.05: #on regarde le taux >30 et pas la p value !
                corr_avec_target.append(col)
            else:
                pass
            result_df.loc[col] = [tau, p_value]
        return corr_avec_target

    var_macro_lags = var_macro_lags.drop(["DR"], axis = 1)
    default_rate1 = default_rate[8:]
    result_df = kendall_tau(var_macro_lags, default_rate1, 'DR') 
    st.markdown("<br/><br/>", unsafe_allow_html=True)
    
    st.write("""
        <div style="text-align: justify;"><span style="font-family: Lucida Caligraphy; font-size: 25px;">
        We use Kendall's tau to determine the variables that are correlated the most with the default rate and we obtain the following variables. We will use them to build our models: 
        </span></div>
    """, unsafe_allow_html=True)



    left, middle, rigth = st.columns(3)
    with left:
        st.write("""
    
    <div align="left"><span style="font-family:Lucida Caligraphy;font-size:16px">
    <ul>
        <li>gr_RGDP</li>
        <li>gr_IRLT</li>
        <li>gr_UNR</li>
        <li>gr_RGDP_lag1</li>
        <li>DR_lag1</li>
        <li>gr_HICP_lissee_lag4</li>
    </ul>
    </span></div>
    
            """,  unsafe_allow_html=True)
    with middle:
        st.write("""
    
    <div align="left"><span style="font-family:Lucida Caligraphy;font-size:16px">
    <ul>
        <li>gr_RGDP_lag2</li>
        <li>gr_RGDP_lag3</li>
        <li>gr_RGDP_lag4</li>
        <li>gr_IRLT_lag1</li>
        <li>gr_HICP_lissee_lag2</li>
        <li>gr_HICP_lissee_lag3</li>
    </ul>
    </span></div>
    
            """,  unsafe_allow_html=True)
        
        with rigth:
            st.write("""
    
    <div align="left"><span style="font-family:Lucida Caligraphy;font-size:16px">
    <ul>
        <li>gr_IRLT_lag1</li>
        <li>gr_IRLT_lag2</li>
        <li>gr_UNR_lag1</li>
        <li>gr_UNR_lag2</li>
        <li>gr_UNR_lag3</li>
        <li>gr_UNR_lag4</li>
    </ul>
    </span></div>
    
            """,  unsafe_allow_html=True)

    explicatives = var_macro_lags[result_df]
    #explicatives
    variables = explicatives.columns
    df = pd.concat([explicatives, default_rate], axis = 1)
    df = df.dropna()
    
    # ____________________        Traitement des bases de test

    scenarios_taux = taux_croissance(var_macro_adverse)
    var_macro_adverse = pd.concat([scenarios_taux["gr_UNR"], var_macro_adverse.drop(columns = ["UNR"])], axis = 1).dropna()
    var_macro_adverse = var_macro_adverse.rename(columns= {"RGDP":"gr_RGDP", "HICP":"gr_HICP", "RREP":"gr_RREP", "IRLT":"gr_IRLT"})
    var_macro_adverse[["gr_RGDP" ,"gr_HICP" ,"gr_RREP"]] = var_macro_adverse[["gr_RGDP" ,"gr_HICP" ,"gr_RREP"]] / 100
    var_macro_adverse["gr_IRLT"] = var_macro_adverse["gr_IRLT"] / 10

    scenarios_taux = taux_croissance(var_macro_baseline)
    var_macro_baseline = pd.concat([scenarios_taux["gr_UNR"], var_macro_baseline.drop(columns = ["UNR"])], axis = 1).dropna()
    var_macro_baseline = var_macro_baseline.rename(columns= {"RGDP":"gr_RGDP", "HICP":"gr_HICP", "RREP":"gr_RREP", "IRLT":"gr_IRLT"})
    var_macro_baseline[["gr_RGDP" ,"gr_HICP" ,"gr_RREP"]] = var_macro_baseline[["gr_RGDP" ,"gr_HICP" ,"gr_RREP"]] / 100
    var_macro_baseline["gr_IRLT"] = var_macro_baseline["gr_IRLT"] / 10

    # ______________________________________ Entrainement du modèle   _____
    
    df_train = df[df.index.year < 2018]
    df_test = df[df.index.year >= 2018]

    model_86 = smf.ols(formula='DR ~ gr_RGDP_lag2 + DR_lag1  + 1', data=df_train).fit() 

    y_hat_86 = model_86.predict(df_test)


    st.markdown("<br/>", unsafe_allow_html=True)
    image_m = Image.open(r"image\methode.jpg")
    col1, col2, col3, col4, col5= st.columns(5)
    with col2:
        st.image(image_m, width=700, use_column_width=False)

    st.title("OLS results for the model")
    st.markdown("<br/>", unsafe_allow_html=True)
    st.write(model_86.summary())
    # ____________________________________  select scenario __________________________

    st.title("3. Default rate forecast")
    dfs = {"Variable Macroeconomiques adverses": var_macro_adverse, "Variable Macroeconomiques baselines": var_macro_baseline}
    selected_df_key = st.selectbox("Choice a scenario / DataFrame", list(dfs.keys()))
    selected_df = dfs[selected_df_key]

    # _________________________________  fonction de prediction + plot _________
    
    def prediction_modele_86(scenario) : 
        sc = create_lags_diffs(scenario, num_lags = 3)
        X1 = sc["gr_RGDP_lag2"]
        X = pd.concat([X1, df[-1:]["DR_lag1"], df[-1:]["DR"]], axis = 1) #on rajoute la 2ème variable explicative
        dates =  X.index.date.astype(str).tolist()
        
        for i in range (len(dates)-1) :
            X_1 = X.loc[dates[i]] # au début on retient juste 2019-10-31 pour calculer la date d'apres, etc.
            sc_predictions = model_86.predict(X_1)
            sc_predictions = pd.DataFrame(sc_predictions, columns = ["DR_pred"])
            X.loc[dates[i+1]]['DR'] = sc_predictions["DR_pred"]
            X["DR_lag1"] = X["DR"].shift(1)

        serie_predite = X.loc[X.index > '2019-10-31']["DR"]
        serie_predite_et_val_precente = X.loc[X.index > '2019-07-31']["DR"] #pour qu'il y ait pas de trou dans le plot dû aux dates
    
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=default_rate.index, y=default_rate.values, name='Série originale'))
        fig1.add_trace(go.Scatter(x=serie_predite_et_val_precente.index, y=serie_predite_et_val_precente.values, name='Série prédite'))
        fig1.update_layout(
            width=900, 
            height=500, 
            title={
                'text': f"Prédicted Time Series Plot for  {scenario}",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 15}
            }
        )
        #st.plotly_chart(fig)

        plt.rcParams["axes.edgecolor"] = "white"
        fig1, ax = plt.subplots()
        ax.plot(default_rate, label='Série originale')
        ax.plot(serie_predite_et_val_precente, label='Série prédite')
        ax.legend()
        st.plotly_chart(fig1)
        return serie_predite, serie_predite_et_val_precente
    
    prediction_modele_86(selected_df)
    

# ___________________________________________________ button 6 _____________________________________________

elif st.session_state.current_button =="button6":

    var_macro = pd.read_excel('Bases/variables_macroeconomiques.xlsx', index_col='Date')
    var_macro_baseline = pd.read_excel('Bases/variables_macroeconomiques_baseline.xlsx',index_col='Date')
    var_macro_adverse = pd.read_excel('Bases/variables_macroeconomiques_adverse.xlsx', index_col='Date')
    
    # ______________________________ Bases macro et DR ____________
    def taux_croissance(df):
        series_en_volume = df.columns
        df = df.dropna()
        for col in df.columns :
            df[f'gr_{col}'] = (df[col] - df[col].shift(4)) / df[col].shift(4)
        df = df.dropna()
        df = df.drop(list(series_en_volume), axis = 1)
        return df
    
    def create_lags_diffs(data, num_lags=1):
        new_data = pd.DataFrame()
        for col in data.columns:
            for i in range(1, num_lags + 1):
                new_data[f"{col}_lag{i}"] = data[col].shift(i)
        new_data = pd.concat([data, new_data], axis=1)
        new_data = new_data.dropna()
        return new_data
    
    var_macro = taux_croissance(var_macro)
    var_macro = pd.concat([var_macro, default_rate], axis = 1)
    var_macro_lags = create_lags_diffs(var_macro, num_lags = 4)
    var_macro_lags = var_macro_lags.drop(columns = ["DR_lag2", "DR_lag3", "DR_lag4"])

    # ____________________        Traitement des bases de test
    
    scenarios_taux = taux_croissance(var_macro_adverse)
    var_macro_adverse = pd.concat([scenarios_taux["gr_UNR"], var_macro_adverse.drop(columns = ["UNR"])], axis = 1).dropna()
    var_macro_adverse = var_macro_adverse.rename(columns= {"RGDP":"gr_RGDP", "HICP":"gr_HICP", "RREP":"gr_RREP", "IRLT":"gr_IRLT"})
    var_macro_adverse[["gr_RGDP" ,"gr_HICP" ,"gr_RREP"]] = var_macro_adverse[["gr_RGDP" ,"gr_HICP" ,"gr_RREP"]] / 100
    var_macro_adverse["gr_IRLT"] = var_macro_adverse["gr_IRLT"] / 10

    scenarios_taux = taux_croissance(var_macro_baseline)
    var_macro_baseline = pd.concat([scenarios_taux["gr_UNR"], var_macro_baseline.drop(columns = ["UNR"])], axis = 1).dropna()
    var_macro_baseline = var_macro_baseline.rename(columns= {"RGDP":"gr_RGDP", "HICP":"gr_HICP", "RREP":"gr_RREP", "IRLT":"gr_IRLT"})
    var_macro_baseline[["gr_RGDP" ,"gr_HICP" ,"gr_RREP"]] = var_macro_baseline[["gr_RGDP" ,"gr_HICP" ,"gr_RREP"]] / 100
    var_macro_baseline["gr_IRLT"] = var_macro_baseline["gr_IRLT"] / 10
    
    
    # ____________________________________  select scenario __________________________

    ## _________________________ Modele challenger: Random Forest
    X_df = var_macro_lags[['DR_lag1', 'gr_UNR', 'gr_UNR_lag4', 'gr_RGDP', 'gr_RREP_lag4', 'gr_UNR_lag3', 'gr_IRLT_lag4', 'gr_RGDP_lag1']]
    y = var_macro_lags['DR']
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    significant_features = ['DR_lag1', 'gr_UNR', 'gr_UNR_lag4', 'gr_RGDP', 'gr_RREP_lag4', 'gr_UNR_lag3', 'gr_IRLT_lag4', 'gr_RGDP_lag1']
    significant_features.remove('DR_lag1')

    ## Entrainement du modèle 
    np.random.seed(42)
    gb = RandomForestRegressor()
    gb.fit(X_train, y_train)
    
    pred_gb = gb.predict(X_test)
    
    # Afficher le data frame avec la colonne Conclusion
    st.title("1. Metrics")
    

    st.title("Results for the Random forest model")
    st.markdown("<br/>", unsafe_allow_html=True)
    st.write()

    one, two, three = st.columns(3)
    with one:
        st.write("Score Test:",gb.score(X_test,y_test))

    with two:
        st.write("MAE:",mean_absolute_error(y_test,pred_gb))
        
    with three:
        st.write("MSE:",mean_squared_error(y_test,pred_gb))
    
    image_1 = Image.open(r"image\1.png")
    image_2 = Image.open(r"image\2.png")
    one1, two2  = st.columns(2)
    with one1:
        st.image(image_1, width=500, use_column_width=False)
    with two2:
        st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.image(image_2, width=600, use_column_width=False)

    un, deux, trois, quatre = st.columns(4)
    with un:
        st.write("""
    
        <div align="left"><span style="font-family:Lucida Caligraphy;font-size:16px">
        <ul>
            <li>DR_lag1 (0.806636) </li>
            <li>gr_UNR_lag4 (0.023210) </li>
        </ul>
        </span></div>
    
                """,  unsafe_allow_html=True)
    with deux:
        st.write("""
    
        <div align="left"><span style="font-family:Lucida Caligraphy;font-size:16px">
        <ul>
            <li>gr_RGDP (0.020173)</li>
            <li>gr_RGDP_lag1 (0.010018)</li>
        </ul>
        </span></div>
    
                """,  unsafe_allow_html=True)
    with trois:
        st.write("""
    
        <div align="left"><span style="font-family:Lucida Caligraphy;font-size:16px">
        <ul>
            <li>gr_RREP_lag4 (0.011815)</li>
            <li>gr_UNR_lag3 (0.011038)</li>
        </ul>
        </span></div>
    
                """,  unsafe_allow_html=True)
    with quatre:
        st.write("""
    
        <div align="left"><span style="font-family:Lucida Caligraphy;font-size:16px">
        <ul>
            <li>gr_IRLT_lag4 (0.010678)</li>
            <li>gr_UNR (0.027692)</li>
        </ul>
        </span></div>
    
                """,  unsafe_allow_html=True)
            
    def prediction_modele_challenge(scenario) : 
        sc = create_lags_diffs(scenario, num_lags = 4)
        X1 = pd.concat([X_df[-1:]["gr_UNR"], sc["gr_UNR"]]) #on concatene la derniere valeur dispo dans la base (2019-10-31) avec les valeurs du scénario
        X2 = pd.concat([X_df[-1:]["gr_RGDP"], sc["gr_RGDP"]])
        X3 = pd.concat([X_df[-1:]["gr_RGDP_lag1"], sc["gr_RGDP_lag1"]])
        X4 = pd.concat([X_df[-1:]["gr_UNR_lag3"], sc["gr_UNR_lag3"]])
        X5 = pd.concat([X_df[-1:]["gr_RREP_lag4"], sc["gr_RREP_lag4"]])
        X6 = pd.concat([X_df[-1:]["gr_UNR_lag4"], sc["gr_UNR_lag4"]])
        X7 = pd.concat([X_df[-1:]["gr_IRLT_lag4"], sc["gr_IRLT_lag4"]])#on concatene la derniere valeur dispo dans la base (2019-10-31) avec les valeurs du scénario
        X = pd.concat([X1,X2, X3, X4, X5, X6, X7, X_df[-1:]["DR_lag1"], y[-1:]], axis = 1 ) #on rajoute la 2ème et 3eme variable explicative
        dates =  X.index.date.astype(str).tolist()

        for i in range (len(dates)-1) :
            X_1 = pd.DataFrame(X.loc[dates[i]][['DR_lag1', 'gr_UNR', 'gr_UNR_lag4', 'gr_RGDP', 'gr_RREP_lag4', 'gr_UNR_lag3', 'gr_IRLT_lag4', 'gr_RGDP_lag1']]).T # au début on retient juste 2019-10-31 pour calculer la date d'apres, etc.
            sc_predictions = gb.predict(X_1)
            sc_predictions = pd.DataFrame(sc_predictions, columns = ["DR_pred"])
            X.loc[dates[i+1]]['DR'] = sc_predictions["DR_pred"]
            X["DR_lag1"] = X["DR"].shift(1)
            
        serie_predite = X.loc[X.index > '2019-10-31']["DR"]
        serie_predite_et_val_precente = X.loc[X.index > '2019-07-31']["DR"] #pour qu'il y ait pas de trou dans le plot dû aux dates
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=default_rate.index, y=default_rate.values, name='Série originale'))
        fig2.add_trace(go.Scatter(x=serie_predite_et_val_precente.index, y=serie_predite_et_val_precente.values, name='Série prédite'))
        fig2.update_layout(
            width=900, 
            height=500, 
            title={
                'text': f"Prédicted Time Series Plot for  {scenario}",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 15}
            }
        )

        plt.rcParams["axes.edgecolor"] = "white"
        fig2, ax = plt.subplots()
        ax.plot(y, label='Série originale')
        ax.plot(serie_predite_et_val_precente, label='Série prédite')
        ax.legend()
        st.plotly_chart(fig2)
        
        return serie_predite_et_val_precente

    st.title("2. Predictions plot")
    dfs = {"Variable Macroeconomiques adverses": var_macro_adverse, "Variable Macroeconomiques baselines": var_macro_baseline}
    selected_df_key = st.selectbox("Choose a scenario", list(dfs.keys()))
    selected_df = dfs[selected_df_key]
    prediction_modele_challenge(selected_df)
    
    #_____________________Afficher les tests des modèles 


    # Calculer les résidus
    residus = y_test - pred_gb

    # Effectuer le test de Jarque-Bera sur les résidus
    jb_stat, jb_p_value = stats.jarque_bera(residus)

    # Effectuer le test de Shapiro-Wilk sur les résidus
    shapiro_stat, shapiro_p_value = stats.shapiro(residus)

    # Effectuer le test de Breusch-Pagan pour l'hétéroscédasticité
    bp_stat, bp_p_value, _, _ = het_breuschpagan(residus, X_test)

    # Effectuer le test de Durbin-Watson
    dw_stat = durbin_watson(residus)

    # Créer le data frame
    df = pd.DataFrame({
        'Test': ['Jarque-Bera', 'Shapiro-Wilk', 'Breusch-Pagan', 'Durbin-Watson'],
        'Use':['Normality test', 'Normality test', 'Heteroscedasticity test', 'Autoccorélation test'],
        'Statistique': [jb_stat, shapiro_stat, bp_stat, dw_stat],
        'Valeur p': [jb_p_value, shapiro_p_value, bp_p_value, '']
    })

    def get_conclusion(row):
        if row['Test'] == 'Jarque-Bera' and pd.to_numeric(row['Valeur p']) > 0.05:
            return 'The residuals follow a normal distribution'
        elif row['Test'] == 'Shapiro-Wilk' and pd.to_numeric(row['Valeur p']) > 0.05:
            return 'The residuals follow a normal distribution'
        elif row['Test'] == 'Breusch-Pagan' and pd.to_numeric(row['Valeur p']) > 0.05:
            return 'The residuals do not exhibit heteroscedasticity'
        elif row['Test'] == 'Durbin-Watson' and pd.to_numeric(row['Statistique']) > 1.5 or row['Test'] == 'Durbin-Watson' and pd.to_numeric(row['Statistique']) < 2.5:
            return 'The residuals do not exhibit autocorrelation'
        else:
            return 'The residuals do not appear random'

    df['Conclusion'] = df.apply(get_conclusion, axis=1)

    # Display the data frame with the Conclusion column
    st.title("3. Hypothesis Tests")
    st.write("""
    <div style="overflow-x: auto; max-width: none; text-align: center;">
    <table style="margin: auto;">
    {}
    </table>
    </div>
    """.format(df.to_html(index=False, justify="center")), unsafe_allow_html=True)

    st.title("4. Areas for improvement")
    st.write("""
    <div style="overflow-x: auto; max-width: none; text-align: left;">
    \n
    - LSTM model and data augmentation \n
    - ARIMAX
    </div>
    """.format(df.to_html(index=False, justify="center")), unsafe_allow_html=True)


    # _________________________________________________  button 7________________________

elif st.session_state.current_button =="button7":
    st.markdown("<br/>", unsafe_allow_html=True)
    st.write("""
    <div align="justify"><span style="font-family:Lucida Caligraphy;font-size:18px">
    In conclusion, our econometric model is a powerful tool for predicting default rates in various macroeconomic situations. By using data from macroeconomic variables such as real GDP, interest rates, unemployment rates, inflation rates, and others, our model has been able to provide reliable forecasts of default rates for individual clients. The model is capable of identifying key economic factors that influence the probability of default, such as economic growth.

    The model we selected for our project, which is based on multiple linear regression, demonstrated a high predictive capacity on both training and testing data, with high prediction accuracy for both baseline and adverse macroeconomic scenarios. The results obtained suggest that our model is capable of providing reliable forecasts of default rates and can be used with confidence by financial institutions to evaluate credit risk and make informed decisions on credit granting.

    Furthermore, our model is easy to use and can be easily implemented on a large scale, making it a valuable tool for financial services companies. We are confident that our model is capable of significantly improving the decision-making of financial institutions in credit risk and contributing to overall financial stability.
        </span></div>
        """,  unsafe_allow_html=True)
  
    image_t = Image.open(r"image\thank.png")
    col1, col2, col3, col4, col5= st.columns(5)
    with col3:
        st.image(image_t, width=300, use_column_width=False)

