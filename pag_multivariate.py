import streamlit as st
import pandas as pd
from datetime import date, datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily
import seaborn as sn
import plotly.graph_objects as go

def main():
    pd.options.display.float_format = '{:,.1f}'.format
    
    df = pd.read_excel('dati.xlsx', index_col='Calendario', parse_dates=True, dtype={"Quantita":int})

    index = df.index

    df_spine = df[df['Tipologia']=='Spine']
    df_cocktail = df[df['Tipologia']=='Cocktail']
    df_burger = df[df['Tipologia']=='Burger']
    df_fritti = df[df['Tipologia']=='Fritti']
    df_bar = df[df['Tipologia']=='Bar']
    
    st.subheader('Grafico complessivo delle vendite')

    # Grafico del dataset fornito da Bifor

    fig = go.Figure()
    #Spine 
    fig.add_trace(go.Scatter(x = index, 
                            y = df_spine['Quantita'],
                            mode = "lines",
                            name = "Spine",
                            line_color='#0000FF',
                            ))
    ##############################################################
    #Cocktail 
    fig.add_trace(go.Scatter(x = index, 
                            y = df_cocktail['Quantita'],
                            mode = "lines", 
                            name = "Cocktail",
                            line_color='#ff8c00',
                            ))
    ##############################################################
    #Burger 
    fig.add_trace(go.Scatter(x = index, 
                            y = df_burger['Quantita'],
                            mode = "lines", 
                            name = "Burger",
                            line_color='#FF00FF',
                            ))
    ##############################################################
    #Fritti 
    fig.add_trace(go.Scatter(x = index, 
                            y = df_fritti['Quantita'],
                            mode = "lines", 
                            name = "Fritti",
                            line_color='#00FF00',
                            ))
    ##############################################################
    #Bar 
    fig.add_trace(go.Scatter(x = index, 
                            y = df_bar['Quantita'],
                            mode = "lines", 
                            name = "Bar",
                            line_color='#FF0000',
                            ))
    ##############################################################
    # adjust layout
    fig.update_layout(title = "Vendite Bifor",
                    xaxis_title = "Data",
                    yaxis_title = "Quantità",
                    width = 1200,
                    height = 700,
                    )
    ####################################################################
    # zoomming
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=15, label="15m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig)




    #Tabella delle correlazioni

    st.subheader('Matrice delle correlazioni')

    st.write("""La matrice delle correlazioni, detta heatmap, mostra il livello di correlazione tra tutte le possibili coppie di variabili. """)

    # Crea una serie di date utilizzando il metodo date_range di Pandas
    date_range = pd.date_range(start='2021-05-01', end=df_bar.index[-1]) #-1 coincide con l'indice finale. In pyhton gli indici negativi iniziano dalla fine dell'array

    # Crea un nuovo DataFrame utilizzando il metodo reindex
    df_reindexed_bar = df_bar.Quantita.reindex(date_range)

    # Sostituisci i valori nulli con 0
    df_reindexed_bar = df_reindexed_bar.fillna(0)

    # Crea una serie di date utilizzando il metodo date_range di Pandas
    date_range = pd.date_range(start='2021-05-01', end=df_fritti.index[-1]) #-1 coincide con l'indice finale. In pyhton gli indici negativi iniziano dalla fine dell'array

    # Crea un nuovo DataFrame utilizzando il metodo reindex
    df_reindexed_fritti = df_fritti.Quantita.reindex(date_range)

    # Sostituisci i valori nulli con 0
    df_reindexed_fritti = df_reindexed_fritti.fillna(0)

    # Crea una serie di date utilizzando il metodo date_range di Pandas
    date_range = pd.date_range(start='2021-05-01', end=df_burger.index[-1]) #-1 coincide con l'indice finale. In pyhton gli indici negativi iniziano dalla fine dell'array

    # Crea un nuovo DataFrame utilizzando il metodo reindex
    df_reindexed_burger = df_burger.Quantita.reindex(date_range)

    # Sostituisci i valori nulli con 0
    df_reindexed_burger = df_reindexed_burger.fillna(0)

    # Crea una serie di date utilizzando il metodo date_range di Pandas
    date_range = pd.date_range(start='2021-05-01', end=df_cocktail.index[-1]) #-1 coincide con l'indice finale. In pyhton gli indici negativi iniziano dalla fine dell'array

    # Crea un nuovo DataFrame utilizzando il metodo reindex
    df_reindexed_cocktail = df_cocktail.Quantita.reindex(date_range)

    # Sostituisci i valori nulli con 0
    df_reindexed_cocktail = df_reindexed_cocktail.fillna(0)

    # Crea una serie di date utilizzando il metodo date_range di Pandas
    date_range = pd.date_range(start='2021-05-01', end=df_spine.index[-1]) #-1 coincide con l'indice finale. In pyhton gli indici negativi iniziano dalla fine dell'array

    # Crea un nuovo DataFrame utilizzando il metodo reindex
    df_reindexed_spine = df_spine.Quantita.reindex(date_range)

    # Sostituisci i valori nulli con 0
    df_reindexed_spine = df_reindexed_spine.fillna(0)

    dataframe=pd.DataFrame({
                    "spine":  df_reindexed_spine,
                    "bar": df_reindexed_bar, 
                    "cocktail": df_reindexed_cocktail,
                    "fritti":df_reindexed_fritti,
                    "burger":df_reindexed_burger})
    
    #Correlazione con la temperatura
    # Import Meteostat library and dependencies

    # Set time period
    start = datetime(2021, 5, 1)
    end = datetime(2022, 11, 22) 

    cities = {'Forli':[44.233334,12.050000]}

    city = Point(list(cities.values())[0][0],list(cities.values())[0][1], 20)

    # Get daily data for 2018
    data = Daily(city, start, end)
    data = data.fetch()
    data['city'] = list(cities.keys())[0]

    df_temp=data['tavg']
    df_prcp = data['prcp']

    # Crea una serie di date utilizzando il metodo date_range di Pandas
    date_range = pd.date_range(start='2021-05-01', end=data.index[-1]) #-1 coincide con l'indice finale. In pyhton gli indici negativi iniziano dalla fine dell'array

    # Crea un nuovo DataFrame utilizzando il metodo reindex
    df_reindexed_temp = df_temp.reindex(date_range)

    # Sostituisci i valori nulli con 0
    df_reindexed_temp = df_reindexed_temp.fillna(0)

    # Crea una serie di date utilizzando il metodo date_range di Pandas
    date_range = pd.date_range(start='2021-05-01', end=data.index[-1]) #-1 coincide con l'indice finale. In pyhton gli indici negativi iniziano dalla fine dell'array

    # Crea un nuovo DataFrame utilizzando il metodo reindex
    df_reindexed_prcp = df_prcp.reindex(date_range)

    # Sostituisci i valori nulli con 0
    df_reindexed_prcp = df_reindexed_prcp.fillna(0)

    dataframe_t = pd.DataFrame({
                    "spine":  df_reindexed_spine,
                    "bar": df_reindexed_bar, 
                    "cocktail": df_reindexed_cocktail,
                    "fritti":df_reindexed_fritti,
                    "burger":df_reindexed_burger,
                    "tavg": df_reindexed_temp,
                    "prcp": df_reindexed_prcp})

    dataframe_t2 = dataframe_t[dataframe_t['burger'] != 0]
    dataframe_t2 = dataframe_t2[dataframe_t2['spine'] != 0] 
    dataframe_t2 = dataframe_t2[dataframe_t2['fritti'] != 0]
    dataframe_t2 = dataframe_t2[dataframe_t2['cocktail'] != 0]
    dataframe_t2 = dataframe_t2[dataframe_t2['bar'] != 0]

    corr_matrix_t2 = dataframe_t2.corr()

    heatmap = sn.heatmap(corr_matrix_t2, annot=True)

    st.pyplot(heatmap.figure)

    st.write("""Da questa heatmap si può notare che la correlazione più alta con la temperatura è quella con le vendite del bar.
    \nInoltre, non si ha alcuna correlazione tra la quantità di precipitazioni e le vendite. """)


if __name__ == "__main__":
    main()