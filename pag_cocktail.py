import streamlit as st
import pandas as pd
from prophet.plot import plot_plotly
import joblib
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import acf
from prophet.diagnostics import cross_validation,performance_metrics
from datetime import date


def main():
    pd.options.display.float_format = '{:,.1f}'.format
    st.subheader('Tabella dati input - Aggregazione giornaliera')
    df = pd.read_excel('dati.xlsx', parse_dates=True,dtype={"Quantita":int})
    copia_df=df.copy()
    df_cocktail=copia_df[copia_df['Tipologia']=='Cocktail']
    df_cocktail=df_cocktail.drop('Tipologia',axis=1)
    df_cocktail['Data'] = df_cocktail['Calendario']
    df_cocktail = df_cocktail.drop('Calendario',axis=1)
    quanti = st.slider("Selezionare la dimensione del dataset",100,len(df_cocktail),250)
    df_cocktail = df_cocktail[['Data','Quantita']]
    #df_cocktail['Quantita']=round(df_cocktail['Quantita'])
    df_cocktail['Data'] = pd.to_datetime(df_cocktail['Data']).dt.date
    df_cocktail['Data']=pd.to_datetime(df_cocktail['Data'],format='%Y/%m/%d')
    df_cocktail['Data']=df_cocktail['Data'].dt.strftime('%d/%m/%Y')
    df_cocktail.columns=['Data','Quantità']
    df_cocktail = df_cocktail.reset_index()
    df_cocktail = df_cocktail.drop('index',axis=1)
    st.dataframe(df_cocktail.head(quanti))

    st.subheader('Tabella statistica riassuntiva')
    st.dataframe(df_cocktail.describe().T)

    st.subheader('Istogramma sui dati di vendita dei cocktails')
    bins_scelti = st.slider(
        'Selezionare il numero di bins',
        40, 100, 60)
    isto = df_cocktail['Quantità'].plot(kind='hist', bins=bins_scelti)
    st.pyplot(isto.figure,clear_figure=True)

    st.subheader('Lag plot dei cocktails')
    st.write ("""Un lag plot è un grafico utilizzato in statistica per individuare la presenza di autocorrelazione nei dati. L'autocorrelazione si riferisce alla dipendenza tra gli elementi di una serie temporale, ossia alla presenza di una relazione tra i valori di una variabile a distanza di un certo intervallo di tempo (detto lag). Se ad esempio utilizziamo un lag di 7 giorni, il primo punto avrà coordinate u = y(1) e v = y(1+7) = y(8), il secondo punto u = y(2) e v = y(2+7) = y(9) e così via, dove y(t) in questo caso è il numero di cocktails venduti al tempo t. Idealmente l'autocorrelazione è, in valore assoluto, uguale a 1, che è il caso in cui tutti i punti giacciono sulla stessa retta.\nTuttavia, è importante ricordare che il lag plot è tanto più inaffidabile quanto più è alto il numero di giorni mancanti nel dataset.""")
    lag = int(st.text_input(f'Scegli il lag (dev\'essere un numero intero compreso tra 1 e 365)',value=1))
    if (lag>365) or (lag<1):
        st.write(f'Devi inserire un numero intero compreso tra 1 e 365!')
    else:
        grafico_lag = lag_plot(df_cocktail['Quantità'],lag)
        st.pyplot(grafico_lag.figure,clear_figure=True)

        autocorrelation_vet = acf(df_cocktail['Quantità'],nlags=lag)
        autocorrelation = autocorrelation_vet[-1]
        st.write(f'L\'autocorrelazione di questo lag plot è del {round(100*autocorrelation,2)}%')    

    st.subheader('Dataset senza outliers')
    # Calcola la media e la deviazione standard
    mean = df_cocktail['Quantità'].mean()
    std = df_cocktail['Quantità'].std()

    #Q1 = df_cocktail['Quantità'].quantile(0.25)
    #Q3 = df_cocktail['Quantità'].quantile(0.75)

    # Crea l'intervallo di deviazione standard
    lower = mean - std   
    upper = mean + 2 * std  
    #lower = Q1-1.5*(Q3-Q1)
    #upper = Q3+1.5*(Q3-Q1)

    df_cocktail_rid = df_cocktail[(df_cocktail['Quantità']>lower) & (df_cocktail['Quantità']<upper)]
    df_cocktail_rid = df_cocktail_rid.reset_index()
    df_cocktail_rid = df_cocktail_rid.drop('index',axis=1)
    quanti2 = st.slider("Selezionare la dimensione del dataset.",100,len(df_cocktail_rid),250)
    st.dataframe(df_cocktail_rid.head(quanti2))
    st.write("""Per evitare di considerare valori troppo grandi o troppo piccoli (outliers) che rischiano di fuorviare le previsioni, sono stati utilizzati solo i dati contenuti entro 2 deviazioni standard a destra della media e entro 1 deviazione standard a sinistra della media""")

    st.subheader('Tabella statistica riassuntiva senza outliers')
    st.dataframe(df_cocktail_rid.describe().T)

    st.subheader('Istogramma dei cocktails senza outliers')
    bins_scelti2 = st.slider(
        'Selezionare il numero bins',
        40, 100, 60)
    isto2 = df_cocktail_rid['Quantità'].plot(kind='hist', bins=bins_scelti2)
    st.pyplot(isto2.figure,clear_figure=True)

    st.subheader('Lag plot dei cocktails senza outliers')
    lag2 = int(st.text_input(f'Scegli il lag (dev\'essere un numero intero compreso tra 1 e 365)',value=7))
    if (lag2>365) or (lag2<1):
        st.write(f'Devi inserire un numero intero compreso tra 1 e 365!')
    else:
        grafico_lag2 = lag_plot(df_cocktail_rid['Quantità'],lag2)
        st.pyplot(grafico_lag2.figure,clear_figure=True)
        
        autocorrelation_vet2 = acf(df_cocktail_rid['Quantità'],nlags=lag2)
        autocorrelation2 = autocorrelation_vet2[-1]
        st.write(f'L\'autocorrelazione di questo lag plot è del {round(100*autocorrelation2,2)}%')

    model = joblib.load('model_cocktail.pkl')

    st.subheader('Componenti dei cocktails senza outliers')
    quanto_trend = st.slider('Scegliere il numero di giorni di cui vedere le componenti future',0,180,60)
    future = model.make_future_dataframe(periods=quanto_trend)
    forecast = model.predict(future)
    comp = model.plot_components(forecast)
    st.pyplot(comp.figure,clear_figure=True)

    st.subheader('Forecasting')
    da_pred = st.slider('Scegliere il numero di giorni da prevedere',0,180,60)
    future = model.make_future_dataframe(da_pred, freq='D')
    forecast = model.predict(future)
    fig = plot_plotly(model, forecast)
    fig.update_layout(title="Previsione dei cocktails venduti",
                    yaxis_title='Cocktails venduti',
                    xaxis_title="Data",
                    )
    fig.add_vline(x=date.today(), line_width=3, line_dash="dash", line_color="red")
    fig.update_layout(width=1200)
    st.plotly_chart(fig)

    df_cv_final=cross_validation(model,
                            horizon="60 days",
                            period='10 days',
                            initial='450 days',
                            )
    df_performance=performance_metrics(df_cv_final)
    mape = df_performance['mape'].mean()
    st.write(f'L\'errore percentuale medio della previsione calcolato sugli ultimi 60 giorni è del {round(mape*100,2)}%')

if __name__ == "__main__":
    main()
