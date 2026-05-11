# ======================
# IMPORT
# ======================

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import joblib
import pickle
from sklearn.metrics import accuracy_score, recall_score


# Scarichiamo il df finale che abbiamo generato dopo tutte le fasi di pulizia e feature engineering

df = pd.read_csv(os.path.join('.', 'modelli/df_final.csv'))

# Titolo dashboard

st.title("📊 Analisiincidenti stradali UK")

# PARTE DI ANALISI ESPLORATIVA

st.header("📈 Data Exploration")

# Facciamo scegliere all'utente se vuole analizzare la severità originale (3 classi) o quella aggregata (2 classi)

option = st.selectbox(
    "Seleziona la variabile target",
    ["Originale (3 classi)", "Aggregata (2 classi)"],
    index = 1
)

# Impostiamo le variabili e i titoli in base alla scelta dell'utente

if option == "Originale (3 classi)":
    col = "collision_severity"
    mapping = {
        1: "Mortale",
        2: "Grave",
        3: "Lieve"
    }
    title1 = "Distribuzione target originale"
    title2 = "Evoluzione temporale per target originale"
    order = ["Mortale", "Grave", "Lieve"]

else:
    col = "collision_severity_coarse"
    mapping = {
        2: "Grave o Mortale",
        3: "Lieve"
    }
    title1 = "Distribuzione target aggregata"
    title2 = "Evoluzione temporale per target aggregata"
    order = ["Grave o Mortale", "Lieve"]


# Estraiamo la distribuzione della variabile target scelta dall'utente per il primo grafico a torta

dist = (
    df[col]
    .value_counts(normalize=True)
    .reset_index()
)

dist.columns = ["collision severity", "fraction"]

dist["collision severity"] = dist["collision severity"].map(mapping)

# Grafico a torta con Plotly Express

fig = px.pie(
    dist,
    values="fraction",
    names="collision severity",
    color="collision severity",
    color_discrete_map={
        "Mortale": "red",
        "Grave": "darkblue",
        "Grave o Mortale": "purple",
        "Lieve": "lightgray",
    },
    category_orders={"collision severity": order}
)

fig.update_layout(
    title={'text': title1, 'font': {'size': 30}},
    legend=dict(font=dict(size=20))
)

fig.update_traces(textinfo="percent+label", textfont_size=16, rotation=276)

st.plotly_chart(fig, use_container_width=True)



# Per il secondo grafico a barre, creiamo un df temporaneo per mappare la variabile target e poi raggruppiamo per anno e classe

df_temp = df.copy()

# Mappa le classi
df_temp["severity label"] = df_temp[col].map(mapping)

# Conteggi per anno e classe
agg = (
    df_temp
    .groupby(["collision_year", "severity label"])
    .size()
    .reset_index(name="count")
)

# Grafico a barre con Plotly Express

fig = px.bar(
    agg,
    x="collision_year",
    y="count",
    color="severity label",
    barmode="group",
    color_discrete_map={
        "Mortale": "red",
        "Grave": "darkblue",
        "Grave o Mortale": "purple",
        "Lieve": "lightgray",
    },
    category_orders={"severity label": order}
)

# Aggiungiamo una linea con il totale degli incidenti per anno

tot = (
    df_temp
    .groupby("collision_year")
    .size()
    .reset_index(name="total")
)

fig.add_scatter(
    x=tot["collision_year"],
    y=tot["total"],
    mode="lines+markers",
    name="Totale incidenti",
    line=dict(color="green", width=3)
)

fig.update_layout(
    title={'text': title2, 'font': {'size': 30}},
    legend=dict(font=dict(size=20), title=""),
    xaxis_title="Anno",
    yaxis_title="Numero di incidenti"
)

st.plotly_chart(fig, use_container_width=True)

# Definiamo le varibaili tra cui scegliere per i grafici della target condizionata e il relativo mapping per migliorare la leggibilità dei grafici

plot_cols = [
    'number_of_vehicles',
    'number_of_casualties',
    'first_road_class',
    'speed_limit',
    'junction_detail_historic',
    'road_surface_conditions',
    'pedestrian_crossing_physical_facilities_historic',
    'light_conditions',
    'weather_conditions',
    'urban_or_rural_area',
    'range_hour',
    'country',
    'hazard_grouped',
    'flag_driver/rider',
    'flag_passenger',
    'flag_pedestrian',
    'flag_age_<10',
    'flag_age_11-55',
    'flag_age_>55',
    'flag_pedestrian_age_>55',
    'flag_moto125',
    'flag_agricolo',
    'flag_tram',
    'flag_van',
    'flag_moto500',
    'flag_moto_heavy',
    'flag_dark_pedestrian',
    'did_police_officer_attend_scene_of_accident'
]

col_labels = {
    'number_of_vehicles': "Numero di veicoli",
    'number_of_casualties': "Numero di feriti",
    'first_road_class': "Classe della strada",
    'speed_limit': "Limite di velocità",
    'junction_detail_historic': "Tipo di incrocio",
    'road_surface_conditions': "Condizioni del manto stradale",
    'pedestrian_crossing_physical_facilities_historic': "Attraversamento pedonale",
    'light_conditions': "Condizioni di luce",
    'weather_conditions': "Condizioni meteo",
    'urban_or_rural_area': "Area urbana o rurale",
    'range_hour': "Fascia oraria",
    'country': "Paese",
    'hazard_grouped': "Tipo di pericolo",
    'did_police_officer_attend_scene_of_accident': "Presenza polizia",

    'flag_driver/rider': "Coinvolgimento conducente",
    'flag_passenger': "Coinvolgimento passeggero",
    'flag_pedestrian': "Coinvolgimento pedone",

    'flag_age_<10': "Età < 10 anni",
    'flag_age_11-55': "Età 11–55 anni",
    'flag_age_>55': "Età > 55 anni",
    'flag_pedestrian_age_>55': "Pedone > 55 anni",

    'flag_moto125': "Moto ≤125cc",
    'flag_moto500': "Moto >125cc",
    'flag_agricolo': "Veicolo agricolo coinvolto",
    'flag_tram': "Tram coinvolto",
    'flag_van': "Furgone coinvolto",
    'flag_moto_heavy': "Moto e veicolo pesante coinvolti",
    'flag_dark_pedestrian': "Pedone in condizioni di oscurità"
}


label_maps = {
    'first_road_class':    {1:'Motorway', 2:'A(M)', 3:'A', 4:'B', 5:'C', 6:'Unclassified'},
    'light_conditions':    {1:'Daylight', 4:'Dark-lit', 5:'Dark-unlit', 6:'Dark-no light', 7:'Dark-unknown'},
    'weather_conditions':  {1:'Fine', 2:'Rain', 3:'Snow', 4:'Fine+wind', 5:'Rain+wind', 6:'Snow+wind', 7:'Fog', 8:'Other', 9:'Unknown'},
    'junction_control':    {0:'No junction', 1:'Auth.person', 2:'Traffic signal', 3:'Stop sign', 4:'Give way', 9:'Unknown'},
    'junction_detail_historic': {0:'No junction', 1:'Roundabout', 2:'Mini-roundabout', 3:'T-junction', 5:'Slip road', 6:'Crossroads', 7:'4+ arms', 8:'Private', 9:'Other', 99:'Unknown'},
    'road_surface_conditions': {1:'Dry', 2:'Wet', 3:'Snow', 4:'Frost/ice', 5:'Flood', 9:'Unknown'},
    'urban_or_rural_area': {1:'Urban', 2:'Rural'},
    'pedestrian_crossing_physical_facilities_historic': {0:'None', 1:'Zebra', 4:'Pelican/Puffin', 5:'Traffic signal', 7:'Footbridge', 8:'Central refuge', 9:'Unknown'},
    'range_hour':          {0:'Night(22-4)', 1:'Morning(4-10)', 2:'Midday(10-16)', 3:'Evening(16-22)'},
    'did_police_officer_attend_scene_of_accident': {2:'Assente', 1:'Presente'}
}


flag_maps = {
    col: {0: "Assente", 1: "Presente"}
    for col in [
        'flag_driver/rider',
        'flag_passenger',
        'flag_pedestrian',
        'flag_age_<10',
        'flag_age_11-55',
        'flag_age_>55',
        'flag_pedestrian_age_>55',
        'flag_moto125',
        'flag_agricolo',
        'flag_tram',
        'flag_van',
        'flag_moto500',
        'flag_moto_heavy',
        'flag_dark_pedestrian'
    ]
}


# UI per selezionare la variabile da analizzare in relazione alla severità

feature_label = st.selectbox(
    "Seleziona la variabile da analizzare in relazione alla severità",
    [col_labels[c] for c in plot_cols],
    index = 3
)

feature = {v: k for k, v in col_labels.items()}[feature_label]

# Funzione per mappare le variabili in base al tipo (target, label o flag)

def apply_labels(df, col):
    if col in label_maps:
        return df[col].map(label_maps[col])
    elif col in flag_maps:
        return df[col].map(flag_maps[col])
    else:
        return df[col]
    


df_temp["feature_mapped"] = apply_labels(df_temp, feature)

# Creiamo una tabella pivot da cui fare il grafico a barre

pivot = (
    df_temp
    .groupby(["feature_mapped", "severity label"])
    .size()
    .unstack(fill_value=0)
)

pivot = pivot[order]  # Assicura l'ordine delle colonne

pivot = pivot.div(pivot.sum(axis=1), axis=0)

# Grafico a barre con Plotly Express

fig = px.bar(
    pivot.reset_index(),
    x="feature_mapped",
    y=pivot.columns, 
    barmode="stack",
    color_discrete_map={
        "Mortale": "red",
        "Grave": "darkblue",
        "Grave o Mortale": "purple",
        "Lieve": "lightgray",
    }
)


fig.update_layout(
    title={'text': f"Distribuzione severità vs {col_labels[feature]}", 'font': {'size': 30}},
    xaxis_title=col_labels[feature],
    yaxis_title="Proporzione",
    legend=dict(font=dict(size=20), title="")
)

fig.update_xaxes(tickangle=30)


st.plotly_chart(fig, use_container_width=True)


# Definiamo un sottogruppo di variabili tra cui scegliere per visualizzare tabelle di contingenza

col_labels2 = {
    
    'first_road_class': "Classe della strada",
    'speed_limit': "Limite di velocità",
    'junction_detail_historic': "Tipo di incrocio",
    'road_surface_conditions': "Condizioni del manto stradale",
    'pedestrian_crossing_physical_facilities_historic': "Attraversamento pedonale",
    'light_conditions': "Condizioni di luce",
    'weather_conditions': "Condizioni meteo",
    'urban_or_rural_area': "Area urbana o rurale",
    'range_hour': "Fascia oraria",
    'country': "Paese",
    'hazard_grouped': "Tipo di pericolo",
    'did_police_officer_attend_scene_of_accident': "Presenza polizia",

    'flag_driver/rider': "Coinvolgimento conducente",
    'flag_passenger': "Coinvolgimento passeggero",
    'flag_pedestrian': "Coinvolgimento pedone",

    'flag_age_<10': "Età < 10 anni",
    'flag_age_11-55': "Età 11–55 anni",
    'flag_age_>55': "Età > 55 anni",
    'flag_pedestrian_age_>55': "Pedone > 55 anni",

    'flag_moto125': "Moto ≤125cc",
    'flag_moto500': "Moto >125cc",
    'flag_agricolo': "Veicolo agricolo coinvolto",
    'flag_tram': "Tram coinvolto",
    'flag_van': "Furgone coinvolto",
    'flag_moto_heavy': "Moto e veicolo pesante coinvolti",
    'flag_dark_pedestrian': "Pedone in condizioni di oscurità"
}

target = col

col_labels2[target] = "Severità incidente"

label_to_col2 = {v: k for k, v in col_labels2.items()}

# UI per selezionare le due variabili da mettere in relazione nella tabella di contingenza

col1_label = st.selectbox(
    "Seleziona la variabile riga",
    list(label_to_col2.keys()),
    key="col1",
    index = 7
)


col2_label = st.selectbox(
    "Seleziona la variabile colonna",
    list(label_to_col2.keys()),
    key="col2",
    index = 1
)

col1 = label_to_col2[col1_label]
col2 = label_to_col2[col2_label]

if col1 == col2:
    st.warning("Seleziona due variabili diverse")
    st.stop()

# Creiamo un df temporaneo per mappare le variabili in base al tipo (target, label o flag) e poi generare la tabella di contingenza

df_temp = df.copy()

# mapping target SOLO se selezionata

def map_column(df, col):
    if col == target:
        return df[col].map(mapping)
    elif col in label_maps:
        return df[col].map(label_maps[col])
    elif col in flag_maps:
        return df[col].map(flag_maps[col])
    else:
        return df[col]

df_temp["col1_mapped"] = map_column(df_temp, col1)
df_temp["col2_mapped"] = map_column(df_temp, col2)

df_temp = df_temp.dropna(subset=["col1_mapped", "col2_mapped"])


pivot = pd.crosstab(
    df_temp["col1_mapped"],
    df_temp["col2_mapped"]
)

# UI per scegliere il tipo di normalizzazione della tabella di contingenza da visualizzare nella heatmap

mode = st.radio(
    "Normalizzazione",
    ["Percentuale per riga", "Percentuale per colonna", "Percentuale totale"],
    horizontal=True
)

if mode == "Percentuale per riga":
    pivot = pivot.div(pivot.sum(axis=1), axis=0)

elif mode == "Percentuale per colonna":
    pivot = pivot.div(pivot.sum(axis=0), axis=1)

elif mode == "Percentuale totale":
    pivot = pivot / pivot.values.sum()

# Grafico a barre con Plotly Express

fig = px.imshow(
    pivot.astype(float),
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="Blues"
)

fig.update_layout(
    title={'text': f"{col1_label} vs {col2_label}", 'font': {'size': 30}},
    xaxis_title=col2_label,
    yaxis_title=col1_label,
)

st.plotly_chart(fig, use_container_width=True)



#  PARTE SUL MODELLO PREDITTIVO

st.header("🤖 Test del modello predittivo ")

# Carichiamo la pipeline che abbiamo salvato dopo la fase di training

pipeline = joblib.load("modelli/xgboost_pipeline_2024.pkl")

# Definiamo le feature che il modello si aspetta in input

feature_cols = [
    'number_of_casualties',
    'number_of_vehicles',
    'speed_limit',
    'country',
    'did_police_officer_attend_scene_of_accident',
    'flag_pedestrian',
    'flag_pedestrian_age_>55',
    'flag_age_>55',
    'flag_moto125',
    'urban_or_rural_area',
    'flag_dark_pedestrian',
    'flag_moto_heavy'
]

# Mappature per migliorare la leggibilità dei dropdown nella UI, sfruttando le stesse mappature usate in precedenza

all_maps = {}
all_maps.update(label_maps)
all_maps.update(flag_maps)

# Dizionario per salvare gli input dell'utente

user_input = {}

# UI per inserire i valori delle feature 1 per 1, con il widget più adatto in base al tipo di variabile

with st.sidebar:

    st.header("Parametri incidente")

    for col in feature_cols:

        label = col_labels.get(col, col)

        if col in all_maps:

            value_map = all_maps[col]

            reverse_map = {
                v: k for k, v in value_map.items()
            }

            selected_label = st.selectbox(
                label,
                options=list(reverse_map.keys())
            )

            user_input[col] = reverse_map[selected_label]


        elif df[col].dtype == "str":

            options = sorted(df[col].dropna().unique())

            user_input[col] = st.selectbox(
                label,
                options
            )

        else:

            min_val = int(df[col].min())
            max_val = int(df[col].max())
            med_val = int(df[col].median())

            user_input[col] = st.slider(
                label,
                min_value=min_val,
                max_value=max_val,
                value=med_val
            )

# Creiamo un df con un'unica riga a partire dagli input dell'utente, da passare alla pipeline per la predizione

input_df = pd.DataFrame([user_input])

# Visualizziamo all'utente i valori che ha inserito prima di fare la predizione, mappando le variabili per migliorare la leggibilità

st.subheader("Input modello")

preview_df = input_df.copy()

for col in preview_df.columns:

    if col in all_maps:
        preview_df[col] = preview_df[col].map(all_maps[col])

st.dataframe(preview_df)

# Inizializziamo una variabile di sessione per salvare la predizione e visualizzarla dopo che l'utente ha cliccato il bottone, in modo da non farla sparire se l'utente vuole modificare un input e rifare la predizione

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Bottone per fare la predizione, che salva il risultato in una variabile di sessione

if st.button("Predici"):
    prob = pipeline.predict_proba(input_df)[0][1]

    threshold = 0.46
    pred_bin = int(prob >= threshold)
    pred_class = 2 if pred_bin == 1 else 3

    st.session_state.prediction = {
        "prob": prob,
        "class": pred_class
    }

# Visualizziamo il risultato della predizione solo se è stata fatta, con una mappatura per rendere più leggibile la classe prevista e un messaggio di rischio basato sulla probabilità prevista

if st.session_state.prediction is not None:

    st.subheader("Risultato")

    mapping_result = {2: "Grave", 3: "Lieve"}

    st.metric(
        "Classe prevista",
        mapping_result[st.session_state.prediction["class"]]
    )

    st.metric(
        "Probabilità classe grave",
        f"{st.session_state.prediction['prob']*100:.1f}%"
    )

    prob = st.session_state.prediction["prob"]

    if prob > 0.7:
        st.error("ALTO RISCHIO")
    elif prob > 0.4:
        st.warning("RISCHIO MEDIO")
    else:
        st.success("RISCHIO BASSO")


