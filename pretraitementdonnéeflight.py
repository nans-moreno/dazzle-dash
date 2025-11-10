import polars as pl
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# 1. Chargement des données
# =========================

print("📂 Chargement des données...")

flights_pl = pl.read_csv(
    "flights.csv",
    ignore_errors=True,
    try_parse_dates=True,
    null_values=["", "NA", "NaN"]
)

airlines_df = pd.read_csv("airlines.csv")
airports_df = pd.read_csv("airports.csv")

flights = flights_pl.to_pandas()

print(f"✅ {len(flights):,} vols chargés")

# =========================
# 2. Nettoyage et filtrage
# =========================

# Supprimer les lignes avec valeurs manquantes
flights = flights.dropna(subset=['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
flights = flights[flights['ORIGIN_AIRPORT'] != flights['DESTINATION_AIRPORT']]

# SUPPRIMER LES DESTINATIONS AVEC DES CHIFFRES
def contains_digit(code):
    """Vérifie si un code contient des chiffres"""
    if pd.isna(code):
        return True
    return any(char.isdigit() for char in str(code))

# Filtrer les destinations et origines avec des chiffres
flights = flights[~flights['DESTINATION_AIRPORT'].apply(contains_digit)]
flights = flights[~flights['ORIGIN_AIRPORT'].apply(contains_digit)]
flights = flights[~flights['AIRLINE'].apply(contains_digit)]

print(f"✅ Après suppression des codes avec chiffres : {len(flights):,} vols")

# =========================
# 3. Dictionnaires de mapping
# =========================

airline_names = dict(zip(airlines_df['IATA_CODE'], airlines_df['AIRLINE']))
airport_names = dict(zip(airports_df['IATA_CODE'], airports_df['AIRPORT']))
airport_cities = dict(zip(airports_df['IATA_CODE'], airports_df['CITY']))
airport_latitudes = dict(zip(airports_df['IATA_CODE'], airports_df['LATITUDE']))
airport_longitudes = dict(zip(airports_df['IATA_CODE'], airports_df['LONGITUDE']))

MOIS_NOMS = {
    1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin',
    7: 'Juillet', 8: 'Août', 9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
}

SAISONS = {
    1: 'Hiver', 2: 'Hiver', 3: 'Hiver',
    4: 'Printemps', 5: 'Printemps', 6: 'Printemps',
    7: 'Été', 8: 'Été', 9: 'Été',
    10: 'Automne', 11: 'Automne', 12: 'Automne'
}

# =========================
# 4. Enrichissement
# =========================

flights['AIRLINE_NAME'] = flights['AIRLINE'].map(airline_names).fillna(flights['AIRLINE'])
flights['ORIGIN_NAME'] = flights['ORIGIN_AIRPORT'].map(airport_names).fillna(flights['ORIGIN_AIRPORT'])
flights['ORIGIN_CITY'] = flights['ORIGIN_AIRPORT'].map(airport_cities).fillna('')
flights['DESTINATION_NAME'] = flights['DESTINATION_AIRPORT'].map(airport_names).fillna(flights['DESTINATION_AIRPORT'])
flights['DESTINATION_CITY'] = flights['DESTINATION_AIRPORT'].map(airport_cities).fillna('')

flights['ORIGIN_LAT'] = flights['ORIGIN_AIRPORT'].map(airport_latitudes)
flights['ORIGIN_LON'] = flights['ORIGIN_AIRPORT'].map(airport_longitudes)
flights['DEST_LAT'] = flights['DESTINATION_AIRPORT'].map(airport_latitudes)
flights['DEST_LON'] = flights['DESTINATION_AIRPORT'].map(airport_longitudes)

for col in ["DEPARTURE_DELAY", "ARRIVAL_DELAY", "MONTH", "DEPARTURE_TIME", "SCHEDULED_DEPARTURE"]:
    if col in flights.columns:
        flights[col] = pd.to_numeric(flights[col], errors="coerce")

flights['MONTH_NAME'] = flights['MONTH'].map(MOIS_NOMS)
flights['SEASON'] = flights['MONTH'].map(SAISONS)

if "SCHEDULED_DEPARTURE" in flights.columns:
    flights["DEPARTURE_HOUR"] = (flights["SCHEDULED_DEPARTURE"] // 100).fillna(0).astype(int)
else:
    flights["DEPARTURE_HOUR"] = 0

flights["ON_TIME"] = flights["ARRIVAL_DELAY"] <= 15
flights["CANCELLED"] = flights.get("CANCELLED", 0).fillna(0).astype(int)

# Calcul des directions
lat_diff = flights['DEST_LAT'] - flights['ORIGIN_LAT']
lon_diff = flights['DEST_LON'] - flights['ORIGIN_LON']

flights['DIRECTION_VERTICAL'] = np.where(
    lat_diff > 0, 'Sud → Nord',
    np.where(lat_diff < 0, 'Nord → Sud', 'Horizontal')
)

flights['DIRECTION_HORIZONTAL'] = np.where(
    lon_diff > 0, 'Ouest → Est',
    np.where(lon_diff < 0, 'Est → Ouest', 'Vertical')
)

mask_invalid = (flights['ORIGIN_LAT'].isna() | flights['DEST_LAT'].isna())
flights.loc[mask_invalid, 'DIRECTION_VERTICAL'] = 'Inconnu'
flights.loc[mask_invalid, 'DIRECTION_HORIZONTAL'] = 'Inconnu'

print(f"✅ Données enrichies : {len(flights):,} vols prêts")

# =========================
# 5. Création des graphiques (CODES sur axes, NOMS au survol)
# =========================

print("📊 Création des graphiques...")

# KPI 1: Délai par compagnie (CODE sur axe)
delay_by_airline = flights.groupby(['AIRLINE', 'AIRLINE_NAME'])['ARRIVAL_DELAY'].mean().sort_values(ascending=False).head(15).reset_index()
fig1 = px.bar(delay_by_airline, x='AIRLINE', y='ARRIVAL_DELAY',
              title="Délai moyen par compagnie (Top 15)",
              labels={'ARRIVAL_DELAY': 'Délai moyen (min)', 'AIRLINE': 'Compagnie'},
              color='ARRIVAL_DELAY', color_continuous_scale='Reds')
fig1.update_traces(
    customdata=delay_by_airline[['AIRLINE_NAME']],
    hovertemplate='<b>%{customdata[0]}</b><br>Code: %{x}<br>Délai: %{y:.2f} min<extra></extra>'
)

# KPI 2: Délai par destination (CODE sur axe)
delay_by_dest = flights.groupby(['DESTINATION_AIRPORT', 'DESTINATION_NAME', 'DESTINATION_CITY'])['ARRIVAL_DELAY'].mean().sort_values(ascending=False).head(15).reset_index()
fig2 = px.bar(delay_by_dest, x='DESTINATION_AIRPORT', y='ARRIVAL_DELAY',
              title="Délai moyen par destination (Top 15)",
              labels={'ARRIVAL_DELAY': 'Délai moyen (min)', 'DESTINATION_AIRPORT': 'Code'},
              color='ARRIVAL_DELAY', color_continuous_scale='Oranges')
fig2.update_traces(
    customdata=delay_by_dest[['DESTINATION_NAME', 'DESTINATION_CITY']],
    hovertemplate='<b>%{customdata[0]}</b><br>Ville: %{customdata[1]}<br>Code: %{x}<br>Délai: %{y:.2f} min<extra></extra>'
)

# KPI 3: Délai par provenance (CODE sur axe)
delay_by_origin = flights.groupby(['ORIGIN_AIRPORT', 'ORIGIN_NAME', 'ORIGIN_CITY'])['ARRIVAL_DELAY'].mean().sort_values(ascending=False).head(15).reset_index()
fig3 = px.bar(delay_by_origin, x='ORIGIN_AIRPORT', y='ARRIVAL_DELAY',
              title="Délai moyen par provenance (Top 15)",
              labels={'ARRIVAL_DELAY': 'Délai moyen (min)', 'ORIGIN_AIRPORT': 'Code'},
              color='ARRIVAL_DELAY', color_continuous_scale='Blues')
fig3.update_traces(
    customdata=delay_by_origin[['ORIGIN_NAME', 'ORIGIN_CITY']],
    hovertemplate='<b>%{customdata[0]}</b><br>Ville: %{customdata[1]}<br>Code: %{x}<br>Délai: %{y:.2f} min<extra></extra>'
)

# KPI 4: Délai par heure
delay_by_hour = flights.groupby('DEPARTURE_HOUR')['ARRIVAL_DELAY'].mean().reset_index()
delay_by_hour = delay_by_hour[delay_by_hour['DEPARTURE_HOUR'] <= 23]
fig4 = px.line(delay_by_hour, x='DEPARTURE_HOUR', y='ARRIVAL_DELAY',
               title="Délai moyen par heure de départ",
               labels={'ARRIVAL_DELAY': 'Délai moyen (min)', 'DEPARTURE_HOUR': 'Heure'},
               markers=True)
fig4.update_traces(line_color='purple', line_width=3,
                  hovertemplate='Heure: %{x}:00<br>Délai: %{y:.2f} min<extra></extra>')

# KPI 5: Délai par mois
delay_by_month = flights.groupby(['MONTH', 'MONTH_NAME'])['ARRIVAL_DELAY'].mean().reset_index().sort_values('MONTH')
fig5 = px.bar(delay_by_month, x='MONTH_NAME', y='ARRIVAL_DELAY',
              title="Délai moyen par mois",
              labels={'ARRIVAL_DELAY': 'Délai moyen (min)', 'MONTH_NAME': 'Mois'},
              color='ARRIVAL_DELAY', color_continuous_scale='Greens')
fig5.update_traces(hovertemplate='%{x}<br>Délai: %{y:.2f} min<extra></extra>')
fig5.update_xaxes(categoryorder='array', categoryarray=[MOIS_NOMS[i] for i in range(1, 13)])

# KPI 6: Annulation par compagnie
cancel_by_airline = flights.groupby(['AIRLINE', 'AIRLINE_NAME']).agg({
    'CANCELLED': ['sum', 'count']
}).reset_index()
cancel_by_airline.columns = ['AIRLINE', 'AIRLINE_NAME', 'Cancelled', 'Total']
cancel_by_airline['Taux_Annulation'] = (cancel_by_airline['Cancelled'] / cancel_by_airline['Total'] * 100).round(2)
cancel_by_airline = cancel_by_airline.sort_values('Taux_Annulation', ascending=False).head(15)
fig6 = px.bar(cancel_by_airline, x='AIRLINE', y='Taux_Annulation',
              title="Taux d'annulation par compagnie (%)",
              labels={'Taux_Annulation': 'Taux (%)', 'AIRLINE': 'Compagnie'},
              color='Taux_Annulation', color_continuous_scale='Reds')
fig6.update_traces(
    customdata=cancel_by_airline[['AIRLINE_NAME', 'Cancelled', 'Total']],
    hovertemplate='<b>%{customdata[0]}</b><br>Code: %{x}<br>Taux: %{y:.2f}%<br>Annulés: %{customdata[1]}<extra></extra>'
)

# KPI 7: Annulation par destination (CODE sur axe)
cancel_by_dest = flights.groupby(['DESTINATION_AIRPORT', 'DESTINATION_NAME', 'DESTINATION_CITY']).agg({
    'CANCELLED': ['sum', 'count']
}).reset_index()
cancel_by_dest.columns = ['DESTINATION_AIRPORT', 'DESTINATION_NAME', 'DESTINATION_CITY', 'Cancelled', 'Total']
cancel_by_dest['Taux_Annulation'] = (cancel_by_dest['Cancelled'] / cancel_by_dest['Total'] * 100).round(2)
cancel_by_dest = cancel_by_dest.sort_values('Taux_Annulation', ascending=False).head(15)
fig7 = px.bar(cancel_by_dest, x='DESTINATION_AIRPORT', y='Taux_Annulation',
              title="Taux d'annulation par destination (%)",
              labels={'Taux_Annulation': 'Taux (%)', 'DESTINATION_AIRPORT': 'Code'},
              color='Taux_Annulation', color_continuous_scale='Oranges')
fig7.update_traces(
    customdata=cancel_by_dest[['DESTINATION_NAME', 'DESTINATION_CITY', 'Cancelled', 'Total']],
    hovertemplate='<b>%{customdata[0]}</b><br>Ville: %{customdata[1]}<br>Code: %{x}<br>Taux: %{y:.2f}%<extra></extra>'
)

# Raisons annulation
if 'CANCELLATION_REASON' in flights.columns:
    cancel_reasons = flights[flights['CANCELLED'] == 1]['CANCELLATION_REASON'].value_counts().reset_index()
    cancel_reasons.columns = ['Reason', 'Count']
    reason_labels = {'A': 'Compagnie aérienne', 'B': 'Météo', 'C': 'Système aérien', 'D': 'Sécurité'}
    cancel_reasons['Reason'] = cancel_reasons['Reason'].map(reason_labels).fillna('Inconnu')
    fig8 = px.pie(cancel_reasons, names='Reason', values='Count', title="Causes d'annulation", hole=0.4)
else:
    fig8 = go.Figure()

# Annulations par mois
cancel_by_month = flights.groupby(['MONTH', 'MONTH_NAME']).agg({
    'CANCELLED': ['sum', 'count']
}).reset_index()
cancel_by_month.columns = ['MONTH', 'MONTH_NAME', 'Cancelled', 'Total']
cancel_by_month['Taux_Annulation'] = (cancel_by_month['Cancelled'] / cancel_by_month['Total'] * 100).round(2)
cancel_by_month = cancel_by_month.sort_values('MONTH')
fig9 = px.line(cancel_by_month, x='MONTH_NAME', y='Taux_Annulation',
               title="Taux d'annulation par mois", markers=True)
fig9.update_traces(line_color='red', line_width=3,
                  hovertemplate='%{x}<br>Taux: %{y:.2f}%<extra></extra>')
fig9.update_xaxes(categoryorder='array', categoryarray=[MOIS_NOMS[i] for i in range(1, 13)])

# Métriques
total_flights = len(flights)
total_cancelled = flights['CANCELLED'].sum()
avg_delay = flights['ARRIVAL_DELAY'].mean()
ontime_rate = (flights['ON_TIME'].sum() / len(flights) * 100)

# Distribution
fig10 = px.histogram(flights, x='ARRIVAL_DELAY', nbins=50,
                    title="Distribution des retards", labels={'ARRIVAL_DELAY': 'Délai (min)'})
fig10.update_xaxes(range=[-60, 180])

# Volume par mois
flights_by_month = flights.groupby(['MONTH', 'MONTH_NAME']).size().reset_index(name='Nombre_Vols').sort_values('MONTH')
fig11 = px.bar(flights_by_month, x='MONTH_NAME', y='Nombre_Vols',
              title="Volume de vols par mois", color='Nombre_Vols', color_continuous_scale='Viridis')
fig11.update_xaxes(categoryorder='array', categoryarray=[MOIS_NOMS[i] for i in range(1, 13)])
fig11.update_traces(hovertemplate='%{x}<br>Vols: %{y:,}<extra></extra>')

# Top routes
top_routes = flights.groupby(['ORIGIN_AIRPORT', 'ORIGIN_NAME', 'ORIGIN_CITY', 
                              'DESTINATION_AIRPORT', 'DESTINATION_NAME', 'DESTINATION_CITY']).size().sort_values(ascending=False).head(10).reset_index(name='Vols')
top_routes['Route'] = top_routes['ORIGIN_AIRPORT'] + ' → ' + top_routes['DESTINATION_AIRPORT']
top_routes['Route_Full'] = top_routes['ORIGIN_NAME'] + ' → ' + top_routes['DESTINATION_NAME']
fig12 = px.bar(top_routes, x='Route', y='Vols', title="Top 10 des routes")
fig12.update_xaxes(tickangle=45)
fig12.update_traces(
    customdata=top_routes[['Route_Full', 'ORIGIN_CITY', 'DESTINATION_CITY']],
    hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]} → %{customdata[2]}<br>Vols: %{y:,}<extra></extra>'
)

# Saisons
season_stats = flights.groupby('SEASON').agg({
    'ARRIVAL_DELAY': 'mean', 'CANCELLED': ['sum', 'count']
}).reset_index()
season_stats.columns = ['SEASON', 'Delai_Moyen', 'Cancelled', 'Total']
season_stats['Taux_Annulation'] = (season_stats['Cancelled'] / season_stats['Total'] * 100).round(2)
season_order = ['Hiver', 'Printemps', 'Été', 'Automne']
season_stats['SEASON'] = pd.Categorical(season_stats['SEASON'], categories=season_order, ordered=True)
season_stats = season_stats.sort_values('SEASON')

fig13 = make_subplots(specs=[[{"secondary_y": True}]])
fig13.add_trace(go.Bar(x=season_stats['SEASON'], y=season_stats['Delai_Moyen'],
                       name='Délai moyen', marker_color='lightblue'), secondary_y=False)
fig13.add_trace(go.Scatter(x=season_stats['SEASON'], y=season_stats['Taux_Annulation'],
                          name='Taux annulation', mode='lines+markers',
                          marker=dict(size=12, color='red'), line=dict(width=3, color='red')), secondary_y=True)
fig13.update_layout(title_text="Analyse par Saison")

# Directions
df_vertical = flights[flights['DIRECTION_VERTICAL'].isin(['Nord → Sud', 'Sud → Nord'])]
if len(df_vertical) > 0:
    vertical_stats = df_vertical.groupby('DIRECTION_VERTICAL').agg({
        'ARRIVAL_DELAY': 'mean', 'CANCELLED': ['sum', 'count']
    }).reset_index()
    vertical_stats.columns = ['Direction', 'Delai_Moyen', 'Cancelled', 'Total']
    vertical_stats['Taux_Annulation'] = (vertical_stats['Cancelled'] / vertical_stats['Total'] * 100).round(2)
    fig14 = make_subplots(specs=[[{"secondary_y": True}]])
    fig14.add_trace(go.Bar(x=vertical_stats['Direction'], y=vertical_stats['Delai_Moyen'],
                           name='Délai', marker_color='#4CAF50'), secondary_y=False)
    fig14.add_trace(go.Scatter(x=vertical_stats['Direction'], y=vertical_stats['Taux_Annulation'],
                              name='Taux', mode='lines+markers', marker=dict(size=12, color='#FF5722')), secondary_y=True)
    fig14.update_layout(title_text="Nord-Sud vs Sud-Nord")
else:
    fig14 = go.Figure()

df_horizontal = flights[flights['DIRECTION_HORIZONTAL'].isin(['Est → Ouest', 'Ouest → Est'])]
if len(df_horizontal) > 0:
    horizontal_stats = df_horizontal.groupby('DIRECTION_HORIZONTAL').agg({
        'ARRIVAL_DELAY': 'mean', 'CANCELLED': ['sum', 'count']
    }).reset_index()
    horizontal_stats.columns = ['Direction', 'Delai_Moyen', 'Cancelled', 'Total']
    horizontal_stats['Taux_Annulation'] = (horizontal_stats['Cancelled'] / horizontal_stats['Total'] * 100).round(2)
    fig15 = make_subplots(specs=[[{"secondary_y": True}]])
    fig15.add_trace(go.Bar(x=horizontal_stats['Direction'], y=horizontal_stats['Delai_Moyen'],
                           name='Délai', marker_color='#2196F3'), secondary_y=False)
    fig15.add_trace(go.Scatter(x=horizontal_stats['Direction'], y=horizontal_stats['Taux_Annulation'],
                              name='Taux', mode='lines+markers', marker=dict(size=12, color='#FF9800')), secondary_y=True)
    fig15.update_layout(title_text="Est-Ouest vs Ouest-Est")
else:
    fig15 = go.Figure()

print("✅ Graphiques créés")

# =========================
# 6. Application Dash
# =========================

app = dash.Dash(__name__)
app.title = "Dashboard Vols"

app.layout = html.Div([
    html.H1("Dashboard d'Analyse des Vols", style={'textAlign': 'center', 'marginBottom': '30px'}),
    dcc.Tabs(id='tabs', value='tab-delays', children=[
        dcc.Tab(label='📊 Analyse des Délais', value='tab-delays'),
        dcc.Tab(label='❌ Analyse des Annulations', value='tab-cancellations'),
        dcc.Tab(label='📈 Vue d\'ensemble', value='tab-overview'),
    ]),
    html.Div(id='tabs-content')
], style={'padding': '20px', 'fontFamily': 'Arial'})

@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-delays':
        return html.Div([
            html.H2("Analyse des Délais", style={'textAlign': 'center'}),
            html.Div([
                html.Div([dcc.Graph(figure=fig1)], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig2)], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            ]),
            html.Div([
                html.Div([dcc.Graph(figure=fig3)], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig4)], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            ]),
            html.Div([dcc.Graph(figure=fig5)]),
        ])
    
    elif tab == 'tab-cancellations':
        return html.Div([
            html.H2("Analyse des Annulations", style={'textAlign': 'center'}),
            html.Div([
                html.Div([dcc.Graph(figure=fig6)], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig7)], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            ]),
            html.Div([
                html.Div([dcc.Graph(figure=fig8)], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig9)], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            ]),
        ])
    
    elif tab == 'tab-overview':
        return html.Div([
            html.H2("Vue d'ensemble", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.H3(f"{total_flights:,}", style={'color': '#1f77b4', 'marginBottom': '0'}),
                    html.P("Total vols", style={'fontSize': '14px'}),
                ], style={'width': '23%', 'display': 'inline-block', 'textAlign': 'center', 
                         'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '10px', 'margin': '1%'}),
                html.Div([
                    html.H3(f"{total_cancelled:,}", style={'color': '#d62728', 'marginBottom': '0'}),
                    html.P("Annulés", style={'fontSize': '14px'}),
                ], style={'width': '23%', 'display': 'inline-block', 'textAlign': 'center', 
                         'padding': '20px', 'backgroundColor': '#ffe8e8', 'borderRadius': '10px', 'margin': '1%'}),
                html.Div([
                    html.H3(f"{avg_delay:.1f} min", style={'color': '#ff7f0e', 'marginBottom': '0'}),
                    html.P("Délai moyen", style={'fontSize': '14px'}),
                ], style={'width': '23%', 'display': 'inline-block', 'textAlign': 'center', 
                         'padding': '20px', 'backgroundColor': '#fff3e0', 'borderRadius': '10px', 'margin': '1%'}),
                html.Div([
                    html.H3(f"{ontime_rate:.1f}%", style={'color': '#2ca02c', 'marginBottom': '0'}),
                    html.P("Ponctualité", style={'fontSize': '14px'}),
                ], style={'width': '23%', 'display': 'inline-block', 'textAlign': 'center', 
                         'padding': '20px', 'backgroundColor': '#e8f5e9', 'borderRadius': '10px', 'margin': '1%'}),
            ], style={'marginBottom': '30px'}),
            html.Div([dcc.Graph(figure=fig13)], style={'marginBottom': '30px'}),
            html.Div([
                html.Div([dcc.Graph(figure=fig14)], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig15)], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            ], style={'marginBottom': '30px'}),
            html.Div([
                html.Div([dcc.Graph(figure=fig10)], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig11)], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            ]),
            html.Div([dcc.Graph(figure=fig12)]),
        ])

if __name__ == '__main__':
    print("\n🚀 Dashboard disponible sur http://127.0.0.1:8050")
    app.run()
