import math

import pandas as pd
import numpy as np
import re
import base64
from collections import Counter
from razdel import sentenize
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Загрузка hedonometer
ru_words = pd.read_csv("Hedonometer_ru.csv", index_col=0)
en_words = pd.read_csv("Hedonometer_en.csv", index_col=0)
words_dict = dict(
    tuple(zip(ru_words["Word"], ru_words["Happiness Score"]))
    + tuple(zip(en_words["Word"], en_words["Happiness Score"]))
)


# Загрузка текста
def load_text_from_file(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    for enc in ['utf-8', 'cp1251']:
        try:
            return decoded.decode(enc).strip().replace("\xa0", " ").replace(
                "…", "...")
        except UnicodeDecodeError:
            continue
    raise ValueError("Не удалось определить кодировку файла.")


# Анализ настроения
def calculate_sentiment(text_chunk, neutral_range=(4.0, 6.0)):
    cleaned = re.sub(r"[^A-Za-zА-Яа-яЁё]", " ", text_chunk).lower()
    tokens = cleaned.split()
    words = [
        w for w in tokens
        if w in words_dict
        and not (neutral_range[0] <= words_dict[w] <= neutral_range[1])
    ]
    if not words:
        return 5.0
    freq = Counter(words)
    total_score = sum(words_dict[w] * c for w, c in freq.items())
    return total_score / sum(freq.values())


# Разбиение текста
def split_text(text_str, n_sentences):
    sentences = [s.text for s in sentenize(text_str)]
    return [" ".join(sentences[i:i + n_sentences]) for i in
            range(0, len(sentences), n_sentences)]


# Сгалживание графика
def moving_average(data, window):
    series = pd.Series(data)
    return series.rolling(window=window, center=True, min_periods=1).mean()


# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Sentiment Book Viewer"

# Разметка сайта
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Анализ динамики эмоциональной окраски книги",
                            className="text-center my-4 text-primary"))),

    # Пояснение шкалы
    dbc.Row(dbc.Col(
        dbc.Card([
            dbc.CardHeader("📊 Шкала Hedonometer (1–9)"),
            dbc.CardBody(
                html.Ul([
                    html.Li(
                        "1.0 – 3.5: Негативный фон (грусть, тревога, конфликты)"),
                    html.Li(
                        "3.5 – 5.5: Нейтральный фон (описания, диалоги, быт)"),
                    html.Li(
                        "5.5 – 9.0: Позитивный фон (радость, праздники, победа)"),
                ], className="mb-0")
            )
        ], className="mb-3 border-info"),
        width=12
    )),

    # Зона загрузки файла
    dbc.Row(dbc.Col(dcc.Upload(
        id='upload-book',
        children=html.Div("Перетащите .txt файл или нажмите для выбора",
                          style={"textAlign": "center", "padding": "1rem"}),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed",
            "borderRadius": "0.5rem",
            "borderColor": "#adb5bd", "backgroundColor": "#f8f9fa",
            "cursor": "pointer", "marginTop": "0.5rem", "marginBottom": "1rem"
        },
        multiple=False,
        accept=".txt,.text,.md"  # Ограничение выбора типа файла
    ), width=12)),

    html.Div(id='upload-status', className="text-center mb-3"),

    dcc.Store(id='book-text', data=None),

    # Управление
    dbc.Row([
        dbc.Col(dbc.InputGroup([
            dbc.InputGroupText("Число предложений в фрагменте:"),
            dbc.Input(id="sentence-input", type="number", value=7, min=1,
                      step=1, className="form-control"),
        ], className="mb-3"), width=6),
        dbc.Col(dbc.InputGroup([
            dbc.InputGroupText("Размер окна скользящего среднего:"),
            dbc.Input(id="window-input", type="number", value=3, min=1, step=1,
                      className="form-control"),
        ], className="mb-3"), width=6),
    ], className="mb-3"),

    # Кнопка подтверждения
    dbc.Row(dbc.Col(
        dbc.Button("Построить график", id="confirm-button", color="primary",
                   className="w-100 mb-4"), width=4)),

    # График
    dbc.Row(dbc.Col(dcc.Graph(id="sentiment-graph"), width=12)),

    # Текст фрагмента
    dbc.Row(dbc.Col(
        html.H3("Выбранный фрагмент:", className="mt-4 text-secondary"))),
    dbc.Row(dbc.Col(
        html.Div(id="text-output",
                 children="Кликни на точку, чтобы увидеть фрагмент текста.",
                 style={"whiteSpace": "pre-wrap", "borderRadius": "0.5rem",
                        "border": "1px solid #dee2e6", "padding": "1rem",
                        "backgroundColor": "#f8f9fa"}),
        width=12)
    ),
], fluid=True)


# Загрузка книги
@app.callback(
    Output('book-text', 'data'),
    Output('upload-status', 'children'),
    Input('upload-book', 'contents'),
    State('upload-book', 'filename'),
    prevent_initial_call=True
)
def update_book(contents, filename):
    if filename is None:
        raise dash.exceptions.PreventUpdate
    # Валидация расширения файла
    if not filename.lower().endswith(('.txt', '.text', '.md')):
        return None, dbc.Alert(
            f"Ошибка: файл '{filename}' не поддерживается. Загрузите текстовый документ.",
            color="danger")

    text = load_text_from_file(contents)
    return text, dbc.Alert(f"✅ Загружено: {filename} ({len(text):,} символов)",
                           color="success")

# Обновление графика
@app.callback(
    Output("sentiment-graph", "figure"),
    Input("confirm-button", "n_clicks"),
    State('book-text', 'data'),
    State("sentence-input", "value"),
    State("window-input", "value"),
    prevent_initial_call=True
)
def update_graph(n_clicks, book_text, n_sentences, window_size):
    if book_text is None:
        text_str = ""
    else:
        text_str = book_text
    chunks = split_text(text_str, n_sentences)
    scores = [calculate_sentiment(ch) for ch in chunks]
    smooth = moving_average(scores, window_size)
    df = pd.DataFrame({
        "idx": range(len(chunks)),
        "score": scores,
        "smooth": smooth,
        "text": chunks
    })
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["idx"], y=df["score"], mode="markers", name="Оценки",
                   marker=dict(size=6, color="#17BECF"),
                   customdata=df["text"]))
    fig.add_trace(go.Scatter(x=df["idx"], y=df["smooth"], mode="lines",
                             name=f"Скользящее среднее (окно={window_size})",
                             line=dict(width=2, dash="dash"),
                             customdata=df["text"]))
    fig.update_layout(
        xaxis_title="Номер фрагмента",
        yaxis_title="Оценка настроения",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


# Отображение фрагмента текста
@app.callback(
    Output("text-output", "children"),
    Input("sentiment-graph", "clickData"),
)
def display_text(clickData):
    if not clickData:
        return "Кликни на точку, чтобы увидеть фрагмент текста."
    return clickData["points"][0]["customdata"]


if __name__ == "__main__":
    app.run(debug=False)
