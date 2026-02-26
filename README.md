# Churn Prediction - System ML End-to-End

## Opis projektu
Projekt przedstawia kompletny system Machine Learning do przewidywania odejścia klienta (churn) w firmie subskrypcyjnej.

System obejmuje:
- Eksploracyjną analizę danych (EDA)
- Feature engineering z użyciem sklearn Pipeline
- Trening i ewaluację modelu
- REST API w FastAPI
- Interaktywną aplikację w Streamlit
- Konteneryzację przy użyciu Docker oraz docker-compose

## Problem biznesowy
Odejście klienta (churn) ma bezpośredni wpływ na przychody firm subskrypcyjnych.
Pozyskanie nowego klienta jest zazwyczaj droższe niż utrzymanie obecnego.

Celem projektu jest:
- identyfikacja klientów zagrożonych odejściem
- estymacja prawdopodobieństwa churn

## Dane
Publiczny zbiór danych zawierający informacje o klientach:
- Dane demograficzne
- Informacje o usługach
- Typ umowy
- Informacje o płatnościach

Zmienna docelowa:
- `Churn` (Yes / No)

Rozkład klas:
- ~73% brak churn
- ~27% churn

Ze względu na niezbalansowane dane jako główną metrykę zastosowano ROC AUC.

## Feature Engineering
Zastosowano:
- `ColumnTransformer`
- `Pipeline`
- `StandardScaler` dla zmiennych numerycznych
- `OneHotEncoder` z `handle_unknown="ignore"` dla zmiennych kategorycznych
Preprocessing jest częścią pipeline zapisanego razem z modelem, co gwarantuje spójność między treningiem a inferencją.

## Wybór modelu
Testowane modele:
- Logistic Regression
- Random Forest

Metryka:
- ROC AUC

Wyniki:
- Logistic Regression 0.83
- Random Forest 0.81

Wybrano Logistic Regression ze względu na:
- lepszy wynik
- prostotę
- interpretowalność

## Architektura systemu
Projekt posiada rozdzielenie warstw:
- interfejs użytkownika
- warstwa API
- logika ML

Takie podejście odzwierciedla architekturę stosowaną w realnych systemach.

## Konteneryzacja (Docker)
Projekt zawiera:
- Dockerfile dla API
- Dockerfile dla aplikacji
- docker-compose do orkiestracji usług