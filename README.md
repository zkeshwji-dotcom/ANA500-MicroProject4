# Forecasting Electricity Demand with Recurrent Neural Networks

ANA500 Micro-Project 4 by Zeena Keshwji.

This project forecasts the next 24 hours of electricity demand for the PJM East region from 16 years of hourly load data. It compares a Simple RNN, LSTM, GRU, sequence-to-sequence LSTM and a dense network against naive, ridge regression and gradient boosting baselines.

**Main notebook:** [`energy_demand_forecasting.ipynb`](energy_demand_forecasting.ipynb)

## Result

The Simple RNN with one week of history was chosen on the validation years. On the test period (January 2017 to August 2018) its mean absolute error was 1,337 MW (4.1%), 62% lower than the seasonal naive forecast.

## Files

| File | What it is |
|---|---|
| `energy_demand_forecasting.ipynb` | Full analysis, executed with outputs |
| `ANA500_MicroProject4_Energy_Demand.pptx` | Presentation slides |
| `ANA500_MicroProject4_Code_Report.docx` | Code and outputs compiled into Word |
| `figures/` | Charts used in the slides |
| `build_notebook.py`, `build_slides.py`, `build_code_report.py` | Scripts that generate the notebook, slides and report |

## Data

[Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) on Kaggle, file `PJME_hourly.csv`. The notebook downloads it automatically into `data/` on first run.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook energy_demand_forecasting.ipynb
```

A full run trains 30 neural networks and took about 1 hour 45 minutes on a laptop CPU.
