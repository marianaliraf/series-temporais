import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from statsmodels.tsa.api import Holt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import pmdarima as pm
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Configuração de diretório para salvar os gráficos
output_dir = "C:/Users/mlfarias/Documents/Projetos Python/previsao receita/apresentacao"
os.makedirs(output_dir, exist_ok=True)

# Função para calcular MAPE
def calcular_mape(real, previsto):
    real, previsto = np.array(real), np.array(previsto)
    return np.mean(np.abs((real - previsto) / real)) * 100

# Função para plotar, salvar e incluir MAPE no gráfico
def plot_and_save(nome, previsao, cor, save_path, mape):
    plt.figure(figsize=(8, 4))
    plt.title(f"Previsão com {nome}")
    plt.plot(serie_treino, label='Série Treinamento', color="blue")
    plt.plot([None] * 90 + list(serie_teste), label='Série Real', color="pink")
    plt.plot([None] * 90 + list(previsao), label=f'{nome} (MAPE: {mape:.2f}%)', color=cor)
    plt.legend(loc='lower left')  # Legenda no canto inferior esquerdo
    plt.savefig(save_path)  # Salva o gráfico como imagem
    plt.show()
    plt.close()  # Fecha a figura para liberar memória

# 1. Geração da série temporal
serie_temporal = [random.randint(10, 100) for _ in range(100)]
serie_treino = serie_temporal[:90]
serie_teste = serie_temporal[90:]
intervalo = 10

# 2. Previsão com Holt-Winters
model_hw = Holt(serie_treino, exponential=False, damped=False)
model_fit_hw = model_hw.fit(smoothing_level=0.5, smoothing_slope=0.3, optimized=False, damping_slope=0.1)
previsto_hw = model_fit_hw.predict(start=len(serie_teste), end=len(serie_teste) + intervalo - 1)

# 3. Previsão com ARIMA
model_arima = pm.auto_arima(
    serie_treino, start_p=1, start_q=1, test='adf',
    max_p=3, max_q=3, m=1, d=None, seasonal=False,
    trace=True, error_action='ignore', suppress_warnings=True, stepwise=True
)
previsto_arima = model_arima.predict(n_periods=intervalo)

# 4. Previsão com SVR
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
x_train = np.arange(len(serie_treino)).reshape(-1, 1)
svr.fit(x_train, serie_treino)
x_forecast = np.arange(len(serie_treino), len(serie_treino) + intervalo).reshape(-1, 1)
previsto_svr = svr.predict(x_forecast)

# 5. Previsão com ETS
model_ets = ETSModel(serie_treino, error='add', trend='add', seasonal=None)
model_fit_ets = model_ets.fit()
previsto_ets = model_fit_ets.forecast(intervalo)

# 6. Previsão com LSTM
serie_array = np.array(serie_treino).reshape(-1, 1)
generator = TimeseriesGenerator(serie_array, serie_array, length=1, batch_size=1)
model_lstm = Sequential([
    LSTM(2, activation='relu', input_shape=(1, 1)),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(generator, epochs=5, verbose=0)
previsto_lstm = []
input_seq = serie_array[-1].reshape(1, 1, 1)
for _ in range(intervalo):
    pred = model_lstm.predict(input_seq, verbose=0)
    previsto_lstm.append(pred[0][0])
    input_seq = np.array(pred).reshape(1, 1, 1)

# 7. Calcular MAPE para cada previsão
mape_hw = calcular_mape(serie_teste, previsto_hw)
mape_arima = calcular_mape(serie_teste, previsto_arima)
mape_svr = calcular_mape(serie_teste, previsto_svr)
mape_ets = calcular_mape(serie_teste, previsto_ets)
mape_lstm = calcular_mape(serie_teste, previsto_lstm)

# 8. Gráfico comparativo com MAPE na legenda
plt.figure(figsize=(10, 6))
plt.title("Comparativo de Previsões")
plt.plot(serie_treino, label='Série Treinamento', color="blue")
plt.plot([None] * 90 + list(serie_teste), label='Série Real', color="pink")
plt.plot([None] * 90 + list(previsto_hw), label=f'Holt-Winters (MAPE: {mape_hw:.2f}%)', color="green")
plt.plot([None] * 90 + list(previsto_arima), label=f'Modelo Autoregressivo (MAPE: {mape_arima:.2f}%)', color="red")
plt.plot([None] * 90 + list(previsto_svr), label=f'SVR (MAPE: {mape_svr:.2f}%)', color="orange")
plt.plot([None] * 90 + list(previsto_ets), label=f'ETS (MAPE: {mape_ets:.2f}%)', color="purple")
plt.plot([None] * 90 + list(previsto_lstm), label=f'LSTM (MAPE: {mape_lstm:.2f}%)', color="cyan")
plt.legend(loc='lower left')  # Legenda no canto inferior esquerdo
plt.savefig(os.path.join(output_dir, "comparativo_previsoes_com_mape.png"))
plt.show()

# 9. Gráficos individuais com MAPE
plot_and_save("Holt-Winters", previsto_hw, "green", os.path.join(output_dir, "previsao_holt_winters.png"), mape_hw)
plot_and_save("Modelo Autoregressivo", previsto_arima, "red", os.path.join(output_dir, "previsao_arima.png"), mape_arima)
plot_and_save("SVR", previsto_svr, "orange", os.path.join(output_dir, "previsao_svr.png"), mape_svr)
plot_and_save("ETS", previsto_ets, "purple", os.path.join(output_dir, "previsao_ets.png"), mape_ets)
plot_and_save("LSTM", previsto_lstm, "cyan", os.path.join(output_dir, "previsao_lstm.png"), mape_lstm)

print(f"Gráficos individuais e comparativo salvos na pasta: {output_dir}")
