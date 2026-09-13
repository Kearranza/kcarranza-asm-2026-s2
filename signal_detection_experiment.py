import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.fft import fft, ifft

# Parámetros
duration = 0.1  # segundos
sample_rate = 10000  # Hz
start_freq = 1000  # Hz
end_freq = 2000  # Hz
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# 1. Generación de la señal transmitida (Chirp)
transmitted_signal = chirp(t, start_freq, duration, end_freq, method='linear')

# 2. Generación de la señal recibida con ecos y ruido
echo_delays = [0.2, 0.4, 0.6]  # segundos
echo_amplitudes = [0.8, 0.6, 0.5]  # Amplitud relativa
received_signal = np.zeros_like(transmitted_signal)
for delay, amp in zip(echo_delays, echo_amplitudes):
    delayed_echo = np.roll(transmitted_signal, int(delay * sample_rate))
    received_signal += amp * delayed_echo

# Añadir ruido gaussiano
noise = np.random.normal(0, 0.1, received_signal.shape)
received_signal += noise

# 3. Correlación directa
correlation_direct = np.correlate(received_signal, transmitted_signal, mode='full')
delay_index_direct = np.argmax(np.abs(correlation_direct))
delay_time_direct = delay_index_direct / sample_rate

# 4. Correlación mediante FFT (convolución en frecuencia)
N = len(transmitted_signal)
received_fft = fft(received_signal, N)
transmitted_fft = fft(transmitted_signal, N)
correlation_fft = ifft(received_fft * np.conj(transmitted_fft)).real
delay_index_fft = np.argmax(np.abs(correlation_fft))
delay_time_fft = delay_index_fft / sample_rate

# 5. Comparación y visualización
# Plot de la señal transmitida
plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
plt.plot(t, transmitted_signal)
plt.title("Señal Transmitida (Chirp)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()

# Plot de la señal recibida
plt.subplot(3, 2, 2)
plt.plot(t, received_signal)
plt.title("Señal Recibida (Ecos + Ruido)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()

# Plot de correlación directa
plt.subplot(3, 2, 3)
plt.plot(np.arange(len(correlation_direct)) / sample_rate, correlation_direct)
plt.title("Correlación Directa")
plt.xlabel("Retardo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.axvline(x=delay_time_direct, color='r', linestyle='--', label=f"Retardo: {delay_time_direct:.4f} s")
plt.legend()

# Plot de correlación via FFT
plt.subplot(3, 2, 4)
plt.plot(np.arange(len(correlation_fft)) / sample_rate, correlation_fft)
plt.title("Correlación via FFT")
plt.xlabel("Retardo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.axvline(x=delay_time_fft, color='r', linestyle='--', label=f"Retardo: {delay_time_fft:.4f} s")
plt.legend()

# Plot de magnitud de la señal transmitida
plt.subplot(3, 2, 5)
transmitted_freq = np.fft.fftfreq(len(transmitted_fft), 1/sample_rate)
plt.plot(transmitted_freq[:N//2], np.abs(transmitted_fft[:N//2]))
plt.title("Magnitud de la Señal Transmitida")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()

# Plot de fase de la señal transmitida
plt.subplot(3, 2, 6)
plt.plot(transmitted_freq[:N//2], np.angle(transmitted_fft[:N//2]))
plt.title("Fase de la Señal Transmitida")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Ángulo [rad]")
plt.grid()

plt.tight_layout()
plt.show()

# Resultados
print(f"Retardo estimado (directo): {delay_time_direct:.4f} s")
print(f"Retardo estimado (FFT): {delay_time_fft:.4f} s")
print(f"Diferencia: {abs(delay_time_direct - delay_time_fft):.6f} s")
