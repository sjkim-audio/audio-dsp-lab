# Audio-DSP-Lab
**A hands-on repository for exploring Digital Signal Processing (DSP) concepts using Python.** 
파이썬을 활용한 오디오 디지털 신호 처리 기초 이론 학습 및 실습 기록 저장소입니다.

---

## Tech Stack (사용 기술)
* **Language:** Python 3.x
* **Main Libraries:** `Librosa`, `NumPy`, `SciPy`, `Matplotlib`
* **Environment:** Google Colab / Jupyter Notebook

---

## Curriculum & 실습 현황

| Step (단계) | Topic (주제) | Notebook (실습 파일) | Key Concepts / Description (핵심 내용) | Status (상태) |
| :---: | :--- | :--- | :--- | :---: |
| 01 | **Audio Visualization** | [`01_Visualization.ipynb`](./notebooks/01_Audio_Signal_Visualization.ipynb) | Waveform, FFT, Mel-Spectrogram Analysis | ✅ Done |
| 02 | **Filtering & Noise** | [`02_Filtering.ipynb`](./notebooks/02_Audio_Filtering_and_Noise_Reduction.ipynb) | White Noise Generation, Butterworth Filter | ✅ Done |

<details>
<summary><b>📚 Learning Notes: Lab 01 & 02 (이론 및 핵심 정리)</b> - <i>Click to expand</i></summary>
<br>

### Lab 01. Audio Signal Visualization
**목표:** 오디오 신호의 물리적 특성(Waveform)을 인간이 인지하는 형태(Mel-Spectrogram)로 변환하는 과정을 이해합니다.

1. **Waveform (Time Domain)**
   - 시간($x$)에 따른 진폭($y$)의 변화를 보여줍니다. 소리의 크기(Loudness)는 알 수 있지만, 음색이나 음높이를 파악하기 어렵습니다.
2. **FFT (Frequency Domain)**
   - 퓨리에 변환을 통해 소리를 주파수 성분으로 분해합니다. 어떤 음높이가 포함되었는지 알 수 있지만, **시간 정보가 소실**됩니다.
3. **Spectrogram (STFT)**
   - 짧은 시간 단위로 FFT를 반복 수행하여 **시간, 주파수, 강도**를 동시에 시각화합니다.
4. **Mel-Spectrogram**
   - 주파수 축을 **Mel-Scale(멜 스케일)**로 변환한 것입니다.
   - *Why?* 인간은 고음보다 저음의 변화에 훨씬 민감하기 때문에, 이를 반영해야 AI 모델이 소리를 사람처럼 인식할 수 있습니다.

---

### Lab 02. Filtering & Noise Reduction
**목표:** 인위적인 노이즈를 섞고, 디지털 필터를 설계하여 원본 소리를 복원해 봅니다.

1. **White Noise (백색 소음)**
   - 모든 주파수 대역에 걸쳐 일정한 에너지를 가진 무작위 잡음입니다.
2. **Butterworth Filter**
   - 통과 대역(Passband)이 평탄하여 오디오 신호 처리에 가장 널리 쓰이는 필터입니다.
3. **Low-pass Filter (LPF)**
   - 차단 주파수(Cutoff Frequency)보다 낮은 저음은 통과시키고, 높은 고음(노이즈)은 걸러냅니다.
   - *Trade-off:* 노이즈를 많이 제거하려고 차단 주파수를 너무 낮추면, 원본 악기의 고음 배음(Harmonics)까지 깎여 소리가 뭉툭해질 수 있습니다.

</details>
