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
<summary><b>📚 Learning Notes: Lab 01 ~ 03 (이론 및 핵심 정리)</b> - <i>Click to expand</i></summary>
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

---

### Lab 03. Feature Extraction (MFCC)
**목표:** 소리의 고유한 '지문(Fingerprint)'인 MFCC를 추출하고, 음성 인식 및 오디오 분류 모델의 핵심 원리를 이해합니다.

1. **MFCC (Mel-Frequency Cepstral Coefficients)**
   - **정의:** 멜-스펙트로그램에서 소리의 높낮이(Pitch) 정보는 제거하고, **음색(Timbre)과 공명 구조(Envelope)** 정보만 남겨 압축한 데이터입니다.
   - **압축 효율:** 128개 이상의 주파수 대역을 단 13~20개의 숫자로 요약하여 AI 모델의 연산 효율을 극대화합니다.

2. **Source-Filter Theory (소스-필터 이론)**
   - 소리는 **Source(떨림/에너지)**와 **Filter(공명/모양)**의 결합입니다.
   - MFCC는 여기서 Source를 배제하고 Filter 특성만 추출합니다.

3. **Applications (다양한 활용 분야)**
   - ** Speech Recognition (음성 인식):** 성도(Vocal Tract)의 모양을 분석하여 발음(Phoneme)을 구분합니다.
   - ** Music Information Retrieval (악기 분류):** 악기 고유의 공명통(Body) 특성을 분석하여 피아노, 기타, 바이올린 등의 음색을 구별합니다.
   - ** Anomaly Detection (이상 감지):** 기계의 모터 소리나 환경음의 패턴 변화를 감지하여 고장 여부를 진단합니다.
   - ** Voice Conversion (음성 변조):** 특정인의 MFCC(음색 지문) 특성을 다른 소리에 입혀 AI 커버 곡 등을 생성합니다.

---

### Lab 04. Musical Analysis (Chroma & Harmony)
**목표:** 주파수 분석을 넘어, 음악적 구조인 음계(Pitch Class)와 화음(Chord)을 시각화합니다.

1. **HPSS (Harmonic-Percussive Source Separation)**
   - 오디오 신호를 **화성 성분(Harmonic, 멜로디/반주)**과 **타악 성분(Percussive, 리듬/어택)**으로 분리합니다.
   - 스펙트로그램 상에서 가로줄(지속음)은 화성 성분으로, 세로줄(일시적 충격음)은 타악 성분으로 분류하는 기법입니다.
   - 크로마 분석 전 전처리 단계로 사용하여 화음 추출의 정확도를 높입니다.

2. **Chroma Feature (Chromagram)**
   - 인간의 청각은 옥타브가 달라도 같은 계이름(C, D, E...)을 비슷하게 인지합니다.
   - 전체 주파수 대역을 12개의 음계(Pitch Class)로 투영하여, 어떤 음(Note)이 연주되고 있는지 보여줍니다.
   - 활용: 코드 인식(Chord Recognition), 커버 곡 탐색(Cover Song Identification).

3. **Tonnetz (Tonal Centroids)**
   - 화성학적 관계(5도권 등)를 기반으로 화음의 변화를 시각화한 공간입니다. 곡의 화성 진행 분위기를 파악하는 데 쓰입니다.

</details>
