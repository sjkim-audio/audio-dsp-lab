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
| 03 | **Feature Extraction** | [`03_MFCC.ipynb`](./notebooks/03_Audio_Feature_Extraction.ipynb) | MFCC, Source-Filter Theory, Cepstrum | ✅ Done |
| 04 | **Musical Analysis** | [`04_Chroma.ipynb`](./notebooks/04_Musical_Features_Chroma.ipynb) | HPSS, Chroma Feature, Tonnetz | ✅ Done |
| 05 | **Audio Manipulation** | [`05_Manipulation.ipynb`](./notebooks/05_Audio_Manipulation.ipynb) | Time Stretching, Pitch Shifting, Phase Vocoder | ✅ Done |
| 06 | **Convolution Reverb** | [`06_Reverb.ipynb`](./notebooks/06_Convolution_Reverb.ipynb) | Impulse Response, Convolution, Unity Gain Normalization | ✅ Done |
| 07 | **Advanced Denoising** | ['07_Denising.ipynb'](./notebooks/07_Advanced_Denoising.ipynb) | Spectral Subtraction, Noise Profiling, STFT/ISTFT | ✅ Done |

<details>
<summary><b>📚 Learning Notes: Lab 01 ~ 07 (이론 및 핵심 정리)</b> - <i>Click to expand</i></summary>
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

---

### Lab 05. Audio Manipulation (Time Stretch & Pitch Shift)
**목표:** 오디오 신호의 주파수 특성(Pitch)과 시간 특성(Duration)을 상호 간섭 없이 독립적으로 제어하는 신호 처리 기법을 실습합니다.

1. **Time Stretching (타임 스트레칭)**
   - **정의:** 소리의 고유한 음높이(Pitch)는 유지하면서 재생 속도만 빠르게 하거나 느리게 조절하는 기술입니다.
   - **핵심 문제 (Chipmunk Effect):** 아날로그 방식처럼 단순히 재생 속도를 높이면 파형이 압축되어 주파수가 올라가고, 목소리가 변조되는 현상이 발생합니다.
   - **해결 원리 (Phase Vocoder):** 신호를 STFT로 분해한 뒤, 주파수 강도(Magnitude)는 유지하고 위상(Phase) 정보를 시간 비율에 맞춰 재계산하여 합성함으로써 음정 변화 없는 속도 조절을 구현합니다.

2. **Pitch Shifting (피치 쉬프팅)**
   - **정의:** 소리의 길이(재생 시간)는 유지한 채 음높이만 올리거나 내리는 기술입니다.
   - **원리:** 내부적으로 Time Stretching과 Resampling(재샘플링)을 결합하여 수행합니다. 주파수 축에서 에너지를 수직 이동시키는 것과 유사한 효과를 냅니다.
   - **활용:** 오토튠(Auto-tune), 노래방의 키 조절(Key Change), 익명 인터뷰를 위한 음성 변조 등에 사용됩니다.

---

### Lab 06. Convolution Reverb (Space & Math)
**목표:** DSP와 딥러닝의 핵심 연산인 컨볼루션(Convolution)의 수학적 원리를 이해하고, 이를 통해 신호에 공간감(Reverb)을 합성합니다.

1. **Impulse Response (IR, 임펄스 응답)**
   - **정의:** 시스템(공간)에 단위 충격(Impulse)이 가해졌을 때 나타나는 출력 반응입니다.
   - **의미:** 공간의 음향적 고유 특성(System Characteristic)을 나타냅니다. 실습에서는 백색 소음(White Noise)과 감쇠 곡선(Decay)을 결합해 가상의 콘서트홀 IR을 생성하여 사용합니다.

2. **Convolution (합성곱)**
   - **수학적 정의:** 두 함수(입력 신호와 IR) 중 하나를 반전 및 이동(Shift)시킨 후, 겹치는 구간을 곱하여 적분하는 연산입니다.
   - **물리적 의미:** 과거와 현재의 입력 신호들이 IR 값에 따라 가중 합(Weighted Sum)되어 선형 중첩(Linear Superposition)되는 과정입니다. 이로 인해 소리의 시간적 정보가 공간의 특성과 결합되어 풍성한 잔향이 형성됩니다.

3. **Issue & Solution: Signal Explosion (문제 해결)**
   - **문제 발견:** 단순 합성곱 적용 시, 출력 신호의 진폭이 입력 대비 수십 배 이상 증폭되어 클리핑(Clipping) 현상이 발생했습니다.
   - **원인 분석:** 컨볼루션 연산 과정에서 IR의 수만 개 샘플 크기가 누적(Accumulation)되면서, 시스템 이득(System Gain)이 1을 크게 초과했기 때문입니다.
   - **해결 방안:** IR 계수들의 절댓값 합(L1 Norm)으로 나누어주는 **Unity Gain Normalization**을 적용하여, 시스템의 최대 이득을 1로 제한하고 신호의 안정성을 확보했습니다.

---

### Lab 07. Advanced Denoising (Spectral Subtraction)
**목표:** 신호와 노이즈가 동일한 주파수 대역을 공유할 때, 노이즈의 패턴을 통계적으로 추정하여 제거하는 스펙트럼 차감법(Spectral Subtraction)을 구현합니다.

1. **Noise Profiling (노이즈 프로파일링)**
   - **정의:** 오디오 내에서 음성 신호가 없는 '묵음 구간(Silence/Noise-only)'을 식별하여 FFT를 수행합니다.
   - **원리:** 해당 구간의 주파수별 평균 에너지를 계산하여, 현재 환경에 깔려 있는 노이즈의 '주파수 지문(Fingerprint)'을 추출합니다.

2. **Spectral Subtraction (스펙트럼 차감)**
   - **알고리즘:** 전체 오디오 신호의 주파수 스펙트럼(Magnitude)에서 앞서 계산한 노이즈 프로파일을 수학적으로 차감(Subtraction)합니다.
   - **핵심:** 단순한 필터(LPF)와 달리, 목소리 주파수 대역(중음역)에 섞여 있는 화이트 노이즈까지 정교하게 제거할 수 있습니다.
   - **Phase Reconstruction:** 스펙트럼 차감은 '소리의 크기(Magnitude)'만 처리하므로, 위상(Phase) 정보는 원본(Noisy Signal)의 것을 그대로 사용하여 시간 영역 신호(Waveform)로 복원합니다.

</details>
