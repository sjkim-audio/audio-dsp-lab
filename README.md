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
| 08 | **Data Augmentation** | [`08_Augmentation.ipynb`](./notebooks/08_Data_Augmentation.ipynb) | Noise Injection, Time Shift, SpecAugment (Masking) | ✅ Done |
| 09 | **CNN Classification** | [`09_CNN_Model.ipynb`](./notebooks/09_Audio_Classification_CNN.ipynb) | 2D CNN Architecture, Model Summary | ✅ Done |
| 10 | **Model Evaluation** | ['10 Model Evaluation.ipynb'](./notebooks/10_Model_Evaluation.ipynb) | Confusion Matrix, Loss Curve | ✅ Done |

<details>
<summary><b>📚 Learning Notes (이론 및 핵심 정리)</b> - <i>Click to expand</i></summary>
<br>

### Lab 01. Audio Signal Visualization
**목표:** 오디오 신호의 물리적 특성(Waveform)을 인간이 인지하는 형태(Mel-Spectrogram)로 변환하는 전 과정을 이해합니다.

1. **Waveform (Time Domain)**
   - 시간에 따른 진폭의 변화를 보여줍니다. 소리의 크기(Loudness)는 알 수 있지만, 음색이나 음높이를 직관적으로 파악하기 어렵습니다.
2. **FFT (Frequency Domain)**
   - 퓨리에 변환을 통해 소리를 주파수 성분으로 분해합니다. 어떤 음높이가 포함되었는지 알 수 있지만, **시간 정보가 소실**되는 한계가 있습니다.
3. **Spectrogram (STFT)**
   - 짧은 시간 단위로 FFT를 반복 수행(STFT)하여 **시간, 주파수, 강도**를 동시에 시각화한 2차원 데이터입니다.
4. **Mel-Spectrogram**
   - 스펙트로그램의 주파수 축을 **Mel-Scale(멜 스케일)**로 변환한 것입니다.
   - **이유:** 인간의 청각은 고음보다 저음의 변화에 훨씬 민감하므로, 이 특성을 반영해야 AI 모델이 소리를 사람처럼 인식할 수 있습니다.

---

### Lab 02. Filtering & Noise Reduction
**목표:** 인위적인 노이즈를 주입하고, 디지털 필터를 설계하여 원본 소리를 복원하는 과정을 실습합니다.

1. **White Noise (백색 소음)**
   - 모든 주파수 대역에 걸쳐 일정한 에너지를 가진 무작위 잡음입니다.
2. **Butterworth Filter**
   - 통과 대역(Passband)의 응답이 평탄(Flat)하여, 왜곡 없이 오디오 신호를 처리하는 데 가장 널리 쓰이는 필터입니다.
3. **Low-pass Filter (LPF)**
   - 차단 주파수(Cutoff Frequency)보다 낮은 저음은 통과시키고, 높은 고음(노이즈)은 걸러냅니다.
   - **Trade-off:** 노이즈 제거를 위해 차단 주파수를 과도하게 낮추면, 원본 악기의 배음(Harmonics)까지 손실되어 소리가 먹먹해질 수 있습니다.

---

### Lab 03. Feature Extraction (MFCC)
**목표:** 소리의 고유한 '지문(Fingerprint)'인 MFCC를 추출하고, 오디오 분류 모델의 핵심 원리를 이해합니다.

1. **MFCC (Mel-Frequency Cepstral Coefficients)**
   - **정의:** 멜-스펙트로그램에서 소리의 높낮이(Pitch) 정보는 배제하고, **음색(Timbre)과 공명 구조(Envelope)** 정보만 남겨 압축한 데이터입니다.
   - **효율성:** 128개 이상의 주파수 대역을 단 13~20개의 계수로 요약하여 AI 모델의 연산 효율을 극대화합니다.
2. **Source-Filter Theory (소스-필터 이론)**
   - 소리는 **Source(성대의 떨림/에너지)**와 **Filter(성도의 공명/모양)**의 결합이라는 이론입니다.
   - MFCC는 여기서 Source를 분리해내고, 고유한 Filter 특성만을 추출합니다.
3. **Applications (활용 분야)**
   - **Speech Recognition:** 성도(Vocal Tract)의 모양을 분석하여 발음(Phoneme)을 구분합니다.
   - **Music Information Retrieval:** 악기 고유의 공명통(Body) 특성을 분석하여 악기 종류를 구별합니다.
   - **Anomaly Detection:** 기계 모터음이나 환경음의 미세한 패턴 변화를 감지하여 고장을 진단합니다.
   - **Voice Conversion (음성 변조):** 특정인의 MFCC(음색 지문) 특성을 다른 소리에 입혀 AI 커버 곡 등을 생성합니다.

---

### Lab 04. Musical Analysis (Chroma & Harmony)
**목표:** 주파수 분석을 넘어, 음악적 구조인 음계(Pitch Class)와 화음(Chord)을 시각화합니다.

1. **HPSS (Harmonic-Percussive Source Separation)**
   - 오디오 신호를 **화성 성분(Harmonic, 멜로디/반주)**과 **타악 성분(Percussive, 리듬/어택)**으로 분리합니다.
   - 스펙트로그램 상에서 가로줄(지속음)은 화성 성분으로, 세로줄(일시적 충격음)은 타악 성분으로 분류합니다.
2. **Chroma Feature (Chromagram)**
   - 인간의 청각은 옥타브가 달라도 같은 계이름(C, D, E...)을 유사하게 인지한다는 점에 착안했습니다.
   - 전체 주파수 대역을 12개의 음계(Pitch Class)로 투영하여, 현재 어떤 화음이 연주되고 있는지 보여줍니다.
3. **Tonnetz (Tonal Centroids)**
   - 화성학적 관계(5도권 등)를 기반으로 화음 간의 거리를 시각화한 공간입니다. 곡의 전반적인 조성 변화와 분위기를 파악하는 데 사용됩니다.

---

### Lab 05. Audio Manipulation (Time Stretch & Pitch Shift)
**목표:** 오디오 신호의 주파수(Pitch)와 시간(Duration) 특성을 상호 간섭 없이 독립적으로 제어합니다.

1. **Time Stretching (타임 스트레칭)**
   - **정의:** 소리의 음높이는 유지하면서 재생 속도만 빠르게 하거나 느리게 조절하는 기술입니다.
   - **Phase Vocoder:** 단순히 재생 속도를 높이면 발생하는 주파수가 변형되는(Chipmunk Effect) 문제를 해결하기 위해, STFT 후 주파수 강도는 유지하고 위상(Phase) 정보만 시간 비율에 맞춰 재계산합니다.
2. **Pitch Shifting (피치 쉬프팅)**
   - **정의:** 소리의 길이는 유지한 채 음높이만 올리거나 내리는 기술입니다.
   - **원리:** 내부적으로 Time Stretching과 Resampling(재샘플링)을 결합하여 구현하며, 주파수 축에서 에너지를 수직 이동시키는 효과를 냅니다.
   - **활용:** 오토튠(Auto-tune), 키 조절(Key Change), 음성 변조 등.

---

### Lab 06. Convolution Reverb (Space & Math)
**목표:** DSP와 딥러닝의 핵심 연산인 컨볼루션(Convolution)의 수학적 원리를 이해하고, 이를 응용해 공간감(Reverb)을 합성합니다.

1. **Impulse Response (IR, 임펄스 응답)**
   - **정의:** 공간에 단위 충격(Impulse)이 가해졌을 때 나타나는 음향적 반응입니다. 공간의 고유한 지문 역할을 합니다.
2. **Convolution (합성곱)**
   - 두 함수(입력 신호와 IR) 중 하나를 반전 및 이동(Shift)시킨 후, 겹쳐서 적분하는 연산입니다.
   - 과거와 현재의 소리(입력 신호들)가 IR 값에 따라 가중 합(Weighted Sum)되어 선형 중첩(Linear Superposition)되면서 공간의 특성(IR)과 섞여(Smearing) 풍성한 잔향이 만들어집니다.
3. **Issue & Solution: Signal Explosion (문제 해결)**
   - **문제 발견:** 단순 합성곱 적용 시, 출력 신호의 진폭이 입력 대비 수십 배 이상 증폭되어 클리핑(Clipping) 현상이 발생했습니다.
   - **원인 분석 :** 컨볼루션 연산 과정에서 IR의 수만 개 샘플 크기가 누적(Accumulation)되면서, 시스템 이득(System Gain)이 1을 크게 초과했기 때문입니다.
   - **해결 방안:** IR 계수들의 절댓값 합(L1 Norm)으로 나누어주는 **Unity Gain Normalization**을 적용하여, 시스템의 최대 이득을 1로 제한하고 신호의 안정성을 확보했습니다.

---

### Lab 07. Advanced Denoising (Spectral Subtraction)
**목표:** 노이즈의 패턴을 통계적으로 추정하여 제거하는 스펙트럼 차감법(Spectral Subtraction)을 구현합니다.

1. **Noise Profiling (노이즈 프로파일링)**
   - 오디오 내에서 음성 신호가 없는 '묵음 구간(Silence/Noise-only)'을 식별하여 FFT를 수행합니다.
   - 해당 구간의 주파수별 평균 에너지를 계산하여, 현재 환경 노이즈의 주파수 특성을 추출합니다.
2. **Spectral Subtraction (스펙트럼 차감)**
   - 전체 오디오 신호의 주파수 스펙트럼(Magnitude)에서 노이즈 프로파일을 수학적으로 차감(Subtraction)합니다.
   - 단순한 필터(LPF)와 달리, 목소리 주파수 대역(중음역)에 섞여 있는 화이트 노이즈까지 정교하게 제거할 수 있습니다.
   - **Phase Reconstruction:** 스펙트럼 차감은 '소리의 크기(Magnitude)'만 처리하므로, 위상(Phase) 정보는 원본(Noisy Signal)의 것을 그대로 사용하여 시간 영역 신호(Waveform)로 복원합니다.

---

### Lab 08. Data Augmentation (AI Training Prep)
**목표:** 딥러닝 모델의 과적합(Overfitting)을 방지하고 일반화 성능을 높이기 위해, 원본 데이터를 인위적으로 변형하여 학습 데이터의 양을 증강시키는 기법을 실습합니다.

1. **Noise Injection (노이즈 주입)**
   - 원본 신호에 백색 소음(White Noise) 등을 임의로 섞는 기법입니다.
   - AI 모델이 깨끗한 환경뿐만 아니라 잡음이 섞인 환경에서도 핵심 신호를 잘 추출하도록 내성(Robustness)을 길러줍니다.
2. **Time Shifting (시간 이동)**
   - **Circular Shift:** 파형을 시간 축에서 이동시키되, 잘려 나간 끝부분을 다시 앞으로 연결하는 순환 이동 방식을 사용합니다.
   - **Shift Invariance:** 소리의 시작 위치가 달라져도 모델이 동일한 소리로 인식하도록 학습시킵니다.
3. **SpecAugment (Frequency Masking)**
   - 시간 영역이 아닌 스펙트로그램(주파수 영역) 상에서 특정 주파수 대역이나 시간 구간을 통째로 지워버리는(Masking) 기법입니다.
   - 특정 주파수 정보가 유실된 상황에서도 남은 정보만으로 전체 내용을 추론하는 능력을 강화합니다.

---

### Lab 09. Audio Classification with CNN (Model Design)
**목표:** 시계열 오디오 데이터를 이미지 처리 기술인 CNN(Convolutional Neural Network)에 적용하기 위한 차원 변환 전략과 모델 아키텍처를 설계합니다.

1. **Input Representation (입력 데이터의 시각화 전략)**
   - **Audio as Image:** 오디오의 주파수(Frequency) 축을 이미지의 높이(Height)로, 시간(Time) 축을 이미지의 너비(Width)로 매핑하여 처리합니다.
   - **Channel Dimension:** 컬러 이미지가 RGB 3채널을 갖는 것과 달리, 스펙트로그램(MFCC)은 흑백 이미지와 유사한 단일 채널(Monophonic Channel) 구조를 갖도록 차원을 확장하여 CNN 입력 규격을 충족시킵니다.
2. **CNN Architecture Design (모델 설계 원리)**
   - **Feature Extraction (특징 추출부):** `Conv2D` 레이어를 통해 소리의 지역적 패턴(Time-Frequency Texture)을 스캔하고, `MaxPooling`으로 데이터의 차원을 축소하여 핵심 특징(Feature Map)만을 요약합니다.
   - **Classification (분류부):** 2차원으로 추출된 특징 맵을 1차원 벡터로 변환(Flatten)한 후, 완전연결층(Dense Layer)과 `Softmax` 활성화 함수를 통해 각 클래스에 속할 확률 분포를 계산합니다.

---

### Lab 10. Model Evaluation & Metrics
**목표:** 학습된 모델의 성능을 객관적인 지표로 검증하고, 시각화 도구를 통해 모델의 과적합 여부와 취약 클래스를 진단합니다.

1. **Learning Curve Analysis (학습 곡선 분석)**
   - **Training vs Validation:** Epoch 진행에 따른 Accuracy와 Loss 변화를 비교 시각화합니다.
   - **Overfitting Detection:** Validation Loss가 다시 증가하기 시작하는 변곡점을 찾아내어 모델의 일반화 성능이 저하되는 시점을 포착합니다.
2. **Confusion Matrix (오차 행렬)**
   - 모델의 예측값과 실제 정답을 교차 비교하여 히트맵(Heatmap)으로 시각화합니다.
   - 단순 오답률을 넘어, "어떤 클래스가 서로 혼동되는지" 오인 패턴(Misclassification Pattern)을 분석하여 추후 데이터 정제 방향을 설정합니다.
3. **Quantitative Metrics (정량적 지표)**
   - 단순 정확도(Accuracy)의 한계를 보완하기 위해 Precision(정밀도), Recall(재현율), F1-Score를 종합적으로 산출하여 모델의 신뢰성을 평가합니다.

</details>
