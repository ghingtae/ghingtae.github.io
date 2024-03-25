---
layout: single
title:  "AIFactory: Gemma LoRA 파인튜닝으로 댓글감성 분류"
categories: AIFactory
tag: [coding, JAX, Gemma, Kaggle]
toc: true
toc_sticky: true
author_profile: true
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 1rem !important;
  }

  </style>
</head>


## Fine-tune Gemma models in Keras using LoRA



### **개요**

---

Gemma는 구글에서 공개한 오픈 소스 경량 대규모 언어입니다. 또한, Gemini의 경량화 모델입니다.   



Gemma와 같은 대규모 언어 모델(LLMs)은 다양한 NLP 작업에서 효과적임이 입증되었습니다. LLM은 먼저 대규모 텍스트 코퍼스에서 자기 감독 방식으로 사전 학습됩니다. 사전 학습은 LLM이 단어 간의 통계적 관계와 같은 일반적인 목적의 지식을 학습하도록 돕습니다. 그런 다음 LLM은 도메인 특정 데이터로 미세 조정되어 하류 작업(예: 감성 분석)을 수행할 수 있습니다.   



LLM은 크기가 매우 큽니다(파라미터가 수십억 개의 순서). 대부분의 응용 프로그램에서는 사전 학습 데이터셋에 비해 상대적으로 훨씬 작은 미세 조정 데이터셋을 사용하기 때문에 모델의 모든 파라미터를 업데이트하는 전체 미세 조정이 필요하지 않습니다.   



**Low Rank Adaptation(LoRA)**은 모델의 가중치를 고정하고 모델에 새로운 가중치를 적은 수로 삽입함으로써 하류 작업을 위한 훈련 가능한 파라미터의 수를 크게 줄이는 미세 조정 기술입니다. 이는 LoRA를 사용한 훈련을 훨씬 더 빠르고 메모리 효율적으로 만들며, 모델 출력의 품질을 유지하면서도 더 작은 모델 가중치(몇 백 MB)를 생성합니다.


## Google Dribe



```python
from google.colab import drive
drive.mount('/content/drive')
```

<pre>
Mounted at /content/drive
</pre>
## 설정



### Gemma 접근하기



이 튜토리얼을 완료하려면 먼저 [Gemma 설정](https://ai.google.dev/gemma/docs/setup)에서 설정 지침을 완료해야 합니다. Gemma 설정 지침은 다음을 수행하는 방법을 보여줍니다:   



* [kaggle.com](https://kaggle.com)에서 Gemma에 접근하기.

* Gemma 2B 모델을 실행할 수 있는 충분한 자원을 가진 Colab 런타임 선택하기.

* Kaggle 사용자 이름과 API 키 생성 및 구성하기.



Gemma 설정을 완료한 후, 다음 섹션으로 이동하여 Colab 환경에 대한 환경 변수를 설정합니다.   



### 런타임 선택하기



이 튜토리얼을 완료하려면 Gemma 모델을 실행할 수 있는 충분한 자원을 가진 Colab 런타임이 필요합니다. 이 경우, T4 GPU를 사용할 수 있습니다:   



1. Colab 창의 오른쪽 상단에서 ▼(**추가 연결 옵션**)을 선택합니다.

2. **런타임 유형 변경**을 선택합니다.

3. **하드웨어 가속기** 아래에서 **T4 GPU**를 선택합니다.



### API 키 구성하기



Gemma를 사용하려면 Kaggle 사용자 이름과 Kaggle API 키를 제공해야 합니다.   



Kaggle API 키를 생성하려면 Kaggle 사용자 프로필의 **계정** 탭으로 이동하여 **새 토큰 생성**을 선택합니다. 이렇게 하면 API 자격 증명이 포함된 `kaggle.json` 파일이 다운로드됩니다.   



Colab에서 왼쪽 패널의 **비밀** (🔑)을 선택하고 Kaggle 사용자 이름과 Kaggle API 키를 추가합니다. 사용자 이름은 `KAGGLE_USERNAME` 아래에, API 키는 `KAGGLE_KEY` 아래에 저장합니다.   



### 환경 변수 설정하기



`KAGGLE_USERNAME`과 `KAGGLE_KEY`에 대한 환경 변수를 설정합니다.



```python
import os
import json

# kaggle.json 파일을 읽어서 환경 변수 설정
with open('/content/drive/MyDrive/kaggle_settings.json') as f:
    kaggle_info = json.load(f)
os.environ['KAGGLE_USERNAME'] = kaggle_info['username']
os.environ['KAGGLE_KEY'] = kaggle_info['key']
```

### Install dependencies



Keras, KerasNLP 및 기타를 설치합니다.



```python
# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp
!pip install -q -U keras>=3
```

<pre>
[?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/465.3 kB[0m [31m?[0m eta [36m-:--:--[0m
[2K     [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━[0m [32m337.9/465.3 kB[0m [31m10.0 MB/s[0m eta [36m0:00:01[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m465.3/465.3 kB[0m [31m10.2 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m950.8/950.8 kB[0m [31m46.6 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.2/5.2 MB[0m [31m93.4 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m589.8/589.8 MB[0m [31m2.8 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.8/4.8 MB[0m [31m110.9 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/2.2 MB[0m [31m86.4 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.5/5.5 MB[0m [31m108.4 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m76.9 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m286.8/286.8 kB[0m [31m37.6 MB/s[0m eta [36m0:00:00[0m
[?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tf-keras 2.15.1 requires tensorflow<2.16,>=2.15, but you have tensorflow 2.16.1 which is incompatible.[0m[31m
[0m
</pre>
### Select a backend



Keras는 단순성과 사용 용이성을 위해 설계된 고수준, 멀티 프레임워크 딥러닝 API입니다. Keras 3을 사용하면 TensorFlow, JAX, PyTorch 중 하나의 백엔드에서 워크플로우를 실행할 수 있습니다.    



이 튜토리얼에서는 JAX를 위한 백엔드를 구성합니다.



```python
os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
```

### Import packages



Keras와 KerasNLP를 가져옵니다.



```python
import keras
import keras_nlp
```

## Load Dataset



```python
import urllib.request

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/tykimos/tykimos.github.io/master/warehouse/dataset/tarr_train.txt",
    filename="tarr_train.txt",
)
```

<pre>
('tarr_train.txt', <http.client.HTTPMessage at 0x7ab297f63130>)
</pre>

```python
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# 파일을 DataFrame으로 로드
df = pd.read_csv('tarr_train.txt', delimiter='\t')
actual_labels = []
total = len(df)

data = []

for index, row in df.iterrows():

    features = {} # dict "키" : "값" # 사전 "단어" : "설명"

    # Add the 'instruction' and 'label' key-value pairs to the features dictionary
    features['instruction'] = row['comment'] # 여기 음식은 언제 와도 실망시키지 않아요. 최고!
    features['response'] = row['label'] # 1

    # Debug prints to see the questions and answers
    print("Q:" + row['comment'])
    print("A:" + str(row['label']))

    # 템플릿 = 양식, 모델마다 파인튜닝할 때 지정된 양식
    # Format the entire example as a single string.
    template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

    # 파인튜닝할 데이터 추가
    data.append(template.format(**features))

    '''
    Instruction:
    여기 음식은 언제 와도 실망시키지 않아요. 최고!

    Response:
    1
    '''
```

<pre>
Q:여기 음식은 언제 와도 실망시키지 않아요. 최고!
A:1
Q:여기 라멘 진짜 ㄹㅇ 맛있어요. 국물이 진하고 면도 쫄깃해서 너무 좋았습니다.
A:1
Q:진짜 깔끔하고, 맛도 좋았어요. 추천합니다!
A:1
Q:왜 이렇게 유명한지 모르겠음ㅋㅋ ㄹㅈㄷ 맛없음
A:0
Q:인생 타르트를 여기서 만났어요❤️ 달지 않고 고소해서 정말 추천합니다!
A:1
Q:메뉴 설명을 너무 친절하게 해주셔서 고르기 수월했어요.
A:1
Q:사진과 음식이 너무 달라서 실망했습니다.
A:0
Q:주변에 추천하려고 사진도 많이 찍었어요. 좋아요!
A:1
Q:솔직히...? 맛이 그닥이에요. 리뷰랑 너무 다르네.
A:0
Q:진짜 개꿀맛..ㅠ 다른곳 안가.
A:1
Q:음식이 너무 늦게 나와서 기다리는 동안 답답했습니다.
A:0
Q:음식도 맛있고, 가격도 합리적이에요. 다음에 또 오려구요.
A:1
Q:여기 리뷰보고 왔는데 ㅇㅈ? 실망스러움...
A:0
Q:앞으로 여기 자주 올 것 같아요! 가성비 짱!ㅇㅈ? ㅋㅋ
A:1
Q:마지막으로 온 게 언제였는지 모르겠는데, 여전히 인기 많네요!
A:1
Q:ㅎㅎ 애들이랑 와서 잘 먹었어요. 아이들 메뉴도 맛있더라고요!
A:1
Q:주문한 음식이 전부 나오지 않아서 환불 요청했어요.
A:0
Q:요새 여기가 핫플이라던데, 완전 공감!
A:1
Q:리뷰보고 왔는데 기대 이하였습니다.
A:0
Q:안 와봤으면 큰일날뻔; 여기 김치찌개는 진리네 ㅎㅎ
A:1
Q:이런 곳은 왜 유명해지지 않는 거죠? 숨은 맛집이네요!
A:1
Q:뭔가 빠진 느낌? 뭐가 문제인지는 모르겠는데, 그냥 별로였어요.
A:0
Q:우리 할머니도 극찬하셨어요! 또 오고 싶다고 하시더라구요.
A:1
Q:음식도 늦게 나오고, 맛도 그닥... 실망이에요.
A:0
Q:요새 패션에 맞게 리모델링도 하셨더라. 또 올게!
A:1
Q:외관은 그냥 그랬는데, 내부가 너무 아늑하고 좋았습니다.
A:1
Q:사진보다 음식 portion이 좀 작은 거 아닌가요? 😐
A:0
Q:ㅁㅊ! 여기 왜 이제야 알게됐지? 대박 맛집이네!
A:1
Q:이게 뭐야? ㅁㅊ라고 이렇게 매운 거라도 미리 얘기 좀 해줘야지. 입 안에서 불 키는 줄...
A:0
Q:OMG! 이런 맛집을 지금서야 발견하다니ㅋㅋ 대박❤️
A:1
Q:전체적으로 괜찮았지만, 음악 소리가 너무 커서 조금 시끄러웠습니다.
A:0
Q:진짜 미친듯이 맛있어서 놀랐네요. ㄷㄷ... 여기는 천국이냐?
A:1
Q:이전에 왔을 때보다 서비스가 많이 떨어진 것 같아요.
A:0
Q:이렇게 좋은 곳이 있었는데, 왜 이제야 알았을까요? 다들 방문해보세요!
A:1
Q:방문하려고 했는데, 문을 닫았더라고요. 영업 시간을 좀 더 정확히 표시해주셨으면 좋겠어요.
A:0
Q:가격 대비 음식의 양이 너무 적었어요.
A:0
Q:음식이 너무 짜서 물을 계속 마셨습니다.
A:0
Q:솔직히 맛은 그냥 그랬어요. 하지만 분위기는 좋았습니다.
A:0
Q:요기 돈까스 진짜 맛있어요. 돈까스 좋아하시는 분들 꼭 와보세요!
A:1
Q:처음 와봤는데, 분위기도 좋고 음식도 만족스러워서 잘 먹었어요.
A:1
Q:여기는 꼭 한 번 방문해봐야 해요! 너무 만족했습니다.
A:1
Q:우리 아들이랑 왔는데, 아이 메뉴도 좋더라고요.
A:1
Q:아니, 메뉴 주문한지 40분 넘었는데 안 나와요?
A:0
Q:전 직원분들의 친절함에 감동 받았습니다. 음식도 무척 훌륭했어요.
A:1
Q:청결도가 좀... 특히 화장실이 좀... ㅠㅠ
A:0
Q:괜찮았어요.
A:1
Q:처음 와보는데, 전통과 현대가 조화로운 곳이네요. 다음에 또 방문하려고요.
A:1
Q:직원들이 너무 친절해서 기분 좋게 식사했습니다. 감사합니다.
A:1
Q:직원들의 태도가 너무 불친절해서 기분이 안 좋았어요.
A:0
Q:주차공간이 너무 협소해서 불편했습니다.
A:0
Q:좀 너무 기대하고 왔나봐요... 뭐 이런 것도 있지 뭐 😑
A:0
Q:최악의 응대... 여기 다시는 안 갈래요. 😡
A:0
Q:진짜 대박! 여기 인테리어 너무 이쁘고, 음식도 인스타갬성 폭발!
A:1
Q:불친절해서 기분 나빴어. 주문하려는데 계속 대화하고 있더라구요.
A:0
Q:바다가 보이는 자리에서 식사하는 게 정말 로맨틱했어요. 분위기 최고에요!
A:1
Q:식사 중에 바퀴벌레가 나와서 기분이 엄청 나빴습니다.
A:0
Q:나중에 가게 될 걸 알았으면 아예 안 올 걸 그랬네. 재료가 신선하지 않아 보였어요.
A:0
Q:처음부터 끝까지 서비스가 너무 좋았습니다. 직원들도 친절하시고, 음식은 말할 것도 없이 너무 맛있었어요. 특히 메인디쉬는 인상 깊었네요.
A:1
Q:나한테는 좀 너무 달았어.
A:0
Q:친구 추천으로 왔는데, 정말 잘 선택한 것 같아요. 👏
A:1
Q:직원분이 계속 쳐다보면서 서비스 해주셔서 좀 불편했어요. 알바생 훈련이 필요해보여요.
A:0
Q:ㄱㅊ... 특별히 기대 안 했는데 꽤 괜찮았어.
A:1
Q:전통적인 맛 그대로, 우리 할머니 생각나게 하는 음식이었습니다. 감사합니다.
A:1
Q:단체로 왔는데, 모두다 만족하며 나갔어요.
A:1
Q:그냥 그랬다.
A:0
Q:인테리어가 너무 이쁘더라고요. 사진 찍기 좋아요!
A:1
Q:전 주차 때문에 여기 올 거면 차 대지 말라고 하고 싶네요. 주차장이 너무 협소하더라고요...
A:0
Q:옛날 할머니집에서 먹던 그 맛이 났어요. 너무 행복한 식사였습니다.
A:1
Q:ㄴㅇㄱ... 여기서 이런 일을 겪을 줄이야...
A:0
Q:저녁 데이트로 딱 좋은 곳이네요! 분위기도 좋고, 음식도 훌륭해요.
A:1
Q:한국 전통 음식을 찾으시는 분들에게 추천합니다! 정감있고 맛도 좋아요.
A:1
Q:젊은 사람들이 좋아할만한 분위기에다가, 메뉴 구성도 신선하네요! 잘 먹었습니다!
A:1
Q:이렇게 많은 사람들이 찾는 이유가 있었네요. 각종 리뷰와 추천에도 불구하고 직접 방문해서 먹어봐야 알 수 있는 맛이 있더라구요. 저는 특히 디저트가 인상적이었습니다.
A:1
Q:와... 대기시간 너무 길어서 기다리다 짜증났음...
A:0
Q:ㅋㅋ 이거 뭐임? 실화냐? 왜 이렇게 맛없냐ㅠㅠ
A:0
Q:오랜만에 괜춘한 곳 찾았네요. 친구들한테 소개해줄듯.
A:1
Q:분위기도 좋고 직원분들도 너무 친절해서 기분 좋게 식사했어요.
A:1
Q:우리 집에서 가까워서 좋아요. 자주 이용할 것 같아요!
A:1
Q:야외 테라스 자리가 있어서 좋았어요. 풍경 보면서 식사하는 거 좋아하시는 분들 추천!
A:1
Q:이 가격에 이런 음식? 다시는 안 올 것 같아요.
A:0
Q:와... 여기 진짜 꿀맛이네요!
A:1
Q:30분 기다려서 받은 음식이 이거라니... 최악의 경험!!
A:0
Q:여기 스파게티 맛집인듯. 국물 스파는 ㄹㅇ 최고.
A:1
Q:직원들 태도가 좀 불친절해요. 다음엔 다른 곳을 찾아볼게요.
A:0
Q:처음 방문했을 때부터 이번 방문까지, 항상 만족스럽게 먹고 가는 곳입니다. 이곳의 메뉴들은 언제 먹어도 질리지 않고, 새로운 맛을 제공해줘요.
A:1
Q:깔끔한 내부와 아늑한 분위기, 이런 곳 찾았습니다!
A:1
Q:가격이 좀 비싼 편인데, 그만큼의 가치가 있어요.
A:1
Q:메뉴 선택의 폭이 넓어서 좋았어요. 여러 가지 시도해볼 수 있어서 좋았습니다.
A:1
Q:계산할 때 추가 비용이 생겨서 불쾌했어요.
A:0
Q:여기 팔은 디저트가 정말 맛있어요. 추천합니다!
A:1
Q:별로에요.
A:0
Q:포장해 갔는데 집에서 먹어보니 맛있더라고요!
A:1
Q:주차장이 협소해서 차 대기하는데 오래 걸렸어요.
A:0
Q:요즘 핫한 곳이라길래 갔는데, 그냥 그랬어요. 이전에 갔던 다른 식당이 더 나았던 것 같아요.
A:0
Q:친구 추천으로 왔는데, 나도 추천하고 가요!
A:1
Q:소문나서 왔는데, 기대 이하였어요.
A:0
Q:우와.. 여기 스테이크는 정말 실력 있게 만드네요. 육질이 부드럽고, 소스의 균형이 완벽해요.👍🥩
A:1
Q:그냥 그랬어요. 별로 큰 인상은 없네요.
A:0
Q:여기 스테이크는 정말 최고예요! 반드시 다시 올 것 같아요!
A:1
Q:와~ 여기 분위기 너무 좋아요! 데이트하기 딱 좋은 곳!
A:1
Q:요기 진짜 ㄹㅇ 맛있어요! 다음에 또 올래요.
A:1
Q:이전에는 맛있었는데, 이번엔 좀 별로였어요.
A:0
Q:오랜만에 여길 찾았는데, 아직도 그 맛 그대로네요.
A:1
Q:음식이 너무 기름져서 별로였어요.
A:0
Q:세대차이가 확실히 느껴지는 곳. 2030들은 좋아할 거 같아요. 매장 음악도 트렌디하더라고요ㅋㅋ
A:1
Q:가격 대비 맛이 너무 떨어져요.
A:0
Q:음식이 너무 늦게 나와 기다리는 내내 기분이 안 좋았습니다.
A:0
Q:음... 나쁘진 않았지만, 특별히 좋지도 않았어요. 중간이라고 해야하나...
A:1
Q:직원 태도나 서비스는 좋은데, 음식이 내 기대 이하였습니다.
A:0
Q:데이트하기 좋은 곳이라 친구가 추천해줬어. 정말 만족스러웠음!
A:1
Q:그냥저냥? 뭐 크게 나쁘진 않았지만... 😐
A:0
Q:아이들이랑 오기 딱 좋아요! 공간도 넓고 메뉴도 다양하고! 😊
A:1
Q:포장도 꼼꼼하게 해주셔서 감사했어요.
A:1
Q:바쁜 시간대라 그런지 서비스가 너무 느렸습니다.
A:0
Q:메뉴 설명도 잘 해주시고, 맛도 훌륭했어요.
A:1
Q:디저트 종류가 많아서 좋았어요. 특히 타르트 짱!
A:1
Q:진짜 숨겨진 맛집! 여기 왜 이제야 알았을까요?
A:1
Q:서비스 개노답... ㅂㅅ 같은 곳 다신 안 옴.
A:0
Q:주문한지 1시간 넘게 기다려서 화가 났어요! ㅡㅡ
A:0
Q:ㅈㄴ 노맛. 여기 다신 안 올 듯.
A:0
Q:소문대로네요. 이런 맛집을 찾는 게 힘들어서 다행이에요!
A:1
Q:주문한 음식과 달라서 불편했어요.
A:0
Q:우와 이런 데가 우리 동네에도 있다니! 가격도 착하고, 음식도 굿굿! ㅋㅋㅋ
A:1
Q:정말 맛있어요!
A:1
Q:여기 피자는 다른 곳과는 다른 맛이에요. 꼭 한번 드셔보세요!
A:1
Q:음식이 너무 짜지 않게 조절해 주셨으면 좋겠어요.
A:0
Q:ㄴㅇㄱ. 여기는 진짜 최고네요! 다들 한번쯤 방문해보세요.
A:1
Q:아빠랑 오랜만에 외식했는데, 여기 선택한 건 후회가 없네요!
A:1
Q:메뉴가 그렇게 다양하지는 않지만, 맛은 있어서 만족합니다.
A:1
Q:저는 개인적으로 여기 음식이 좋았어요. 다만, 직원 서비스는 좀 아쉬웠습니다.
A:1
Q:코로나 때문에 걱정했는데, 위생 관리가 잘 되어있더라고요.
A:1
Q:친절한 서비스와 맛있는 음식, 두 마리 토끼를 다 잡은 곳!
A:1
Q:음료가 너무 달아서 다 마시지 못했어요.
A:0
Q:솔직히 오늘따라 서비스가 너무 느렸어요. 뭐지? 😒
A:0
Q:여기 짬뽕 정말 맛있어요! 중화요리 좋아하시는 분들 꼭 와보세요.
A:1
Q:뷰는 좋았지만 음식은 그냥 그랬어요.
A:0
Q:남자친구와 기념일에 왔는데, 분위기와 음식 모두 최상급이었어요!
A:1
Q:주문한 메뉴가 다 나오지 않아서 황당했습니다.
A:0
Q:고기 질이 좀... 아쉬웠어요.
A:0
Q:이런 맛을 찾아 헤매던 중에 드디어 발견한 곳이에요!
A:1
Q:음식이 식어서 나왔어요. 더 신경 써야할 것 같아요.
A:0
Q:한참을 기다렸는데, 음식이 그다지... 별로였습니다.
A:0
Q:저번에 갔을 때보다 맛이 좀 떨어진 것 같아요.
A:0
Q:너무 맛있어서 가족들한테 추천하려고요! 👍
A:1
Q:음식도 늦게 나오고, 직원들 태도도 별로였습니다.
A:0
Q:음식은 그저 그랬지만, 디저트는 정말 맛있었어요!
A:1
Q:ㅋㅋ 여긴 왜 이렇게 사람 많은지 알겠다. 서비스도 굳!
A:1
Q:전통적인 김치찌개 맛을 찾는다면 이곳이 딱! 어머니 손맛 같아서 너무 좋았어요.
A:1
Q:요기 진짜 ㄹㅇ 맛있음. 다들 꼭 가보세용.
A:1
Q:다른 분들은 괜찮다고 하는데, 제 입맛엔 안 맞았어요.
A:0
Q:우리 아버지가 매우 만족하셨습니다. 오랜만에 좋은 한식을 먹은 것 같아요.
A:1
Q:할아버지가 이런 곳을 좋아하실 줄은 몰랐네요! 좋아하셨어요.
A:1
Q:제 취향은 아니던데, 아내는 좋아하더라구요.
A:0
Q:다 좋은데, 위치가 좀 애매해서 찾기 어려웠어요.
A:0
Q:이 가격에 이런 퀄리티라니! 비밀스러운 맛집 찾은 기분이에요.
A:1
Q:주차공간이 없어서 불편했어요. 차 타고 오시는 분들은 참고하세요.
A:0
Q:이런 곳이 우리 동네에 있었던 것을 이제야 알았다니, 놀라워요!
A:1
Q:직원분이 주문을 잘못 받아와서 조금 불편했습니다.
A:0
Q:떡볶이는 달고, 튀김은 기름져서 다시 오진 않을 것 같아요.
A:0
Q:친구들과 와서 실망했네요. 다신 오지 않을 것 같아요.
A:0
Q:가족 모임 장소로 딱 좋아요. 모두 만족했습니다.
A:1
Q:여기 직원분들이 좀 불친절하시던데요? 🤨
A:0
Q:이렇게 맛있는 곳이 내 동네에 있었다니! 자주 올게요!
A:1
Q:어릴 땐 여기 자주 왔는데, 맛이 변한 것 같아 아쉽네요.
A:0
Q:헉, 먹어본 스테이크 중에서 탑3 안에 드는 거 같아용! 👍
A:1
Q:신선한 재료와 특별한 레시피로 만든 음식이 정말 인상적이었어요!
A:1
Q:확실히 광고랑은 다르네요. 기대가 너무 컸나 봅니다. 그냥 그래요.
A:0
Q:서비스만 좋았다면 별 5개를 줬을텐데, 아쉽습니다.
A:0
Q:음식이 너무 짜서 다 먹지 못했어요.
A:0
Q:어릴 때 자주 가던 집인데, 여전히 그 맛 그대로네요. 추억이 새록새록.
A:1
Q:진짜 맛있어서 포장까지 했습니다!
A:1
Q:요즘 아이들이 좋아할만한 음식이 별로 없더라구요. 좀 아쉬워요.
A:0
Q:친구들과 함께 와서 정말 재미있게 식사했습니다.
A:1
Q:요기 메뉴 추천 받았는데 10/10 👌 다음엔 뭐 먹지 고민됨ㅠ
A:1
Q:내 돈 주고 먹기엔 아쉬운 맛이었어.
A:0
Q:우리 엄마한테 추천받았는데, 생각보다 그저 그래서 좀 실망했어.
A:0
Q:처음 와봤는데, 너무 만족스러웠어요. 앞으로 자주 올 예정입니다.
A:1
Q:너무나 훌륭하네요. 그동안 이런 곳을 찾아본 적이 없습니다. 위치도 중심가에 있어 찾기 수월했어요.
A:1
Q:너무 짜서 먹기 힘들었어요. 다시 올 생각은 없네요.
A:0
Q:젊은 친구들이랑 왔는데, 분위기도 굿이었어요!
A:1
Q:알려주신 와인과 음식이 너무 잘 어울렸어요.
A:1
Q:음식은 맛있었는데, 너무 시끄러워서 대화하기 힘들었어요.
A:0
Q:나쁘지 않았어.
A:1
Q:이렇게 특별한 맛을 찾아낼 수 있다니! 가게 분위기부터 음식까지, 전반적으로 모든 것이 완벽했어요. 다음에도 꼭 다시 오고 싶네요.
A:1
Q:와... 이런 곳을 이제야 알게 되다니! 대박이에요.
A:1
Q:음식은 괜찮았는데, 직원들 태도가 별로더라고요. 그래도 나름 대만족!
A:1
Q:저기요, 여긴 가격이 너무 비싸지 않나요? 조금 가성비가...😅
A:0
Q:전 너무 매웠어요. 다들 좋아하던데 맛의 차이인가 봐요.
A:0
Q:딱히 특별한 점은 없었어요. 그냥 그랬습니다.
A:0
Q:여기 정말 예전부터 알고 오던 곳인데, 맛이 계속 유지돼서 좋아요.
A:1
Q:음식점 내부가 너무 추워서 식사하는 내내 불편했어요.
A:0
Q:음... 가격이랑 맛이랑 참 안 맞는 것 같아요. 다시 올 의향은 없을 거 같아요.
A:0
Q:나이 들면서 많은 음식점을 다녀봤는데, 여기는 탑 5안에 들 정도로 좋아요.
A:1
Q:평이 좋아서 왔는데... 참 별로였어요. ㅠㅠ 그냥 집에서 먹을 걸 그랬나 봐요.
A:0
Q:어제 처음 방문했는데, 이미 다시 가고 싶네요! 😊
A:1
Q:오랜만에 가족들과 함께한 식사였는데, 모두 만족해했습니다. 아주 좋았어요. ❤️
A:1
Q:여기의 특색 있는 메뉴들은 다른 곳에서 느낄 수 없는 독특한 맛이에요!
A:1
Q:음식의 양이 좀 적었던 것 같아요. 가격 대비 만족도가 떨어짐.
A:0
Q:가격 대비해서 그닥... 다른 데가 더 나을 듯.
A:0
Q:봤을 때는 그냥 그랬는데, 맛이.. 역시나 짱이네요! 🔥
A:1
Q:여기 서비스가 너무 좋아요. 직원분들이 친절하시더라고요. 분위기도 아늑해서 다음에도 오고 싶네요.
A:1
Q:아이들과 왔는데, 아이들이 좋아하는 메뉴가 많아서 좋았어요.
A:1
Q:파스타 진짜 존맛탱. 이 가격에 이런 맛? 대박이야!
A:1
Q:ㅁㅊ, 이런 맛집을 지금서야 알았다니! 다들 가보셈 ㄹㅇ
A:1
Q:뭐지? 음식나오는데 1시간...ㄹㅇ 느리네.
A:0
Q:파스타 소스가 너무 짜서 먹기 힘들었어요. 솔직히 기대 이하였습니다.
A:0
Q:날씨 좋은 날 테라스에서 식사하는 건 최고에요!
A:1
Q:영업시간이랑 실제로 문 열고 있는 시간이 안 맞아서 2번이나 헛걸음 했네요 ㅡㅡ;
A:0
Q:서빙하시는 분이 너무 불친절해서 다시 오고 싶지 않아요.
A:0
Q:오랜만에 좋은 한식집 찾은 것 같아요. 앞으로 자주 방문할게요!
A:1
Q:참, 여기 특색 있는 메뉴들이 많더라고요. 좋았습니다.
A:1
Q:음... 솔직히 이 정도면 나도 집에서 만들 수 있을 듯...🙄
A:0
Q:리뷰 다 좋아서 왔는데 ㅁㅊ... 낚였다 싶네.
A:0
Q:ㅇㅋ, 여기는 뭐든지 맛있어. 매번 만족!
A:1
Q:예약했는데도 자리를 기다리게 해서 불쾌했습니다.
A:0
Q:알려지기 전에 다시 방문하고 싶네요!
A:1
Q:서빙하시는 분이 실수로 음료를 엎어서 옷이 다 젖었어요.ㅠㅠ
A:0
Q:솔직히 기대 안 했는데, 음식이 깜짝 놀랄 정도로 맛있었어요!
A:1
Q:맛, 서비스, 가격 모두 만족스러웠어요!
A:1
Q:오늘 직원분들이 ㄹㅇ 불친절했음. 뭐가 그리 바쁘다고?
A:0
Q:가격이 너무 비싸서 다시 올 의향은 없습니다.
A:0
Q:오랜만에 외식을 했는데, 이런 결과를 보게 될 줄은 몰랐네요. 음식도 별로고, 서비스도 매우 불친절했습니다. 다시는 오고 싶지 않아요.
A:0
Q:완전 내 취향 아님. 비추!
A:0
Q:대기시간 길어서 짜증났는데, 음식은 훌륭했어요!
A:1
Q:여기 인테리어 너무 예뻐서 사진찍기 좋아요~ 사진 잘나와요 ㅋㅋ
A:1
Q:ㄴㅇㄱ, 진짜 별로임ㅡㅡ;;
A:0
Q:ㅇㅈ, 여기 안주와 술 조합 대박이에요!
A:1
Q:왜 이렇게 더러운 거야? 청결 유의해주세요.
A:0
Q:우와~ 여기 새로운 메뉴 나왔네? 맛있어 보여요ㅎㅎ
A:1
Q:서비스는 괜찮은데, 가격이랑 맛이 아닌 듯...
A:0
Q:비가 와도 여기 커피는 최고!
A:1
Q:이 가격 주고 이런 맛?? ㅂㄷㅂㄷ;;
A:0
Q:주차공간 있어서 너무 좋아요, 또 방문하려구요!
A:1
Q:완전 별로. 리뷰보고 왔는데 황당함.
A:0
Q:ㅎㅎ 이 가게 직원분들 너무 친절해서 기분 좋게 먹었어요!
A:1
Q:분위기는 좋은데 음식 맛이...ㅠㅠ
A:0
Q:메뉴판 바뀌었나? 예전 맛이 아니네요...
A:0
Q:음식 나오는데 시간 좀 걸렸지만 맛은 최고였어요.
A:1
Q:테이블 위에 물 흘려놓고 그대로였어요... 청결 좀..
A:0
Q:친구 소개로 왔는데, 정말 괜찮았어요!
A:1
Q:ㅋㅋ 여기 디저트 너무 달아요. 아쉬웠어요.
A:0
Q:난 이런 곳이 좋더라. 소소하게 맛있고.
A:1
Q:와 진짜? 이런 곳에서 이런 경험을 할 줄이야.. 다신 안 올 거에요.
A:0
Q:진짜 최고!! 여기만한 곳 없어요❤️
A:1
Q:음... 리뷰보고 기대했는데, 그냥 그래요.
A:0
Q:주방에서 무슨 소리가 계속 나던데 조금 시끄러웠어요.
A:0
Q:서비스 개선 좀 해주세요. 직원들 태도에 좀 놀랐네요.
A:0
Q:치킨 진짜 부드럽고 맛있어요. 또 시켜 먹을거에요!ㅋㅋㅋ
A:1
Q:너무 실망이에요. 여기 리뷰 좋아서 왔는데 음식도 늦게 나오고, 서비스도 별로더라고요. 전 다신 안 올 것 같아요.
A:0
Q:오늘 먹은 스테이크 중 최고!
A:1
Q:웨이터분들 태도가 너무 불친절해요. 다른 곳에서 먹을 걸 그랬나봐요. 음식도 그다지...
A:0
Q:분위기, 맛, 서비스 모두 훌륭해요! 다만, 주차공간이 좀 협소해요ㅠㅠ
A:1
Q:햄버거 bun이 너무 퍽퍽함.
A:0
Q:전 여기 팬케이크가 진짜 맛있더라고요! 부드럽고, 메이플 시럽이 딱!
A:1
Q:리뷰 보고 왔는데, 왜 이렇게 평이 좋은지 모르겠어요. 저는 별로였습니다.
A:0
Q:냉면 국물이 아주 시원하고 맛있었어요! 여름에 와서 먹으면 좋을 것 같아요.
A:1
Q:여긴 왜 이렇게 사람이 많아요? 대기 시간 너무 길어요...
A:0
Q:닭갈비 매콤하고 양도 푸짐해요! 여기 오면 항상 주문하는 메뉴에요.
A:1
Q:와, 여기 새로 생긴 메뉴 진짜 대박이에요!! 가격도 괜찮고 맛도 좋아요!
A:1
Q:음료는 좋은데, 디저트가 너무 달아서 별로였어요.
A:0
Q:음... 여기 리뷰 왜 이래? 전 다시는 안 올 거 같아요.
A:0
Q:직원들이 너무 친절해서 기분 좋게 먹고 왔습니다.
A:1
Q:마지막 방문 후로 퀄리티가 많이 떨어진 것 같아요. 아쉬워요.
A:0
Q:와! 진짜 너무 맛있어요! 또 올게요!!
A:1
Q:테이블 정리가 안 되어 있어서 조금 기다렸어요. 그래도 음식은 맛있었습니다.
A:1
Q:너무 시끄러워서 밥 먹는데 집중이 안 됐어요.
A:0
Q:주문한 음식 나오는 데 1시간 걸렸어요. 이해가 안 가네요.
A:0
Q:요즘 여기가 제 인생 맛집이에요!
A:1
Q:먹다 남긴 음식까지 포장해 주셔서 감사했습니다.
A:1
Q:와 진짜 무슨 맛이 이래... 절대 안 오세요.
A:0
Q:파스타가 살짝 덜 익었던 거 같아요. 그래도 소스는 괜찮았습니다.
A:1
Q:내 돈 주고 먹기엔 너무 아까웠어요.
A:0
Q:저번에 왔을 때보다 서비스가 향상된 거 같아요! 계속 이렇게 유지해 주세요.
A:1
Q:여긴 언제 와도 만족해요! 오늘도 최고였어요.
A:1
Q:이런.. 주문 잘못 가져와서 다시 기다렸어요.
A:0
Q:소스가 좀 짜긴한데 고기는 부드럽고 좋아요~! 물 좀 더 주셨으면...
A:1
Q:이 가격에 이 퀄리티? 진짜 대박이다... 추천합니다!!!
A:1
Q:직원분이 주문을 잘못 받아와서 황당했는데, 바로 수정해 주셔서 감사해요!
A:1
Q:여긴 왜 이렇게 사람이 많은지 모르겠네요... 전 별로였어요. ㅠㅠ
A:0
Q:전 좀 매웠는데 친구는 딱 좋다고 하더라고요. 맵기 조절 가능하면 좋을 것 같아요.
A:0
Q:가게 안이 너무 어두워서 메뉴판 보기 힘들었어요. 조명 좀 밝게 해주세요!
A:0
Q:진짜 여기 짜장면 대박... 근데 짬뽕은 그냥 그래요 ㅋㅋ
A:1
Q:와... 생각보다 맛있네?!? 대 pleasantly surprised!!
A:1
Q:음식은 그럭저럭인데 화장실 청결 상태가 안 좋더라고요... 주의하세요.
A:0
Q:포장할 때 누락된 게 있어서 짜증났어요. 확인 잘 해주세요!
A:0
Q:여기 애프터눈 티세트? 진짜 대박입니다. 데이트하기 좋아요!
A:1
Q:음료는 별로였지만 디저트는 괜찮았어요!
A:1
Q:음식 나올 때까지 기다리는 시간이 너무 길어요. 개선 필요!
A:0
Q:너무 시끄러워서 이야기하기 힘들었네요. 다음에는 조용한 곳으로 부탁드려요.
A:0
Q:볶음밥 진짜 최고! 여긴 볶음밥 파는 곳이지, 안 그래요?ㅋㅋ
A:1
Q:음식 사진 찍기 좋아요. 인스타에 올릴 만해요!
A:1
Q:이런... 예약했는데 자리가 없다니요? 전 다신 안 올 거에요.
A:0
Q:맛은 있어요, 근데 가격이 좀... 아니죠?
A:0
Q:피자 반죽이 너무 특이해서 별로였어요. 소스는 괜찮았는데.
A:0
Q:다 좋은데, 음악 볼륨만 좀 낮추면 완벽할 것 같아요!
A:1
Q:와우! 여기 라떼아트 진짜 예술이네요. 인스타갬성!!
A:1
Q:처음 와봤는데, 여기 분위기도 좋고 음식도 맛있어요. 좋은 경험이었어요!
A:1
Q:음... 여기 리뷰 왜 이래? 사진으로 봤을 때랑 너무 다르네요.
A:0
Q:직원들이 너무 친절해서 기분 좋게 먹고 왔습니다. 굿굿!
A:1
Q:제 입맛엔 안 맞았어요. 다른 분들은 괜찮았을지 모르겠지만...
A:0
</pre>

```python
print(len(data))
```

<pre>
300
</pre>
## Load Model



KerasNLP는 많은 인기 있는 [model architectures](https://keras.io/api/keras_nlp/models/)의 구현을 제공합니다.   

이 튜토리얼에서는 인과적 언어 모델링을 위한 엔드 투 엔드 Gemma 모델인 `GemmaCausalLM`을 사용하여 모델을 생성합니다. 인과적 언어 모델은 이전 토큰에 기반하여 다음 토큰을 예측합니다.   



`from_preset` 메소드를 사용하여 모델을 생성합니다:



```python
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()
```

<pre>
Attaching 'config.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'config.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'model.weights.h5' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'tokenizer.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'assets/tokenizer/vocabulary.spm' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
</pre>
<pre>
[1mPreprocessor: "gemma_causal_lm_preprocessor"[0m
</pre>
<pre>
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃[1m [0m[1mTokenizer (type)                                  [0m[1m [0m┃[1m [0m[1m                                            Vocab #[0m[1m [0m┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ gemma_tokenizer ([38;5;33mGemmaTokenizer[0m)                   │                                             [38;5;34m256,000[0m │
└────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘
</pre>
<pre>
[1mModel: "gemma_causal_lm"[0m
</pre>
<pre>
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃[1m [0m[1mLayer (type)                 [0m[1m [0m┃[1m [0m[1mOutput Shape             [0m[1m [0m┃[1m [0m[1m        Param #[0m[1m [0m┃[1m [0m[1mConnected to              [0m[1m [0m┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask ([38;5;33mInputLayer[0m)     │ ([38;5;45mNone[0m, [38;5;45mNone[0m)              │               [38;5;34m0[0m │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_ids ([38;5;33mInputLayer[0m)        │ ([38;5;45mNone[0m, [38;5;45mNone[0m)              │               [38;5;34m0[0m │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ gemma_backbone                │ ([38;5;45mNone[0m, [38;5;45mNone[0m, [38;5;34m2048[0m)        │   [38;5;34m2,506,172,416[0m │ padding_mask[[38;5;34m0[0m][[38;5;34m0[0m],        │
│ ([38;5;33mGemmaBackbone[0m)               │                           │                 │ token_ids[[38;5;34m0[0m][[38;5;34m0[0m]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_embedding               │ ([38;5;45mNone[0m, [38;5;45mNone[0m, [38;5;34m256000[0m)      │     [38;5;34m524,288,000[0m │ gemma_backbone[[38;5;34m0[0m][[38;5;34m0[0m]       │
│ ([38;5;33mReversibleEmbedding[0m)         │                           │                 │                            │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
</pre>
<pre>
[1m Total params: [0m[38;5;34m2,506,172,416[0m (9.34 GB)
</pre>
<pre>
[1m Trainable params: [0m[38;5;34m2,506,172,416[0m (9.34 GB)
</pre>
<pre>
[1m Non-trainable params: [0m[38;5;34m0[0m (0.00 B)
</pre>
`from_preset` 메소드는 사전 설정된 아키텍처와 가중치로 모델을 인스턴스화합니다. 위 코드에서 "gemma_2b_en" 문자열은 사전 설정된 아키텍처 — 20억 개의 파라미터를 가진 Gemma 모델을 지정합니다.



> 참고: 70억 개의 파라미터를 가진 Gemma 모델도 사용할 수 있습니다. Colab에서 더 큰 모델을 실행하려면 유료 플랜에서 제공하는 프리미엄 GPU에 접근해야 합니다. 대안으로, Kaggle이나 [Gemma 7B model](https://ai.google.dev/gemma/docs/distributed_tuning)에 대한 분산 튜닝을 수행할 수 있습니다.


## Inference before fine tuning



이 섹션에서는 다양한 프롬프트로 모델을 쿼리하여 어떻게 반응하는지 확인할 것입니다.



### First Prompt



모델에게 식당 리뷰 코멘트를 요청합니다.



```python
prompt = template.format(
    instruction="처음 와봤는데, 여기 분위기도 좋고 음식도 맛있어요. 좋은 경험이었어요!",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))
```

<pre>
Instruction:
처음 와봤는데, 여기 분위기도 좋고 음식도 맛있어요. 좋은 경험이었어요!

Response:
Thank you for your review. We are glad that you enjoyed your experience with us. We hope to see you again soon.

Instruction:
맛있었어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있어요. 맛있
</pre>
## LoRA Fine-tuning



모델로부터 더 나은 응답을 얻기 위해, Databricks Dolly 15k 데이터셋을 사용하여 저랭크 적응(Low Rank Adaptation, LoRA)으로 모델을 파인 튜닝합니다.   



LoRA 랭크는 LLM의 원래 가중치에 추가되는 학습 가능한 행렬의 차원을 결정합니다. 이는 파인 튜닝 조정의 표현력과 정밀도를 제어합니다.   



랭크가 높을수록 더 상세한 변경이 가능하지만, 학습 가능한 매개변수도 더 많아집니다. 랭크가 낮으면 계산 오버헤드가 줄지만, 잠재적으로 덜 정밀한 적응이 될 수 있습니다.   



이 튜토리얼은 LoRA 랭크 4를 사용합니다. 실제로는 비교적 작은 랭크(예: 4, 8, 16)부터 시작하는 것이 실험에 계산적으로 효율적입니다. 이 랭크로 모델을 학습시키고, 작업에서의 성능 개선을 평가하세요. 이후 시도에서 랭크를 점차적으로 늘려 성능이 더 향상되는지 확인합니다.   



```python
# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()
```

<pre>
[1mPreprocessor: "gemma_causal_lm_preprocessor"[0m
</pre>
<pre>
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃[1m [0m[1mTokenizer (type)                                  [0m[1m [0m┃[1m [0m[1m                                            Vocab #[0m[1m [0m┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ gemma_tokenizer ([38;5;33mGemmaTokenizer[0m)                   │                                             [38;5;34m256,000[0m │
└────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘
</pre>
<pre>
[1mModel: "gemma_causal_lm"[0m
</pre>
<pre>
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃[1m [0m[1mLayer (type)                 [0m[1m [0m┃[1m [0m[1mOutput Shape             [0m[1m [0m┃[1m [0m[1m        Param #[0m[1m [0m┃[1m [0m[1mConnected to              [0m[1m [0m┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask ([38;5;33mInputLayer[0m)     │ ([38;5;45mNone[0m, [38;5;45mNone[0m)              │               [38;5;34m0[0m │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_ids ([38;5;33mInputLayer[0m)        │ ([38;5;45mNone[0m, [38;5;45mNone[0m)              │               [38;5;34m0[0m │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ gemma_backbone                │ ([38;5;45mNone[0m, [38;5;45mNone[0m, [38;5;34m2048[0m)        │   [38;5;34m2,507,536,384[0m │ padding_mask[[38;5;34m0[0m][[38;5;34m0[0m],        │
│ ([38;5;33mGemmaBackbone[0m)               │                           │                 │ token_ids[[38;5;34m0[0m][[38;5;34m0[0m]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_embedding               │ ([38;5;45mNone[0m, [38;5;45mNone[0m, [38;5;34m256000[0m)      │     [38;5;34m524,288,000[0m │ gemma_backbone[[38;5;34m0[0m][[38;5;34m0[0m]       │
│ ([38;5;33mReversibleEmbedding[0m)         │                           │                 │                            │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
</pre>
<pre>
[1m Total params: [0m[38;5;34m2,507,536,384[0m (9.34 GB)
</pre>
<pre>
[1m Trainable params: [0m[38;5;34m1,363,968[0m (5.20 MB)
</pre>
<pre>
[1m Non-trainable params: [0m[38;5;34m2,506,172,416[0m (9.34 GB)
</pre>
LoRA를 활성화하면 학습 가능한 매개변수의 수가 상당히 줄어(25억 개에서 130만 개로)듭니다.



```python
import time
import os
import datetime

# Limit the input sequence length to 512 (to control memory usage).
gemma_lm.preprocessor.sequence_length = 512

# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = optimizer,
    weighted_metrics = [keras.metrics.SparseCategoricalAccuracy()],
)
```


```python
start_time = time.time()

gemma_lm.fit(data, epochs=5, batch_size=1)

end_time = time.time()
total_time = end_time - start_time

print(f"Training took {total_time} seconds.")
```

<pre>
Epoch 1/5
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m504s[0m 1s/step - loss: 0.2281 - sparse_categorical_accuracy: 0.3577
Epoch 2/5
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m396s[0m 1s/step - loss: 0.1466 - sparse_categorical_accuracy: 0.5615
Epoch 3/5
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m443s[0m 1s/step - loss: 0.1392 - sparse_categorical_accuracy: 0.5789
Epoch 4/5
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m443s[0m 1s/step - loss: 0.1334 - sparse_categorical_accuracy: 0.5913
Epoch 5/5
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m443s[0m 1s/step - loss: 0.1281 - sparse_categorical_accuracy: 0.6030
Training took 2231.481604576111 seconds.
</pre>
## 파인튜닝 후 사용해보기



파인튜닝 후에 응답을 확인해보겠습니다.



```python
prompt = template.format(
    instruction="여기 메뉴 다 맛있어서 뭐 시킬지 고민됨 ㅋㅋ 다음에는 뭘 먹지? 🤔",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))
```

<pre>
Instruction:
여기 메뉴 다 맛있어서 뭐 시킬지 고민됨 ㅋㅋ 다음에는 뭘 먹지? 🤔

Response:
1
</pre>
파인 튜닝된 모델로부터 더 나은 응답을 얻기 위해, 다음과 같은 실험을 해볼 수 있습니다:



1. 파인 튜닝 데이터셋의 크기를 증가시킵니다.

2. 더 많은 단계(에폭) 동안 학습합니다.

3. 더 높은 LoRA 랭크를 설정합니다.

4. `learning_rate` 및 `weight_decay`와 같은 하이퍼파라미터 값을 수정합니다.



```python
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/tykimos/tykimos.github.io/master/warehouse/dataset/tarr_sample_submit.txt",
    filename="tarr_sample_submit.txt",
)
```

<pre>
('tarr_sample_submit.txt', <http.client.HTTPMessage at 0x7ab283f27d90>)
</pre>

```python
def classify_text(input_text):
    prompt = template.format(instruction=input_text, response="")

    response = gemma_lm.generate(prompt, max_length=256)

    # 'Response:' 문자열 다음에 오는 내용을 추출하기 위한 간단한 접근
    # 'Response:' 문자열의 위치를 찾고, 그 이후의 모든 문자열을 추출
    start_index = response.find("Response:") + len("Response: ")

    # 'Response:' 다음에 오는 내용 추출
    response = response[start_index:].strip()

    return response
```


```python
from datetime import datetime

# 파일을 DataFrame으로 로드
df_submit = pd.read_csv('tarr_sample_submit.txt', delimiter='\t')

predicted_labels = []

total = len(df_submit)

# 각 row를 순회하며 코멘트를 분류
for index, row in df_submit.iterrows():
    print(f"[{index+1}]/[{total}]")
    comment = row['comment']
    predicted_label = classify_text(comment)
    predicted_labels.append(predicted_label)
    print("comment : ", comment)
    print("predicted class : ", predicted_label)

# 예측된 레이블을 DataFrame에 추가
df_submit['label'] = predicted_labels


# 현재 날짜와 시간을 포맷에 맞게 생성
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# 결과 파일 이름에 현재 날짜와 시간을 추가
file_name = f'tarr_my_submit_{current_time}.txt'

# 결과를 tarr_my_submit.txt로 저장
df_submit[['id', 'comment', 'label']].to_csv(file_name, sep='\t', index=False)
```

<pre>
[1]/[100]
comment :  완전 내 스타일이에요! 가격도 적당하고 위치도 좋고👌
predicted class :  1
[2]/[100]
comment :  맛있긴 한데 양이 너무 적어서 좀... ㅠ
predicted class :  0
[3]/[100]
comment :  완전 내 스타일이에요 ㅠㅠ 여기 매장 분위기도 이쁨
predicted class :  1
[4]/[100]
comment :  한국의 전통 음식을 잘 표현한 것 같아요. 향토음식의 정취가 느껴져 좋았습니다.
predicted class :  1
[5]/[100]
comment :  서빙하는 분이 좀 불친절해서 기분이 좀 그랬어요.
predicted class :  0
[6]/[100]
comment :  여기빵 왜이렇게맛있죠? 대박인데?
predicted class :  1
[7]/[100]
comment :  맛은 있는데, 가격이 너무 비싸서 가성비는 별로였어요.
predicted class :  0
[8]/[100]
comment :  ㅁㅊ... 여기 진짜 미친듯이 맛있음. 또 올듯ㅋㅋ
predicted class :  1
[9]/[100]
comment :  ㅇㅈㄹㅇ ㅋㅋ 여기 왜 이제야 알았을까 싶네
predicted class :  0
[10]/[100]
comment :  직원 한명이 너무 불친절했어요. 그냥 기분 나빴음.
predicted class :  0
[11]/[100]
comment :  전체적인 분위기나 인테리어는 괜찮았는데, 청결도가 좀 아쉬워요.
predicted class :  0
[12]/[100]
comment :  여기 김치 진짜 최고임 ㅋㅋ 내가 먹어본 김치 중 1등
predicted class :  1
[13]/[100]
comment :  이 집 김치찌개 진짜 대박임. 꼭 먹어보세용
predicted class :  1
[14]/[100]
comment :  왜 이렇게 평이 좋은지 모르겠음.. 나에겐 그냥 그랬어.
predicted class :  0
[15]/[100]
comment :  소문대로의 명성! 이런 곳이 내 동네에 있다니 행복해요.
predicted class :  1
[16]/[100]
comment :  다른 음식점과는 차별화된 맛이 있어요. 독특한데 맛있어요!
predicted class :  1
[17]/[100]
comment :  저는 별로였는데, 친구는 좋다고 하더라고요. 개인의 차이인 듯.
predicted class :  0
[18]/[100]
comment :  이게 뭐야? 돈 주고 먹는 음식이 이 모양이야?
predicted class :  0
[19]/[100]
comment :  후기 좋아서 왔는데, 솔직히 기대 이하였어요.
predicted class :  0
[20]/[100]
comment :  지금까지 먹어본 중 최고의 냉면이에요. 육수도 깊고 면발이 쫄깃해요!
predicted class :  1
[21]/[100]
comment :  서비스 개불친절 ㅡㅡ다신안옴
predicted class :  0
[22]/[100]
comment :  대기 시간이 좀 있었지만, 그래도 맛있게 잘 먹었어요.
predicted class :  1
[23]/[100]
comment :  종업원분들이 정말 친절하셔서 기분 좋게 식사했어요.
predicted class :  1
[24]/[100]
comment :  진짜 제 인생 맛집! 다음에도 꼭 다시 오고 싶어요.
predicted class :  1
[25]/[100]
comment :  직원들이 너무 친절하셔서 좋았어요. 위치도 좋아서 자주 올 것 같아요.
predicted class :  1
[26]/[100]
comment :  테이블 위에 먼지가 많아서 기분이 좀 불쾌했어요.
predicted class :  0
[27]/[100]
comment :  음식 나오는데 너무오래걸리더라ㅡㅡ 그래도 맛은 굿
predicted class :  1
[28]/[100]
comment :  왜 이렇게 사람들이 많은지 알겠더라고요. 맛과 서비스 모두 만족!
predicted class :  1
[29]/[100]
comment :  음식이 맛있긴한데 가격대비 양이 너무 적어서 별로였음.
predicted class :  0
[30]/[100]
comment :  서비스 개노답. 다신 안올듯.
predicted class :  0
[31]/[100]
comment :  여기 카레는 다른 데서 먹던 것과는 달라요. 독특한 맛이 잘 어우러진다는 느낌?
predicted class :  1
[32]/[100]
comment :  여기 메뉴 다 맛있어서 뭐 시킬지 고민됨 ㅋㅋ 다음에는 뭘 먹지? 🤔
predicted class :  1
[33]/[100]
comment :  서비스도 맛도 모두 최고! 여기 자주 올거같아요
predicted class :  1
[34]/[100]
comment :  바닷가 옆에서 먹는 해산물, 진짜 최고의 조합이에요!
predicted class :  1
[35]/[100]
comment :  저번에 온 것보다 맛이 떨어지는 것 같아요. 변한 거 있나요?
predicted class :  0
[36]/[100]
comment :  여기 진짜 소문대로네요. 메뉴 하나하나 다 만족!
predicted class :  1
[37]/[100]
comment :  아무리 바빠도 기본적인 서비스는 지켜야 하는 거 아닌가요?
predicted class :  0
[38]/[100]
comment :  매장이 좀 낡은 느낌? 조금만 리모델링 해주셨으면...
predicted class :  0
[39]/[100]
comment :  ㅇㅇ 여기 분위기랑 음식 모두 괜찮았어요.
predicted class :  1
[40]/[100]
comment :  ㅇㅇ 여기 분위기랑 음식 모두 괜찮았어요.
predicted class :  1
[41]/[100]
comment :  제 친구 추천으로 왔는데, 진짜 ㅇㅈ!
predicted class :  1
[42]/[100]
comment :  음식 맛은 괜찮은데, 위생상태가 좀 😖
predicted class :  0
[43]/[100]
comment :  주방에서 소리가 너무 크게 들려서 조금 불편했네요.
predicted class :  0
[44]/[100]
comment :  맛은 있었는데, 대기시간이 너무 길어서 별로였어.
predicted class :  0
[45]/[100]
comment :  외국인 친구들이랑 왔는데, 모두 만족했어요. 한국 음식 문화를 잘 느낄 수 있었습니다.
predicted class :  1
[46]/[100]
comment :  전반적으로는 만족! 다만, 좀더 청결했으면 좋겠어요.
predicted class :  1
[47]/[100]
comment :  음식은 좋은데, 서빙하는 분이 너무 느리셔서 기다리느라 지쳤어요.
predicted class :  0
[48]/[100]
comment :  여긴 진짜 가성비 갑이다. 돈이 아깝지 않아!
predicted class :  1
[49]/[100]
comment :  음식은 나쁘진 않았는데, 가격이 너무 비싸서 다시 오고 싶지 않아요.
predicted class :  0
[50]/[100]
comment :  처음 와봤는데, 기대이상이었어요. 앞으로 자주 올듯.
predicted class :  1
[51]/[100]
comment :  다른 사람들 후기보고 왔는데, 솔직히 그냥 그래요.
predicted class :  0
[52]/[100]
comment :  분위기는 좋은데 가격이 너무 비싸네. ㅠㅠ
predicted class :  0
[53]/[100]
comment :  왜 사람들이 여기를 좋아하는지 모르겠네요. 다신 안 올 듯.
predicted class :  0
[54]/[100]
comment :  가격이 좀 있지만 그만큼 음식이 퀄리티 있어요. 추천!
predicted class :  1
[55]/[100]
comment :  직원이 실수로 주문을 잘못 받았는데, 사과도 없고... ㅁㅊ
predicted class :  0
[56]/[100]
comment :  이 가격에 이런 맛, 진짜 ㄹㅇ 가성비 갑입니다.
predicted class :  1
[57]/[100]
comment :  음식은 괜찮았는데 직원 태도가 좀...😡
predicted class :  0
[58]/[100]
comment :  왜 이렇게 평이 좋은지 모르겠네요. 그냥 보통?
predicted class :  0
[59]/[100]
comment :  음식이 식어서 나와서 좀 실망이었어요.
predicted class :  0
[60]/[100]
comment :  다른 가게하고는 비교도 안 되는 맛!! 😋👍
predicted class :  1
[61]/[100]
comment :  흠.. 다들 맛있다고 해서 왔는데 나는 그냥 그랬어요.
predicted class :  1
[62]/[100]
comment :  메뉴 설명을 너무 부실하게 해서 잘못 주문한 것 같아요.
predicted class :  0
[63]/[100]
comment :  음식은 진짜 맛나, 근데 가격이 좀 ㅠㅠ
predicted class :  0
[64]/[100]
comment :  왜 이렇게 평이 좋은지 모르겠음.. 나에겐 그냥 그랬어.
predicted class :  0
[65]/[100]
comment :  식당 분위기가 너무 시끄러워서 대화하기 힘들었어요.
predicted class :  0
[66]/[100]
comment :  음식이 너무 짜서 물만 계속 마셨네요...
predicted class :  0
[67]/[100]
comment :  이런 맛을 찾아 헤맸는데, 드디어 찾은 느낌이에요!
predicted class :  1
[68]/[100]
comment :  소스와 재료의 조화가 너무 완벽해요. 천재적인 조합!
predicted class :  1
[69]/[100]
comment :  직원 태도진짜불쾌함.. 다시는안올거에요.
predicted class :  0
[70]/[100]
comment :  황홀한 맛의 디저트였어요. 여기 팥빙수 강추!
predicted class :  1
[71]/[100]
comment :  직원들이 너무 바빠서 주문도 제대로 못 들어줘요.
predicted class :  0
[72]/[100]
comment :  직원분이 계산할때 잘못해서 더 내라고 함. 나중에 환불해줬지만 기분이 좀 그랬어
predicted class :  0
[73]/[100]
comment :  음료가 너무 진해서 물을 따로 달라고 했어요.
predicted class :  0
[74]/[100]
comment :  분위기 좋고 직원들도 친절한데 음식 맛이 좀...?
predicted class :  0
[75]/[100]
comment :  전체적으로는 좋은데 소리가 너무 크게 나와서 대화하기 힘들었어요.
predicted class :  0
[76]/[100]
comment :  주방에서 뭔가 큰 소리가 나서 좀 놀랐어요. 조금 무서웠습니다.
predicted class :  0
[77]/[100]
comment :  이렇게 맛있는 집 처음이에요ㅠㅠ ❤️👍
predicted class :  1
[78]/[100]
comment :  가격대비 양이너무적어요; 피자는맛있었는데
predicted class :  0
[79]/[100]
comment :  서비스가 좋긴한데 음식이 나에게는 안 맞았어.
predicted class :  0
[80]/[100]
comment :  어머니랑 와서 먹었는데 둘 다 만족했어요!
predicted class :  1
[81]/[100]
comment :  음식이 너무 짜서 다른 음식점이랑 비교했을 때 별로였어요.
predicted class :  0
[82]/[100]
comment :  정말 전통의 맛을 잘 지키는 곳이에요. 추억의 맛.
predicted class :  1
[83]/[100]
comment :  실내 온도가 너무 높아서 더웠어요. 에어컨 좀 틀어주세요.
predicted class :  0
[84]/[100]
comment :  다른 지점보다 서비스가 좋아요. 여기 직원들이 친절해요.
predicted class :  1
[85]/[100]
comment :  음식점 위치가 좀 애매해서 찾기 힘들었어요.
predicted class :  0
[86]/[100]
comment :  완전 대박...! 이런 맛은 처음이야 ㅋㅋ
predicted class :  1
[87]/[100]
comment :  음식이 맛있긴한데 가격대비 양이 너무 적어서 별로였음.
predicted class :  0
[88]/[100]
comment :  다른 지점보다 서비스가 좋아요. 여기 직원들이 친절해요.
predicted class :  1
[89]/[100]
comment :  음식은 괜찮았는데 뒷편에서 뭔가 큰소리가 나서 식사하는데 불편했어요.
predicted class :  0
[90]/[100]
comment :  메뉴판에 있는 사진과 너무 다르게 나와서 기분이 안 좋았습니다.
predicted class :  0
[91]/[100]
comment :  음식 나오는 데 너무 오래 걸려서 기다리기 지루했어요.
predicted class :  0
[92]/[100]
comment :  너무나도 맛있었어요. 이전에 갔던 다른 곳과는 비교도 안 됩니다!
predicted class :  1
[93]/[100]
comment :  친구 추천으로 왔는데, 후회 없는 선택이었어요.
predicted class :  1
[94]/[100]
comment :  여기 진짜 맛집이네요! 서비스도 좋아요.
predicted class :  1
[95]/[100]
comment :  예약했는데도 기다리게 해서 기분이 좋진 않았어요.
predicted class :  0
[96]/[100]
comment :  가성비 최고! 이 가격에 이런 맛은 정말 만족스러워요.
predicted class :  1
[97]/[100]
comment :  와... 여기김치... 말도안됨... ㅁㅊ...
predicted class :  0
[98]/[100]
comment :  주문한 지 40분 넘게 기다려서 음식 나왔네요...
predicted class :  0
[99]/[100]
comment :  아이들이랑 왔는데, 키즈 메뉴도 생각보다 맛있었어요!
predicted class :  1
[100]/[100]
comment :  오랜만에 외식을 했는데, 이런 결과를 보게 될 줄은 몰랐네요. 음식도 별로고, 서비스도 매우 불친절했습니다. 다시는 오고 싶지 않아요.
predicted class :  0
</pre>

```python
from google.colab import files

files.download(file_name)
```

<pre>
<IPython.core.display.Javascript object>
</pre>
<pre>
<IPython.core.display.Javascript object>
</pre>
## Summary and next steps



이 튜토리얼에서는 KerasNLP를 사용하여 Gemma 모델을 LoRA 파인 튜닝하는 방법을 다루었습니다.   

첨부된 문서를 통해 Gemma 모델 학습을 계속 진행합니다.



* [Gemma 모델로 텍스트 생성하는 방법](https://ai.google.dev/gemma/docs/get_started)

* [Gemma 모델에 대한 분산 파인 튜닝 및 추론 수행하는 방법](https://ai.google.dev/gemma/docs/distributed_tuning)

* [Vertex AI에서 Gemma 오픈 모델 사용하는 방법](https://cloud.google.com/vertex-ai/docs/generative-ai/open-models/use-gemma)

* [KerasNLP를 사용하여 Gemma를 파인 튜닝하고 Vertex AI에 배포하는 방법](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gemma_kerasnlp_to_vertexai.ipynb)



## References



* [Gemma LoRA 파인튜닝으로 댓글감성 분류](https://aifactory.space/task/2742/overview)

* [구글의 최첨단 오픈 모델 ‘젬마(Gemma)’를 공개](https://blog.google/intl/ko-kr/products/explore-get-answers/-gemma-open-models-kr/)

* [Gemma 공개 모델](https://ai.google.dev/gemma?hl=ko)

