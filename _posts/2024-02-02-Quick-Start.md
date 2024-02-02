---
layout: single
title:  "MLX: Apple silicon Machine Learning - 01.Quick Start Guide"
categories: MLX
tag: [IOS, MLX, coding]
toc: true
author_profile: false
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
    font-size: 0.8rem !important;
  }

  </style>
</head>


# MLX

---

   

MLX 는 Apple silicon 에서 머신러닝을 수행하기 위해 만들어진 array framework 입니다.   

Apple silicon 만의 CPU와 GPU를 사용하여 벡터와 그래프 연산 속도를 크게 높일 수 있습니다.   


## 설치 요구 사항

1. M 시리즈 apple silicon

2. native Python >= 3.8

3. MacOS >= 13.3



```python
!python -c "import platform; print(platform.processor())" # It must be arm
!pip install mlx
```

<pre>
arm
Collecting mlx
  Obtaining dependency information for mlx from https://files.pythonhosted.org/packages/8f/e7/40e631abca0823399ad5f89e2fd849393d7e6a8f3efd2cf1a3ef4ceb0df0/mlx-0.0.11-cp311-cp311-macosx_14_0_arm64.whl.metadata
  Downloading mlx-0.0.11-cp311-cp311-macosx_14_0_arm64.whl.metadata (4.9 kB)
Downloading mlx-0.0.11-cp311-cp311-macosx_14_0_arm64.whl (17.1 MB)
[2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.1/17.1 MB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m[36m0:00:01[0mm eta [36m0:00:01[0m
[?25hInstalling collected packages: mlx
Successfully installed mlx-0.0.11
</pre>
## Basic Quick Start

---

   

array(배열)를 만들기 위해 mlx.core를 import 합니다.



```python
import mlx.core as mx

a = mx.array([1,2,3,4])
print(f'a shape: {a.shape}')
print(f'a dtype: {a.dtype}')

b = mx.array([1.0, 2.0, 3.0, 4.0])
print(f'b shape: {b.shape}')
print(f'b dtype: {b.dtype}')

c = mx.array([[1,2,3,4],[5,6,7,8]])
print(f'c shape: {c.shape}')
print(f'c dtype: {c.dtype}')

d = mx.array([[1,2,3,4],[5.0,6.0,7.0,8.0]])
print(f'd shape: {d.shape}')
print(f'd dtype: {d.dtype}')
```

<pre>
a shape: [4]
a dtype: mlx.core.int32
b shape: [4]
b dtype: mlx.core.float32
c shape: [2, 4]
c dtype: mlx.core.int32
d shape: [2, 4]
d dtype: mlx.core.float32
</pre>
MLX는 lazy evaluation을 사용합니다.

> **lazy evaluation이란?**   

: 실제로 연산 결과가 어딘가에 사용되기 전까지 연산을 미루는 프로그래밍 방법론입니다.

   

lazy evaluation은 성능 관점에서 이득을 보거나 오류를 회피 혹은 무한 자료구조를 사용할 수 있다는 장점이 있습니다.

   

연산이 일어나지 않은 것과 연산이 일어나는 것에 대한 시간 소모를 확인해 보겠습니다.



```python
import time

start = time.time()
for _ in range(100):
    c = a + b # 실제로 연산이 일어나지 않는다, c가 사용되지 않았기 때문이다.
print(f'lazy evaluation time: {time.time()-start}')

start = time.time()
for _ in range(100):
    c = a + b
    mx.eval(c) # 연산이 수행된다. (mx.eval 함수는 강제로 연산을 수행시킨다.)
print(f'forced evaluation time: {time.time()-start}')
```

<pre>
lazy evaluation time: 0.0003581047058105469
forced evaluation time: 0.02145099639892578
</pre>
## Unified Memory

---

   

Apple Silicon은 CPU와 GPU가 별개의 장치로 존재하지 않습니다.   

하나의 Unifired Memory Architecture(UMA)로 구성되어 있으며, CPU와 GPU가 동일한 memory pool에서 직접적으로 접근 가능합니다.   

MLX는 이러한 장점을 누릴 수 있도록 디자인 되었습니다.   

   

*(현재 torch에서 MPS를 사용하여 GPU를 사용할 수 있지만 성능을 제대로 사용하지 못하고 있습니다. MLX는 M3 Macbook pro 성능을 제대로 이끌 것 같아서 기대가 됩니다.)*



```python
# 두개의 array 생성
a = mx.random.normal((100,))
b = mx.random.normal((100,))
```

MLX에서는 operation을 위한 device를 지정해줄 수 있습니다.   

즉, memory 위치의 이동 없이 CPU 연산과 GPU 연산을 모두 할 수 있습니다.



```python
# dependency 존재 X
mx.add(a, b, stream = mx.cpu)
mx.add(a, b, stream = mx.gpu)

# dependency 존재 O
c = mx.add(a, b, stream = mx.cpu)
d = mx.add(a, b, stream = mx.gpu)
```

dependency가 존재하지 않을 경우 병렬적으로 각각 연산이 됩니다.   

하지만 dependency가 존재할 경우 첫 번째 연산이 끝난 후 두 번째 연산이 시작됩니다.   

*('c' 연산 후 'd' 연산)*



---



연산의 종류에 따라서 CPU가 유리할 수도 있고 GPU가 유리할 수도 있습니다.   

matmul 연산은 GPU에세 유리한 연산입니다. 하지만 for loop로 이루어진 연산은 CPU에 유리한 연산입니다.   

   

아래의 내용으로 연산 속도를 확인할 수 있습니다.



```python
def fun(a, b, d1, d2):
    x = mx.matmul(a, b, stream=d1)
    mx.eval(x) # mx.eval 함수는 강제로 연산을 수행시킨다.
    for _ in range(500):
        b = mx.exp(b, stream=d2)
        mx.eval(b)
    return x, b

a = mx.random.uniform(shape=(4096, 512))
b = mx.random.uniform(shape=(512, 4))

start = time.time()
fun(a, b, mx.cpu, mx.cpu)
print(f"cpu elapsed time: {time.time()-start}")

start = time.time()
fun(a, b, mx.gpu, mx.gpu)
print(f"gpu elapsed time: {time.time()-start}")

start = time.time()
fun(a, b, mx.cpu, mx.gpu)
print(f"cpu-gpu elapsed time: {time.time()-start}")

start = time.time()
fun(a, b, mx.gpu, mx.cpu)
print(f"gpu-cpu elapsed time: {time.time()-start}")
```

<pre>
cpu elapsed time: 0.024873018264770508
gpu elapsed time: 0.11129403114318848
cpu-gpu elapsed time: 0.062744140625
gpu-cpu elapsed time: 0.0065081119537353516
</pre>
MXL은 stream을 지정하지 않으면 default_device로 설정되어 있습니다.   

M3 pro 기준은 GPU입니다.



```python
print(mx.default_device())
print(mx.default_stream(mx.default_device()))
```

<pre>
Device(gpu, 0)
Stream(Device(gpu, 0), 0)
</pre>
## References

---

* [MLX](https://ml-explore.github.io/mlx/build/html/index.html)(MLX 홈페이지)   

* [SKT Enterprise](https://www.sktenterprise.com/bizInsight/blogDetail/dev/8107)(MLX 설명)   

* [Medium]('https://medium.com/sjk5766/lazy-evaluation%EC%9D%84-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-411651d5227b')(Lazy computation 설명)

