---
layout: single
title:  "MLX: Apple Silicon MacBook M3 Pro에서 구동하는 Gemma-7B-IT 모델 기반 PDF 요약 서비스 개발"
categories: MLX
tag: [coding, Gemma, MLX, LLM, NLP]
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


## 개요



이 프로젝트에서는 Gemma-7b-it 모델을 사용하여 Apple Silicon Macbook M3 pro에서 돌아가는 pdf 요약 서비스를 만들 것이다.


## 환경설정


필요한 라이브러리를 다운받습니다.



```python
!pip install -U -q langchain pypdf langchain_community
!pip install -U -q mlx-lm
!pip install -U -q huggingface-hub hf-transfer
```

<pre>
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
</pre>
<pre>
[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m
</pre>
<pre>
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
</pre>
<pre>
[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m
</pre>
<pre>
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
</pre>
<pre>
[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m
</pre>
mlx 사용이 가능한 gemma-7b-it 모델을 가져옵니다.



```python
from mlx_lm import load, generate
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
model_id = "mlx-community/quantized-gemma-7b-it"
model, tokenizer = load(model_id)
```

<pre>
Fetching 8 files:   0%|          | 0/8 [00:00<?, ?it/s]
</pre>
## 모델 사용


Gemma는 특정 명령을 사용합니다.   

특정 명령에 맞게 template를 설정합니다.



```python
from jinja2 import Template
from typing import List
from pprint import pprint

def apply_chat_template(messages:List, add_generation:bool=True):
    template_str = "{% for item in messages %}" \
                   "<start_of_turn>{{ item.role }}\n{{ item.content }}<end_of_turn>\n" \
                   "{% if loop.last %}{% if add_generation %}<start_of_turn>model\n{% endif %}{% endif %}" \
                   "{% endfor %}"
    
    template = Template(template_str)
    result = template.render(messages=messages, add_generation=add_generation)
    return result
```

간단한 질문을 해봅니다.



```python
messages = [
    {"content": "Response announcer, Who is Einstein?", "role":"user"},
]
```

응답을 잘한다는 것을 확인해 볼 수 있었습니다.   



```python
response = generate(model, tokenizer, prompt=apply_chat_template(messages), temp=0.1, max_tokens=500, verbose=True)
```

<pre>
==========
Prompt: <start_of_turn>user
Response announcer, Who is Einstein?<end_of_turn>
<start_of_turn>model

Albert Einstein was a German-born physicist who revolutionized the field of physics with his theories of relativity. He is known for his groundbreaking contributions to the theory of motion, gravity, and space-time. Einstein's theories revolutionized our understanding of the universe and paved the way for many modern scientific advancements.
==========
Prompt: 25.011 tokens-per-sec
Generation: 18.912 tokens-per-sec
</pre>

```python
messages = [
    {"content": "아나운서처럼 응답해줘, 아인슈타인 누구입니까?", "role":"user"},
]
response = generate(model, tokenizer, prompt=apply_chat_template(messages), temp=0.1, max_tokens=500, verbose=True)
```

<pre>
==========
Prompt: <start_of_turn>user
아나운서처럼 응답해줘, 아인슈타인 누구입니까?<end_of_turn>
<start_of_turn>model

아인슈타인은 독일 태생의 물리학자로, 20세기 가장 중요한 발견 중 하나인 상대론을 발표했습니다. 그는 빛의 속도를 초과하는 물리적 현상에 대한 이론을 개발했습니다. 아인슈타인은 1921년 노벨 물리학상을 받았습니다.
==========
Prompt: 125.069 tokens-per-sec
Generation: 19.403 tokens-per-sec
</pre>
## PDF 요약 모델



```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_file(file_name, file_type):
    loader = PyPDFLoader(f"./storage/{file_name}.{file_type}")

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=5000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    return loader.load_and_split(text_splitter)
```

저장된 위치의 pdf 파일을 불러오고 지정한 `chunk_size`로 `documents`에 저장합니다.



```python
documents = load_file(file_name='Gemma-report', file_type='pdf')
documents
```

<pre>
[Document(page_content='2024-02-21\nGemma: Open Models Based on Gemini\nResearch and Technology\nGemma Team, Google DeepMind1\nThisworkintroducesGemma,afamilyoflightweight,state-of-theartopenmodelsbuiltfromtheresearch\nand technology used to create Gemini models. Gemma models demonstrate strong performance across\nacademicbenchmarksforlanguageunderstanding, reasoning, andsafety. Wereleasetwosizesofmodels\n(2 billion and 7 billion parameters), and provide both pretrained and fine-tuned checkpoints. Gemma\noutperformssimilarlysizedopenmodelson11outof18text-basedtasks, andwepresentcomprehensive\nevaluations of safety and responsibility aspects of the models, alongside a detailed description of model\ndevelopment. We believe the responsible release of LLMs is critical for improving the safety of frontier\nmodels, and for enabling the next wave of LLM innovations.\nIntroduction\nWe present Gemma, a family of open models\nbased on Google’s Gemini models (Gemini Team,\n2023).\nWe trained Gemma models on up to 6T to-\nkens of text, using similar architectures, data,\nand training recipes as the Gemini model family.\nLike Gemini, these models achieve strong gener-\nalist capabilities in text domains, alongside state-\nof-the-art understanding and reasoning skills at\nscale. With this work, we release both pre-trained\nand fine-tuned checkpoints, as well as an open-\nsource codebase for inference and serving.\nGemma comes in two sizes: a 7 billion param-\neter model for efficient deployment and develop-\nment on GPU and TPU, and a 2 billion param-\neter model for CPU and on-device applications.\nEach size is designed to address different compu-\ntational constraints, applications, and developer\nrequirements. At each scale, we release raw, pre-\ntrained checkpoints, as well as checkpoints fine-\ntuned for dialogue, instruction-following, help-\nfulness, and safety. We thoroughly evaluate the\nshortcomingsofourmodelsonasuiteofquantita-\ntive and qualitative benchmarks. We believe the\nrelease of both pretrained and fine-tuned check-\npoints will enable thorough research and inves-\ntigation into the impact of current instruction-\ntuning regimes, as well as the development of\nincreasingly safe and responsible model develop-\nment methodologies.Gemma advances state-of-the-art performance\nrelative to comparable-scale (and some larger),\nopen models (Almazrouei et al., 2023; Jiang\net al., 2023; Touvron et al., 2023a,b) across a\nwide range of domains including both automated\nbenchmarks and human evaluation. Example do-\nmains include question answering (Clark et al.,\n2019; Kwiatkowski et al., 2019), commonsense\nreasoning (Sakaguchi et al., 2019; Suzgun et al.,\n2022), mathematics and science (Cobbe et al.,\n2021;Hendrycksetal.,2020),andcoding(Austin\net al., 2021; Chen et al., 2021). See complete de-\ntails in the Evaluation section.\nLike Gemini, Gemma builds on recent work\non sequence models (Sutskever et al., 2014) and\ntransformers (Vaswani et al., 2017), deep learn-\ning methods based on neural networks (LeCun\net al., 2015), and techniques for large-scale train-\ning on distributed systems (Barham et al., 2022;\nDean et al., 2012; Roberts et al., 2023). Gemma\nalso builds on Google’s long history of open mod-\nelsandecosystems,includingWord2Vec(Mikolov\net al., 2013), the Transformer (Vaswani et al.,\n2017), BERT (Devlin et al., 2018), and T5 (Raffel\net al., 2019) and T5X (Roberts et al., 2022).\nWe believe the responsible release of LLMs is\ncriticalforimprovingthesafetyoffrontiermodels,\nforensuringequitableaccesstothisbreakthrough\ntechnology, for enabling rigorous evaluation and\nanalysis of current techniques, and for enabling\nthe development of the next wave of innovations.\nWhile thorough testing of all Gemma models has\n1See Contributions and Acknowledgments section for full author list. Please send correspondence to gemma-1-report@google.com .\n©2024 Google DeepMind. All rights reserved', metadata={'source': './storage/Gemma-report.pdf', 'page': 0}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nPerformance by Score\n020406080\nQuestion Answering Reasoning Math / Science CodingLLaMA 2 (7B) LLaMA 2 (13B) Mistral (7B) Gemma (7B)\nFigure 1|Language understanding and generation performance of Gemma 7B across different capa-\nbilities compared to similarly sized open models. We group together standard academic benchmark\nevaluations by capability and average the respective scores; see Table 6 for a detailed breakdown of\nperformance.\nbeen conducted, testing cannot cover all appli-\ncations and scenarios in which Gemma may be\nused. With this in mind, all Gemma users should\nconduct rigorous safety testing specific to their\nuse case before deployment or use. More details\non our approach to safety can be found in section\nResponsible Deployment.\nIn this technical report, we provide a detailed\noverview of the model architecture, training in-\nfrastructure, and pretraining and fine-tuning\nrecipes for Gemma, followed by thorough eval-\nuations of all checkpoints across a wide-variety\nof quantitative and qualitative benchmarks, as\nwell as both standard academic benchmarks and\nhuman-preference evaluations. We then discuss\nin detail our approach to safe and responsible de-\nployment. Finally, we outline the broader impli-\ncations of Gemma, its limitations and advantages,\nand conclusions.\nModel Architecture\nThe Gemma model architecture is based on the\ntransformer decoder (Vaswani et al., 2017). The\ncore parameters of the architecture are summa-\nrized in Table 1. Models are trained on a context\nlength of 8192 tokens.Parameters 2B 7B\nd_model 2048 3072\nLayers 18 28\nFeedforward hidden dims 32768 49152\nNum heads 8 16\nNum KV heads 1 16\nHead size 256 256\nVocab size 256128 256128\nTable 1|Key model parameters.\nWe also utilize several improvements proposed\nafter the original transformer paper. Below, we\nlist the included improvements:\nMulti-Query Attention (Shazeer, 2019). No-\ntably, the 7B model uses multi-head attention\nwhile the 2B checkpoints use multi-query atten-\ntion (with 𝑛𝑢𝑚_𝑘𝑣_ℎ𝑒𝑎𝑑𝑠 =1), based on ablation\nstudies that revealed respective attention variants\nimproved performance at each scale (Shazeer,\n2019).\nRoPEEmbeddings (Su et al., 2021). Rather than\nusing absolute positional embeddings, we use ro-\ntary positional embeddings in each layer; we also\nshare embeddings across our inputs and outputs\nto reduce model size.\n2', metadata={'source': './storage/Gemma-report.pdf', 'page': 1}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nModelEmbedding\nParametersNon-embedding\nParameters\n2B 524,550,144 1,981,884,416\n7B 786,825,216 7,751,248,896\nTable 2|Parameter counts for both sizes of\nGemma models.\nGeGLU Activations (Shazeer, 2020). The stan-\ndardReLUnon-linearityisreplacedbytheGeGLU\nactivation function.\nRMSNorm . We normalize the input of each trans-\nformer sub-layer, the attention layer and the feed-\nforward layer, with RMSNorm (Zhang and Sen-\nnrich, 2019) .\nTraining Infrastructure\nWe train the Gemma models using TPUv5e;\nTPUv5e are deployed in pods of 256 chips, con-\nfigured into a 2D torus of 16 x 16 chips. For the\n7B model, we train our model across 16 pods, to-\ntaling to 4096 TPUv5e. We pretrain the 2B model\nacross2pods, totaling512TPUv5e. Withinapod,\nwe use 16-way model sharding and 16-way data\nreplication for the 7B model. For the 2B, we sim-\nply use 256-way data replication. The optimizer\nstate is further sharded using techniques simi-\nlar to ZeRO-3. Beyond a pod, we perform data-\nreplica reduce over the data-center network, us-\ning Pathways approach of (Barham et al., 2022).\nAs in Gemini, we leverage the ’single controller’\nprogramming paradigm of Jax (Roberts et al.,\n2023) and Pathways (Barham et al., 2022) to\nsimplify the development process by enabling a\nsingle Python process to orchestrate the entire\ntraining run; we also leverage the GSPMD par-\ntitioner (Xu et al., 2021) for the training step\ncomputation and the MegaScale XLA compiler\n(XLA, 2019).\nCarbon Footprint\nWe estimate the carbon emissions from pretrain-\ning the Gemma models to be ∼131𝑡𝐶𝑂 2𝑒𝑞. This\nvalue is calculated based on the hourly energy us-\nage reported directly from our TPU datacenters;we also scale this value to account for the addi-\ntional energy expended to create and maintain\nthe data center, giving us the total energy usage\nfor our training experiments. We convert total\nenergy usage to carbon emissions by joining our\nhourly energy usage against hourly per-cell car-\nbon emission data reported by our data centers.\nIn addition, Google data centers are carbon\nneutral, achieved through a combination of en-\nergy efficiency, renewable energy purchases, and\ncarbon offsets. This carbon neutrality also applies\nto our experiments and the machines used to run\nthem.\nPretraining\nTraining Data\nGemma 2B and 7B are trained on 2T and 6T\ntokensrespectivelyofprimarily-Englishdatafrom\nweb documents, mathematics, and code. Unlike\nGemini, these models are not multimodal, nor are\nthey trained for state-of-the-art performance on\nmultilingual tasks.\nWe use a subset of the SentencePiece tokenizer\n(Kudo and Richardson, 2018) of Gemini for com-\npatibility. It splits digits, does not remove extra\nwhitespace, and relies on byte-level encodings for\nunknown tokens, following the techniques used\nfor both (Chowdhery et al., 2022) and (Gemini\nTeam, 2023). The vocabulary size is 256k tokens.\nFiltering\nWe filter the pre-training dataset to reduce the\nrisk of unwanted or unsafe utterances, and filter\nout certain personal information and other sen-\nsitive data. This includes using both heuristics\nand model-based classifiers to remove harmful or\nlow-quality content. Further, we filter all evalua-\ntion sets from our pre-training data mixture, run\ntargeted contamination analyses to check against\nevaluation set leakage, and reduce the risk of\nrecitation by minimizing proliferation of sensitive\noutputs.\nThefinaldatamixturewasdeterminedthrough\na series of ablations on both the 2B and 7B mod-\nels. Similartotheapproachadvocatedin(Gemini\n3', metadata={'source': './storage/Gemma-report.pdf', 'page': 2}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nTeam, 2023), we stage training to alter the cor-\npus mixture throughout training to increase the\nweight of relevant, high-quality data towards the\nend of training.\nInstruction Tuning\nWe finetune Gemma 2B and 7B with supervised\nfine-tuning (SFT) on a mix of text-only, English-\nonly synthetic and human-generated prompt-\nresponse pairs and reinforcement learning from\nhuman feedback (RLHF) with the reward model\ntrained on labelled English-only preference data\nand the policy based on a set of high-quality\nprompts. We find that both stages are important\nfor improved performance on downstream auto-\nmatic evaluations and human preference evalua-\ntions of model outputs.\nSupervised Fine-Tuning\nWe selected our data mixtures for supervised fine-\ntuning based on LM-based side-by-side evalua-\ntions (Zheng et al., 2023). Given a set of held-\nout prompts, we generate responses from a test\nmodel, generate responses on the same prompts\nfrom a baseline model, shuffle these randomly,\nand ask a larger, high capability model to express\na preference between two responses. Different\nprompt sets are constructed to highlight specific\ncapabilities, such as instruction following, factual-\nity, creativity, and safety. The different automatic\nLM-basedjudgesweuseemployanumberoftech-\nniques, such as chain-of-thought prompting (Wei\net al., 2022) and use of rubrics and constitutions\n(Bai et al., 2022), to be aligned with human pref-\nerences.\nFiltering\nWhen using synthetic data, we run several stages\nof filtering over it, removing examples that show\ncertain personal information, unsafe or toxic\nmodel outputs, mistaken self-identification data,\nor duplicated examples. Following Gemini, we\nfind that including subsets of data that encour-\nage better in-context attribution, hedging, and\nrefusals to minimize hallucinations can improve\nperformance on several factuality metrics, with-out degrading model performance on other met-\nrics.\nThe final data mixtures and supervised fine-\ntuning recipe, which includes tuned hyperparam-\neters, were chosen on the basis of improving help-\nfulness while minimizing model harms related to\nsafety and hallucinations.\nFormatting\nInstruction tuned models are trained with a spe-\ncific formatter that annotates all instruction tun-\ning examples with extra information, both at\ntraining and inference time. It has two purposes:\n1) indicating roles in a conversation, such as the\nUser role, and 2) delineating turns in a conver-\nsation, especially in a multi-turn conversation.\nSpecial control tokens are reserved in the tok-\nenizer for this purpose. While it is possible to\nget coherent generations without the formatter,\nit will be out-of-distribution for the model, and\nwill very likely produce worse generations.\nThe relevant formatting control tokens are pre-\nsented in Table 3, with a dialogue example pre-\nsented in Table 4.\nContext Relevant Token\nUser turn user\nModel turn model\nStart of conversation turn <start_of_turn>\nEnd of conversation turn <end_of_turn>\nTable 3|Relevant formatting control tokens used\nfor both SFT and RLHF of Gemma models.\nUser: <start_of_turn>user\nKnock knock.<end_of_turn>\n<start_of_turn>model\nModel: Who’s there?<end_of_turn>\nUser: <start_of_turn>user\nGemma.<end_of_turn>\n<start_of_turn>model\nModel: Gemma who?<end_of_turn>\nTable 4|Example dialogue with user and model\ncontrol tokens.\n4', metadata={'source': './storage/Gemma-report.pdf', 'page': 3}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nReinforcement Learning from Human Feed-\nback\nWe further finetuned the supervised fine-tuned\nmodel using RLHF (Christiano et al., 2017;\nOuyang et al., 2022). We collected pairs of pref-\nerences from human raters and trained a reward\nfunction under the Bradley-Terry model (Bradley\nand Terry, 1952), similarly to Gemini. The pol-\nicy was trained to optimize this reward function\nusing a variant of REINFORCE (Williams, 1992)\nwith a Kullback–Leibler regularization term to-\nwards the initially tuned model. Similar to the\nSFT phase, and in order to tune hyperparame-\nters and additionally mitigate reward hacking\n(Amodeietal.,2016;Skalseetal.,2022)werelied\non a high capacity model as an automatic rater\nand computed side-by-side comparisons against\nbaseline models.\nEvaluation\nWe evaluate Gemma across a broad range of do-\nmains, using both automated benchmarks and\nhuman evaluation.\nHuman Preference Evaluations\nIn addition to running standard academic bench-\nmarks on the finetuned models, we sent final re-\nlease candidates to human evaluation studies to\nbe compared against the Mistral v0.2 7B Instruct\nmodel (Jiang et al., 2023).\nOn a held-out collection of around 1000\nprompts oriented toward asking models to follow\ninstructions across creative writing tasks, coding,\nand following instructions, Gemma 7B IT has a\n51.7% positive win rate and Gemma 2B IT has\na 41.6% win rate over Mistral v0.2 7B Instruct.\nOn a held-out collection of around 400 prompts\noriented towards testing basic safety protocols,\nGemma 7B IT has a 58% win rate, while Gemma\n2B IT has a 56.5% win rate. We report the corre-\nsponding numbers in Table 5.\nAutomated Benchmarks\nWe measure Gemma models’ performance on do-\nmains including physical reasoning (Bisk et al.,Model Safety Instruction Following\nGemma 7B IT 58% 51.7%\n95% Conf. Interval [55.9%, 60.1%] [49.6%, 53.8%]\nWin / Tie / Loss 42.9% / 30.2% / 26.9% 42.5% / 18.4% / 39.1%\nGemma 2B IT 56.5% 41.6%\n95% Conf. Interval [54.4%, 58.6%] [39.5%, 43.7%]\nWin / Tie / Loss 44.8% / 22.9% / 32.3% 32.7% / 17.8% / 49.5%\nTable 5|Win rate of Gemma models versus Mis-\ntral 7B v0.2 Instruct with 95% confidence inter-\nvals. We report breakdowns of wins, ties, and\nlosses, and we break ties evenly when reporting\nthe final win rate.\n2019), social reasoning (Sap et al., 2019), ques-\ntion answering (Clark et al., 2019; Kwiatkowski\net al., 2019), coding (Austin et al., 2021; Chen\net al., 2021), mathematics (Cobbe et al., 2021),\ncommonsense reasoning (Sakaguchi et al., 2019),\nlanguage modeling (Paperno et al., 2016), read-\ning comprehension (Joshi et al., 2017), and more.\nFor most automated benchmarks we use the\nsame evaluation methodology as in Gemini.\nSpecifically for those where we report perfor-\nmance compared with Mistral, we replicated\nmethodology from the Mistral technical report\nas closely as possible. These specific benchmarks\nare: ARC (Clark et al., 2018), CommonsenseQA\n(Talmor et al., 2019), Big Bench Hard (Suzgun\net al., 2022), and AGI Eval (English-only) (Zhong\netal.,2023). Duetorestrictivelicensing, wewere\nunable to run any evaluations on LLaMA-2 and\ncite only those metrics previously reported (Tou-\nvron et al., 2023b).\nWe compare Gemma 2B and 7B models to sev-\neral external open-source (OSS) LLMs across a\nseries of academic benchmarks, reported in Table\n6 and Table 7.\nOn MMLU (Hendrycks et al., 2020), Gemma\n7B outperforms all OSS alternatives at the same\norsmallerscale; italsooutperformsseverallarger\nmodels, including LLaMA2 13B. However, human\nexpert performance is gauged at 89.8% by the\nbenchmark authors; as Gemini Ultra is the first\nmodel to exceed this threshold, there is signifi-\ncantroomforcontinuedimprovementstoachieve\nGemini and human-level performance.\n5', metadata={'source': './storage/Gemma-report.pdf', 'page': 4}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nLLaMA-2 Mistral Gemma\nBenchmark metric 7B 13B 7B 2B 7B\nMMLU 5-shot, top-1 45.3 54.8 62.5 42.3 64.3\nHellaSwag 0-shot 77.2 80.7 81.0 71.4 81.2\nPIQA 0-shot 78.8 80.5 82.2 77.3 81.2\nSIQA 0-shot 48.3 50.3 47.0∗49.751.8\nBoolq 0-shot 77.4 81.7 83.2∗69.483.2\nWinogrande partial scoring 69.2 72.8 74.2 65.4 72.3\nCQA 7-shot 57.8 67.3 66.3∗65.371.3\nOBQA 58.657.0 52.2 47.8 52.8\nARC-e 75.2 77.3 80.5 73.2 81.5\nARC-c 45.9 49.4 54.9 42.1 53.2\nTriviaQA 5-shot 72.1 79.6 62.5 53.2 63.4\nNQ 5-shot 25.7 31.2 23.2 12.5 23.0\nHumanEval pass@1 12.8 18.3 26.2 22.0 32.3\nMBPP†3-shot 20.8 30.6 40.2∗29.244.4\nGSM8K maj@1 14.6 28.7 35.4∗17.746.4\nMATH 4-shot 2.5 3.9 12.7 11.8 24.3\nAGIEval 29.3 39.1 41.2∗24.241.7\nBBH 32.6 39.4 56.1∗35.2 55.1\nAverage 47.0 52.2 54.0 44.9 56.4\nTable 6|Academic benchmark results, compared to similarly sized, openly-available models trained\non general English text data.†Mistral reports 50.2 on a different split for MBPP and on their split\nour 7B model achieves 54.5.∗evaluations run by us. Note that due to restrictive licensing, we were\nunable to run evals on LLaMA-2; all values above were previously reported in Touvron et al. (2023b).\nGemma models demonstrate particularly\nstrong performance on mathematics and coding\nbenchmarks. On mathematics tasks, which\nare often used to benchmark the general ana-\nlytical capabilities of models, Gemma models\noutperform other models by at least 10 points\non GSM8K (Cobbe et al., 2021) and the more\ndifficult MATH (Hendrycks et al., 2021) bench-\nmark. Similarly, they outperform alternate open\nmodels by at least 6 points on HumanEval (Chen\net al., 2021). They even surpass the performance\nof the code-fine-tuned CodeLLaMA-7B models\non MBPP (CodeLLaMA achieves a score of 41.4%\nwhere Gemma 7B achieves 44.4%).\nMemorization Evaluations\nRecent work has shown that aligned models may\nbe vulnerable to new adversarial attacks that canbypass alignment (Nasr et al., 2023). These at-\ntackscancausemodelstodiverge,andsometimes\nregurgitate memorized training data in the pro-\ncess. We focus on discoverable memorization,\nwhich serves as a reasonable upper-bound on the\nmemorization of a model (Nasr et al., 2023) and\nhas been the common definition used in several\nstudies (Anil et al., 2023; Carlini et al., 2022;\nKudugunta et al., 2023).\nWe test for memorization1of the Gemma pre-\ntrained models with the same methodology per-\nformed in Anil et al. (2023). We sample 10,000\ndocuments from each corpus and use the first\n50 tokens as a prompt for the model. We focus\nmainly on exact memorization, where we classify\ntexts as memorized if the subsequent 50 tokens\n1Our use of “memorization” relies on the definition of\nthat term found at www.genlaw.org/glossary.html.\n6', metadata={'source': './storage/Gemma-report.pdf', 'page': 5}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nMistral Gemma\nBenchmark 7B 7B\nARC-c 60.0 61.9\nHellaSwag 83.3 82.2\nMMLU 64.2 64.6\nTruthfulQA 42.2 44.8\nWinogrande 78.4 79.0\nGSM8K 37.8 50.9\nAverage 61.0 63.8\nTable 7|HuggingFace H6 benchmark. The per-\nformance of small models are sensitive to small\nmodifications in prompts and we further validate\nthe quality of our models on an independent im-\nplementation of multiple known benchmarks. All\nevaluations were run by HuggingFace.\nGemma\n 2B Gemma\n 7B PaLM 2\nSmall\nModel0.11% Exact MemorizedMemorization of\nEnglish Web Content\nGemma\n2BGemma\n7BPaLM\nSmall\nModel0.11% Exact MemorizedMemorization of\nAll Content\nFigure2|Comparingaveragememorizationrates\nacross model families. We compare the Gemma\npretrained models to PaLM and PaLM 2 models\nof comparable size and find similarly low rates of\nmemorization.\ngeneratedbythemodelexactlymatchtheground\ntruth continuation in the text. However, to bet-\nter capture potential paraphrased memorizations,\nwe include approximate memorization (Ippolito\net al., 2022) using an 10% edit distance thresh-\nold. In Figure 2, we compare the results of our\nevaluation with the closest sized PaLM (Chowdh-\nery et al., 2022) and PaLM 2 models (Anil et al.,\n2023).\nVerbatim Memorization PaLM 2 compared\nwith PaLM by evaluating on a shared subset of\ntheir training corpora. However, there is even\nlessoverlapbetweentheGemmapretrainingdatawith the PaLM models, and so using this same\nmethodology, we observe much lower memoriza-\ntion rates (Figure 2 left). Instead, we find that\nestimating the “total memorization” across the\nentire pretraining dataset gives a more reliable\nestimate (Figure 2 right) where we now find the\nGemma memorizes training data at a comparable\nrate to PaLM.\nCode WikiScienceWeb\nMultilingual\nData Source0.11% Exact Memorized2B Model\nCode WikiScienceWeb\nMultilingual\nData Source0.11% Exact Memorized7B Model\nPersonal Data\nYes No\nFigure 3|Measuring personal and sensitive data\nmemorizationrates. Nosensitivedatawasmem-\norized, hence it is omitted from the figure .\nPersonal Data Perhaps of higher importance is\nthe possibility that personal data might be mem-\norized. As part of making Gemma pre-trained\nmodelssafeandreliable,weusedautomatedtech-\nniques to filter out certain personal information\nand other sensitive data from training sets.\nTo identify possible occurrences of personal\ndata, we use Google Cloud Sensitive Data Protec-\ntion2. Thistooloutputsthreeseveritylevelsbased\non many categories of personal data (e.g., names,\nemails, etc.). We classify the highest severity\nas “sensitive” and the remaining two as simply\n“personal”. Then, we measure how many mem-\norized outputs contain any sensitive or personal\ndata. As shown in Figure 3, we observe no cases\nof memorized sensitive data. We do find that the\nmodel memorizes some data we have classified\nas potentially “personal” according to the above,\nthough often at a much lower rate. Further, it\nis important to note that these tools are known\nto have many false positives (because they only\nmatch patterns and do not consider the context),\n2Available at: https://cloud.google.com/\nsensitive-data-protection\n7', metadata={'source': './storage/Gemma-report.pdf', 'page': 6}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nMistral Gemma\nBenchmark metric 7B* 2B 7B\nRealToxicity avg 8.44 6.86 7.90\nBOLD 38.21 45.57 49.08\nCrowS-Pairs top-1 32.76 45.82 51.33\nBBQ Ambig 1-shot, top-1 97.53 62.58 92.54\nBBQ Disambig top-1 84.45 54.62 71.99\nWinogender top-1 64.3 51.25 54.17\nTruthfulQA 44.2 44.84 31.81\nWinobias 1_2 65.72 56.12 59.09\nWinobias 2_2 84.53 91.1 92.23\nToxigen 60.26 29.77 39.59\nTable 8|Safety academic benchmark results, compared to similarly sized, openly-available models.\n∗evaluations run by us. Note that due to restrictive licensing, we were unable to run evals on LLaMA-\n2; we do not report previously-published LLaMA-2 numbers for TruthfulQA, as we use different,\nnon-comparable evaluation set-ups (we use MC2, where LLaMA-2 uses GPT-Judge).\nmeaning that our results are likely overestimates\nof the amount of personal data identified.\nCode WikiScienceWeb\nMultilingual\nData Source0.1110% Memorized2B Model\nCode WikiScienceWeb\nMultilingual\nData Source0.1110% Memorized7B Model\nMemorization Type\nExact Approximate\nFigure 4|Comparing exact and approximate\nmemorization.\nApproximate Memorization In Figure 4, we\nobserve that roughly 50% more data is approxi-\nmately memorized (note the log scale) and that\nthis is nearly consistent across each of the differ-\nent subcategories over the dataset.\nResponsible Deployment\nIn line with previous releases of Google’s AI tech-\nnologies(GeminiTeam,2023;Kavukcuogluetal.,\n2022), wefollowastructuredapproachtorespon-\nsible development and deployment of our models,in order to identify, measure, and manage fore-\nseeable downstream societal impacts. As with\nour recent Gemini release, these are informed\nby prior academic literature on language model\nrisks (Weidinger et al., 2021), findings from simi-\nlar prior exercises conducted across the industry\n(Anil et al., 2023), ongoing engagement with ex-\nperts internally and externally, and unstructured\nattempts to discover new model vulnerabilities.\nBenefits\nWe believe that openness in AI science and tech-\nnology can bring significant benefits. Open-\nsourcing is a significant driver of science and\ninnovation, and a responsible practice in most\ncircumstances. But this needs to be balanced\nagainst the risk of providing actors with the tools\nto cause harm now or in the future.\nGoogle has long committed to providing\nbroader access to successful research innovations\n(GraphCast, Transformer, BERT, T5, Word2Vec),\nand we believe that releasing Gemma into the AI\ndevelopment ecosystem will enable downstream\ndevelopers to create a host of beneficial appli-\ncations, in areas such as science, education and\nthe arts. Our instruction-tuned offerings should\nencourage a range of developers to leverage\nGemma’s chat and code capabilities to support\n8', metadata={'source': './storage/Gemma-report.pdf', 'page': 7}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\ntheir own beneficial applications, while allowing\nforcustomfine-tuningtospecializethemodel’sca-\npabilities for specific use cases. To ensure Gemma\nsupports a wide range of developer needs, we are\nalso releasing two model sizes to optimally sup-\nportdifferentenvironments,andhavemadethese\nmodels available across a number of platforms\n(seeKagglefordetails). Providingbroadaccessto\nGemma in this way should reduce the economic\nand technical barriers that newer ventures or in-\ndependent developers face when incorporating\nthese technologies into their workstreams.\nAs well as serving developers with our\ninstruction-tuned models, we have also provided\naccess to corresponding base pretrained mod-\nels. By doing so, it is our intention to encourage\nfurther AI safety research and community inno-\nvation, providing a wider pool of models avail-\nable to developers to build on various methods of\ntransparency and interpretability research that\nthe community has already benefited from (Pac-\nchiardi et al., 2023; Zou et al., 2023).\nRisks\nIn addition to bringing benefits to the AI devel-\nopment ecosystem, we are aware that malicious\nuses of LLMs, such as the creation of deepfake\nimagery, AI-generated disinformation, and illegal\nand disturbing material can cause harm on both\nan individual and institutional levels (Weidinger\netal.,2021). Moreover,providingaccesstomodel\nweights, rather than releasing models behind an\nAPI, raises new challenges for responsible deploy-\nment.\nFirst, we cannot prevent bad actors from fine\ntuning Gemma for malicious intent, despite their\nuse being subject to Terms of Use that prohibit\nthe use of Gemma models in ways that contra-\nvene our Gemma Prohibited Use Policy. However,\nwe are cognizant that further work is required to\nbuild more robust mitigation strategies against\nintentionalmisuseofopensystems, whichGoogle\nDeepMindwillcontinuetoexplorebothinternally\nand in collaboration with the wider AI commu-\nnity.\nThe second challenge we face is protecting de-\nvelopers and downstream users against the un-intended behaviours of open models, including\ngeneration of toxic language or perpetuation of\ndiscriminatorysocialharms,modelhallucinations\nandleakageofpersonallyidentifiableinformation.\nWhendeployingmodelsbehindanAPI,theserisks\ncan be reduced via various filtering methods.\nMitigations\nWithout this layer of defense for the Gemma fam-\nily of models, we have endeavoured to safeguard\nagainst these risks by filtering and measuring bi-\nases in pre-training data in line with the Gemini\napproach, assessing safety through standardized\nAI safety benchmarks, internal red teaming to\nbetter understand the risks associated with exter-\nnal use of Gemma, and subjecting the models to\nrigorous ethics and safety evaluations, the results\nof which can be seen in 8.\nWhile we’ve invested significantly in improving\nthe model, we recognize its limitations. To en-\nsure transparency for downstream users, we’ve\npublished a detailed model card to provide re-\nsearchers with a more comprehensive under-\nstanding of Gemma.\nWe have also released a Generative AI Respon-\nsible Toolkit to support developers to build AI\nresponsibly. This encompasses a series of assets\nto help developers design and implement respon-\nsible AI best practices and keep their own users\nsafe.\nThe relative novelty of releasing open weights\nmodels means new uses, and misuses, of these\nmodels are still being discovered, which is why\nGoogle DeepMind is committed to the continuous\nresearch and development of robust mitigation\nstrategies alongside future model development.\nAssessment\nUltimately,giventhecapabilitiesoflargersystems\naccessible within the existing ecosystem, we be-\nlieve the release of Gemma will have a negligible\neffect on the overall AI risk portfolio. In light\nof this, and given the utility of these models for\nresearch, auditing and downstream product de-\nvelopment, we are confident that the benefit of\nGemma to the AI community outweighs the risks\ndescribed.\n9', metadata={'source': './storage/Gemma-report.pdf', 'page': 8}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nGoing Forward\nAs a guiding principle, Google DeepMind strives\nto adopt assessments and safety mitigations pro-\nportionate to the potential risks from our mod-\nels. In this case, although we are confident that\nGemma models will provide a net benefit to the\ncommunity, our emphasis on safety stems from\nthe irreversible nature of this release. As the\nharms resulting from open models are not yet\nwell defined, nor does an established evaluation\nframeworkforsuchmodelsexist,wewillcontinue\nto follow this precedent and take a measured and\ncautionary approach to open model development.\nAs capabilities advance, we may need to explore\nextended testing, staggered releases or alterna-\ntive access mechanisms to ensure responsible AI\ndevelopment.\nAs the ecosystem evolves, we urge the wider\nAI community to move beyond simplistic ’open\nvs. closed’ debates, and avoid either exaggerat-\ning or minimising potential harms, as we believe\na nuanced, collaborative approach to risks and\nbenefits is essential. At Google DeepMind we’re\ncommittedtodevelopinghigh-qualityevaluations\nand invite the community to join us in this effort\nfor a deeper understanding of AI systems.\nDiscussion and Conclusion\nWe present Gemma, an openly available family\nof generative language models for text and code.\nGemma advances the state of the art of openly\navailable language model performance, safety,\nand responsible development.\nIn particular, we are confident that Gemma\nmodels will provide a net benefit to the commu-\nnity given our extensive safety evaluations and\nmitigations; however, we acknowledge that this\nrelease is irreversible and the harms resulting\nfrom open models are not yet well defined, so we\ncontinue to adopt assessments and safety mitiga-\ntions proportionate to the potential risks of these\nmodels. In addition, our models outperform com-\npetitors on 6 standard safety benchmarks, and in\nhuman side-by-side evaluations.\nGemma models improve performance on a\nbroad range of domains including dialogue, rea-soning, mathematics, and code generation. Re-\nsults on MMLU (64.3%) and MBPP (44.4%)\ndemonstrate both the high performance of\nGemma, as well as the continued headroom in\nopenly available LLM performance.\nBeyond state-of-the-art performance measures\non benchmark tasks, we are excited to see what\nnew use-cases arise from the community, and\nwhat new capabilities emerge as we advance the\nfield together. We hope that researchers use\nGemma to accelerate a broad array of research,\nand we hope that developers create beneficial\nnew applications, user experiences, and other\nfunctionality.\nGemma benefits from many learnings of the\nGemini model program including code, data,\narchitecture, instruction tuning, reinforcement\nlearning from human feedback, and evaluations.\nAs discussed in the Gemini technical report, we\nreiterate a non-exhaustive set of limitations to\nthe use of LLMs. Even with great performance on\nbenchmark tasks, further research is needed to\ncreate robust, safe models that reliably perform\nas intended. Example further research areas in-\nclude factuality, alignment, complex reasoning,\nand robustness to adversarial input. As discussed\nbyGemini, wenotetheneedformorechallenging\nand robust benchmarks.\n10', metadata={'source': './storage/Gemma-report.pdf', 'page': 9}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nContributions and Acknowledgments\nCore Contributors\nThomas Mesnard\nCassidy Hardin\nRobert Dadashi\nSurya Bhupatiraju\nShreya Pathak\nLaurent Sifre\nMorgane Rivière\nMihir Sanjay Kale\nJuliette Love\nPouya Tafti\nLéonard Hussenot\nContributors\nAakanksha Chowdhery\nAdam Roberts\nAditya Barua\nAlex Botev\nAlex Castro-Ros\nAmbrose Slone\nAmélie Héliou\nAndrea Tacchetti\nAnna Bulanova\nAntonia Paterson\nBeth Tsai\nBobak Shahriari\nCharline Le Lan\nChristopher A. Choquette-Choo\nClément Crepy\nDaniel Cer\nDaphne Ippolito\nDavid Reid\nElena Buchatskaya\nEric Ni\nEric Noland\nGeng Yan\nGeorge Tucker\nGeorge-Christian Muraru\nGrigory Rozhdestvenskiy\nHenryk Michalewski\nIan Tenney\nIvan Grishchenko\nJacob Austin\nJames Keeling\nJane Labanowski\nJean-Baptiste Lespiau\nJeff Stanway\nJenny BrennanJeremy Chen\nJohan Ferret\nJustin Chiu\nJustin Mao-Jones\nKatherine Lee\nKathy Yu\nKatie Millican\nLars Lowe Sjoesund\nLisa Lee\nLucas Dixon\nMachel Reid\nMaciej Mikuła\nMateo Wirth\nMichael Sharman\nNikolai Chinaev\nNithum Thain\nOlivier Bachem\nOscar Chang\nOscar Wahltinez\nPaige Bailey\nPaul Michel\nPetko Yotov\nPier Giuseppe Sessa\nRahma Chaabouni\nRamona Comanescu\nReena Jana\nRohan Anil\nRoss McIlroy\nRuibo Liu\nRyan Mullins\nSamuel L Smith\nSebastian Borgeaud\nSertan Girgin\nSholto Douglas\nShree Pandya\nSiamak Shakeri\nSoham De\nTed Klimenko\nTom Hennigan\nVlad Feinberg\nWojciech Stokowiec\nYu-hui Chen\nZafarali Ahmed\nZhitao Gong\n11', metadata={'source': './storage/Gemma-report.pdf', 'page': 10}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nProduct Management\nTris Warkentin\nLudovic Peran\nProgram Management\nMinh Giang\nExecutive Sponsors\nClément Farabet\nOriol Vinyals\nJeff Dean\nKoray Kavukcuoglu\nDemis Hassabis\nZoubin Ghahramani\nDouglas Eck\nJoelle Barral\nFernando Pereira\nEli Collins\nLeads\nArmand Joulin\nNoah Fiedel\nEvan Senter\nTech Leads\nAlek Andreev†\nKathleen Kenealy†\nAcknowledgements\nOur work is made possible by the dedication and\nefforts of numerous teams at Google. We would\nlike to acknowledge the support from the follow-\ning teams: Gemini, Gemini Safety, Gemini In-\nfrastructure, Gemini Evaluation, Google Cloud,\nGoogle Research Responsible AI, Kaggle, and\nKeras.\nSpecial thanks and acknowledgment to Adrian\nHutter, Andreas Terzis, Andrei Kulik, Angelos Fi-\nlos, Anushan Fernando, Aurelien Boffy, Danila\nSinopalnikov, Edouard Leurent, Gabriela Surita,\nGeoffrey Cideron, Jilin Chen, Karthik Raveen-\ndran, Kathy Meier-Hellstern, Kehang Han, Kevin\nRobinson, KritikaMuralidharan, LeHou, Leonard\nBerrada, Lev Proleev, Luheng He, Marie Pel-\nlat, Mark Sherwood, Matt Hoffman, Matthias\nGrundmann, Nicola De Cao, Nikola Momchev,\nNino Vieillard, Noah Constant, Peter Liu, Piotr\nStanczyk, Qiao Zhang, Ruba Haroun, Seliem El-\nSayed, Siddhartha Brahma, Tianhe (Kevin) Yu,\nTom Le Paine, Yingjie Miao, Yuanzhong Xu, and\nYuting Sun.\n†equal contribution.References\nE. Almazrouei, H. Alobeidli, A. Alshamsi, A. Cap-\npelli, R. Cojocaru, M. Debbah, Étienne Goffinet,\nD. Hesslow, J. Launay, Q. Malartic, D. Mazzotta,\nB. Noune, B. Pannier, and G. Penedo. The fal-\ncon series of open language models, 2023.\nD. Amodei, C. Olah, J. Steinhardt, P. Christiano,\nJ. Schulman, and D. Mané. Concrete problems\nin AI safety. arXiv preprint , 2016.\nR. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lep-\nikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey,\nZ. Chen, et al. Palm 2 technical report. arXiv\npreprint arXiv:2305.10403 , 2023.\nJ. Austin, A. Odena, M. I. Nye, M. Bosma,\nH. Michalewski, D. Dohan, E. Jiang, C. J.\nCai, M. Terry, Q. V. Le, and C. Sutton. Pro-\ngram synthesis with large language models.\nCoRR, abs/2108.07732, 2021. URL https:\n//arxiv.org/abs/2108.07732 .\nY.Bai,S.Kadavath,S.Kundu,A.Askell,J.Kernion,\nA. Jones, A. Chen, A. Goldie, A. Mirhoseini,\nC. McKinnon, C. Chen, C. Olsson, C. Olah,\nD. Hernandez, D. Drain, D. Ganguli, D. Li,\nE. Tran-Johnson, E. Perez, J. Kerr, J. Mueller,\nJ. Ladish, J. Landau, K. Ndousse, K. Lukosuite,\nL. Lovitt, M. Sellitto, N. Elhage, N. Schiefer,\nN. Mercado, N. DasSarma, R. Lasenby, R. Lar-\nson, S. Ringer, S. Johnston, S. Kravec, S. E.\nShowk, S. Fort, T. Lanham, T. Telleen-Lawton,\nT. Conerly, T. Henighan, T. Hume, S. R. Bow-\nman, Z. Hatfield-Dodds, B. Mann, D. Amodei,\nN. Joseph, S. McCandlish, T. Brown, and J. Ka-\nplan. Constitutional ai: Harmlessness from ai\nfeedback, 2022.\nP. Barham, A. Chowdhery, J. Dean, S. Ghemawat,\nS. Hand, D. Hurt, M. Isard, H. Lim, R. Pang,\nS. Roy, B. Saeta, P. Schuh, R. Sepassi, L. E.\nShafey, C. A. Thekkath, and Y. Wu. Path-\nways: Asynchronous distributed dataflow for\nml, 2022.\nY. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi.\nPIQA: reasoning about physical commonsense\nin natural language. CoRR, abs/1911.11641,\n2019. URL http://arxiv.org/abs/1911.\n11641.\n12', metadata={'source': './storage/Gemma-report.pdf', 'page': 11}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nR. A. Bradley and M. E. Terry. Rank analysis\nof incomplete block designs: I. the method of\npaired comparisons. Biometrika , 39, 1952.\nN. Carlini, D. Ippolito, M. Jagielski, K. Lee,\nF. Tramer, and C. Zhang. Quantifying memo-\nrization across neural language models. arXiv\npreprint arXiv:2202.07646 , 2022.\nM. Chen, J. Tworek, H. Jun, Q. Yuan, H. P.\nde Oliveira Pinto, J. Kaplan, H. Edwards,\nY. Burda, N. Joseph, G. Brockman, A. Ray,\nR. Puri, G. Krueger, M. Petrov, H. Khlaaf,\nG. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ry-\nder, M. Pavlov, A. Power, L. Kaiser, M. Bavar-\nian, C. Winter, P. Tillet, F. P. Such, D. Cum-\nmings, M. Plappert, F. Chantzis, E. Barnes,\nA.Herbert-Voss,W.H.Guss,A.Nichol,A.Paino,\nN. Tezak, J. Tang, I. Babuschkin, S. Balaji,\nS. Jain, W. Saunders, C. Hesse, A. N. Carr,\nJ. Leike, J. Achiam, V. Misra, E. Morikawa,\nA.Radford,M.Knight,M.Brundage,M.Murati,\nK. Mayer, P. Welinder, B. McGrew, D. Amodei,\nS. McCandlish, I. Sutskever, and W. Zaremba.\nEvaluating large language models trained on\ncode. CoRR, abs/2107.03374, 2021. URL\nhttps://arxiv.org/abs/2107.03374 .\nA. Chowdhery, S. Narang, J. Devlin, M. Bosma,\nG. Mishra, A. Roberts, P. Barham, H. W.\nChung, C. Sutton, S. Gehrmann, P. Schuh,\nK. Shi, S. Tsvyashchenko, J. Maynez, A. Rao,\nP. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran,\nE. Reif, N. Du, B. Hutchinson, R. Pope, J. Brad-\nbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin,\nT. Duke, A. Levskaya, S. Ghemawat, S. Dev,\nH. Michalewski, X. Garcia, V. Misra, K. Robin-\nson, L. Fedus, D. Zhou, D. Ippolito, D. Luan,\nH. Lim, B. Zoph, A. Spiridonov, R. Sepassi,\nD. Dohan, S. Agrawal, M. Omernick, A. M. Dai,\nT.S.Pillai,M.Pellat,A.Lewkowycz,E.Moreira,\nR. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang,\nB. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei,\nK. Meier-Hellstern, D. Eck, J. Dean, S. Petrov,\nand N. Fiedel. Palm: Scaling language model-\ning with pathways, 2022.\nP. F. Christiano, J. Leike, T. Brown, M. Martic,\nS. Legg, and D. Amodei. Deep reinforcement\nlearning from human preferences. Advancesin Neural Information Processing Systems , 30,\n2017.\nC. Clark, K. Lee, M. Chang, T. Kwiatkowski,\nM. Collins, and K. Toutanova. Boolq: Explor-\ning the surprising difficulty of natural yes/no\nquestions. CoRR, abs/1905.10044, 2019. URL\nhttp://arxiv.org/abs/1905.10044 .\nP. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabhar-\nwal, C. Schoenick, and O. Tafjord. Think you\nhave solved question answering? try arc, the\nai2 reasoning challenge, 2018.\nK. Cobbe, V. Kosaraju, M. Bavarian, M. Chen,\nH. Jun, L. Kaiser, M. Plappert, J. Tworek,\nJ. Hilton, R. Nakano, C. Hesse, and J. Schul-\nman. Training verifiers to solve math word\nproblems. CoRR, abs/2110.14168, 2021. URL\nhttps://arxiv.org/abs/2110.14168 .\nJ.Dean,G.Corrado,R.Monga,K.Chen,M.Devin,\nM. Mao, M. a. Ranzato, A. Senior, P. Tucker,\nK. Yang, Q. Le, and A. Ng. Large scale dis-\ntributeddeepnetworks. InF.Pereira,C.Burges,\nL. Bottou, and K. Weinberger, editors, Advances\nin Neural Information Processing Systems ,\nvolume 25. Curran Associates, Inc., 2012.\nURL https://proceedings.neurips.\ncc/paper_files/paper/2012/file/\n6aca97005c68f1206823815f66102863-Paper.\npdf.\nJ. Devlin, M. Chang, K. Lee, and K. Toutanova.\nBERT: pre-training of deep bidirectional trans-\nformers for language understanding. CoRR,\nabs/1810.04805, 2018. URL http://arxiv.\norg/abs/1810.04805 .\nGemini Team. Gemini: A family of highly capable\nmultimodal models, 2023.\nD. Hendrycks, C. Burns, S. Basart, A. Zou,\nM. Mazeika, D. Song, and J. Steinhardt. Mea-\nsuring massive multitask language understand-\ning. CoRR, abs/2009.03300, 2020. URL\nhttps://arxiv.org/abs/2009.03300 .\nD. Hendrycks, C. Burns, S. Kadavath, A. Arora,\nS. Basart, E. Tang, D. Song, and J. Steinhardt.\nMeasuring mathematical problem solving with\nthe math dataset. NeurIPS , 2021.\n13', metadata={'source': './storage/Gemma-report.pdf', 'page': 12}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nD. Ippolito, F. Tramèr, M. Nasr, C. Zhang,\nM. Jagielski, K. Lee, C. A. Choquette-Choo, and\nN. Carlini. Preventing verbatim memorization\nin language models gives a false sense of pri-\nvacy. arXiv preprint arXiv:2210.17546 , 2022.\nA. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bam-\nford, D. S. Chaplot, D. de las Casas, F. Bressand,\nG.Lengyel,G.Lample,L.Saulnier,L.R.Lavaud,\nM.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril,\nT. Wang, T. Lacroix, and W. E. Sayed. Mistral\n7b, 2023.\nM. Joshi, E. Choi, D. S. Weld, and L. Zettle-\nmoyer. Triviaqa: A large scale distantly su-\npervised challenge dataset for reading compre-\nhension. CoRR, abs/1705.03551, 2017. URL\nhttp://arxiv.org/abs/1705.03551 .\nK. Kavukcuoglu, P. Kohli, L. Ibrahim, D. Bloxwich,\nandS.Brown. Howourprincipleshelpeddefine\nalphafold’s release, 2022.\nT. Kudo and J. Richardson. SentencePiece: A\nsimple and language independent subword to-\nkenizeranddetokenizerforneuraltextprocess-\ning. InE.BlancoandW.Lu,editors, Proceedings\nof the 2018 Conference on Empirical Methods in\nNatural Language Processing: System Demon-\nstrations , pages 66–71, Brussels, Belgium, Nov.\n2018. Association for Computational Linguis-\ntics. doi: 10.18653/v1/D18-2012. URL\nhttps://aclanthology.org/D18-2012 .\nS. Kudugunta, I. Caswell, B. Zhang, X. Garcia,\nC. A. Choquette-Choo, K. Lee, D. Xin, A. Kusu-\npati, R. Stella, A. Bapna, et al. Madlad-400:\nA multilingual and document-level large au-\ndited dataset. arXiv preprint arXiv:2309.04662 ,\n2023.\nT. Kwiatkowski, J. Palomaki, O. Redfield,\nM. Collins, A. Parikh, C. Alberti, D. Epstein,\nI. Polosukhin, J. Devlin, K. Lee, K. Toutanova,\nL. Jones, M. Kelcey, M.-W. Chang, A. M. Dai,\nJ. Uszkoreit, Q. Le, and S. Petrov. Natural ques-\ntions: A benchmark for question answering\nresearch. Transactions of the Association for\nComputational Linguistics , 7:452–466, 2019.\ndoi: 10.1162/tacl_a_00276. URL https://\naclanthology.org/Q19-1026 .Y. LeCun, Y. Bengio, and G. Hinton. Deep learn-\ning. nature, 521(7553):436–444, 2015.\nT. Mikolov, K. Chen, G. Corrado, and J. Dean. Ef-\nficient estimation of word representations in\nvector space. In Y. Bengio and Y. LeCun, edi-\ntors, 1st International Conference on Learning\nRepresentations, ICLR 2013, Scottsdale, Arizona,\nUSA, May 2-4, 2013, Workshop Track Proceed-\nings, 2013. URL http://arxiv.org/abs/\n1301.3781 .\nM. Nasr, N. Carlini, J. Hayase, M. Jagielski, A. F.\nCooper, D. Ippolito, C. A. Choquette-Choo,\nE. Wallace, F. Tramèr, and K. Lee. Scal-\nable extraction of training data from (pro-\nduction) language models. arXiv preprint\narXiv:2311.17035 , 2023.\nL. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wain-\nwright, P. Mishkin, C. Zhang, S. Agarwal,\nK. Slama, A. Ray, et al. Training language mod-\nels to follow instructions with human feedback.\nAdvances in Neural Information Processing Sys-\ntems, 35, 2022.\nL. Pacchiardi, A. J. Chan, S. Mindermann,\nI. Moscovitz, A. Y. Pan, Y. Gal, O. Evans, and\nJ. Brauner. How to catch an ai liar: Lie de-\ntection in black-box llms by asking unrelated\nquestions, 2023.\nD. Paperno, G. Kruszewski, A. Lazaridou, Q. N.\nPham, R. Bernardi, S. Pezzelle, M. Baroni,\nG. Boleda, and R. Fernández. The LAMBADA\ndataset: Word prediction requiring a broad\ndiscourse context. CoRR, abs/1606.06031,\n2016. URL http://arxiv.org/abs/1606.\n06031.\nC. Raffel, N. Shazeer, A. Roberts, K. Lee,\nS. Narang, M. Matena, Y. Zhou, W. Li, and P. J.\nLiu. Exploring the limits of transfer learning\nwith a unified text-to-text transformer. CoRR,\nabs/1910.10683, 2019. URL http://arxiv.\norg/abs/1910.10683 .\nA. Roberts, H. W. Chung, A. Levskaya, G. Mishra,\nJ. Bradbury, D. Andor, S. Narang, B. Lester,\nC. Gaffney, A. Mohiuddin, C. Hawthorne,\nA. Lewkowycz, A. Salcianu, M. van Zee,\nJ. Austin, S. Goodman, L. B. Soares, H. Hu,\n14', metadata={'source': './storage/Gemma-report.pdf', 'page': 13}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nS. Tsvyashchenko, A. Chowdhery, J. Bast-\nings, J. Bulian, X. Garcia, J. Ni, A. Chen,\nK. Kenealy, J. H. Clark, S. Lee, D. Garrette,\nJ. Lee-Thorp, C. Raffel, N. Shazeer, M. Rit-\nter, M. Bosma, A. Passos, J. Maitin-Shepard,\nN. Fiedel, M. Omernick, B. Saeta, R. Sepassi,\nA. Spiridonov, J. Newlan, and A. Gesmundo.\nScaling up models and data with t5xand\nseqio, 2022.\nA. Roberts, H. W. Chung, G. Mishra, A. Levskaya,\nJ. Bradbury, D. Andor, S. Narang, B. Lester,\nC. Gaffney, A. Mohiuddin, et al. Scaling up\nmodels and data with t5x and seqio. Jour-\nnal of Machine Learning Research , 24(377):1–8,\n2023.\nK. Sakaguchi, R. L. Bras, C. Bhagavatula, and\nY. Choi. WINOGRANDE: an adversarial\nwinograd schema challenge at scale. CoRR,\nabs/1907.10641, 2019. URL http://arxiv.\norg/abs/1907.10641 .\nM. Sap, H. Rashkin, D. Chen, R. L. Bras,\nand Y. Choi. Socialiqa: Commonsense\nreasoning about social interactions. CoRR,\nabs/1904.09728, 2019. URL http://arxiv.\norg/abs/1904.09728 .\nN.Shazeer. Fasttransformerdecoding: Onewrite-\nhead is all you need. CoRR, abs/1911.02150,\n2019. URL http://arxiv.org/abs/1911.\n02150.\nN. Shazeer. GLU variants improve transformer.\nCoRR, abs/2002.05202, 2020. URL https:\n//arxiv.org/abs/2002.05202 .\nJ. M. V. Skalse, N. H. R. Howe, D. Krasheninnikov,\nand D. Krueger. Defining and characterizing\nreward gaming. In NeurIPS , 2022.\nJ. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu. Roformer:\nEnhanced transformer with rotary position em-\nbedding. CoRR, abs/2104.09864, 2021. URL\nhttps://arxiv.org/abs/2104.09864 .\nI. Sutskever, O. Vinyals, and Q. V. Le. Sequence to\nsequence learning with neural networks. CoRR,\nabs/1409.3215, 2014. URL http://arxiv.\norg/abs/1409.3215 .M. Suzgun, N. Scales, N. Schärli, S. Gehrmann,\nY. Tay, H. W. Chung, A. Chowdhery, Q. V. Le,\nE. H. Chi, D. Zhou, and J. Wei. Challenging\nbig-bench tasks and whether chain-of-thought\ncan solve them, 2022.\nA. Talmor, J. Herzig, N. Lourie, and J. Be-\nrant. Commonsenseqa: A question answering\nchallenge targeting commonsense knowledge,\n2019.\nH. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-\nA. Lachaux, T. Lacroix, B. Rozière, N. Goyal,\nE. Hambro, F. Azhar, A. Rodriguez, A. Joulin,\nE. Grave, and G. Lample. Llama: Open and\nefficient foundation language models, 2023a.\nH. Touvron, L. Martin, K. Stone, P. Albert,\nA. Almahairi, Y. Babaei, N. Bashlykov, S. Batra,\nP. Bhargava, S. Bhosale, D. Bikel, L. Blecher,\nC. C. Ferrer, M. Chen, G. Cucurull, D. Es-\niobu, J. Fernandes, J. Fu, W. Fu, B. Fuller,\nC. Gao, V. Goswami, N. Goyal, A. Hartshorn,\nS. Hosseini, R. Hou, H. Inan, M. Kardas,\nV. Kerkez, M. Khabsa, I. Kloumann, A. Korenev,\nP. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee,\nD. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mi-\nhaylov,P.Mishra,I.Molybog,Y.Nie,A.Poulton,\nJ. Reizenstein, R. Rungta, K. Saladi, A. Schel-\nten, R. Silva, E. M. Smith, R. Subramanian,\nX. E. Tan, B. Tang, R. Taylor, A. Williams, J. X.\nKuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan,\nM. Kambadur, S. Narang, A. Rodriguez, R. Sto-\njnic, S. Edunov, and T. Scialom. Llama 2: Open\nfoundation and fine-tuned chat models, 2023b.\nA. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit,\nL. Jones, A. N. Gomez, L. Kaiser, and I. Polo-\nsukhin. Attention is all you need. CoRR,\nabs/1706.03762, 2017. URL http://arxiv.\norg/abs/1706.03762 .\nJ. Wei, X. Wang, D. Schuurmans, M. Bosma, E. H.\nChi, Q. Le, and D. Zhou. Chain of thought\nprompting elicits reasoning in large language\nmodels. CoRR, abs/2201.11903, 2022. URL\nhttps://arxiv.org/abs/2201.11903 .\nL. Weidinger, J. Mellor, M. Rauh, C. Griffin,\nJ. Uesato, P. Huang, M. Cheng, M. Glaese,\nB. Balle, A. Kasirzadeh, Z. Kenton, S. Brown,\nW. Hawkins, T. Stepleton, C. Biles, A. Birhane,\n15', metadata={'source': './storage/Gemma-report.pdf', 'page': 14}),
 Document(page_content='Gemma: Open Models Based on Gemini Research and Technology\nJ. Haas, L. Rimell, L. A. Hendricks, W. Isaac,\nS. Legassick, G. Irving, and I. Gabriel. Eth-\nical and social risks of harm from language\nmodels. CoRR, abs/2112.04359, 2021. URL\nhttps://arxiv.org/abs/2112.04359 .\nR. J. Williams. Simple statistical gradient-\nfollowing algorithms for connectionist rein-\nforcement learning. Machine learning , 8, 1992.\nXLA. Xla: Optimizing compiler for tensor-\nflow, 2019. URL https://www.tensorflow.\norg/xla .\nY. Xu, H. Lee, D. Chen, B. A. Hechtman, Y. Huang,\nR. Joshi, M. Krikun, D. Lepikhin, A. Ly, M. Mag-\ngioni, R. Pang, N. Shazeer, S. Wang, T. Wang,\nY. Wu, and Z. Chen. GSPMD: general and\nscalable parallelization for ML computation\ngraphs. CoRR, abs/2105.04663, 2021. URL\nhttps://arxiv.org/abs/2105.04663 .\nB. Zhang and R. Sennrich. Root mean square\nlayer normalization. CoRR, abs/1910.07467,\n2019. URL http://arxiv.org/abs/1910.\n07467.\nL. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang,\nZ. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing,\nH. Zhang, J. E. Gonzalez, and I. Stoica. Judg-\ning llm-as-a-judge with mt-bench and chatbot\narena, 2023.\nW. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang,\nA. Saied, W. Chen, and N. Duan. Agieval: A\nhuman-centric benchmark for evaluating foun-\ndation models, 2023.\nA. Zou, L. Phan, S. Chen, J. Campbell, P. Guo,\nR. Ren, A. Pan, X. Yin, M. Mazeika, A.-K. Dom-\nbrowski, S. Goel, N. Li, M. J. Byun, Z. Wang,\nA. Mallen, S. Basart, S. Koyejo, D. Song,\nM. Fredrikson, J. Z. Kolter, and D. Hendrycks.\nRepresentation engineering: A top-down ap-\nproach to ai transparency, 2023.\n16', metadata={'source': './storage/Gemma-report.pdf', 'page': 15})]
</pre>
저장된 `documents`의 파일들을 `prompts` 설정에 따라 `summaries`에 저장합니다.   

> "Gemma-report.pdf"는 11page 부터 참조기 때문에 10페이지까지 저장합니다.



```python
prompts = [{"content": f"Create a summary of the following document: '{doc.page_content}'", "role":"user"} for doc in documents[:9]]
```


```python
from tqdm.auto import tqdm
summaries = []
for prompt in tqdm(prompts):
    summaries.append(generate(model, tokenizer=tokenizer, prompt=apply_chat_template([prompt]), temp=0.1, max_tokens=500))
```

<pre>
  0%|          | 0/9 [00:00<?, ?it/s]
</pre>

```python
# 전체 문자 수 
print(len("\n".join(summaries)))

summaries_text = "\n".join(summaries)
```

<pre>
12529
</pre>
나눠서 요약한 것을 하나로 묶어서 다시 요약을 시킵니다.



```python
prompt_for_summaries_text = {"content": f"Give me a summary of the following document: '{summaries_text}'", "role":"user"}
final_summary = generate(model, tokenizer=tokenizer, prompt=apply_chat_template([prompt_for_summaries_text]), temp=0.1, max_tokens=3000)
```


```python
pprint(final_summary)
```

<pre>
('## Summary of "Gemma: Open Models Based on Gemini Research and Technology"\n'
 '\n'
 'This document summarizes the "Gemma: Open Models Based on Gemini Research '
 'and Technology" paper. It highlights the key points of the paper, '
 'including:\n'
 '\n'
 '**Key findings:**\n'
 '\n'
 '* Gemma models are highly accurate and efficient on a wide range of tasks.\n'
 '* The models are well-designed and have low memorization vulnerabilities.\n'
 '* The models are open-source and can be easily fine-tuned for specific use '
 'cases.\n'
 '\n'
 '**Challenges:**\n'
 '\n'
 '* The potential for malicious use of LLMs.\n'
 '* The potential for unintended biases in the data.\n'
 '\n'
 '**Mitigations:**\n'
 '\n'
 '* Data filtering and bias mitigation techniques.\n'
 '* Safety benchmarks and evaluations.\n'
 '* Red teaming and ethical considerations.\n'
 '\n'
 '**Overall, Gemma is a promising open-source model family that has the '
 'potential to improve the accessibility and safety of large language '
 'models.**')
</pre>
## 참조



* [Gemma Report](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)

* [huggingface mlx Gemma-7b-it](https://huggingface.co/mlx-community/quantized-gemma-7b-it)

