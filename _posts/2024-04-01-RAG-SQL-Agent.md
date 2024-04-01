---
layout: single
title:  "RAG: 원하는 SQL 쿼리를 LLM 모델을 사용하여 작성하는 시스템 개발"
categories: SQL
tag: [RAG, coding, SQL]
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



Q&A 시스템을 구축할 수 있는 가장 일반적인 유형의 데이터베이스 중 하나는 SQL 데이터베이스입니다.   

LangChain에는 SQLAlchemy에서 지원하는 모든 SQL 언어(예: MySQL, PostgreSQL, Oracle SQL, Databricks, SQLite)와 호환되는 여러 내장 체인 및 에이전트가 함께 제공됩니다.   

다음과 같은 사용 사례를 가능하게 합니다.   



* 자연어 질문을 기반으로 실행될 **SQL 쿼리 생성**

* 데이터베이스 데이터를 기반으로 질문에 답할 수 있는 **챗봇 개발**

* 사용자가 분석하고 싶은 인사이트를 바탕으로 **맞춤형 대시보드를 구축**


> 이번 튜토리얼에서는 SQL 데이터베이스를 통해 Q&A 체인과 에이전트를 만드는 기본 방법을 살펴보겠습니다.   

> 이러한 시스템을 통해 우리는 SQL 데이터베이스의 데이터에 대해 질문하고 자연어 답변을 얻을 수 있습니다.   


먼저 필수 패키지를 가져오고 환경 변수를 설정합니다.



```python
!pip install --upgrade --quiet langchain langchain-community langchain-openai
```

<pre>
[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m
</pre>

```python
import getpass
import os
import json

# api_key.json 파일을 읽어서 환경 변수 설정
with open('./api_key.json') as f:
    api_key_info = json.load(f)
os.environ["OPENAI_API_KEY"] = api_key_info['OPENAI_API_KEY']

# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

아래 예는 Chinook DB와의 SQLite 연결을 수행합니다.



* [이 파일](https://www.sqlitetutorial.net/sqlite-sample-database/)을 디렉터리에 `Chinook_Sqlite.sql`로 저장합니다.

* `sqlite3 Chinook.db`를 실행합니다.

* `.read Chinook_Sqlite.sql` 실행

* 테스트 `SELECT * FROM Artist LIMIT 10;`



이제 `Chinhook.db`가 디렉토리에 있습니다.



SQL 쿼리를 생성하고 실행하기 위해 `SQLDatabaseChain`을 생성해 보겠습니다.



```python
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///chinook.db")
llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
```

<pre>
/Users/ghingtae/miniforge3/envs/tf29_py39/lib/python3.9/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The class `langchain_community.llms.openai.OpenAI` was deprecated in langchain-community 0.0.10 and will be removed in 0.2.0. An updated version of the class exists in the langchain-openai package and should be used instead. To use it run `pip install -U langchain-openai` and import as `from langchain_openai import OpenAI`.
  warn_deprecated(
</pre>

```python
db_chain.run("몇명의 직원이 있어?")
```

## Case 1: Text-to-SQL query




```python
from langchain.chains import create_sql_query_chain
from langchain.chat_models import ChatOpenAI
```

SQL 쿼리를 작성할 체인을 만들어 보겠습니다:



```python
chain = create_sql_query_chain(ChatOpenAI(temperature=0), db)
response = chain.invoke({"question": "How many employees are there"})
print(response)
```

사용자 질문을 기반으로 SQL 쿼리를 작성하고 나면 쿼리를 실행할 수 있습니다:



```python
db.run(response)
```

<pre>
'[(8,)]'
</pre>
보시다시피 SQL 쿼리 빌더 체인은 쿼리를 **생성만**하고 **쿼리 실행**은 별도로 처리했습니다.


## Case 2: Text-to-SQL query and execution



`langchain_experiment`의 `SQLDatabaseChain`을 사용하여 SQL 쿼리를 생성하고 실행할 수 있습니다.



```python
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain

llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
```


```python
db_chain.run("직원이 몇명이나 있어?")
```

<pre>


[1m> Entering new SQLDatabaseChain chain...[0m
직원이 몇명이나 있어?
SQLQuery:[32;1m[1;3mSELECT COUNT(*) FROM employees;[0m
SQLResult: [33;1m[1;3m[(8,)][0m
Answer:[32;1m[1;3m직원은 8명입니다.[0m
[1m> Finished chain.[0m
</pre>
<pre>
'직원은 8명입니다.'
</pre>
보시다시피 이전 사례와 동일한 결과를 얻을 수 있습니다.



여기서 체인은 **쿼리 실행도 처리**하고 사용자 질문과 쿼리 결과를 기반으로 최종 답변을 제공합니다.



이 방식은 'SQL 인젝션'에 취약하기 때문에 사용 시 **주의**해야 합니다:



* 체인이 LLM에 의해 생성되고 검증되지 않은 쿼리를 실행하고 있습니다.

* 예: 레코드가 의도치 않게 생성, 수정 또는 삭제될 수 있음_.



이것이 바로 `SQLDatabaseChain`이 `랭체인_실험` 안에 있는 이유입니다.


## Case 3: SQL Agent



LangChain에는 `SQLDatabaseChain`보다 SQL 데이터베이스와 상호 작용하는 더 유연한 방법을 제공하는 SQL 에이전트가 있습니다.



SQL 에이전트 사용의 주요 장점은 다음과 같습니다:



- 데이터베이스의 스키마뿐만 아니라 데이터베이스의 콘텐츠(예: 특정 테이블 설명)를 기반으로 질문에 답변할 수 있습니다.

- 생성된 쿼리를 실행하고 트레이스백을 포착하여 올바르게 다시 생성함으로써 오류로부터 복구할 수 있습니다.



에이전트를 초기화하기 위해 `create_sql_agent` 함수를 사용합니다.



이 에이전트에는 다음과 같은 도구가 포함된 `SQLDatabaseToolkit`이 포함되어 있습니다:



* 쿼리 생성 및 실행

* 쿼리 구문 확인

* 테이블 설명 검색

* ... 등



```python
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

# from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType

db = SQLDatabase.from_uri("sqlite:///chinook.db")

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0)),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
```

### Agent task example #1 - Running queries




```python
agent_executor.run(
    "국가별 총 매출을 나열합니다. 어느 국가의 고객이 가장 많이 지출했나요?"
)
```

<pre>


[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3mAction: sql_db_list_tables
Action Input: [0m
Observation: [38;5;200m[1;3malbums, artists, customers, employees, genres, invoice_items, invoices, media_types, playlist_track, playlists, tracks[0m
Thought:[32;1m[1;3m I should query the schema of the customers and invoices tables.
Action: sql_db_schema
Action Input: customers, invoices[0m
Observation: [33;1m[1;3m
CREATE TABLE customers (
	"CustomerId" INTEGER NOT NULL, 
	"FirstName" NVARCHAR(40) NOT NULL, 
	"LastName" NVARCHAR(20) NOT NULL, 
	"Company" NVARCHAR(80), 
	"Address" NVARCHAR(70), 
	"City" NVARCHAR(40), 
	"State" NVARCHAR(40), 
	"Country" NVARCHAR(40), 
	"PostalCode" NVARCHAR(10), 
	"Phone" NVARCHAR(24), 
	"Fax" NVARCHAR(24), 
	"Email" NVARCHAR(60) NOT NULL, 
	"SupportRepId" INTEGER, 
	PRIMARY KEY ("CustomerId"), 
	FOREIGN KEY("SupportRepId") REFERENCES employees ("EmployeeId")
)

/*
3 rows from customers table:
CustomerId	FirstName	LastName	Company	Address	City	State	Country	PostalCode	Phone	Fax	Email	SupportRepId
1	Luís	Gonçalves	Embraer - Empresa Brasileira de Aeronáutica S.A.	Av. Brigadeiro Faria Lima, 2170	São José dos Campos	SP	Brazil	12227-000	+55 (12) 3923-5555	+55 (12) 3923-5566	luisg@embraer.com.br	3
2	Leonie	Köhler	None	Theodor-Heuss-Straße 34	Stuttgart	None	Germany	70174	+49 0711 2842222	None	leonekohler@surfeu.de	5
3	François	Tremblay	None	1498 rue Bélanger	Montréal	QC	Canada	H2G 1A7	+1 (514) 721-4711	None	ftremblay@gmail.com	3
*/


CREATE TABLE invoices (
	"InvoiceId" INTEGER NOT NULL, 
	"CustomerId" INTEGER NOT NULL, 
	"InvoiceDate" DATETIME NOT NULL, 
	"BillingAddress" NVARCHAR(70), 
	"BillingCity" NVARCHAR(40), 
	"BillingState" NVARCHAR(40), 
	"BillingCountry" NVARCHAR(40), 
	"BillingPostalCode" NVARCHAR(10), 
	"Total" NUMERIC(10, 2) NOT NULL, 
	PRIMARY KEY ("InvoiceId"), 
	FOREIGN KEY("CustomerId") REFERENCES customers ("CustomerId")
)

/*
3 rows from invoices table:
InvoiceId	CustomerId	InvoiceDate	BillingAddress	BillingCity	BillingState	BillingCountry	BillingPostalCode	Total
1	2	2009-01-01 00:00:00	Theodor-Heuss-Straße 34	Stuttgart	None	Germany	70174	1.98
2	4	2009-01-02 00:00:00	Ullevålsveien 14	Oslo	None	Norway	0171	3.96
3	8	2009-01-03 00:00:00	Grétrystraat 63	Brussels	None	Belgium	1000	5.94
*/[0m
Thought:[32;1m[1;3m I should query the customers and invoices tables to get the total sales by country.
Action: sql_db_query
Action Input: SELECT customers.Country, SUM(invoices.Total) AS TotalSales FROM customers INNER JOIN invoices ON customers.CustomerId = invoices.CustomerId GROUP BY customers.Country ORDER BY TotalSales DESC LIMIT 10[0m
Observation: [36;1m[1;3m[('USA', 523.0600000000004), ('Canada', 303.96), ('France', 195.09999999999994), ('Brazil', 190.1), ('Germany', 156.48), ('United Kingdom', 112.85999999999999), ('Czech Republic', 90.24), ('Portugal', 77.24), ('India', 75.25999999999999), ('Chile', 46.62)][0m
Thought:[32;1m[1;3m I now know the final answer
Final Answer: 가장 많이 지출한 국가는 미국입니다.[0m

[1m> Finished chain.[0m
</pre>
<pre>
'가장 많이 지출한 국가는 미국입니다.'
</pre>
### Agent task example #2 - Describing a Table



```python
agent_executor.run("playlisttrack 테이블에 대해서 설명해줄래?")
```

<pre>


[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3mAction: sql_db_list_tables
Action Input: [0m
Observation: [38;5;200m[1;3malbums, artists, customers, employees, genres, invoice_items, invoices, media_types, playlist_track, playlists, tracks[0m
Thought:[32;1m[1;3m The most relevant table is playlist_track, so I should query the schema of that table.
Action: sql_db_schema
Action Input: playlist_track[0m
Observation: [33;1m[1;3m
CREATE TABLE playlist_track (
	"PlaylistId" INTEGER NOT NULL, 
	"TrackId" INTEGER NOT NULL, 
	PRIMARY KEY ("PlaylistId", "TrackId"), 
	FOREIGN KEY("TrackId") REFERENCES tracks ("TrackId"), 
	FOREIGN KEY("PlaylistId") REFERENCES playlists ("PlaylistId")
)

/*
3 rows from playlist_track table:
PlaylistId	TrackId
1	3402
1	3389
1	3390
*/[0m
Thought:[32;1m[1;3m I now know the final answer
Final Answer: playlist_track 테이블은 PlaylistId와 TrackId를 가지고 있는 테이블이며, PlaylistId와 TrackId는 각각 playlists 테이블과 tracks 테이블과 연결되어 있습니다.[0m

[1m> Finished chain.[0m
</pre>
<pre>
'playlist_track 테이블은 PlaylistId와 TrackId를 가지고 있는 테이블이며, PlaylistId와 TrackId는 각각 playlists 테이블과 tracks 테이블과 연결되어 있습니다.'
</pre>
### SQL 툴킷 확장하기



기본 제공되는 SQL 툴킷에는 데이터베이스 작업을 시작하는 데 필요한 도구가 포함되어 있지만, 에이전트의 기능을 확장하는 데 몇 가지 추가 도구가 유용할 수 있는 경우가 종종 있습니다. 이는 솔루션의 전반적인 성능을 개선하기 위해 솔루션에서 **도메인별 지식**을 사용하려고 할 때 특히 유용합니다.



몇 가지 예는 다음과 같습니다:



- Dynamic Few shot 예시 포함

- 열 필터로 사용할 고유명사의 철자 오류 찾기



이러한 특정 사용 사례를 처리하는 별도의 도구를 만들어 표준 SQL 도구 키트에 보완용으로 포함할 수 있습니다. 이 두 가지 사용자 정의 도구를 포함하는 방법을 살펴보겠습니다.



#### Dynamic Few shot 예제 포함



Dynamic Few shot 예제를 포함하려면 사용자의 질문과 의미적으로 유사한 예제를 검색하기 위해 벡터 데이터베이스를 처리하는 사용자 지정 **검색 도구**가 필요합니다.



몇 가지 예제가 포함된 사전을 만드는 것부터 시작하겠습니다:



```python
few_shots = {
    "List all artists.": "SELECT * FROM artists;",
    "Find all albums for the artist 'AC/DC'.": "SELECT * FROM albums WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'AC/DC');",
    "List all tracks in the 'Rock' genre.": "SELECT * FROM tracks WHERE GenreId = (SELECT GenreId FROM genres WHERE Name = 'Rock');",
    "Find the total duration of all tracks.": "SELECT SUM(Milliseconds) FROM tracks;",
    "List all customers from Canada.": "SELECT * FROM customers WHERE Country = 'Canada';",
    "How many tracks are there in the album with ID 5?": "SELECT COUNT(*) FROM tracks WHERE AlbumId = 5;",
    "Find the total number of invoices.": "SELECT COUNT(*) FROM invoices;",
    "List all tracks that are longer than 5 minutes.": "SELECT * FROM tracks WHERE Milliseconds > 300000;",
    "Who are the top 5 customers by total purchase?": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM invoices GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    "Which albums are from the year 2000?": "SELECT * FROM albums WHERE strftime('%Y', ReleaseDate) = '2000';",
    "How many employees are there": 'SELECT COUNT(*) FROM "employee"',
}
```

그런 다음 질문 목록을 사용하여 검색기를 생성하고 대상 SQL 쿼리를 메타데이터로 할당할 수 있습니다:



```python
!pip install tiktoken faiss-cpu
```

<pre>
Collecting tiktoken
  Downloading tiktoken-0.5.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.0 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m10.1 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting faiss-cpu
  Downloading faiss_cpu-1.7.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.6 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.6/17.6 MB[0m [31m47.1 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2023.6.3)
Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2.31.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2023.7.22)
Installing collected packages: faiss-cpu, tiktoken
[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
llmx 0.0.15a0 requires cohere, which is not installed.[0m[31m
[0mSuccessfully installed faiss-cpu-1.7.4 tiktoken-0.5.1
</pre>

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()

few_shot_docs = [
    Document(page_content=question, metadata={"sql_query": few_shots[question]})
    for question in few_shots.keys()
]
vector_db = FAISS.from_documents(few_shot_docs, embeddings)
retriever = vector_db.as_retriever()
```

이제 고유한 사용자 지정 도구를 만들어 'create_sql_agent' 함수에 새 도구로 추가할 수 있습니다:



```python
from langchain.agents.agent_toolkits import create_retriever_tool

tool_description = """
이 도구는 유사한 예시를 이해하여 사용자 질문에 적용하는 데 도움이 됩니다.
이 도구에 입력하는 내용은 사용자 질문이어야 합니다.
"""

retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
)
custom_tool_list = [retriever_tool]
```

이제 사용 사례를 고려하여 표준 SQL 에이전트 접미사를 조정하여 에이전트를 만들 수 있습니다. 이를 처리하는 가장 간단한 방법은 도구 설명에 포함시키는 것이지만, 이것만으로는 충분하지 않은 경우가 많으므로 생성자의 '접미사' 인수를 사용하여 에이전트 프롬프트에서 이를 지정해야 합니다.



```python
from langchain.agents import AgentType, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///chinook.db")
llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

custom_suffix = """
먼저 제가 알고 있는 비슷한 예제를 가져와야 합니다.
예제가 쿼리를 구성하기에 충분하다면 쿼리를 작성할 수 있습니다.
그렇지 않으면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인할 수 있습니다.
그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
"""

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools=custom_tool_list,
    suffix=custom_suffix,
)
```

Let's try it out:



```python
agent.run("How many employees do we have?")
```

<pre>


[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `sql_get_similar_examples` with `{'query': 'How many employees do we have?'}`


[0m[33;1m[1;3m[Document(page_content='How many employees are there', metadata={'sql_query': 'SELECT COUNT(*) FROM "employee"'}), Document(page_content='Find the total number of invoices.', metadata={'sql_query': 'SELECT COUNT(*) FROM invoices;'}), Document(page_content='Who are the top 5 customers by total purchase?', metadata={'sql_query': 'SELECT CustomerId, SUM(Total) AS TotalPurchase FROM invoices GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;'}), Document(page_content='List all customers from Canada.', metadata={'sql_query': "SELECT * FROM customers WHERE Country = 'Canada';"})][0m[32;1m[1;3m
Invoking: `sql_db_query_checker` with `SELECT COUNT(*) FROM employee`


[0m[36;1m[1;3mSELECT COUNT(*) FROM employee[0m[32;1m[1;3m
Invoking: `sql_db_query` with `SELECT COUNT(*) FROM employee`


[0m[36;1m[1;3mError: (sqlite3.OperationalError) no such table: employee
[SQL: SELECT COUNT(*) FROM employee]
(Background on this error at: https://sqlalche.me/e/20/e3q8)[0m[32;1m[1;3m
Invoking: `sql_db_list_tables` with ``


[0m[38;5;200m[1;3malbums, artists, customers, employees, genres, invoice_items, invoices, media_types, playlist_track, playlists, tracks[0m[32;1m[1;3m
Invoking: `sql_db_query` with `SELECT COUNT(*) FROM employees`


[0m[36;1m[1;3m[(8,)][0m[32;1m[1;3mWe have a total of 8 employees.[0m

[1m> Finished chain.[0m
</pre>
<pre>
'We have a total of 8 employees.'
</pre>
보시다시피, 에이전트는 먼저 `sql_get_similar_examples` 도구를 사용하여 유사한 예제를 검색했습니다. 질문이 다른 몇 개의 샷 예제와 매우 유사했기 때문에 에이전트는 표준 툴킷의 다른 툴을 사용할 필요가 없었기 때문에 **시간과 토큰을 절약**할 수 있었습니다.


#### 고유명사의 맞춤법 오류 찾기 및 수정하기



주소, 노래 이름 또는 아티스트와 같은 고유명사가 포함된 열을 필터링하려면 먼저 철자를 다시 확인하여 데이터를 올바르게 필터링해야 합니다.



데이터베이스에 존재하는 모든 고유 고유명사를 사용하여 벡터 저장소를 생성하면 됩니다. 그런 다음 사용자가 질문에 고유 명사를 포함할 때마다 상담원이 해당 벡터 저장소를 쿼리하여 해당 단어의 올바른 철자를 찾도록 할 수 있습니다. 이러한 방식으로 에이전트는 대상 쿼리를 작성하기 전에 사용자가 어떤 엔티티를 참조하는지 이해할 수 있습니다.



메타데이터 없이 고유명사를 임베드한 다음 철자가 틀린 사용자 질문과 가장 유사한 것을 쿼리하는 방식으로 몇 가지 샷과 유사한 접근 방식을 따라 해 보겠습니다.



먼저 원하는 각 엔티티에 대한 고유 값이 필요하며, 이를 위해 결과를 요소 목록으로 파싱하는 함수를 정의합니다:



```python
print(db.table_info)
```

<pre>

CREATE TABLE albums (
	"AlbumId" INTEGER NOT NULL, 
	"Title" NVARCHAR(160) NOT NULL, 
	"ArtistId" INTEGER NOT NULL, 
	PRIMARY KEY ("AlbumId"), 
	FOREIGN KEY("ArtistId") REFERENCES artists ("ArtistId")
)

/*
3 rows from albums table:
AlbumId	Title	ArtistId
1	For Those About To Rock We Salute You	1
2	Balls to the Wall	2
3	Restless and Wild	2
*/


CREATE TABLE artists (
	"ArtistId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("ArtistId")
)

/*
3 rows from artists table:
ArtistId	Name
1	AC/DC
2	Accept
3	Aerosmith
*/


CREATE TABLE customers (
	"CustomerId" INTEGER NOT NULL, 
	"FirstName" NVARCHAR(40) NOT NULL, 
	"LastName" NVARCHAR(20) NOT NULL, 
	"Company" NVARCHAR(80), 
	"Address" NVARCHAR(70), 
	"City" NVARCHAR(40), 
	"State" NVARCHAR(40), 
	"Country" NVARCHAR(40), 
	"PostalCode" NVARCHAR(10), 
	"Phone" NVARCHAR(24), 
	"Fax" NVARCHAR(24), 
	"Email" NVARCHAR(60) NOT NULL, 
	"SupportRepId" INTEGER, 
	PRIMARY KEY ("CustomerId"), 
	FOREIGN KEY("SupportRepId") REFERENCES employees ("EmployeeId")
)

/*
3 rows from customers table:
CustomerId	FirstName	LastName	Company	Address	City	State	Country	PostalCode	Phone	Fax	Email	SupportRepId
1	Luís	Gonçalves	Embraer - Empresa Brasileira de Aeronáutica S.A.	Av. Brigadeiro Faria Lima, 2170	São José dos Campos	SP	Brazil	12227-000	+55 (12) 3923-5555	+55 (12) 3923-5566	luisg@embraer.com.br	3
2	Leonie	Köhler	None	Theodor-Heuss-Straße 34	Stuttgart	None	Germany	70174	+49 0711 2842222	None	leonekohler@surfeu.de	5
3	François	Tremblay	None	1498 rue Bélanger	Montréal	QC	Canada	H2G 1A7	+1 (514) 721-4711	None	ftremblay@gmail.com	3
*/


CREATE TABLE employees (
	"EmployeeId" INTEGER NOT NULL, 
	"LastName" NVARCHAR(20) NOT NULL, 
	"FirstName" NVARCHAR(20) NOT NULL, 
	"Title" NVARCHAR(30), 
	"ReportsTo" INTEGER, 
	"BirthDate" DATETIME, 
	"HireDate" DATETIME, 
	"Address" NVARCHAR(70), 
	"City" NVARCHAR(40), 
	"State" NVARCHAR(40), 
	"Country" NVARCHAR(40), 
	"PostalCode" NVARCHAR(10), 
	"Phone" NVARCHAR(24), 
	"Fax" NVARCHAR(24), 
	"Email" NVARCHAR(60), 
	PRIMARY KEY ("EmployeeId"), 
	FOREIGN KEY("ReportsTo") REFERENCES employees ("EmployeeId")
)

/*
3 rows from employees table:
EmployeeId	LastName	FirstName	Title	ReportsTo	BirthDate	HireDate	Address	City	State	Country	PostalCode	Phone	Fax	Email
1	Adams	Andrew	General Manager	None	1962-02-18 00:00:00	2002-08-14 00:00:00	11120 Jasper Ave NW	Edmonton	AB	Canada	T5K 2N1	+1 (780) 428-9482	+1 (780) 428-3457	andrew@chinookcorp.com
2	Edwards	Nancy	Sales Manager	1	1958-12-08 00:00:00	2002-05-01 00:00:00	825 8 Ave SW	Calgary	AB	Canada	T2P 2T3	+1 (403) 262-3443	+1 (403) 262-3322	nancy@chinookcorp.com
3	Peacock	Jane	Sales Support Agent	2	1973-08-29 00:00:00	2002-04-01 00:00:00	1111 6 Ave SW	Calgary	AB	Canada	T2P 5M5	+1 (403) 262-3443	+1 (403) 262-6712	jane@chinookcorp.com
*/


CREATE TABLE genres (
	"GenreId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("GenreId")
)

/*
3 rows from genres table:
GenreId	Name
1	Rock
2	Jazz
3	Metal
*/


CREATE TABLE invoice_items (
	"InvoiceLineId" INTEGER NOT NULL, 
	"InvoiceId" INTEGER NOT NULL, 
	"TrackId" INTEGER NOT NULL, 
	"UnitPrice" NUMERIC(10, 2) NOT NULL, 
	"Quantity" INTEGER NOT NULL, 
	PRIMARY KEY ("InvoiceLineId"), 
	FOREIGN KEY("TrackId") REFERENCES tracks ("TrackId"), 
	FOREIGN KEY("InvoiceId") REFERENCES invoices ("InvoiceId")
)

/*
3 rows from invoice_items table:
InvoiceLineId	InvoiceId	TrackId	UnitPrice	Quantity
1	1	2	0.99	1
2	1	4	0.99	1
3	2	6	0.99	1
*/


CREATE TABLE invoices (
	"InvoiceId" INTEGER NOT NULL, 
	"CustomerId" INTEGER NOT NULL, 
	"InvoiceDate" DATETIME NOT NULL, 
	"BillingAddress" NVARCHAR(70), 
	"BillingCity" NVARCHAR(40), 
	"BillingState" NVARCHAR(40), 
	"BillingCountry" NVARCHAR(40), 
	"BillingPostalCode" NVARCHAR(10), 
	"Total" NUMERIC(10, 2) NOT NULL, 
	PRIMARY KEY ("InvoiceId"), 
	FOREIGN KEY("CustomerId") REFERENCES customers ("CustomerId")
)

/*
3 rows from invoices table:
InvoiceId	CustomerId	InvoiceDate	BillingAddress	BillingCity	BillingState	BillingCountry	BillingPostalCode	Total
1	2	2009-01-01 00:00:00	Theodor-Heuss-Straße 34	Stuttgart	None	Germany	70174	1.98
2	4	2009-01-02 00:00:00	Ullevålsveien 14	Oslo	None	Norway	0171	3.96
3	8	2009-01-03 00:00:00	Grétrystraat 63	Brussels	None	Belgium	1000	5.94
*/


CREATE TABLE media_types (
	"MediaTypeId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("MediaTypeId")
)

/*
3 rows from media_types table:
MediaTypeId	Name
1	MPEG audio file
2	Protected AAC audio file
3	Protected MPEG-4 video file
*/


CREATE TABLE playlist_track (
	"PlaylistId" INTEGER NOT NULL, 
	"TrackId" INTEGER NOT NULL, 
	PRIMARY KEY ("PlaylistId", "TrackId"), 
	FOREIGN KEY("TrackId") REFERENCES tracks ("TrackId"), 
	FOREIGN KEY("PlaylistId") REFERENCES playlists ("PlaylistId")
)

/*
3 rows from playlist_track table:
PlaylistId	TrackId
1	3402
1	3389
1	3390
*/


CREATE TABLE playlists (
	"PlaylistId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("PlaylistId")
)

/*
3 rows from playlists table:
PlaylistId	Name
1	Music
2	Movies
3	TV Shows
*/


CREATE TABLE tracks (
	"TrackId" INTEGER NOT NULL, 
	"Name" NVARCHAR(200) NOT NULL, 
	"AlbumId" INTEGER, 
	"MediaTypeId" INTEGER NOT NULL, 
	"GenreId" INTEGER, 
	"Composer" NVARCHAR(220), 
	"Milliseconds" INTEGER NOT NULL, 
	"Bytes" INTEGER, 
	"UnitPrice" NUMERIC(10, 2) NOT NULL, 
	PRIMARY KEY ("TrackId"), 
	FOREIGN KEY("MediaTypeId") REFERENCES media_types ("MediaTypeId"), 
	FOREIGN KEY("GenreId") REFERENCES genres ("GenreId"), 
	FOREIGN KEY("AlbumId") REFERENCES albums ("AlbumId")
)

/*
3 rows from tracks table:
TrackId	Name	AlbumId	MediaTypeId	GenreId	Composer	Milliseconds	Bytes	UnitPrice
1	For Those About To Rock (We Salute You)	1	1	1	Angus Young, Malcolm Young, Brian Johnson	343719	11170334	0.99
2	Balls to the Wall	2	2	1	None	342562	5510424	0.99
3	Fast As a Shark	3	2	1	F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman	230619	3990994	0.99
*/
</pre>

```python
import ast
import re


def run_query_save_results(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return res


artists = run_query_save_results(db, "SELECT name FROM artists")
albums = run_query_save_results(db, "SELECT title FROM albums")
```

이제 사용자 지정 **리트리버 도구**와 최종 에이전트 생성을 진행할 수 있습니다:



```python
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

texts = artists + albums

embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_texts(texts, embeddings)
retriever = vector_db.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    name="name_search",
    description="이름, 성 주소 등 데이터가 실제로 어떻게 쓰여졌는지 알아내는 데 사용합니다.",
)

custom_tool_list = [retriever_tool]
```


```python
from langchain.agents import AgentType, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase

# db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

custom_suffix = """
사용자가 고유명사를 기준으로 필터링해 달라고 요청하는 경우, 먼저 name_search 도구를 사용하여 철자를 확인해야 합니다.
그렇지 않으면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인할 수 있습니다.
그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
"""

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools=custom_tool_list,
    suffix=custom_suffix,
)
```

Let's try it out:



```python
agent.run("alice in chains는 몇 개의 앨범을 가지고 있나요?")
```

<pre>


[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `sql_db_list_tables` with ``


[0m[38;5;200m[1;3malbums, artists, customers, employees, genres, invoice_items, invoices, media_types, playlist_track, playlists, tracks[0m[32;1m[1;3m
Invoking: `sql_db_schema` with `albums, artists`


[0m[33;1m[1;3m
CREATE TABLE albums (
	"AlbumId" INTEGER NOT NULL, 
	"Title" NVARCHAR(160) NOT NULL, 
	"ArtistId" INTEGER NOT NULL, 
	PRIMARY KEY ("AlbumId"), 
	FOREIGN KEY("ArtistId") REFERENCES artists ("ArtistId")
)

/*
3 rows from albums table:
AlbumId	Title	ArtistId
1	For Those About To Rock We Salute You	1
2	Balls to the Wall	2
3	Restless and Wild	2
*/


CREATE TABLE artists (
	"ArtistId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("ArtistId")
)

/*
3 rows from artists table:
ArtistId	Name
1	AC/DC
2	Accept
3	Aerosmith
*/[0m[32;1m[1;3m
Invoking: `sql_db_query_checker` with `SELECT COUNT(*) FROM albums WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'Alice In Chains')`


[0m[36;1m[1;3mSELECT COUNT(*) FROM albums WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'Alice In Chains')[0m[32;1m[1;3m
Invoking: `sql_db_query` with `SELECT COUNT(*) FROM albums WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'Alice In Chains')`


[0m[36;1m[1;3m[(1,)][0m[32;1m[1;3mAlice In Chains는 1개의 앨범을 가지고 있습니다.[0m

[1m> Finished chain.[0m
</pre>
<pre>
'Alice In Chains는 1개의 앨범을 가지고 있습니다.'
</pre>
보시다시피 에이전트는 이 특정 아티스트에 대한 데이터베이스를 올바르게 쿼리하는 방법을 확인하기 위해 `name_search` 도구를 사용했습니다.

