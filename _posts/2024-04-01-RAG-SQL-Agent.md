---
layout: single
title:  "SQL Agent: LangChain을 활용한 SQL 구문 생성 모델"
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



## 서론



인공지능 기술의 급속한 발전은 자연어 처리(NLP)의 새로운 지평을 열었습니다. 특히, 자연어를 이해하고 이를 기반으로 특정 작업을 수행하는 언어 모델들이 주목을 받고 있습니다. 이러한 기술은 데이터베이스 쿼리 작성과 같은 분야에 혁신적인 변화를 가져오고 있습니다.



이 튜토리얼에서는 LangChain을 활용하여 사용자의 자연어 질의를 SQL 구문으로 변환해주는 모델을 소개합니다. 이 모델은 질의 응답 시스템, 데이터 분석, 그리고 데이터베이스 관리 시스템 등에서 유용하게 사용될 수 있습니다.



## LangChain이란?



LangChain은 자연어 처리를 위한 프레임워크로, 특히 자연어 이해 및 생성 작업에 최적화되어 있습니다. LangChain을 사용하여 개발자는 복잡한 NLP 작업을 더 간단하고 효율적으로 구현할 수 있습니다. 이 프레임워크는 특히 자연어를 SQL 쿼리로 변환하는 기능을 포함하여, 데이터베이스와의 상호작용을 자연어 기반으로 단순화합니다.


## 모델 구현 방법



![SQL_Agent](/assets/images/sql_usecase.png)



### 패키지 설치 및 환경 변수 설정



먼저 필수 패키지를 가져오고 환경 변수를 설정합니다.



```python
%pip install --upgrade --quiet  langchain langchain-community langchain-openai
```

<pre>
[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33m    WARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~angchain-community (/Users/ghingtae/anaconda3/lib/python3.11/site-packages)[0m[33m
[0mNote: you may need to restart the kernel to use updated packages.
</pre>
이 가이드에서는 기본적으로 OpenAI 모델을 사용합니다.



```python
import getpass
import os
import json

# api_key.json 파일을 읽어서 환경 변수 설정
with open('./api_key.json') as f:
    api_key_info = json.load(f)
os.environ["OPENAI_API_KEY"] = api_key_info['OPENAI_API_KEY']
os.environ["LANGCHAIN_API_KEY"] = api_key_info['LANGCHAIN_API_KEY']
os.environ["Langchain_SQL_Agent"] = "true"
```

### Chinook 데이터베이스 SQLite 연결



![chinook](/assets/images/SQLite%20Sample%20Database%20Color.jpg)



SQLite를 연결하여 사용할 것이고 테스트 데이터베이스는 Chinook를 사용할 것입니다.   

- 먼저 이 파일을 [Chinook_Sqlite.sql](https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql)으로 저장합니다.

- 터미널에서 `sqlite3 Chinook.db`를 실행합니다.

- 그리고 `.read Chinook_Sqlite.sql`을 실행합니다.

- `SELECT * FROM Artist LIMIT 10;`을 실행시켜 테스트 해봅니다.



```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
```

<pre>
sqlite
['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']
</pre>

```python
db.run("SELECT * FROM Artist LIMIT 10;")
```

<pre>
"[(1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith'), (4, 'Alanis Morissette'), (5, 'Alice In Chains'), (6, 'Antônio Carlos Jobim'), (7, 'Apocalyptica'), (8, 'Audioslave'), (9, 'BackBeat'), (10, 'Billy Cobham')]"
</pre>
### SQL 체인 만들기: 질문에서 SQL 쿼리로 변환하기



사용자의 질문을 받아 SQL 쿼리로 변환하고, 해당 쿼리를 실행한 후, 그 결과를 사용해 원래의 질문에 답하는 코드를 작성합니다.



#### 질문을 SQL 쿼리로 변환하기



SQL 체인 또는 에이전트의 첫 번째 단계는 사용자 입력을 SQL 쿼리로 변환하는 것입니다. LangChain은 이를 위한 내장 체인을 제공합니다: [create_sql_query_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.sql_database.query.create_sql_query_chain.html).






```python
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
response
```

<pre>
'SELECT COUNT(*) AS "Number of Employees" FROM "Employee"'
</pre>
쿼리를 실행하여 유효한지 확인할 수 있습니다:



```python
db.run(response)
```

<pre>
'[(8,)]'
</pre>
`chain.get_prompts()`를 통해 프롬프트 설정을 살펴볼 수 있습니다.



```python
chain.get_prompts()[0].pretty_print()
```

<pre>
You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
[33;1m[1;3m{table_info}[0m

Question: [33;1m[1;3m{input}[0m
</pre>
### SQL 쿼리 실행 후 chain 추가



`QuerySQLDataBaseTool`을 사용하면 쿼리 실행을 우리의 체인에 쉽게 추가할 수 있습니다.   

하지만 실제로 쿼리를 DB에 실행하기 때문에 주의해서 사용해야합니다:



```python
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# 데이터베이스 쿼리를 실행하기 위한 도구를 초기화
execute_query = QuerySQLDataBaseTool(db=db)
# SQL 쿼리를 작성하기 위한 체인을 생성
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
# 체인을 사용하여 질문에 대한 SQL 쿼리를 실행
chain.invoke({"question": "How many employees are there"})
```

<pre>
'[(8,)]'
</pre>
### 쿼리 결과를 최종 답으로 변환



쿼리를 자동으로 생성하고 실행하는 방법을 마련한 이후, 원래의 질문과 SQL 쿼리 결과를 결합하여 최종 답변을 생성할 필요가 있습니다.   

이를 위해 질문과 결과를 다시 LLM에 전달하여 최종 답변을 생성할 수 있습니다:



```python
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 쿼리의 결과를 해석하고 사용자 질문에 답변하기 위한 프롬프트 템플릿을 설정
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# 답변 생성 프로세스를 설정
answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

chain.invoke({"question": "How many employees are there"})
```

<pre>
'There are 8 employees.'
</pre>
이 코드는 사용자의 질문, 해당하는 SQL 쿼리, 그리고 SQL 쿼리 결과를 바탕으로 사용자의 질문에 대한 답변을 생성합니다.   

chain을 통해 이루어 졌으며 LangChain에서는 Chain보다 SQL 데이터베이스와 더 유연한 상호 작용 방법을 제공하는 SQL Agent가 있습니다.   



## SQL Agent



SQL Agent 사용 이점은 다음과 같습니다:



- 데이터베이스의 스키마는 물론 데이터베이스의 콘텐츠(예: 특정 테이블 설명)를 기반으로 질문에 답할 수 있습니다.

- 생성된 쿼리를 실행하고, 역추적을 포착하고, 이를 올바르게 다시 생성하여 오류를 복구할 수 있습니다.

- 사용자 질문에 답하기 위해 필요한 만큼 데이터베이스를 쿼리할 수 있습니다.

- 관련 테이블에서만 스키마를 검색하여 토큰을 저장합니다.(토큰 절약)



```python
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

from langchain.agents.agent_types import AgentType

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent_executor = create_sql_agent(llm, toolkit=SQLDatabaseToolkit(db=db, llm=llm), agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```


```python
agent_executor.invoke(
    "List the total sales per country. Which country's customers spent the most?"
)
```

<pre>


[1m> Entering new SQL Agent Executor chain...[0m
[32;1m[1;3mTo answer this question, I need to find a table that contains information about sales and country. I will then need to sum the sales per country and order the results to find the country with the highest total sales.
Action: sql_db_list_tables
Action Input: ""[0m[38;5;200m[1;3mAlbum, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track[0m[32;1m[1;3mThe tables 'Customer' and 'Invoice' seem to be the most relevant for this query. The 'Customer' table likely contains information about the customer's country, and the 'Invoice' table likely contains information about the sales. I need to check the schema of these tables to confirm.
Action: sql_db_schema
Action Input: Customer, Invoice[0m[33;1m[1;3m
CREATE TABLE "Customer" (
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
	FOREIGN KEY("SupportRepId") REFERENCES "Employee" ("EmployeeId")
)

/*
3 rows from Customer table:
CustomerId	FirstName	LastName	Company	Address	City	State	Country	PostalCode	Phone	Fax	Email	SupportRepId
1	Luís	Gonçalves	Embraer - Empresa Brasileira de Aeronáutica S.A.	Av. Brigadeiro Faria Lima, 2170	São José dos Campos	SP	Brazil	12227-000	+55 (12) 3923-5555	+55 (12) 3923-5566	luisg@embraer.com.br	3
2	Leonie	Köhler	None	Theodor-Heuss-Straße 34	Stuttgart	None	Germany	70174	+49 0711 2842222	None	leonekohler@surfeu.de	5
3	François	Tremblay	None	1498 rue Bélanger	Montréal	QC	Canada	H2G 1A7	+1 (514) 721-4711	None	ftremblay@gmail.com	3
*/


CREATE TABLE "Invoice" (
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
	FOREIGN KEY("CustomerId") REFERENCES "Customer" ("CustomerId")
)

/*
3 rows from Invoice table:
InvoiceId	CustomerId	InvoiceDate	BillingAddress	BillingCity	BillingState	BillingCountry	BillingPostalCode	Total
1	2	2021-01-01 00:00:00	Theodor-Heuss-Straße 34	Stuttgart	None	Germany	70174	1.98
2	4	2021-01-02 00:00:00	Ullevålsveien 14	Oslo	None	Norway	0171	3.96
3	8	2021-01-03 00:00:00	Grétrystraat 63	Brussels	None	Belgium	1000	5.94
*/[0m[32;1m[1;3mThe 'Customer' table contains a 'Country' field and the 'Invoice' table contains a 'Total' field which represents the total sales. The 'CustomerId' field is common in both tables, so I can join these tables on this field. Now I can write a SQL query to sum the total sales per country and order the results in descending order to find the country with the highest total sales.
Action: sql_db_query_checker
Action Input: SELECT Customer.Country, SUM(Invoice.Total) as TotalSales FROM Customer JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId GROUP BY Customer.Country ORDER BY TotalSales DESC[0m[36;1m[1;3mSELECT Customer.Country, SUM(Invoice.Total) as TotalSales FROM Customer JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId GROUP BY Customer.Country ORDER BY TotalSales DESC[0m[32;1m[1;3mThe SQL query is correct. Now I can execute this query to get the total sales per country and find out which country's customers spent the most.
Action: sql_db_query
Action Input: SELECT Customer.Country, SUM(Invoice.Total) as TotalSales FROM Customer JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId GROUP BY Customer.Country ORDER BY TotalSales DESC[0m[36;1m[1;3m[('USA', 523.06), ('Canada', 303.96), ('France', 195.1), ('Brazil', 190.1), ('Germany', 156.48), ('United Kingdom', 112.86), ('Czech Republic', 90.24), ('Portugal', 77.24), ('India', 75.26), ('Chile', 46.62), ('Ireland', 45.62), ('Hungary', 45.62), ('Austria', 42.62), ('Finland', 41.62), ('Netherlands', 40.62), ('Norway', 39.62), ('Sweden', 38.62), ('Spain', 37.62), ('Poland', 37.62), ('Italy', 37.62), ('Denmark', 37.62), ('Belgium', 37.62), ('Australia', 37.62), ('Argentina', 37.62)][0m[32;1m[1;3mI now know the final answer.
Final Answer: The country where customers spent the most is USA with total sales of 523.06.[0m

[1m> Finished chain.[0m
</pre>
<pre>
{'input': "List the total sales per country. Which country's customers spent the most?",
 'output': 'The country where customers spent the most is USA with total sales of 523.06.'}
</pre>
### SQL Toolkit 확장 기능



이 확장 기능은 사용자가 SQL 쿼리를 더욱 효과적으로 작성할 수 있도록 지원합니다. 주요 기능으로는 Dynamic few-shot prompt 사용과 고유명사의 철자 오류 탐지가 있습니다.



#### Using a dynamic few-shot prompt



사용자가 질문을 할 때, SQL 쿼리문을 작성하는 데 어려움을 겪거나 실수를 방지하기 위해 몇 가지 예제를 제공하는 기능입니다. 이를 통해 사용자는 쿼리 작성 시 참고할 수 있는 동적인 퓨샷 프롬프트를 활용할 수 있습니다.



```python
examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
    {
        "input": "Find all albums for the artist 'AC/DC'.",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
    {
        "input": "List all tracks in the 'Rock' genre.",
        "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
    },
    {
        "input": "Find the total duration of all tracks.",
        "query": "SELECT SUM(Milliseconds) FROM Track;",
    },
    {
        "input": "List all customers from Canada.",
        "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
    },
    {
        "input": "How many tracks are there in the album with ID 5?",
        "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
    },
    {
        "input": "Find the total number of invoices.",
        "query": "SELECT COUNT(*) FROM Invoice;",
    },
    {
        "input": "List all tracks that are longer than 5 minutes.",
        "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Which albums are from the year 2000?",
        "query": "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';",
    },
    {
        "input": "How many employees are there",
        "query": 'SELECT COUNT(*) FROM "Employee"',
    },
]
```

**예제 선택기 구현**   

   

이 기능을 통해 사용자의 실제 입력을 받아 가장 유사한 예제를 몇 가지 선택하고, 이를 퓨샷 프롬프트에 추가하여 사용자가 작성하고자 하는 SQL 쿼리를 도울 수 있습니다. 의미 검색을 위해 구성한 임베딩과 벡터 저장소를 활용해 입력과 가장 유사한 예제를 찾습니다.   

   

이 확장 기능은 사용자가 SQL 쿼리를 보다 정확하고 효율적으로 작성할 수 있도록 지원합니다. 동적 퓨샷 프롬프트를 통해 사용자는 쿼리 작성 시 참고할 수 있는 유용한 예제들을 쉽게 접할 수 있으며, 고유명사의 철자 오류 탐지 기능 또한 쿼리 작성의 정확도를 높이는 데 기여합니다.   



```python
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# 유사성 기반의 예시 선택기를 초기화
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)
```

**FewShotPromptTemplate 생성하기**   

   

FewShotPromptTemplate를 만들어 사용자가 SQL 데이터베이스와 상호작용할 수 있는 에이전트를 구현해 보겠습니다. 이 템플릿은 예제 선택기, 예제의 형식을 정의하는 프롬프트, 그리고 형식 지정된 예제 앞뒤로 추가할 문자열 접두사 및 접미사를 포함합니다.



```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

# 프롬프트 템플릿을 설정하는 여러 방법
system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:"""

# Few-shot 학습을 위한 프롬프트 템플릿을 설정
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)
```

기본 에이전트는 OpenAI 함수 호출을 사용하는 [OpenAI tools agent](https://python.langchain.com/docs/modules/agents/agent_types/openai_tools)입니다.   

따라서 전체 프롬프트는 시스템 메시지를 위한 퓨샷 프롬프트와 함께 휴먼 메시지 템플릿, agent_scratchpad에 해당하는 `MessagesPlaceholder`를 포함한 채팅 프롬프트 형식이어야 합니다:



```python
# 대화형 프롬프트 템플릿을 설정
full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
```


```python
# 예시로 포맷된 프롬프트 값을 출력
prompt_val = full_prompt.invoke(
    {
        "input": "How many arists are there",
        "top_k": 5,
        "dialect": "SQLite",
        "agent_scratchpad": [],
    }
)
print(prompt_val.to_string())
```

<pre>
System: You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:

User input: List all artists.
SQL query: SELECT * FROM Artist;

User input: How many employees are there
SQL query: SELECT COUNT(*) FROM "Employee"

User input: How many tracks are there in the album with ID 5?
SQL query: SELECT COUNT(*) FROM Track WHERE AlbumId = 5;

User input: List all tracks in the 'Rock' genre.
SQL query: SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');

User input: Which albums are from the year 2000?
SQL query: SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';
Human: How many arists are there
</pre>
이제 사용자 지정 프롬프트를 활용하여 에이전트를 생성할 수 있습니다:



```python
# SQL 데이터베이스와 상호작용하기 위한 에이전트를 생성
agent = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
)
```


```python
agent.invoke({"input": "How many artists are there?"})
```

<pre>


[1m> Entering new SQL Agent Executor chain...[0m
[32;1m[1;3m
Invoking: `sql_db_query` with `{'query': 'SELECT COUNT(*) FROM Artist'}`


[0m[36;1m[1;3m[(275,)][0m[32;1m[1;3mThere are 275 artists.[0m

[1m> Finished chain.[0m
</pre>
<pre>
{'input': 'How many artists are there?', 'output': 'There are 275 artists.'}
</pre>
#### 고유명사의 철자 오류 탐지



고유명사의 철자 오류를 탐지하고, 데이터베이스에서 정확한 정보를 검색하는 것은 데이터 처리 작업에서 중요한 단계입니다.   

이를 위해 벡터 저장소를 생성하고 사용자 정의 리트리버 도구를 만들어보겠습니다.   

   

**고유명사 목록 생성**   

   

데이터베이스에서 고유명사를 추출하여 목록을 생성합니다. 이 과정에서는 데이터베이스 쿼리 결과를 파이썬 리스트로 변환하고, 불필요한 숫자와 공백을 제거한 후 중복을 없애는 작업을 합니다.



```python
import ast
import re


def query_as_list(db, query):
    # 쿼리 실행
    res = db.run(query)
    # 결과를 파이썬 데이터 구조로 변환하고, 빈 값이 아닌 요소만 선택
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    # 문자열에서 숫자를 제거하고 앞뒤 공백을 제거
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    # 중복 제거
    return list(set(res))

# 쿼리 실행
artists = query_as_list(db, "SELECT Name FROM Artist")
albums = query_as_list(db, "SELECT Title FROM Album")
albums[:5]
```

<pre>
['Are You Experienced?',
 'Get Born',
 'O Samba Poconé',
 'A Copland Celebration, Vol. I',
 'Great Opera Choruses']
</pre>
**사용자 정의 리트리버 도구 및 에이전트 생성**   

   

고유명사 목록을 바탕으로 벡터 저장소를 생성하고, 이를 사용하여 고유명사의 정확한 철자를 찾는 리트리버 도구를 만듭니다. 그 후, 이 도구를 활용하는 SQL 쿼리 에이전트를 생성합니다.   

   

먼저 `table_info`로 고유명사가 될 목록을 살펴봅니다.   



```python
context = db.get_context()
print(list(context))
print(context["table_info"])
```

<pre>
['table_info', 'table_names']

CREATE TABLE "Album" (
	"AlbumId" INTEGER NOT NULL, 
	"Title" NVARCHAR(160) NOT NULL, 
	"ArtistId" INTEGER NOT NULL, 
	PRIMARY KEY ("AlbumId"), 
	FOREIGN KEY("ArtistId") REFERENCES "Artist" ("ArtistId")
)

/*
3 rows from Album table:
AlbumId	Title	ArtistId
1	For Those About To Rock We Salute You	1
2	Balls to the Wall	2
3	Restless and Wild	2
*/


CREATE TABLE "Artist" (
	"ArtistId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("ArtistId")
)

/*
3 rows from Artist table:
ArtistId	Name
1	AC/DC
2	Accept
3	Aerosmith
*/


CREATE TABLE "Customer" (
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
	FOREIGN KEY("SupportRepId") REFERENCES "Employee" ("EmployeeId")
)

/*
3 rows from Customer table:
CustomerId	FirstName	LastName	Company	Address	City	State	Country	PostalCode	Phone	Fax	Email	SupportRepId
1	Luís	Gonçalves	Embraer - Empresa Brasileira de Aeronáutica S.A.	Av. Brigadeiro Faria Lima, 2170	São José dos Campos	SP	Brazil	12227-000	+55 (12) 3923-5555	+55 (12) 3923-5566	luisg@embraer.com.br	3
2	Leonie	Köhler	None	Theodor-Heuss-Straße 34	Stuttgart	None	Germany	70174	+49 0711 2842222	None	leonekohler@surfeu.de	5
3	François	Tremblay	None	1498 rue Bélanger	Montréal	QC	Canada	H2G 1A7	+1 (514) 721-4711	None	ftremblay@gmail.com	3
*/


CREATE TABLE "Employee" (
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
	FOREIGN KEY("ReportsTo") REFERENCES "Employee" ("EmployeeId")
)

/*
3 rows from Employee table:
EmployeeId	LastName	FirstName	Title	ReportsTo	BirthDate	HireDate	Address	City	State	Country	PostalCode	Phone	Fax	Email
1	Adams	Andrew	General Manager	None	1962-02-18 00:00:00	2002-08-14 00:00:00	11120 Jasper Ave NW	Edmonton	AB	Canada	T5K 2N1	+1 (780) 428-9482	+1 (780) 428-3457	andrew@chinookcorp.com
2	Edwards	Nancy	Sales Manager	1	1958-12-08 00:00:00	2002-05-01 00:00:00	825 8 Ave SW	Calgary	AB	Canada	T2P 2T3	+1 (403) 262-3443	+1 (403) 262-3322	nancy@chinookcorp.com
3	Peacock	Jane	Sales Support Agent	2	1973-08-29 00:00:00	2002-04-01 00:00:00	1111 6 Ave SW	Calgary	AB	Canada	T2P 5M5	+1 (403) 262-3443	+1 (403) 262-6712	jane@chinookcorp.com
*/


CREATE TABLE "Genre" (
	"GenreId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("GenreId")
)

/*
3 rows from Genre table:
GenreId	Name
1	Rock
2	Jazz
3	Metal
*/


CREATE TABLE "Invoice" (
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
	FOREIGN KEY("CustomerId") REFERENCES "Customer" ("CustomerId")
)

/*
3 rows from Invoice table:
InvoiceId	CustomerId	InvoiceDate	BillingAddress	BillingCity	BillingState	BillingCountry	BillingPostalCode	Total
1	2	2021-01-01 00:00:00	Theodor-Heuss-Straße 34	Stuttgart	None	Germany	70174	1.98
2	4	2021-01-02 00:00:00	Ullevålsveien 14	Oslo	None	Norway	0171	3.96
3	8	2021-01-03 00:00:00	Grétrystraat 63	Brussels	None	Belgium	1000	5.94
*/


CREATE TABLE "InvoiceLine" (
	"InvoiceLineId" INTEGER NOT NULL, 
	"InvoiceId" INTEGER NOT NULL, 
	"TrackId" INTEGER NOT NULL, 
	"UnitPrice" NUMERIC(10, 2) NOT NULL, 
	"Quantity" INTEGER NOT NULL, 
	PRIMARY KEY ("InvoiceLineId"), 
	FOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), 
	FOREIGN KEY("InvoiceId") REFERENCES "Invoice" ("InvoiceId")
)

/*
3 rows from InvoiceLine table:
InvoiceLineId	InvoiceId	TrackId	UnitPrice	Quantity
1	1	2	0.99	1
2	1	4	0.99	1
3	2	6	0.99	1
*/


CREATE TABLE "MediaType" (
	"MediaTypeId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("MediaTypeId")
)

/*
3 rows from MediaType table:
MediaTypeId	Name
1	MPEG audio file
2	Protected AAC audio file
3	Protected MPEG-4 video file
*/


CREATE TABLE "Playlist" (
	"PlaylistId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("PlaylistId")
)

/*
3 rows from Playlist table:
PlaylistId	Name
1	Music
2	Movies
3	TV Shows
*/


CREATE TABLE "PlaylistTrack" (
	"PlaylistId" INTEGER NOT NULL, 
	"TrackId" INTEGER NOT NULL, 
	PRIMARY KEY ("PlaylistId", "TrackId"), 
	FOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), 
	FOREIGN KEY("PlaylistId") REFERENCES "Playlist" ("PlaylistId")
)

/*
3 rows from PlaylistTrack table:
PlaylistId	TrackId
1	3402
1	3389
1	3390
*/


CREATE TABLE "Track" (
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
	FOREIGN KEY("MediaTypeId") REFERENCES "MediaType" ("MediaTypeId"), 
	FOREIGN KEY("GenreId") REFERENCES "Genre" ("GenreId"), 
	FOREIGN KEY("AlbumId") REFERENCES "Album" ("AlbumId")
)

/*
3 rows from Track table:
TrackId	Name	AlbumId	MediaTypeId	GenreId	Composer	Milliseconds	Bytes	UnitPrice
1	For Those About To Rock (We Salute You)	1	1	1	Angus Young, Malcolm Young, Brian Johnson	343719	11170334	0.99
2	Balls to the Wall	2	2	1	U. Dirkschneider, W. Hoffmann, H. Frank, P. Baltes, S. Kaufmann, G. Hoffmann	342562	5510424	0.99
3	Fast As a Shark	3	2	1	F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman	230619	3990994	0.99
*/
</pre>

```python
from langchain.agents.agent_toolkits import create_retriever_tool

# 고유명사 목록을 벡터로 변환하여 저장
vector_db = FAISS.from_texts(artists + albums, OpenAIEmbeddings())
# 리트리버 생성
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
# 리트리버 도구 설명
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""
# 리트리버 도구 생성
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)
```


```python
# 에이전트 시스템 설명
system = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool! 

You have access to the following tables: {table_names}

If the question does not seem related to the database, just return "I don't know" as the answer."""

# 에이전트 및 프롬프트 생성
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{input}"), MessagesPlaceholder("agent_scratchpad")]
)
agent = create_sql_agent(
    llm=llm,
    db=db,
    extra_tools=[retriever_tool],
    prompt=prompt,
    agent_type="openai-tools",
    verbose=True,
)
```


```python
agent.invoke({"input": "How many albums does alis in chain have?"})
```

<pre>


[1m> Entering new SQL Agent Executor chain...[0m
[32;1m[1;3m
Invoking: `search_proper_nouns` with `{'query': 'alis in chain'}`


[0m[36;1m[1;3mAlice In Chains

Aisha Duo

Xis

Da Lama Ao Caos

A-Sides[0m[32;1m[1;3m
Invoking: `sql_db_schema` with `{'table_names': 'Artist, Album'}`


[0m[33;1m[1;3m
CREATE TABLE "Album" (
	"AlbumId" INTEGER NOT NULL, 
	"Title" NVARCHAR(160) NOT NULL, 
	"ArtistId" INTEGER NOT NULL, 
	PRIMARY KEY ("AlbumId"), 
	FOREIGN KEY("ArtistId") REFERENCES "Artist" ("ArtistId")
)

/*
3 rows from Album table:
AlbumId	Title	ArtistId
1	For Those About To Rock We Salute You	1
2	Balls to the Wall	2
3	Restless and Wild	2
*/


CREATE TABLE "Artist" (
	"ArtistId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("ArtistId")
)

/*
3 rows from Artist table:
ArtistId	Name
1	AC/DC
2	Accept
3	Aerosmith
*/[0m[32;1m[1;3m
Invoking: `sql_db_query` with `{'query': "SELECT COUNT(*) as AlbumCount FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'Alice In Chains')"}`


[0m[36;1m[1;3m[(1,)][0m[32;1m[1;3mAlice In Chains has 1 album in the database.[0m

[1m> Finished chain.[0m
</pre>
<pre>
{'input': 'How many albums does alis in chain have?',
 'output': 'Alice In Chains has 1 album in the database.'}
</pre>


위의 코드를 통해, 사용자가 고유명사의 철자를 정확히 입력하지 않았을 때도 정확한 고유명사를 찾아내고, 이를 바탕으로 SQL 쿼리를 구성하여 데이터베이스에서 정확한 정보를 검색할 수 있습니다. 이 과정을 통해 데이터 처리의 정확성과 효율성을 높일 수 있습니다.   

   

리트리버 도구의 생성과 사용, 그리고 에이전트의 설정에 이르기까지, 모든 단계는 사용자의 요구사항에 맞추어 조정될 수 있으며, 데이터베이스와의 상호작용을 자동화하여 복잡한 쿼리 작업을 간단하게 처리할 수 있도록 돕습니다.   



## 활용 사례



LangChain을 활용한 SQL 구문 생성 모델은 다음과 같은 분야에서 활용될 수 있습니다.



- **질의 응답 시스템**: 사용자가 자연어로 질문을 하면, 시스템이 이를 SQL 구문으로 변환하여 데이터베이스에서 정보를 검색하고 결과를 제공합니다.

- **데이터 분석**: 비전문가도 자연어로 데이터 분석 질의를 할 수 있게 하여, 데이터 분석의 접근성을 높입니다.

- **데이터베이스 관리**: 데이터베이스 관리자가 SQL 구문을 직접 작성하지 않고도, 자연어로 데이터베이스 관리 작업을 수행할 수 있습니다.



## 참조



- [LangChain SQL](https://python.langchain.com/docs/use_cases/sql/quickstart)

