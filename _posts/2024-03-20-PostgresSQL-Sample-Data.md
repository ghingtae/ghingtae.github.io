---
layout: single
title:  "PostgreSQL: Quick Start - Load Sample Database"
categories: SQL
tag: [coding, PostgreSQL, SQL]
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

## Load the sample database
PostgreSQL에 샘플 데이터베이스를 로드하는 방법에 대해서 알아보도록 하겠습니다.

1. pgAdmin을 시작합니다.

2. postgres 사용자의 비밀번호를 입력합니다.

3. PostgreSQL 12 마우스 오른쪽 버튼으로 클릭하고 **Create > Database**를 선택하여 새 데이터베이스를 생성하기 위한 대화 상자를 엽니다.   
![Restore-Sample-Database-Step-1](/assets/images/Restore-Sample-Database-Step-1.png)

4. **Database**에 dvdrental, **Owner**에 postgres를 입력하고 **Save** 버튼을 클릭하여 dvdrental 데이터베이스를 생성합니다.   
![Restore-Sample-Database-Step-2](/assets/images/Restore-Sample-Database-Step-2.png)

5. [샘플 데이터베이스를 다운로드](https://www.postgresqltutorial.com/postgresql-getting-started/postgresql-sample-database/) 하고 압축을 풉니다. 많은 파일이 포함된 디렉토리를 얻게 됩니다.   
![DVD-Rental](/assets/images/DVD%20Rental.png)   
다이어그램에서 필드 앞에 나타나는 별표(*)는 기본 키를 나타냅니다.

7. **dvdrental** 데이터베이스를 마우스 오른쪽 버튼으로 클릭하고 **Restore** 메뉴 항목을 선택합니다.   
![Restore-Sample-Database-Step-3](/assets/images/Restore-Sample-Database-Step-3.png)

8. **Format**에 Directory 선택 후, **Filename**을 샘플 데이터베이스가 포함된 위치로 해줍니다. 마지막으로 **Role name**을 postgres로 해주고 **Restore** 버튼을 클릭합니다.   
![Sample-Database-Step(1)](/assets/images/Sample%20Database%20Step.png)

9. 샘플 데이터베이스를 복원하는데 몇 초 정도 걸리며, 복원이 완료되면 다음과 같은 알람이 표시됩니다.   
![Sample-Database-Step(2)](/assets/images/Sample%20Database%20Step%20(1).png)   
이것은 샘플 데이터베이스를 성공적으로 생성하고 다운로드한 파일에서 복원 했음을 의미합니다.

---

이 튜토리얼에서는 PostgreSQL에서 샘플 데이터베이스를 복원하는 방법에 대해서 알아보았습니다.   
다음 튜토리얼에서는 샘플 데이터베이스를 사용하여 PostgreSQL의 다양한 기능을 실습할 것입니다.
* QUERYING DATA
* FILTERING DATA
* JOIN TABLES
* GROUP DATA
* SET OPERATIONS    
...
위와 같은 내용을 실습할 것입니다.

## References

* [POSTGRESQL TUTORIAL](https://www.postgresqltutorial.com/postgresql-getting-started/install-postgresql-macos/)

* [Sample Database](https://www.postgresqltutorial.com/postgresql-getting-started/postgresql-sample-database/)

* [Download PostgreSQL](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)
