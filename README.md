# Simple GraphRAG LLM

## 프로젝트 소개

이 프로젝트는 텍스트 chunks에서 entity와 relationship을 추출하여 Neo4j 데이터베이스에 지식 그래프(Knowledge Graph)를 구축하는 간단한 GraphRAG 시스템입니다. 그리고 지식 그래프(Knowledge Graph)를 구축하고 Prompt Engineering으로 적절한 prompt을 LLM에 주어서 Cypher Query 생성을 실험합니다.

주요 기능은 다음과 같습니다:
- LLM을 활용한 텍스트 기반 엔티티 및 관계 추출
- Neo4j를 사용한 지식 그래프 구축 및 저장
- 텍스트 청크를 순차적으로 연결 (`FIRST_CHUNK`, `NEXT_CHUNK` 관계)
- 추출된 엔티티를 원본 청크와 연결 (`HAS_ENTITY` 관계)
- incremental 데이터 추가 지원
- 엔티티에 대한 전체 텍스트 검색(full-text search) 인덱스 생성

## 요구 사항

- Python 3.x
- Docker
- OpenAI 호환 API 키 및 엔드포인트

## 설치 및 설정

1.  **환경 변수 설정:**
    프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래 내용을 채워주세요.

    ```env
    # Neo4j Database
    NEO4J_URL="bolt://localhost:7687"
    NEO4J_USERNAME="neo4j"
    NEO4J_PASSWORD="password"

    # OpenAI-compatible LLM
    OPENAI_MODEL="o4-mini"
    OPENAI_API_KEY="your-api-key"
    OPENAI_URL="your-openai-compatible-api-base-url" # (Optional)
    ```

2.  **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```

## 실행 방법

1.  **Neo4j 데이터베이스 시작:**
    Docker를 사용하여 Neo4j 데이터베이스 컨테이너를 실행합니다. `apoc` 플러그인이 포함되어야 합니다.

    ```bash
    docker run -d --rm \
        -e NEO4J_AUTH=neo4j/password \
        -e NEO4J_PLUGINS='["apoc"]' \
        -p 7474:7474 -p 7687:7687 \
        --name graphrag_db \
        neo4j:5.24
    ```

2.  **샘플 그래프 생성:**
    `src/extract_server.py`를 실행하여 데이터베이스를 초기화하고 샘플 데이터를 기반으로 지식 그래프를 구축합니다.
    ```bash
    python src/extract_server.py
    ```

3.  **그래프 쿼리 실습:**
    그래프 생성이 완료된 후, 다음 스크립트를 통해 그래프를 쿼리하는 방법을 실습할 수 있습니다.
    - `retrieval_server1.py` 으로 기본적인 검색 실습
    - `retrieval_server2.py` 으로 노드 및 관계 유형을 동적으로 파악하여 더 정확한 쿼리를 생성하는 심화 실습
