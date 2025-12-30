import streamlit as st
import pandas as pd
import numpy as np
import openai

st.set_page_config(
    page_title="LLM 기반 고객 데이터 EDA 리포트",
    layout="wide"
)

st.title("고객 데이터 자동 EDA 해설 리포트")
st.caption("통계 계산은 Python, 해석은 LLM이 담당합니다.")
st.caption("수치형, 범주형 컬럼들은 각각 최대 3개씩만 분석합니다.")

# CSV 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is None:
    st.info("데이터 CSV 파일을 업로드하세요.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ID/PK 추정 컬럼 자동 제거 로직
find_pk = []
for col in df.columns:
    # 컬럼명에 id, no, code, pk 등이 포함되어 있고, 모든 값이 고유(Unique)한 경우 PK로 간주
    if any(k in col.lower() for k in ['id', 'no', 'code', 'pk']) and df[col].nunique() == len(df):
        find_pk.append(col)

if find_pk:
    df = df.drop(columns=find_pk)
    st.warning(f"PK(ID)로 추정되는 컬럼을 자동으로 제거했습니다. 대상 컬럼 : {', '.join(find_pk)}")

st.success("데이터 로드 완료")

st.divider()

st.subheader("결측치 처리")


if st.button("결측치 제거"):
    df = df.dropna(axis=0)
    st.success("결측치 제거 완료")

st.divider()

# 데이터 개요
st.subheader("데이터 개요")


st.write(f"- 행 수: {df.shape[0]}")
st.write(f"- 열 수: {df.shape[1]}")
st.dataframe(df.head())

# 컬럼 유형 분리
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# 수치형 EDA 요약 텍스트 생성
numeric_report = "수치형 데이터 분석:\n"

for col in numeric_cols[:3]:
    c = df[col]
    numeric_report += f"""
- 컬럼: {col}
  평균: {c.mean():.2f}
  중앙값: {c.median():.2f}
  표준편차: {c.std():.2f}
  최소값: {c.min():.2f}
  최대값: {c.max():.2f}
"""

# 범주형 EDA 요약 텍스트 생성
categorical_report = "Categorical Columns Analysis:\n"

for col in categorical_cols[:3]:
    vc = df[col].value_counts(dropna=False)
    ratio = (vc / len(df)) * 100

    categorical_report += f"\n- 컬럼: {col}\n"
    categorical_report += f"  결측치: {df[col].isna().sum()}\n"

    for k in vc.index[:5]:
        categorical_report += f"  {k}: {vc[k]} ({ratio[k]:.2f}%)\n"

# 리포트 문체·역할 고정 프롬프트
REPORT_PROMPT = """
당신은 데이터 분석 보고서를 작성하는 분석 보조자입니다.

규칙:
1. 계산을 수행하지 마십시오.
2. 제공된 통계 수치만 해석하십시오.
3. 단정적인 표현을 사용하지 마십시오.
4. 불확실한 경우 반드시 '추측입니다' 또는 '확실하지 않음'을 명시하십시오.
5. 문체는 한국어 데이터 분석 보고서 스타일을 유지하십시오.
6. 개인적인 의견이나 조언은 포함하지 마십시오.

문체 예시:
- "~로 해석될 수 있습니다"
- "~한 경향이 관찰됩니다"
- "~일 가능성이 있습니다 (추측입니다)"
"""

final_eda_text = f"""
{REPORT_PROMPT}

[Dataset]
고객 데이터셋

{numeric_report}

{categorical_report}
"""


# OpenAI 호출 함수
def generate_eda_report(prompt_text):
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        return "OPENAI_API_KEY가 Streamlit Secrets에 설정되지 않았습니다."

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": REPORT_PROMPT
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        temperature=0.2,
        max_tokens=700
    )

    return response.choices[0].message.content


# LLM 입력 내용 미리보기
with st.expander("LLM에 전달되는 EDA 요약 텍스트"):
    st.text(final_eda_text)


# 리포트 생성 버튼
st.divider()

st.subheader("EDA 해설 리포트")

if "final_report" not in st.session_state:
    st.session_state.final_report = None

if st.button("전체 EDA 리포트 생성"):
    with st.spinner("EDA 리포트를 생성 중입니다..."):
        try:
            st.session_state.final_report = generate_eda_report(final_eda_text)
        except Exception as e:
            st.session_state.final_report = f"리포트 생성 중 오류 발생: {e}"


# 11. 리포트 출력
with st.container(height=500):
    if st.session_state.final_report:
        st.markdown(st.session_state.final_report)
    else:
        st.markdown("EDA 리포트 공간입니다.")


# 12. 분석 기준 명시
with st.expander("리포트 작성 기준"):
    st.markdown("""
- 본 리포트는 자동 생성된 EDA 해설입니다.
- 모든 통계 계산은 Python에서 수행됩니다.
- LLM은 해석 역할만 수행하며 단정적 결론을 배제했습니다.
""")
