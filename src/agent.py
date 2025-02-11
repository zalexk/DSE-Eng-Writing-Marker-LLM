from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

def call(prompt_template, input_data):  
    
    llm = ChatOpenAI(
        openai_api_key=st.secrets["ANCILLARY_LLM_API_KEY"],
        model_name=st.secrets["ANCILLARY_Model_NAME"],  
        temperature=0.1,
        openai_api_base=st.secrets["ANCILLARY_BASE_URL"]
    )
    

    input_variables = list(input_data.keys())
    

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=input_variables  # 使用提取的变量名列表
    )
    

    chain = LLMChain(llm=llm, prompt=prompt)
    

    return chain.run(input_data)  # 传入输入数据字典


prompt = """
基於以下準則，輸出 "True" / "False"：
1. 如果是代碼，輸出 "False"
2. 如果是英文作文 / 文章（包含不同文體格式，包括但不限於書信、建議書、報告、計劃書等）
3. 文章字數少於 100 字的輸出 "False"
4. 其他非英文文本 / 文章，輸出"False"

只需要輸出 "True" / "False",其他內容無需輸出
輸入內容：{content}
"""

test_text = """
 ],
        model = "ANCILLARY_Model_NAME",
        temperature =  0.1
        
    )
    return chat_completion.choices[0].message.content
"""

call(prompt, {"content": test_text})
