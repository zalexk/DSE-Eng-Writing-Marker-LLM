import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from src import agent
import os

# Power by DeepSeek R1

# 定义 PromptTemplate
common_prompt_template = PromptTemplate(
    input_variables=["system_prompt", "user_prompt"],
    template="""
'system_prompt' : {system_prompt}
'user_prompt' : {user_prompt}
"""
)

# 初始化组件
def init_components():
    if 'corrections' not in st.session_state:
        st.session_state.corrections = {}

# LLM处理模块
def process_with_llm(prompt_template, input_vars):
    llm = ChatOpenAI(
        openai_api_key=st.secrets["LLM_API_KEY"], 
        model=st.secrets["Model_NAME"],
        temperature=0.5,
        base_url=st.secrets["BASE_URL"]
    )  
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(input_vars)

# 界面布局
def main():
    st.set_page_config(
        page_title="HKDSE English Writing AI Marker",
        page_icon="👋",
        initial_sidebar_state="collapsed",
        layout="wide"
    )
    
    st.title("HKDSE English Writing AI Marker")
    
    init_components()
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("input_form"):
            question = st.text_area("Question", height=100)
            content = st.text_area("Your Article:", height=500)
            submitted = st.form_submit_button("Submit")
            
    if submitted:     
        with col2:
            progress_bar = st.progress(0)
            
            with open("prompts/Grammar Correction.txt", "r") as g:
                grammar_prompt = g.read()
            
            with open("prompts/Vocab Suggestion.txt", "r") as v:
                vocab_prompt = v.read()
            
            with open("prompts/Evaluation.txt", "r") as e:
                eval_prompt = e.read()
            
            with open("prompts/Rewrite.txt", "r") as r:
                rewrite_prompt = r.read()

            # 构建user_prompt内容
            user_prompt_content = f"question: {question}\ncontent: {content}"
            
            # Grammar Correction
            st.session_state.corrections['grammar'] = process_with_llm(
                common_prompt_template,
                {'system_prompt': grammar_prompt, 'user_prompt': user_prompt_content}
            )
            progress_bar.progress(25, "LLM is checking ...")
            
            # Vocab Suggestion
            st.session_state.corrections['vocab'] = process_with_llm(
                common_prompt_template,
                {'system_prompt': vocab_prompt, 'user_prompt': content}
            )
            progress_bar.progress(50, "LLM is answering ...")
            
            # Evaluation
            st.session_state.corrections['evaluation'] = process_with_llm(
                common_prompt_template,
                {'system_prompt': eval_prompt, 'user_prompt': user_prompt_content}
            )
            progress_bar.progress(75, "LLM is evaluating ...")

            # Rewrite
            st.session_state.corrections['rewrite'] = process_with_llm(
                common_prompt_template,
                {'system_prompt': rewrite_prompt, 'user_prompt': user_prompt_content}
            )
            progress_bar.progress(100, "LLM is writing the response ...")
            
            # 显示结果
            with st.expander("Grammar Correction", expanded=True):
                st.markdown(st.session_state.corrections['grammar'])
                
            with st.expander("Vocab/Sentence Suggestions", expanded=True):
                st.markdown(st.session_state.corrections['vocab'])
            
            with st.expander("Evaluation", expanded=True):
                st.markdown(st.session_state.corrections['evaluation'])

            with st.expander("Rewritten Writing", expanded=True):
                st.markdown(st.session_state.corrections['rewrite'])
            
            # 导出数据
            data_content = f"Grammar Corrections:\n{st.session_state.corrections['grammar']}\n\nVocabulary Suggestions:\n{st.session_state.corrections['vocab']}\n\nEvaluation:\n{st.session_state.corrections['evaluation']}"
            st.download_button(
                label="Export as text",
                data=data_content,
                file_name="corrections.txt"
            )

if __name__ == "__main__":
    main()