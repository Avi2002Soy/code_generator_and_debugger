import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

@st.cache_resource
def load_model(model_name="codellama/CodeLlama-7b-Instruct-hf"):
    try:
        from accelerate import Accelerator  # Just to check installation
    except ImportError:
        raise ImportError("🚨 Please install 'accelerate>=0.26.0' using: pip install 'accelerate>=0.26.0'")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def call_local_llm(prompt, pipe, code_type="python", special_instructions="Implement a script in a single code block to perform this task: "):
    full_prompt = f"{special_instructions}{prompt}"
    result = pipe(full_prompt, max_new_tokens=512, do_sample=True, temperature=0.7)[0]['generated_text']
    
    # Extract code block
    start_tag = f"```{code_type}\n"
    start_index = result.find(start_tag)
    end_index = result.find("\n```", start_index + len(start_tag))

    if start_index != -1 and end_index != -1:
        return result[start_index + len(start_tag):end_index]
    return result

def save_code(code, code_type="python"):
    if not os.path.exists('./scripts'):
        os.makedirs('./scripts')
    ext = 'py' if code_type == 'python' else code_type
    file_path = f"./scripts/generated_script.{ext}"
    with open(file_path, 'w') as f:
        f.write(code)
    return file_path

# Streamlit UI
def main():
    st.set_page_config(page_title="Code Generator with LLM", layout="wide")
    st.title("💻 Code Generator using Code LLaMA")

    pipe = load_model()

    with st.form("code_form"):
        prompt = st.text_area("📝 Enter your task prompt", height=150)
        code_type = st.selectbox("Select code type", ["python", "html", "latex", "javascript"], index=0)
        submit = st.form_submit_button("Generate Code")

    if submit and prompt.strip():
        with st.spinner("Generating code..."):
            code = call_local_llm(prompt, pipe, code_type)
        st.success("✅ Code generated successfully!")

        st.code(code, language=code_type)

        file_path = save_code(code, code_type)
        with open(file_path, "rb") as file:
            st.download_button(label="📥 Download Code", data=file, file_name=os.path.basename(file_path), mime="text/plain")

if __name__ == "__main__":
    main()
