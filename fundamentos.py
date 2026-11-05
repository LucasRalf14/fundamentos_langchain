from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

prompt_template_str = """
Sua tarefa é explicar o conceito de {input_text} para mim de forma:

1. Clara e intuitiva
2. Concisa, em menos de 100 palavras
3. Adaptada ao que eu já conheço

Use estas informações para personalizar a explicação:
- Interesses profissionais: engenharia de dados, engenharia de IA,
  automações e sistemas de segurança

Personalize de forma sutil e natural. Evite referências forçadas que
não ajudem a compreender o conceito.
"""

prompt_template = PromptTemplate.from_template(prompt_template_str)
model = init_chat_model("gpt-6-luna", model_provider="openai")

def generate_explanation(input_text):
    prompt = prompt_template.format(input_text=input_text)
    response = model.invoke(prompt)
    return response.text
  
import gradio as gr

demo = gr.Interface(
    fn=generate_explanation,
    inputs=[gr.Textbox(label="Input Text", lines=1)],
    outputs=[gr.Textbox(label="Response", lines=5)],
    flagging_mode="never",
    title="LLM Demo",
    description="This is a demo of a simple LLM application"
)

# demo.launch()