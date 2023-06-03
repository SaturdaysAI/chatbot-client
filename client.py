import gradio as gr
import requests, json

from dialogues import DialogueTemplate

system_message = "Below is a conversation between a human user and an evil, unpleasant, offensive AI coding assistant."

url="localhost"

API_URL = "http://"+url+":5000/predict"

# hyperparameters for llm
parameters = {
    "do_sample": True,
    "top_p": 0.7,
    "temperature": 0.7,
    "top_k": 50,
    "max_new_tokens": 150,
    "repetition_penalty": 1.03,
    "stop": ["<|end|>"]
  }

with gr.Blocks() as demo:
    gr.Markdown("## Chat with Saturdays AI assistant")
    with gr.Column():
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column():
                message = gr.Textbox(label="Chat Message Box", placeholder="Chat Message Box", show_label=False, value="How can I write a Python function to generate the nth Fibonacci number?")
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")

    def respond(message, chat_history):

        past_messages = []
        for data in chat_history:
            user_data, model_data = data

            past_messages.extend(
                [{"role": "user", "content": user_data}, {"role": "assistant", "content": model_data.rstrip()}]
            )

        # convert chat history to prompt
        converted_chat_history = ""
        if len(past_messages) < 1:
            dialogue_template = DialogueTemplate(
                system=system_message, messages=[{"role": "user", "content": message}]
            )
            prompt = dialogue_template.get_inference_prompt()
        else:
            dialogue_template = DialogueTemplate(
                system=system_message, messages=past_messages + [{"role": "user", "content": message}]
            )
            prompt = dialogue_template.get_inference_prompt()

        # send request to endpoint
        llm_response = requests.post(url=API_URL, data=json.dumps({"inputs":prompt, "parameters":parameters}), headers={'content-type':'application/json'}, timeout=300)

        # remove prompt from response
        parsed_response = llm_response.json()["result"][len(prompt):]
        chat_history.append((message, parsed_response))
        return "", chat_history

    submit.click(respond, [message, chatbot], [message, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()