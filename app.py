import os
import gradio as gr
from gradio import ChatMessage
import time
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from utils import format_chat_history, stream_gemini_response, user_message




with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="slate", neutral_hue="neutral"),
    css="""
        .chatbot-wrapper .message {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    """
) as demo:
    gr.Markdown("# 💭 PharmAI: Inference-based Expert Pharmacology AI Service 💭")
    # gr.HTML("""<a href="https://visitorbadge.io/status?path=https%3A%2F%2Faiqcamp-Gemini2-Flash-Thinking.hf.space">
    #            <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Faiqcamp-Gemini2-Flash-Thinking.hf.space&countColor=%23263759" />
    #            </a>""")

    with gr.Tabs() as tabs:
        with gr.TabItem("Expert", id="chat_tab"): 
            chatbot = gr.Chatbot(
                type="messages",
                label="PharmAI Chatbot (Streaming Output)",
                render_markdown=True,
                scale=1,
                avatar_images=(None, "https://lh3.googleusercontent.com/oxz0sUBF0iYoN4VvhqWTmux-cxfD1rxuYkuFEfm1SFaseXEsjjE4Je_C_V3UQPuJ87sImQK3HfQ3RXiaRnQetjaZbjJJUkiPL5jFJ1WRl5FKJZYibUA=w214-h214-n-nu"),
                elem_classes="chatbot-wrapper"
            )

            with gr.Row(equal_height=True):
                input_box = gr.Textbox(
                    lines=1,
                    label= "Conversation message",
                    placeholder="Enter your message here...",
                    scale=4
                )
                clear_button = gr.Button("Reset conversation", scale=1)

            example_prompts = [
                ["Describe the interaction between CYP450 enzymes and drug metabolism, with particular emphasis on how enzyme induction or inhibition can affect the therapeutic efficacy of drugs such as warfarin."],
                ["Provide a detailed analysis of the pharmacokinetic and pharmacodynamic properties of erythropoietin agents used for the treatment of anemia in patients with chronic kidney disease, and explain the factors that influence dosing and dosing interval decisions."],
                ["Extract natural plants for new drug development aimed at treating liver cirrhosis (reversing liver fibrosis), and from the perspective of traditional Korean medicine (Hanbang), provide an optimal response by reasoning through their specific pharmacological mechanisms, the rationale behind them, and how they should be combined to achieve the best effect."],
                ["Explain and provide information on natural plant substances effective in treating Alzheimer's disease, including their pharmacological mechanisms, from the perspective of traditional Korean medicine (Hanbang)."],
                ["Explain and provide information on highly promising natural plant substances and their pharmacological mechanisms for the development of new drugs effective in treating and relieving symptoms of hypertension, from the perspective of traditional Korean medicine (Hanbang)."],
                ["Compare and contrast the mechanisms of action of ACE inhibitors and ARBs in hypertension management, taking into account their effects on the renin-angiotensin-aldosterone system (RAAS)."],
                ["Describe the pathophysiology of type 2 diabetes and explain how metformin achieves its glucose-lowering effect, including key considerations for patients with renal impairment."],
                ["Discuss the mechanism of action and clinical significance of beta-blockers in the treatment of heart failure, and refer to the specific beta-receptor subtypes and their effects on the cardiovascular system."],
                ["Explain the pathophysiological mechanisms of Alzheimer's disease and describe the main targets of currently used drugs. Specifically, compare and analyze the mechanisms of action and clinical significance of acetylcholinesterase inhibitors and NMDA receptor antagonists."],
                ["Please explain the FDA-approved treatments for liver cirrhosis and their mechanisms of action.""Tell me about the FDA-approved treatments for hypertension."]
            ]

            gr.Examples(
                examples=example_prompts,
                inputs=input_box,
                label="Example: Try using the following prompt to view Gemini's thoughts!",
                examples_per_page=3
            )

            msg_store = gr.State("") 
            input_box.submit(
                lambda msg: (msg, msg, ""), 
                inputs=[input_box],
                outputs=[msg_store, input_box, input_box],
                queue=False
            ).then(
                user_message,
                inputs=[msg_store, chatbot],
                outputs=[input_box, chatbot],
                queue=False
            ).then(
                stream_gemini_response,
                inputs=[msg_store, chatbot],
                outputs=chatbot,
                queue=True
            )

            clear_button.click(
                lambda: ([], "", ""),
                outputs=[chatbot, input_box, msg_store],
                queue=False
            )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
