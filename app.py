import os
import gradio as gr
from agent_module import run_deep_research, RAW_MODEL_NAME, _extract_text

# Custom CSS for high-end styling (Glassmorphism, Indigo/Violet gradients, Custom typography)
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Apply Outfit font globally */
body, .gradio-container, button, input, textarea, select {
    font-family: 'Outfit', sans-serif !important;
}

/* Glassmorphism sidebar & container */
.glass-panel {
    background: rgba(15, 23, 42, 0.4) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}

/* Page Title with Gradient Text */
.gradient-title {
    background: linear-gradient(135deg, #c084fc 0%, #6366f1 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    letter-spacing: -0.03em !important;
    text-align: center;
    margin-bottom: 5px !important;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 25px;
    font-weight: 400;
}

/* Glow effects and styling for inputs */
input[type="text"], input[type="password"], textarea {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    transition: all 0.3s ease !important;
}
input[type="text"]:focus, input[type="password"]:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.3) !important;
}

/* Style the chatbot panel */
.chatbot-container {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    overflow: hidden;
}

/* Custom quick start suggestion chips */
.suggestion-chip {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
    padding: 6px 15px !important;
    transition: all 0.2s ease !important;
    cursor: pointer;
    text-align: left;
}
.suggestion-chip:hover {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}
"""

async def predict(message: str, history: list, api_key: str):
    """
    Asynchronous generator that runs deep research and streams progress logs,
    and then outputs the final synthesized report.
    """
    query_str = _extract_text(message)
    async for update in run_deep_research(query_str, api_key):
        yield update

# Build a premium dark theme
theme = gr.themes.Default(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_800",
    button_primary_background_fill="linear-gradient(135deg, #a855f7 0%, #6366f1 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #c084fc 0%, #818cf8 100%)",
    button_primary_text_color="#ffffff",
)

with gr.Blocks() as demo:
    
    # Custom Header
    gr.HTML("<h1 class='gradient-title'>🔍 Deep Research Agent</h1>")
    gr.HTML("<p class='subtitle'>Autonomous multi-step research assistant powered by Pydantic AI & Google Gemini</p>")
    
    with gr.Row():
        # Left Sidebar (Configurations and Tips)
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("### ⚙️ Configurations")
            
            # API Key Input
            api_key_input = gr.Textbox(
                label="Gemini API Key",
                placeholder="Paste key (leaves env active if empty)...",
                type="password",
                info="If not set in your .env file, you can enter it here."
            )
            
            gr.Markdown(f"""
            ### 🤖 Model Info
            - **Provider:** Google AI Studio
            - **Default Model:** `{RAW_MODEL_NAME}`
            - **Methodology:** Multi-turn intent detection, parallel Google searches, document text fetches, source fact extraction, and report synthesis.
            """)
            
            gr.Markdown("### 💡 Quick Prompt Suggestions")
            
            # Suggestion Chips (Pre-defined prompts for testing)
            sug1 = gr.Button("NVDA", elem_classes=["suggestion-chip"])
            sug2 = gr.Button("Explain TSMC's position in the global semiconductor supply chain.", elem_classes=["suggestion-chip"])
            sug3 = gr.Button("Latest developments and clinical trial results of cancer immunotherapies.", elem_classes=["suggestion-chip"])
            
        # Right Chat Area
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
            
            # Build the ChatInterface
            chat_interface = gr.ChatInterface(
                fn=predict,
                additional_inputs=[api_key_input],
                chatbot=gr.Chatbot(
                    height=600,
                    elem_classes=["chatbot-container"],
                )
            )

    # Suggestion click handlers (loads prompt into textbox and submits)
    def load_prompt(prompt):
        return prompt

    sug1.click(load_prompt, inputs=[sug1], outputs=[chat_interface.textbox])
    sug2.click(load_prompt, inputs=[sug2], outputs=[chat_interface.textbox])
    sug3.click(load_prompt, inputs=[sug3], outputs=[chat_interface.textbox])

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=theme, css=custom_css)
