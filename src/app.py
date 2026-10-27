import gradio as gr
from rag_pipeline import initialize_rag

query_engine = initialize_rag()


def ask_research_assistant(question: str) -> str:
    try:
        response = query_engine.query(question)
    except Exception as e:
        return f"Error: {e}"
    sources = "\n".join(
        f"- {node.metadata.get('file_name', 'unknown')} (similarity {node.score:.2f})"
        for node in response.source_nodes
    )
    return f"**Answer:**\n{response}\n\n**Sources:**\n{sources}"


examples = [
    "What is the main contribution of this paper?",
    "Summarize the methodology section.",
    "What datasets were used in this study?"
]

interface = gr.Interface(
    fn=ask_research_assistant,
    inputs=gr.Textbox(label="Research Question", placeholder="Enter your query..."),
    outputs=gr.Markdown(label="Answer"),
    title="Research Assistant",
    description="Ask questions about your own collection of papers. Everything runs locally.",
    examples=examples,
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", share=False)
