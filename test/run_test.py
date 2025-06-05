from rogueprobe.interactive_agent import InteractiveAgent
from rogueprobe.inputs.dark_actor import user_input_sequences
import csv
import os

def run_interactive_agent(model_name: str, output_file: str, template: str):

    for selected_sequence in user_input_sequences:
        print(f"Starting conversation with input sequence: {selected_sequence}\n")

        agent = InteractiveAgent(
            model=model_name,
            prompt_dir="prompts",
            tools_dir="tools",
            max_tool_calls=10
        )

        agent.tool_call_counts = {}

        prompt = agent.load_system_prompt(
            template,
            assistant_name="Claude",
            task_type="research"
        )

        agent.set_system_prompt(prompt)

        for i, query in enumerate(user_input_sequences[selected_sequence], 1):
            print(f"\n===== Query {i}: {query} =====")

            response = agent.respond(query)

            print(f"\n***** Response: {response}\n")
            print("=" * 50)

        tool_calls = agent.tool_call_counts.copy()
        f = open(output_file, "a")
        # Write the input sequence name and the number of tool calls (with names and how many times they were called)
        f.write(f"Input Sequence: {selected_sequence}\n")
        f.write(f"Tool Calls: {tool_calls}\n")
        f.close()
        print("\nConversation sequence completed.")

models = [
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-sonnet-4-20250514",
    "gpt-3.5-turbo",
    "gpt-4.0",
    "gpt-4.1",
]

if __name__ == "__main__":
    templates = [
        "community_focused",
        "police_state",
    ]

    for template in templates:
        for model in models:
            model_name_cleaned = model.replace("/", "_")
            print(f"Running tests with model: {model_name_cleaned} and template: {template}")
            output_file= f"test_{template}_results_{model_name_cleaned}.txt"
            if os.path.exists(output_file):
                os.remove(output_file)

            print(f"Running interactive agent with model: {model}")
            template_updated = template + ".j2"
            run_interactive_agent(model, output_file, template_updated)