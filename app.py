import argparse
import json

import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template, add_model_args
from fastchat.serve.inference import generate_stream
from fastchat.conversation import Conversation, SeparatorStyle

app = Flask(__name__)

BUFFER_SIZE = 5

# Load the model and tokenizer
parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--debug", action="store_true")
add_model_args(parser)
args = parser.parse_args()
print(args)

class FlaskChatIO():
    def prompt_for_input(self, role) -> str:
        pass

    def prompt_for_output(self, role: str):
        pass

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)


conv = Conversation(
        name="one_shot",
        system=""" PUT_YOUR_INSTRUCTION_PROMPT_HERE """,
        roles=("Human", "Assistant"),
        messages=[
            (
              "Human", 
              """ PUT_AN_EXAMPLE_REQUEST_HERE """
            ),
            (
                "Assistant",
                """ EXAMPLE_RESPONSE_HERE """
            )
        ],
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )

model, tokenizer = load_model(
    args.model_path,
    args.device,
    args.num_gpus,
    args.max_gpu_memory,
    args.load_8bit,
    args.cpu_offloading,
    debug=args.debug,
)

chatio = FlaskChatIO()

@app.route("/answer", methods=["POST"])
@torch.inference_mode()
def answer():
    data = request.get_json()
    message = data["message"].replace("\n", " ")

    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    gen_params = {
                "model": args.model_path,
                "prompt": prompt,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "stop": conv.stop_str,
                "stop_token_ids": conv.stop_token_ids,
                "echo": False,
            }

    output_stream = generate_stream(model, tokenizer, gen_params, args.device)
    outputs = chatio.stream_output(output_stream)
    # NOTE: strip is important to align with the training data.
    conv.messages[-1][-1] = outputs.strip()

    # previous answers buffer size
    if len(conv.messages) > BUFFER_SIZE:
        conv.messages.pop(0)

    return jsonify({"response": outputs})

if __name__ == "__main__":
    app.run()