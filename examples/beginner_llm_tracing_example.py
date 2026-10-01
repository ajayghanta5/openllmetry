"""
Beginner-friendly OpenLLMetry example: trace a basic LLM call step by step.

What this does:
  1. Initializes OpenLLMetry's Traceloop SDK (built on OpenTelemetry).
  2. Makes one LLM call (via Ollama running locally - free, no API key needed).
  3. Captures the request/response as a trace, printed to your console.

Prerequisites:
  pip install traceloop-sdk openai

  Install Ollama from https://ollama.com and pull a model:
      ollama pull llama3.1
  (Ollama serves an OpenAI-compatible API at http://localhost:11434/v1,
   so the OpenAI instrumentation traces it automatically.)
"""

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from openai import OpenAI

# ---------------------------------------------------------------------------
# Step 1: Initialize tracing.
# disable_batch=True sends spans immediately instead of batching them,
# so you can see the trace output right away while experimenting.
# ---------------------------------------------------------------------------
Traceloop.init(app_name="beginner_tracing_example", disable_batch=True)

# Point the OpenAI client at your local Ollama server.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# ---------------------------------------------------------------------------
# Step 2: Wrap your LLM logic in a @workflow.
# This creates a parent span; every instrumented LLM call inside it
# becomes a child span, so you get the full request -> response picture.
# ---------------------------------------------------------------------------
@workflow(name="joke_workflow")
def tell_me_a_joke() -> str:
    response = client.chat.completions.create(
        model="llama3.1",
        messages=[{"role": "user", "content": "Tell me a one-line joke about observability."}],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Step 3: Run it. The SDK prints the captured trace spans to the console,
# showing the workflow span plus the LLM call span with its
# request (prompt) and response (completion) attributes.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    joke = tell_me_a_joke()
    print("\nLLM response:", joke)
    print("\nDone! Check the console output above for the exported trace spans.")
