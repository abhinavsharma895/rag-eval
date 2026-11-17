import os
import sys
import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from groq import Groq
from pydantic import BaseModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.configs import AsyncConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing from .env")


class GroqJudge(DeepEvalBaseLLM):

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = "openai/gpt-oss-120b"

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel = None):

        messages = [
    {
        "role": "system",
        "content": """
You are a precise evaluation judge.

Follow the supplied JSON schema exactly.
Every required property must be included.

For every verdict, provide BOTH:
1. verdict
2. reason

Never omit the reason field.
Never return incomplete JSON.
"""
    },
    {
        "role": "user",
        "content": prompt,
    },
]
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
        }

        if schema is not None:

            json_schema = schema.model_json_schema()

            # Groq strict mode requires all object fields
            # to be required and additionalProperties=false.
            def prepare_schema(node):
                if isinstance(node, dict):
                    if node.get("type") == "object" or "properties" in node:
                        properties = node.get("properties", {})
                        node["required"] = list(properties.keys())
                        node["additionalProperties"] = False

                    for value in node.values():
                        prepare_schema(value)

                elif isinstance(node, list):
                    for value in node:
                        prepare_schema(value)

            prepare_schema(json_schema)

            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "deepeval_response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        response = self.client.chat.completions.create(**kwargs)

        output = response.choices[0].message.content

        if schema is not None:
            return schema.model_validate_json(output)

        return output

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


model = GroqJudge()




# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in .env")


# 1. Groq as DeepEval judge
# model = LiteLLMModel(
#     model="groq/openai/gpt-oss-120b",
#     api_key=GROQ_API_KEY,
#     temperature=0,
# )

# 2. Gemini embeddings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# 3. Load test cases
with open("cases.json", encoding="utf-8") as f:
    all_cases = json.load(f)


def make_test_cases(set_name):
    test_cases = []

    for c in all_cases[set_name]:
        test_cases.append(
            LLMTestCase(
                input=c["input"],
                actual_output=c["actual_output"],
                expected_output=c["expected_output"] or None,
                retrieval_context=c["retrieval_context"],
            )
        )

    return test_cases


# 4. Configure metrics
RUNS = {}

RUNS["answer_relevancy"] = (
    AnswerRelevancyMetric(
        threshold=0.7,
        model=model,
        include_reason=True,
        async_mode=False,
    ),
    "answer_relevancy",
)

RUNS["faithfulness"] = (
    FaithfulnessMetric(
        threshold=0.7,
        model=model,
        include_reason=True,
        async_mode=False,
    ),
    "faithfulness",
)

RUNS["contextual_recall"] = (
    ContextualRecallMetric(
        threshold=0.7,
        model=model,
        include_reason=True,
        async_mode=False,
    ),
    "contextual_recall",
)


# 5. Run evaluation
which = sys.argv[1] if len(sys.argv) > 1 else "all"

all_results = []

for name, (metric, set_name) in RUNS.items():

    if which != "all" and which != name:
        continue

    print(f"\n===== {name} =====")

    evaluation_result = evaluate(
        test_cases=make_test_cases(set_name),
        metrics=[metric],
        async_config=AsyncConfig(run_async=False),
    )

    for test in evaluation_result.test_results:
        for metric_result in test.metrics_data:

            all_results.append({
                "test_case": test.name,
                "metric": metric_result.name,
                "score": metric_result.score,
                "threshold": metric_result.threshold,
                "passed": metric_result.success,
                "reason": metric_result.reason,
            })


# 6. Save results
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(
        all_results,
        f,
        indent=4,
        ensure_ascii=False,
    )

print("\nEvaluation results saved successfully!")
print("File: results2.json")
