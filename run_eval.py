import sys
import json
import time
from google.genai.errors import ServerError
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel
from deepeval.evaluate.configs import AsyncConfig
from dotenv import load_dotenv
load_dotenv()


# The judge model (same as in class)

model = GeminiModel(
    model="gemini-3.5-flash-lite",
    temperature=0,
)


# Load the test cases that build_cases.py saved

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


# Every metric we add is stored here: name -> (metric, which set of questions to use)
RUNS = {}


# ===== METRICS: paste new metrics here =====
# Query-time metric 1: Answer Relevancy
# Metric 1: Answer Relevancy
RUNS["answer_relevancy"] = (
    AnswerRelevancyMetric(
        threshold=0.7,
        model=model,
        include_reason=True,
        async_mode=False
    ),
    "answer_relevancy",
)


# Metric 2: Faithfulness
RUNS["faithfulness"] = (
    FaithfulnessMetric(
        threshold=0.7,
        model=model,
        include_reason=True,
        async_mode=False
    ),
    "faithfulness",
)


# Metric 3: Contextual Precision
# Retrieval-time metric 2: Contextual Recall
RUNS["contextual_recall"] = (
    ContextualRecallMetric(threshold=0.7, model=model, include_reason=True, async_mode=False),
    "contextual_recall",
)



# ===== RUN: KEEP THIS AT THE VERY BOTTOM =====

which = sys.argv[1] if len(sys.argv) > 1 else "all"

all_results = []

for name, (metric, set_name) in RUNS.items():
    if which != "all" and which != name:
        continue

    print(f"\n===== {name} =====")

    evaluatioin_result = evaluate(
        test_cases=make_test_cases(set_name),
        metrics=[metric],
        async_config=AsyncConfig(run_async=False),
    )
    # Extract evaluation results
    for test in evaluatioin_result.test_results:

        for metric_result in test.metrics_data:

            all_results.append({
                "test_case": test.name,
                "metric": metric_result.name,
                "score": metric_result.score,
                "threshold": metric_result.threshold,
                "passed": metric_result.success,
                "reason": metric_result.reason,
            })


# Save results
with open("results_abhinav.json", "w", encoding="utf-8") as f:


    json.dump(
        all_results,
        f,
        indent=4,
        ensure_ascii=False,
    )

print("\nEvaluation results saved successfully!")
print("File: results_abhinav.json")