"""Hand-checked toy examples for the Ragas metrics wired to Gemma. Expected values are derived by hand below.

  python -m arretrieval.rageval.toy_ragas
Expected (Gemma may deviate slightly; deviations are recorded in RESULTS.md):
  faithfulness  full support            -> 1.0
  faithfulness  1 of 2 claims invented  -> 0.5
  context recall: reference has 2 facts, context covers 1 -> 0.5
  context precision with reference: [useful, useless] -> 1.0 ; [useless, useful] -> 0.5
  factual correctness identical answer  -> ~1.0 ; answer with wrong number -> < 0.7
"""

from __future__ import annotations

import asyncio
import json

from ragas.metrics.collections import ContextPrecision, ContextRecall, FactualCorrectness, Faithfulness

from .llm import get_llm
from .ragas_adapters import CachedRagasLLM

CTX = [
    "The Panthers defense gave up just 308 points, ranking sixth in the league.",
    "The Eiffel Tower is located in Paris and was completed in 1889.",
]


async def main():
    llm = CachedRagasLLM(get_llm("gemma"))
    out = {}
    f = Faithfulness(llm=llm)
    out["faith_full"] = (
        await f.ascore("How many points did the Panthers defense give up?", "The Panthers defense gave up 308 points.", CTX[:1])
    ).value
    out["faith_half"] = (
        await f.ascore(
            "How many points did the Panthers defense give up?",
            "The Panthers defense gave up 308 points. They won the Super Bowl in 2016.",
            CTX[:1],
        )
    ).value
    rec = ContextRecall(llm=llm)
    out["recall_half"] = (
        await rec.ascore(
            "Tell me about the Panthers and the Eiffel Tower.",
            CTX[:1],
            "The Panthers defense gave up 308 points. The Eiffel Tower was completed in 1889.",
        )
    ).value
    out["recall_full"] = (
        await rec.ascore(
            "Tell me about the Panthers and the Eiffel Tower.",
            CTX,
            "The Panthers defense gave up 308 points. The Eiffel Tower was completed in 1889.",
        )
    ).value
    cp = ContextPrecision(llm=llm)
    q, ref = "How many points did the Panthers defense give up?", "The Panthers defense gave up 308 points."
    out["prec_useful_first"] = (await cp.ascore(q, ref, [CTX[0], CTX[1]])).value
    out["prec_useful_second"] = (await cp.ascore(q, ref, [CTX[1], CTX[0]])).value
    fc = FactualCorrectness(llm=llm)
    out["fc_same"] = (await fc.ascore("The Panthers defense gave up 308 points.", ref)).value
    out["fc_wrong_number"] = (await fc.ascore("The Panthers defense gave up 380 points.", ref)).value
    out["fc_short_ref"] = (await fc.ascore("The Panthers defense gave up 308 points.", "308")).value
    out["fc_arabic_same"] = (await fc.ascore("دفاع بانثرز استقبل 308 نقطة فقط.", "استقبل دفاع بانثرز 308 نقطة فقط.")).value
    out["fc_arabic_indic"] = (await fc.ascore("دفاع بانثرز استقبل ٣٠٨ نقطة فقط.", "استقبل دفاع بانثرز 308 نقطة فقط.")).value
    print(json.dumps(out, indent=2))
    print("parse_failures", llm.parse_failures, "calls", llm.calls)


if __name__ == "__main__":
    asyncio.run(main())
