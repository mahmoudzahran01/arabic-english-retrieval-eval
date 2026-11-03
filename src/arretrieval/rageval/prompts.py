"""Generator prompts. v1 = the baseline grounded prompt; v2 = stricter grounding (used in the A/B experiment)."""

ABSTAIN_EN = "Not in the documents."
ABSTAIN_AR = "غير موجود في المستندات."

_V1 = """You answer questions using ONLY the documents provided below.
Rules:
1. Answer in the same language as the question.
2. After each fact you state, cite the document id that supports it in square brackets, for example [P012].
3. If the documents do not contain the answer, reply with exactly "{abstain}" and nothing else.
4. Keep the answer to one or two sentences."""

_V2_EXTRA = """
5. Before answering, check that a document states the answer explicitly about the SAME person, event or thing that the question names. A document about a related but different subject does not count.
6. Never use outside knowledge, never guess, and never infer beyond what the text says.
7. If you are not certain that the answer is written in the documents, reply with exactly "{abstain}"."""


# v3 = v1 + an explicit, language-specific output instruction. Motivated by a defect the eval surfaced in config A:
# 27% of the baseline's real answers (all to English questions) were written in Arabic because the retrieved passage was Arabic.
_V3_EXTRA = """
5. IMPORTANT: write your answer in {lang_name} (the language of the question), even when the documents are in another language. Translate the facts if needed, but keep every number and name accurate."""


def system_prompt(version: str, q_lang: str) -> str:
    abstain = ABSTAIN_AR if q_lang == "ar" else ABSTAIN_EN
    base = _V1.format(abstain=abstain)
    if version == "v1":
        return base
    if version == "v2":
        return base + _V2_EXTRA.format(abstain=abstain)
    if version == "v3":
        return base + _V3_EXTRA.format(lang_name="Arabic" if q_lang == "ar" else "English")
    raise ValueError(version)


def user_prompt(question: str, passages: list[dict]) -> str:
    docs = "\n\n".join(f"[{p['id']}] ({p['lang']}) {p['text']}" for p in passages)
    return f"Documents:\n{docs}\n\nQuestion: {question}\nAnswer:"


def messages(question: str, passages: list[dict], version: str, q_lang: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt(version, q_lang)}, {"role": "user", "content": user_prompt(question, passages)}]
