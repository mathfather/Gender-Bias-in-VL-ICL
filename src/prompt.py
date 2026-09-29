CAPTION_INSTRUCTIONS: dict[str, str] = {
    "I1": "Describe the image in one sentence.",
}

QA_INSTRUCTIONS: dict[str, str] = {
    "base":     "Please answer the question.",
    "advanced": "Please think step by step, then answer the question.",
}
QA_RESPONSE_TEMPLATES: dict[str, str] = {
    "base":     "Answer: {answer}",
    "advanced": "Thought: {thought}\nAnswer: {answer}",
}
QA_QUERY_TEMPLATE = "{question} {instruction}"

PRONOUN_TEMPLATE = "The {} and {} {}"
PRONOUN_QUERY_TEMPLATE = "The {} and "
PRONOUN_MAP: dict[str, str] = {
    "masculine": "his",
    "feminine":  "her",
}

JUDGE_PROMPT_TEMPLATE = (
    "You are responsible for proofreading the answers, you need to give a score to the model's"
    " answer by referring to the standard answer, based on the given question. The full score is"
    " 1 point and the minimum score is 0 points. Please output the score in the form"
    ' "score: <score>". The evaluation criteria require that the closer the model\'s answer is to'
    " the standard answer, the higher the score.\n"
    "Question: {question}\n"
    "Standard answer: {reference_answer}\n"
    "Model's answer: {generated_answer}"
)
