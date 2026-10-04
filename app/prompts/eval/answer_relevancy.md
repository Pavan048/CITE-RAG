You are scoring how directly a generated answer addresses the question that was actually asked —
independent of whether the answer is factually correct. A relevant answer stays on-topic and
addresses what was asked; an irrelevant one is evasive, off-topic, or answers a different question.

Question: {{ question }}

Generated answer:
{{ answer }}

Respond with only this JSON object, no other text:
{"relevancy_score": <float between 0.0 and 1.0>, "reasoning": <short string>}
