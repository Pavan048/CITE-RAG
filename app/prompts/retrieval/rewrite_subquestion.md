You are producing the next targeted retrieval query in a multi-hop question-answering loop.

Original question: {{ question }}

Sub-questions already asked, in order:
{% for sq in sub_question_history %}
{{ loop.index }}. {{ sq }}
{% endfor %}

What's still missing from the evidence gathered so far: {{ missing }}

Write the next sub-question: a single, concrete, independently-retrievable question that targets
exactly the gap described above. It should read as a standalone question (name the specific
entity/document/metric involved — don't rely on pronouns referring back to the original question),
since it will be used on its own to search the knowledge base.

Respond with only this JSON object, no other text:
{"next_sub_question": <string>}
