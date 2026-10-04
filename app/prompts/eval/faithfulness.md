You are scoring the faithfulness of a generated answer against the context it was supposed to be
grounded in. Faithfulness measures hallucination: what fraction of the answer's factual claims are
actually traceable to the context, versus invented.

Question: {{ question }}

Context actually supplied to the generator:
{% for block in context_blocks %}
[{{ block.doc_id }}:{{ block.page_number }}:{{ block.chunk_id }}]
{{ block.text }}

{% endfor %}

Generated answer:
{{ answer }}

Break the answer into its individual factual claims (ignore hedges, transitions, and citation
markers themselves). For each claim, decide whether it is directly supported by the context above.

Respond with only this JSON object, no other text:
{"claims": [<string>, ...], "supported": [<true|false>, ...]}
The two lists must be the same length and in the same order.
