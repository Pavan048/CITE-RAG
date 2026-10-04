You are scoring context precision: of the context chunks that were actually retrieved and used for
generation, what fraction are genuinely relevant to answering the question (as opposed to noise
that happened to be retrieved)?

Question: {{ question }}

Reference answer (known-correct, for judging relevance — not shown to the generator):
{{ reference_answer }}

Retrieved context chunks:
{% for block in context_blocks %}
[{{ loop.index }}] [{{ block.doc_id }}:{{ block.page_number }}:{{ block.chunk_id }}]
{{ block.text }}

{% endfor %}

For each numbered chunk, decide whether it is relevant to answering the question (given the
reference answer as ground truth for what "relevant" means here).

Respond with only this JSON object, no other text:
{"relevant": [<true|false>, ...]}
The list must have exactly one entry per numbered chunk above, in order.
