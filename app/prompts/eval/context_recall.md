You are scoring context recall: of the individual factual claims in the reference (known-correct)
answer, what fraction are actually backed by the retrieved context? This catches cases where
retrieval missed evidence the generator would have needed, even if it wasn't asked about here.

Question: {{ question }}

Reference answer (known-correct):
{{ reference_answer }}

Retrieved context chunks:
{% for block in context_blocks %}
[{{ block.doc_id }}:{{ block.page_number }}:{{ block.chunk_id }}]
{{ block.text }}

{% endfor %}

Break the reference answer into its individual factual claims. For each claim, decide whether the
retrieved context above contains evidence supporting it.

Respond with only this JSON object, no other text:
{"claims": [<string>, ...], "supported_by_context": [<true|false>, ...]}
The two lists must be the same length and in the same order.
