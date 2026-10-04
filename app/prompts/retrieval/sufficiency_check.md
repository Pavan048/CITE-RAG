You are checking whether enough evidence has been gathered to fully answer a question, across
possibly several rounds of retrieval. Judge against the ORIGINAL question below — not against any
one sub-question that was used to fetch a particular round of evidence.

Original question: {{ question }}

Sub-questions asked so far, in order:
{% for sq in sub_question_history %}
{{ loop.index }}. {{ sq }}
{% endfor %}

Accumulated evidence (from all rounds so far):
{% for item in evidence_blocks %}
[{{ item.doc_id }}:{{ item.page_number }}:{{ item.chunk_id }}]
{{ item.text }}

{% endfor %}

Decide: does this evidence, taken together, contain everything needed to fully and accurately
answer the ORIGINAL question? Partial coverage of one part of a comparative question while another
named entity/document has no evidence yet counts as NOT sufficient.

If not sufficient, name specifically what is still missing (e.g. "no evidence yet about
[entity/document Y]'s figures for the same metric") so the next retrieval round can be targeted at
exactly that gap.

Respond with only this JSON object, no other text:
{"sufficient": <true|false>, "missing": <string describing the gap, or null iff sufficient is true>}
