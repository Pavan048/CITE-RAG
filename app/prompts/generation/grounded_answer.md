You are answering a question using ONLY the context blocks below, retrieved from a knowledge base
of PDF documents. Do not use any outside knowledge, and do not guess at anything not stated in the
context.

Question: {{ question }}

Context blocks:
{% for block in context_blocks %}
[Block {{ loop.index }} | document: {{ block.doc_id }} | citation options: {% for c in block.citation_options %}chunk_id={{ c.chunk_id }} (page {{ c.page_number }}){% if not loop.last %}, {% endif %}{% endfor %}]
{{ block.text }}

{% endfor %}
{% if not context_blocks %}
(No context was retrieved — the knowledge base has nothing relevant to this question.)
{% endif %}

Rules:
1. If the context blocks are empty, or none of them actually discuss what the question is asking
   about at all, say plainly that you could not find relevant information in the knowledge base to
   answer this question. Do not fabricate an answer to avoid saying so.
   This is different from the question being broad, open-ended, or loosely worded (e.g. "tell me
   about X", "get information about X", a request with a typo, or a general summary request) — if
   the context blocks are clearly about the thing being asked about, synthesize a general answer
   from whatever is there instead of treating vagueness as a reason to abstain. Abstention is for
   "the knowledge base doesn't cover this topic," not "the question wasn't phrased narrowly."
2. Otherwise, write the answer in plain prose. After every sentence that states a fact from the
   context, add a citation immediately after it: one square-bracketed group containing exactly
   three colon-separated values, in order — that block's document id, then the page number, then
   the chunk id — with no other words or labels inside the brackets. For example, if a block's
   document is `acme-corp-2025` and one of its citation options is `chunk_id=8f3a1c2e (page 12)`,
   the citation looks exactly like this: [acme-corp-2025:12:8f3a1c2e]
   Use one of that block's listed citation options — pick whichever chunk_id/page_number pair most
   directly backs that sentence; if you can't tell which one specifically, use the first listed.
   Never write a chunk_id that isn't listed under some block above, and never write the words
   "doc_id", "page_number", or "chunk_id" themselves inside the brackets — only the real values.
3. If the question is comparative (e.g. asks you to compare two documents, entities, or time
   periods), attribute each side of the comparison to its own source: cite the block(s) for
   document A right after the sentence(s) about A, and the block(s) for document B right after the
   sentence(s) about B. Do not write one citation covering both sides of a comparison.
4. Do not add a references list, summary of sources, or any citation not immediately following the
   sentence it supports.
