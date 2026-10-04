You are the planning step of a retrieval system over a knowledge base of PDF documents. Decide
whether this question can be answered from a single, focused retrieval, or whether it genuinely
needs evidence gathered from more than one document (or from clearly separate parts of the
knowledge base) before it can be answered.

Question: {{ question }}

Signals that point toward multi-hop: comparative language ("compare", "versus", "difference
between"), multiple named entities or documents that must each be looked up separately, words like
"across" or "all of", or a question that bundles two distinct sub-asks together. A question that
just asks a single fact, definition, or summary — even a detailed one — is single-hop, even if
answering it well requires several retrieved chunks from the *same* area of the knowledge base.

If multi-hop, also produce the first sub-question to retrieve for: usually the most concrete,
independently-answerable piece of the original question — often just the original question itself
restated to name one specific document or entity, when the question already does that.

Respond with only this JSON object, no other text:
{"is_multi_hop": <true|false>, "first_sub_question": <string or null, null iff is_multi_hop is false>}
