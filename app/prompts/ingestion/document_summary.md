You are summarizing one document for a knowledge-base index. This summary becomes the vector used
to decide, at query time, whether this document is even worth searching — so it must reflect what
the document is actually *about* (its subject, scope, and the kinds of questions it could answer),
not a compressed retelling of its content.

Document filename: {{ filename }}

Document text (may be truncated):
---
{{ document_text }}
---

Write a single summary of {{ min_tokens }}-{{ max_tokens }} tokens. Cover: the document's subject
matter, its type (e.g. financial report, technical manual, research paper, contract), and the main
topics or sections it contains. Do not include citations, headers, or meta-commentary about this
task. Output only the summary text.
