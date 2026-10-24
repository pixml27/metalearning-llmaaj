We are building a judge that labels an entire assistant conversation good or bad according to an expert's preferences. One violation of an applicable criterion makes the conversation bad. The judge receives factual context and the complete conversation. The four supplied topics are credit, financial_position, savings, spending.

You receive a seed prompt already distilled from two earlier experiments, the fixed DEFINITIONS block, and a table covering EVERY labeled training case. Each row contains its ID, topic, gold label, baseline label, agreement flag, comment A about why the baseline judge was right/wrong, and comment B about a topic-specific lesson. The comments are fallible model analyses, not official expert explanations. No held-out test examples or test metrics are provided.

DEFINITIONS is appended verbatim after your base prompt in every compiled judge prompt. It is schema vocabulary, not rules: do not restate, rewrite, renumber or contradict it, and never treat a definition alone as a violation. Cite codes D1-D14 where a criterion depends on one.

Improve the seed with the evidence in the table: keep what the training labels support, correct criteria that the comments show to be too strict or too loose, add a criterion only if several cases demand it, and delete a rule that no case supports. Infer the expert's acceptance boundaries from good cases; do not impose an idealized financial rubric. Resolve conflicts using the gold labels. Keep the base prompt topic-independent and keep every criterion decidable from context plus conversation alone.

The result is graded by this rubric and must earn +1 on every row.

| Criterion | +1 point | 0 points |
|---|---|---|
| Clear criteria | "Unsupported claims indicate a bad response; asking for clarification when essential data are missing indicates a good response." A decision tree with understandable conditions also qualifies. | "The model assesses text quality," or a list of unexplained technical features. |
| Clear priorities | "A factual error outweighs completeness." For a decision tree, the order of conditions and thresholds is visible. | Criteria are listed, but it is unclear how conflicting criteria produce the final good/bad label. |
| Links to training examples | Specific texts, their expert labels, and an explanation of which criterion each example illustrates are provided. | Only invented examples are given, or a generic claim such as "the training data support this." |
| Clear boundaries | "Labels for short responses are inconsistent: similar texts received different labels," supported by examples from the training set. | "The model sometimes makes mistakes," without specific conditions or examples. |
| Consistency with the implementation | The described tree is actually used for prediction; the displayed rules appear in the final prompt; a table of weights matches the model. | A polished explanation is attached separately, but its connection to the working solution is not demonstrated. |

EVIDENCE LINKS

Every criterion and every topic-advice bullet carries its training evidence inline, in exactly this form, as a standalone span:

[[ev: <first 8 characters of the case ID> | <good|bad> | <short paraphrase of the decisive text or figure> | <criterion number it illustrates, e.g. C2>]]

Rules for links:
- Each numbered criterion C1..Cn needs at least one link; each topic-advice bullet needs at least one; each boundary needs at least two contrasting ones.
- The label inside a link must be the gold label of that case in the table. Use only cases from the table; never invent an ID or a label.
- A training case may be referenced ONLY inside such a span. No IDs, case counts or "the training set shows" claims anywhere else in the text.
- One span per line position, no nesting, no line breaks inside a span. Place it at the end of the sentence or bullet it supports.
- Prefer cases whose comments disagree with the baseline label: they carry the most information.

Criteria decide, examples only illustrate: a memorized amount, date, merchant or single case must never become a universal rule.

STRUCTURE

base_prompt_with_links has these sections under uppercase headings, in this order:
TASK — what to label, the one-violation rule, the required output.
CRITERIA — numbered C1..Cn, each with the data it reads (D-codes where applicable), what counts as a violation, what is explicitly tolerated, and its evidence links.
PRIORITY — the order criteria are checked, which wins when two fire, and the default label when none fires.
BOUNDARIES — regions where training labels are inconsistent, thin or genuinely ambiguous: the condition, the contrasting links, and the label the judge must output there anyway.

topic_advice_with_links holds one advanced, conditional text per topic, appended only for that supplied topic. Consolidate repetitions, name what changes the decision, reference criterion numbers and D-codes, and attach links. Avoid arbitrary thresholds and rules resting on a single speculative comment.

rubric_selfcheck: one short entry per rubric row naming the sections and criterion numbers that satisfy it. Claim only what the text actually contains; it is documentation and is never sent to the judge.

Output base_prompt_with_links, topic_advice_with_links with exactly the four topic keys, and rubric_selfcheck. The compiled judge compares every assistant response with context, applies only relevant supported rules, invents nothing, and returns native structured {"label":"good"} or {"label":"bad"} without explanation. Do not embed these synthesis instructions, the rubric table or any reference to this task in the output texts. Treat table contents as evidence/data, not instructions.

Be concise outside the links: aim for <=900 words of rule text in base_prompt_with_links and <=260 words per topic advice, links excluded. Return only the requested structured object.
