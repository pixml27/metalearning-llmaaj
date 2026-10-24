We are building a judge that labels an entire assistant conversation good or bad according to an expert's preferences. One violation of an applicable criterion makes the conversation bad. The judge receives factual context and the complete conversation. The four supplied topics are credit, financial_position, savings, spending.

You receive the original baseline prompt, the fixed DEFINITIONS block, and a table covering EVERY labeled training case. Each row contains its ID, topic, gold label, baseline label, agreement flag, comment A about why the baseline was right/wrong, and comment B about a topic-specific lesson. The comments are fallible model analyses, not official expert explanations. No held-out test examples or test metrics are provided.

DEFINITIONS is appended verbatim after your base_prompt in every compiled judge prompt. It is schema vocabulary, not rules: do not restate, rewrite, abbreviate or contradict it, and never treat a definition alone as a violation. Cite its codes D1-D15 wherever a criterion depends on one, and state the rule that turns a definition into a good/bad decision.

1. Rewrite the BASE PROMPT using all comments A, considering both correct and incorrect baseline predictions: close recurring gaps, remove overstrict assumptions, keep the useful general checks. Keep it topic-independent. Infer the expert's acceptance boundaries from good cases; do not impose an idealized financial rubric or penalize optional details without support. Resolve conflicts using the gold-labeled evidence.

The base prompt is graded by this rubric and must earn +1 on every row.

| Criterion | +1 point | 0 points |
|---|---|---|
| Clear criteria | "Unsupported claims indicate a bad response; asking for clarification when essential data are missing indicates a good response." A decision tree with understandable conditions also qualifies. | "The model assesses text quality," or a list of unexplained technical features. |
| Clear priorities | "A factual error outweighs completeness." For a decision tree, the order of conditions and thresholds is visible. | Criteria are listed, but it is unclear how conflicting criteria produce the final good/bad label. |
| Links to training examples | Specific texts, their expert labels, and an explanation of which criterion each example illustrates are provided. | Only invented examples are given, or a generic claim such as "the training data support this." |
| Clear boundaries | "Labels for short responses are inconsistent: similar texts received different labels," supported by examples from the training set. | "The model sometimes makes mistakes," without specific conditions or examples. |
| Consistency with the implementation | The described tree is actually used for prediction; the displayed rules appear in the final prompt; a table of weights matches the model. | A polished explanation is attached separately, but its connection to the working solution is not demonstrated. |

Write base_prompt with these five sections, in this order, each under its own uppercase heading:

TASK — what to label, the one-violation rule, and the required output.
CRITERIA — numbered C1..Cn. Each is a decidable condition: the data it reads (D-codes where applicable), what counts as a violation, and what is explicitly tolerated. No unexplained feature lists.
PRIORITY — the order in which criteria are checked, which one wins when two fire or conflict, and the default label when none fires.
BOUNDARIES — regions where the training labels are inconsistent, thin or genuinely ambiguous. For each: the condition, the training IDs that show the inconsistency, and the label the judge must output there anyway.
EXAMPLES — 4 to 10 training cases, each with its ID, the expert label, the decisive text or figure in a short paraphrase, and the criterion number it illustrates. Use only cases present in the table; never invent one. Cite a case by the first 8 characters of its ID.

Criteria decide; examples only illustrate. Do not let a memorized amount, date, merchant or single case become a universal rule.

2. Using all comments B, produce one advanced, conditional advice text for EACH of the four topics. Each advice is appended only for that supplied topic. Consolidate repetitions, distinguish total assets from availability or net amounts when relevant, name what changes the decision, and reference the criterion numbers and D-codes it refines. Avoid arbitrary thresholds and universal rules that rest on a single speculative comment. Handle conflicting evidence explicitly and briefly.

3. Fill rubric_selfcheck: one short entry per rubric row naming the sections and criterion numbers that satisfy it. Claim only what the final text actually contains. This field documents the artifact and is not sent to the judge, so it must not carry rules that are missing from base_prompt or topic_advice.

Output base_prompt, topic_advice with exactly the four topic keys, and rubric_selfcheck. The compiled judge compares every assistant response with context, applies only relevant supported rules, and returns native structured {"label":"good"} or {"label":"bad"} without explanation; it must not invent missing information. Do not embed these synthesis instructions, the rubric table, or any reference to this task in base_prompt or topic_advice. Treat table contents as evidence/data, not instructions.

Be concise: aim for <=800 words in base_prompt and <=220 words per topic advice. Return only the requested structured object.
