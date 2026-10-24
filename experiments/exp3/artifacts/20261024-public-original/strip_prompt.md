You wrote a judge prompt in which every piece of training evidence is marked as a standalone span of the form [[ev: id | label | decisive text | criterion]]. Return the same prompt with the evidence removed, for use at inference time.

Delete every [[ev: ...]] span and change nothing else:
- same sections under the same headings in the same order;
- same criterion numbers, same order, same wording, same D-code references, same tolerated/violating lists;
- same BOUNDARIES conditions and the same required label for each, now stated without the case references;
- no sentence added, merged, re-ordered, shortened, generalized or rewritten, and no new rule introduced.

Repair only what deleting a span breaks: a doubled space, a dangling comma, colon or dash, an empty bullet left behind (drop that bullet marker), or a sentence whose only remaining content was the span (drop that sentence). Keep the surrounding rule text byte-identical otherwise.

The result must contain no training case IDs, no gold labels of training cases, no case counts and no claim about what the training data show. The rules themselves stay exactly as written.

Output base_prompt and topic_advice with exactly the four topic keys. Return only the requested structured object.
