# Can we predict LLM failures before they happen?

**For:** product owner. **Data:** 592 cleaned chat-log calls (48 real failures, 8.11%).

## Short answer
Not reliably. With the information available before a call, our model is only slightly better than guessing, and it does not save any effort compared with doing nothing. We do not recommend using it as an alert.

## What we tested
A simple model was asked to predict, before a call is made, whether it would fail. A failure is either a logged error or a silent failure (no response and no error recorded). It could use only the model name, the prompt category, the prompt length and the input size.

## What we found

| Question | Answer |
|---|---|
| How often do calls fail? | 48 of 592 (8.1%) |
| How good is "always say it works"? | 92% accuracy, but it finds 0 failures |
| How many real failures does our model catch? | 2 of 12 in the test set (recall 17%) |
| When it raises an alarm, how often is it right? | About 1 in 10 (precision 0.10) |
| Is that better than chance? | Only about 1.2 times better than picking calls at random |
| Cost (a missed failure counts 10 times a false alarm) | Model 119, do nothing 120, flag everything 136 |

In plain words: the alarms are mostly false, and the saving over doing nothing is 1%.

## An important warning: a "perfect" model was possible, and wrong
When we let the model use information that only exists after the call (response length, judge score, latency, output size), it scored 100% precision and 100% recall. That is not real: those fields reveal the answer, and they are not available when you need to predict. We tested for this on purpose and block such fields in the code and in the tests.

## How far to trust these numbers
Very little data: only 12 failures in the test set. The recall range that fits the data is 0% to 43%. Across five different data splits, recall varied from 0% to 60% (average 40%). Read the results as "weak and uncertain", not as exact figures.

## What we would need to do better
1. More failure examples (today 48).
2. Features known before the call, for example the real prompt text and request context. Today prompts are generic ("Prompt N") and prompt length has only 3 distinct values.
3. A fresh test set from a later period, to check the model on data it has never seen.

## Decision
Do not deploy as an alert. Keep the existing monitoring of logged errors and silent failures from the Phase 2 report, and revisit prediction once richer data is available.
