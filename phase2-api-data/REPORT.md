# API Evaluation Report

## Summary

We analysed 592 chatbot requests (two models, four types of prompt) after
removing 12 duplicates and 8 rows with impossible timings. Most requests are
handled well. Tricky ("adversarial") prompts fail more often and score lower,
and 10 requests failed without any error message.

## What is healthy

- About 94% of requests finished without a logged error (38 errors in 592). Whether 6% is acceptable depends on your target, which this data does not tell us.
- Factual, creative and out-of-scope prompts all score about 4.0 out of 5,
  with error rates between 3% and 7%.
- A typical request takes 2.6 seconds, and 95% finish within 8.3 seconds.

## What is not healthy

- **Adversarial prompts.** 12.8% of them end in an error (14 of 109), against
  5.0% for all other prompts (24 of 483). The automatic judge also scores them
  lower: 3.13 against 4.04 out of 5. With 93 and 451 scored requests, a gap
  this large (about 0.9 points, plausible range 0.6 to 1.2) is very unlikely
  to be chance. We do not yet know why.
- **Silent failures.** 10 requests (1.7%) returned nothing and logged no
  error. The judge still gave all 10 a score, and seven of them got a 4 or 5,
  so these failures can hide inside quality numbers. We excluded them from
  every score above.
- **Slow outliers.** The three slowest requests took 90, 171 and 241 seconds.
  They barely move the typical time but distort the average, which is why we
  quote the median. gpt-5 is also slower at the median: 4.1 seconds against
  2.2 for gpt-5-mini. The 95th-percentile latency among successful calls is
  also higher: 9.7 seconds for gpt-5 versus 5.1 for gpt-5-mini.

## What to investigate next

1. Adversarial failures: which error types, which prompt patterns, which model.
2. The 10 silent failures: check raw logs, and find out why the judge scored
   empty answers.
3. The three slowest requests: review timeout and retry behaviour.
4. Re-run on a larger, representative set and track error rate, silent-failure
   rate, slow-request time and score for each prompt type.

## How far to trust this

The judge is an automated model whose scores have not been validated, so
differences show where to look, not what is broken. The data covers about one
week and may not hold for future traffic. The two models saw similar mixes of
prompt types, but not identical prompts.
