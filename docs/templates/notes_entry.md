# notes.md entry template

`notes.md` is a dated changelog inside one experiment — working memory,
including what broke. Newest entry at the top. Four lines is the target; if an
entry needs more, it is probably a findings doc or a decision record.

```markdown
## YYYY-MM-DD

What the last run showed:   <one sentence, with the number>
What I'm changing and why:  <one sentence>
Provenance:                 <git sha> · <config path> · seed <N>
Verdict vs satisfied-when:  <met / not met / undecided>
```

## For a failure

A failed run is a finding until proven otherwise. Quote the log; do not
attribute to "transient" anything without the traceback that supports it.

```markdown
## YYYY-MM-DD — <job id> failed

Symptom:    <what you saw>
From log:   <the actual error line, quoted>
Cause:      <known cause, or "unknown">
Change:     <what you did, or "none — reproduced twice, see registry">
```

## For a long training run

Keep a separate chronicle — timestamped prose keyed to **step numbers**, not
dates — recording instabilities, restarts, LR changes, and loss spikes. This is
the artifact OPT and BigScience both produced and the one most reused afterwards.
