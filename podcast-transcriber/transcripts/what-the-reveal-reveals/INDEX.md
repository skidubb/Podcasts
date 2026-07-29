# Plastic Surgery: What the Reveal Reveals — Episode Index

Patient-voice corpus for customer-journey research: how people decide on a procedure, what
they feared, what recovery was actually like, and how they describe the result in their own
words. 328 episodes, Sep 2023 – Jul 2026, hosted by Dr. R. Brannon Claytor. Episodes are
short (3.3 min average), so each file is a single patient's account rather than a discussion.

Indexed into Pinecone index `beauty-and-the-biz`, namespace `patient-voice` — the namespace
that patient-perspective shows share, separate from the practitioner-facing corpora
(Beauty and the Biz in the default namespace, The Face Podcast in `the-face-podcast`).

## Read this before drawing conclusions

**Every patient here is a satisfied patient of one practice.** The show is produced by
Claytor Noone Plastic Surgery (Philadelphia) and features its own patients, selected for
publication. Two consequences:

1. **Not sentiment data.** Dissatisfied patients, revisions, complications, and people who
   chose a different surgeon are structurally absent. A query like "what do patients
   complain about" will return a misleadingly positive picture, because the negative cases
   were never recorded.
2. **Not geographically or demographically representative.** One practice, one metro area,
   one surgeon's case mix (heavy on tummy tuck, breast, and deep plane facelift).

**Valid uses:** the language patients use for their triggers and hesitations, how they
describe the consult and decision, recovery expectations vs. experience, what they say
convinced them. **Invalid uses:** satisfaction rates, complication rates, comparative
outcomes, or any claim about patients in general.

For the practice-business perspective on the same market, use Beauty and the Biz; for
clinical technique, use The Face Podcast.

## Known metadata caveat

The `guest` field is unreliable on this corpus — patients are usually unnamed and the
recurring voice is Dr. Claytor, so the entity extractor has no reliable name to attach.
Filter on `topics` or `products` instead; do not filter on `guest`.

## Theme map

To be added once the backfill completes — see the `ratchet-wrench-radio/INDEX.md` theme map
for the format.
