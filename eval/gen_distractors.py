#!/usr/bin/env python3
"""Generate confusable distractor documents for the CI fixture corpus.

WHY THIS EXISTS

    The 19-item CI golden set scored recall@10 = 1.000 against 25 fixture
    chunks, which sounds like a passing grade and is actually a broken
    instrument: TOP_K is 10, so retrieval returned 40% of the corpus and
    recall@10 could not have come out any other way. Measured on that corpus
    the cross-encoder -- the single biggest quality win in this pipeline --
    contributed a rerank_lift of 0.048, against 0.164 on the 300-chunk real
    corpus. A gate that cannot see the reranker cannot catch its regression.

    Volume alone does not fix that. Random filler is trivially separable and
    leaves the ranking unchanged; what forces a reranker to work is a corpus
    full of documents that a lexical or dense match plausibly confuses with
    the answer. So every distractor here is a near-miss for a specific golden
    item: same component family, same vocabulary, same shape, different
    specifics.

CONTRACT

    Deterministic. No RNG, no timestamps -- the same input always produces
    byte-identical output, because chunk ids are the eval contract and a
    fixture that shifted between runs would invalidate every golden item.

    Additive only. This writes NEW files and never touches the five
    hand-written fixtures, so the chunk ids the golden set references
    (error-codes.csv#0, runbooks.md#9, ...) keep pointing at the same text.

    Answer-preserving. Distractors deliberately avoid answering any golden
    question. They use different components, regions, magnitudes and
    identifiers, so a retrieval that ranks one above the true chunk is a
    genuine ranking error rather than a second correct answer scored wrong.

    That property is asserted, not assumed -- see check(). The first version of
    this file generated ORB-6203 and ORB-3010, both of which the hand-written
    fixture already defines, and ci-code-6203 asks about ORB-6203 by name. Two
    contradictory definitions of one code turn a benchmark item into a coin
    flip, and it would have scored as a retrieval regression forever after.
    A distractor may look like the answer; it must never be one.

    Run from the repo root:  python eval/gen_distractors.py
"""

import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "fixtures"

# Component families mirroring error-codes.csv, so generated codes sit in the
# same numeric neighbourhoods as ORB-4412 (auth) and ORB-6203 (telemetry).
FAMILIES = [
    ("ingest", 1000, [
        ("Payload accepted but its declared encoding did not match the sniffed type", "Re-submit with an explicit content-type header"),
        ("Upload stream closed before the declared length was reached", "Retry the upload; partial objects are discarded automatically"),
        ("Manifest referenced a part that was never uploaded", "Re-upload the missing part before finalising"),
        ("Payload checksum matched but the archive index was unreadable", "Rebuild the archive locally and re-upload"),
        ("Declared size was below the minimum for a multipart upload", "Send it as a single-part upload instead"),
    ]),
    ("scheduler", 2200, [
        ("Job was placed on a worker that drained before it started", "Requeue the job; placement is not sticky"),
        ("Placement window expired while the pool was scaling up", "Widen the window or pre-warm the pool"),
        ("Job requested an affinity tag no worker advertises", "Correct the tag; unmatched affinity never falls back"),
        ("Two placements raced and the later one won", "No action; the earlier result is discarded by design"),
        ("Worker accepted the job then failed its readiness probe", "Investigate the worker before returning it to the pool"),
    ]),
    ("storage", 3000, [
        ("Read quorum not reached on a secondary replica", "Retry against the primary; secondaries lag under load"),
        ("Compaction lagged far enough to reject new writes", "Wait for compaction; writes resume without intervention"),
        ("Key existed but its value failed a checksum on read", "Escalate with the key; the replica needs rebuilding"),
        ("Snapshot referenced a segment removed by retention", "Restore from an older snapshot instead"),
        ("Write succeeded on the primary but no replica acknowledged", "Treat the write as lost and re-issue it"),
    ]),
    # Deliberately dense around 4412: same family, adjacent numbers, and
    # meanings a dense retriever will find genuinely close.
    ("auth", 4400, [
        ("Bearer token carried permission tags that no longer exist", "Reissue the token against the current tag list"),
        ("Bearer token was well formed but had already been revoked", "Issue a fresh token; revocation is immediate"),
        ("Token carried permission tags but none matched the resource", "Grant the resource tag explicitly; wildcards are not expanded"),
        ("Token was valid but presented over an untrusted transport", "Re-issue the request over TLS; the token is now burned"),
        ("Token signature verified but its audience claim was wrong", "Request a token scoped to this service"),
        ("Token was valid but its permission tags were empty after filtering", "Check the group mapping; filtered tags fail closed"),
    ]),
    ("billing", 5000, [
        ("Usage record arrived twice for the same metering window", "No action; duplicates are collapsed on the record id"),
        ("Invoice total disagreed with the sum of its line items", "Refuse the invoice and re-run aggregation"),
        ("Currency conversion succeeded against a deprecated rate table", "Refresh the table; deprecated rates expire in 30 days"),
        ("Usage arrived for a tenant closed mid-period", "Record is held for manual review"),
        ("Discount applied after tax rather than before it", "Re-issue the invoice; ordering is not adjustable"),
    ]),
    # Dense around 6203, same reason as the auth family.
    ("telemetry", 6200, [
        ("Metric cardinality approached but did not exceed the tenant limit", "Reduce labels proactively; no data has been dropped"),
        ("Exporter was re-enabled before the offending label was removed", "Remove the label first; re-enabling alone loops"),
        ("Metric was dropped because its name collided with a reserved prefix", "Rename the metric; reserved prefixes are not configurable"),
        ("Trace was sampled in but its parent span was missing", "Fix the propagation header at the caller"),
        ("Histogram bucket boundaries changed mid-series", "Start a new series; buckets cannot be migrated in place"),
        ("Per-tenant limit was raised but the exporter cached the old value", "Restart the exporter to pick up the new limit"),
    ]),
]

# Bench inventory sharing the RF-00xxx serial space with lab-inventory.txt,
# so "RF-00417" has close neighbours rather than being the only RF serial.
INSTRUMENTS = [
    ("Signal generator", "SG", "4200"), ("Spectrum analyser", "SA", "2050"),
    ("Vector network analyser", "VNA", "8720"), ("Oscilloscope", "DSO", "6104"),
    ("Logic analyser", "LA", "1670"), ("Power meter", "PM", "4419"),
    ("Frequency counter", "FC", "5335"), ("Noise figure meter", "NFM", "8970"),
    ("Arbitrary waveform generator", "AWG", "3252"), ("Bench multimeter", "DMM", "3458"),
    ("DC power supply", "PSU", "6632"), ("Electronic load", "LOAD", "6060"),
    ("Temperature chamber", "TC", "9023"), ("Vibration table", "VT", "4410"),
    ("Vacuum pump", "VP", "1400"), ("Thermal camera", "TCAM", "7800"),
]

# Runbook procedures that are ordering-sensitive or duration-bearing, the two
# things the rerank-stage golden items turn on.
RUNBOOKS = [
    ("Rotating a storage encryption key", "storage",
     "the previous key stays readable for 30 days", "rotate, verify, then retire"),
    ("Rotating a webhook signing secret", "integrations",
     "the previous secret stays valid for 7 days", "publish the new secret before revoking the old"),
    ("Rotating a database credential", "storage",
     "the previous credential is revoked immediately", "swap the connection pool first"),
    ("Rotating a TLS certificate", "edge",
     "the previous certificate is served until expiry", "stage the chain before cutting over"),
    ("Failing over a read replica", "storage",
     "promote only after replication lag is under 5 seconds", "check lag, promote, then repoint"),
    ("Failing over a message broker", "messaging",
     "promote after the in-flight queue drains", "drain, promote, then resume producers"),
    ("Failing over an edge PoP", "edge",
     "withdraw the route before draining connections", "withdraw, drain, then decommission"),
    ("Restoring a database from snapshot", "storage",
     "restore into a new instance, never over the original", "restore aside, verify, then repoint"),
    ("Restoring object storage from versioning", "storage",
     "restore in place using version markers", "list versions, select, then promote"),
    ("Restoring a configuration bundle", "platform",
     "apply to a canary before the fleet", "canary, observe, then roll forward"),
    ("Draining a worker pool", "scheduler",
     "stop accepting new jobs before terminating workers", "cordon, drain, then terminate"),
    ("Draining a database connection pool", "storage",
     "quiesce writes before closing connections", "quiesce, close, then scale down"),
    ("Enabling a new metrics exporter", "telemetry",
     "validate label cardinality before enabling", "validate, enable, then alert"),
    ("Disabling a noisy alert rule", "telemetry",
     "silence before deleting so history is preserved", "silence, observe, then delete"),
    ("Expanding a shard set", "storage",
     "add capacity before rebalancing", "add, rebalance, then verify"),
    ("Shrinking a shard set", "storage",
     "rebalance away before removing capacity", "rebalance, verify, then remove"),
    ("Cutting over a DNS record", "edge",
     "lower the TTL a full TTL period in advance", "lower TTL, wait, then cut over"),
    ("Rolling back a schema migration", "storage",
     "roll back the application before the schema", "app first, schema second"),
    ("Promoting a canary deployment", "platform",
     "hold the canary for one full traffic cycle", "hold, compare, then promote"),
    ("Quarantining a compromised token", "auth",
     "revoke before notifying so the window closes first", "revoke, notify, then audit"),
    ("Rotating a service account key", "auth",
     "the previous key stays valid for 21 days", "issue, migrate callers, then retire"),
    ("Rotating an API gateway secret", "edge",
     "the previous secret is honoured for 48 hours", "deploy, verify, then revoke"),
    ("Failing over a search cluster", "search",
     "promote only once the index is fully warmed", "warm, promote, then repoint"),
    ("Failing over a cache tier", "platform",
     "warm the replacement before cutting traffic", "warm, cut, then retire"),
    ("Restoring a search index from snapshot", "search",
     "restore into a new index and swap the alias", "restore aside, verify, then swap alias"),
    ("Restoring a secrets store", "auth",
     "restore into an isolated namespace first", "isolate, restore, then re-key"),
    ("Draining an ingest pipeline", "ingest",
     "stop producers before flushing buffers", "stop, flush, then verify"),
    ("Draining a scheduler queue", "scheduler",
     "pause admission before cancelling in-flight jobs", "pause, wait, then cancel"),
    ("Enabling a new trace sampler", "telemetry",
     "shadow the sampler before making it authoritative", "shadow, compare, then enable"),
    ("Disabling a deprecated exporter", "telemetry",
     "confirm no dashboard depends on it before removal", "audit, disable, then delete"),
    ("Expanding a broker partition set", "messaging",
     "add partitions before repartitioning consumers", "add, repartition, then verify"),
    ("Shrinking a broker partition set", "messaging",
     "drain the partition before removing it", "drain, verify, then remove"),
    ("Cutting over a load balancer", "edge",
     "health-check the new target before shifting weight", "check, shift, then retire"),
    ("Rolling forward a failed migration", "storage",
     "roll the schema forward before the application", "schema first, app second"),
    ("Promoting a shadow deployment", "platform",
     "compare error budgets over a full week", "shadow, compare, then promote"),
    ("Revoking a compromised signing key", "auth",
     "revoke immediately; there is no grace period", "revoke, re-sign, then notify"),
    ("Rebuilding a corrupt replica", "storage",
     "detach before rebuilding so writes are not accepted", "detach, rebuild, then reattach"),
    ("Migrating a tenant between shards", "storage",
     "freeze writes for the tenant during the copy", "freeze, copy, then unfreeze"),
    ("Decommissioning a region", "edge",
     "withdraw routes a full day before shutting services", "withdraw, wait, then shut down"),
    ("Recovering from a bad config push", "platform",
     "roll back before diagnosing; diagnosis is not urgent", "roll back, stabilise, then diagnose"),
]

# API semantics adjacent to the pagination and quota golden items.
API_TOPICS = [
    ("Conditional requests", "An `If-Match` header that does not match returns 412 and the resource is untouched."),
    ("Partial responses", "A `fields` parameter narrows the representation; omitted fields are absent, not null."),
    ("Bulk endpoints", "A bulk call is not atomic: each element reports its own status in the response array."),
    ("Long-running operations", "A 202 returns an operation handle; polling it does not consume request quota."),
    ("Webhook delivery", "Delivery is at-least-once with exponential backoff for 24 hours, then the event is dropped."),
    ("Webhook ordering", "Events are not ordered across resources; use the sequence field within a resource."),
    ("Field deprecation", "A deprecated field keeps returning values for two minor versions before removal."),
    ("Error envelopes", "Every 4xx carries a machine-readable code; the human message is not part of the contract."),
    ("Sorting", "Sort keys must be total; ties are broken by resource id so pages cannot repeat an element."),
    ("Filtering", "Filters compose with AND only; OR requires separate calls merged client-side."),
    ("Time parameters", "All timestamps are RFC 3339 in UTC; a naive local time is rejected rather than assumed."),
    ("Compression", "Responses are gzipped above 1 KB; the client must send an Accept-Encoding header."),
    ("Client versioning", "The major version is in the path; minor changes are additive and unannounced."),
    ("Sandbox environment", "Sandbox shares the schema but not the data, and its quotas are a tenth of production."),
    ("Request tracing", "A client-supplied trace id is echoed on every response, including errors."),
    ("Cursor lifetime", "A cursor expires 15 minutes after issue; an expired cursor returns 410, not an empty page."),
    ("Page size limits", "Page size is capped at 200; a larger request is clamped silently rather than rejected."),
    ("Concurrent writes", "Two writers to one resource are resolved last-writer-wins unless a version is supplied."),
    ("Soft deletes", "A deleted resource is retrievable by id for 30 days and absent from list endpoints immediately."),
    ("Quota accounting", "Quota is consumed on request receipt, not on success; a 500 still costs a unit."),
    ("Burst allowance", "A short burst above the sustained rate is permitted once per hour and is not announced."),
    ("Retry semantics", "Only 429 and 503 are safe to retry automatically; a 409 means the request will never succeed."),
    ("Idempotency scope", "An idempotency key is scoped to one endpoint; reusing it elsewhere is a separate operation."),
    ("Batch limits", "A batch is capped at 50 elements; exceeding it fails the whole batch rather than truncating."),
    ("Nested expansion", "Expansion is one level deep; a nested expand parameter is ignored without warning."),
    ("Null semantics", "An explicit null clears a field; omitting the field leaves it unchanged."),
    ("List consistency", "List endpoints are eventually consistent; a resource may be absent for up to 5 seconds after creation."),
    ("Header casing", "Header names are case-insensitive, but custom header values are not normalised."),
    ("Content negotiation", "Only JSON is served; an Accept header requesting anything else returns 406."),
    ("Rate limit headers", "Remaining quota is reported per window, and the window resets on a fixed boundary, not a sliding one."),
]

# Regional policy variants: same policy vocabulary as the handbook, different
# jurisdictions and numbers, so a handbook question has plausible rivals.
REGIONS = [
    ("EMEA", "euro", "EUR", 2, "Frankfurt"),
    ("APAC", "yen", "JPY", 3, "Singapore"),
    ("LATAM", "real", "BRL", 2, "Sao Paulo"),
    ("NORDIC", "krona", "SEK", 2, "Stockholm"),
    ("ANZ", "dollar", "AUD", 3, "Sydney"),
    ("MENA", "dirham", "AED", 4, "Dubai"),
    ("CEE", "zloty", "PLN", 2, "Warsaw"),
    ("CANADA", "dollar", "CAD", 3, "Toronto"),
]
POLICY_TOPICS = [
    ("Procurement thresholds", "A purchase order above {n},000 {cur} needs a second signature from the {region} finance lead."),
    ("Capital expenditure", "Capex above {n}0,000 {cur} goes to the quarterly review board rather than a line manager."),
    ("Contractor engagement", "A contractor engaged for more than {n} months is reclassified and needs a headcount slot."),
    ("Data residency", "{region} customer records stay in the {city} region and are not replicated outside it."),
    ("Local holidays", "{region} observes {n} additional public holidays that are not in the global calendar."),
    ("Expense receipts", "Receipts are required above {n}0 {cur}; below that a line item on the report is enough."),
    ("Travel class", "Flights over {n} hours may be booked in premium economy without prior approval."),
    ("Equipment refresh", "Laptops are refreshed every {n} years, or sooner if the warranty lapses."),
]


def real_codes():
    """Codes defined by the hand-written fixture, which must never be redefined.

    Generating ORB-6203 here once produced a second, contradictory definition
    of a code that ci-code-6203 asks about by name — turning a distractor into
    a rival correct answer and the item into a coin flip. Read the real file
    rather than hardcoding the list, so this stays true when the fixture grows.
    """
    text = (OUT / "error-codes.csv").read_text(encoding="utf-8")
    return {line.split(",", 1)[0] for line in text.splitlines()[1:] if line.strip()}


def gen_error_codes():
    taken = real_codes()
    rows = ["code,component,meaning,first action"]
    for comp, base, entries in FAMILIES:
        n = 0
        for meaning, action in entries:
            # Step past any code the real fixture already defines.
            while f"ORB-{base + (n * 7) + 3}" in taken:
                n += 1
            code = f"ORB-{base + (n * 7) + 3}"
            taken.add(code)
            rows.append(f"{code},{comp},{meaning},{action}")
            n += 1
    return "\n".join(rows) + "\n"


def gen_inventory():
    out = ["Orbital Dynamics - Annexe Bench Inventory (fictional)", ""]
    serial = 500  # starts above lab-inventory.txt's range; no serial collides
    for bench, (name, prefix, model) in enumerate(INSTRUMENTS, start=3):
        out.append(f"Bench {bench} (annexe)")
        for unit in range(3):
            serial += 13
            out.append(f"  1x  {name} {prefix}-{model}, serial RF-{serial:05d}")
        out.append(f"  Calibration due: bench {bench} is recertified every {12 + bench % 6} months")
        out.append("")
    return "\n".join(out)


def gen_runbooks():
    out = ["# Orbital Dynamics - Legacy Runbooks (fictional)", "",
           "Superseded procedures retained for audit. Do not follow these for", "current systems.", ""]
    for title, owner, rule, order in RUNBOOKS:
        out.append(f"## {title}")
        out.append("")
        out.append(f"Owner: {owner} team.")
        out.append("")
        out.append(f"Constraint: {rule}.")
        out.append("")
        out.append(f"Order of operations: {order}. Performing these out of order "
                   f"leaves the system in a state this runbook cannot recover, and "
                   f"the recovery path is a full rebuild.")
        out.append("")
        out.append(f"Verification: confirm the change took effect before closing the "
                   f"ticket. An unverified {title.lower()} is treated as not done.")
        out.append("")
    return "\n".join(out)


def gen_api_notes():
    out = ["# Orbital Dynamics API - v1 Notes (fictional)", "",
           "The v1 API is frozen. These notes describe its behaviour only.", ""]
    for topic, body in API_TOPICS:
        out.append(f"## {topic}")
        out.append("")
        out.append(body)
        out.append("")
        out.append("This behaviour is specific to v1 and was changed in v2. Client "
                   "libraries pinned to v1 keep the semantics described here.")
        out.append("")
    return "\n".join(out)


def gen_regional_handbook():
    out = ["# Orbital Dynamics - Regional Policy Supplements (fictional)", "",
           "Regional supplements to the global handbook. Where a supplement and",
           "the global handbook disagree, the global handbook governs.", ""]
    for region, _money, cur, n, city in REGIONS:
        for topic, tmpl in POLICY_TOPICS:
            out.append(f"## {region} - {topic}")
            out.append("")
            out.append(tmpl.format(n=n + len(topic) % 5, cur=cur, region=region, city=city))
            out.append("")
            out.append(f"This supplement applies to staff whose employing entity is "
                       f"registered in {region}. It does not apply to visitors or to "
                       f"staff on short-term assignment from another region.")
            out.append("")
    return "\n".join(out)


FILES = {
    "error-codes-legacy.csv": gen_error_codes,
    "lab-inventory-annexe.txt": gen_inventory,
    "runbooks-legacy.md": gen_runbooks,
    "api-notes-v1.md": gen_api_notes,
    "handbook-regional.md": gen_regional_handbook,
}

def check():
    """The one property that makes these distractors and not a second answer key.

    A distractor may look like the answer; it must never BE the answer. The
    cheapest way for that to go wrong is an identifier collision, so assert on
    the two identifier spaces the golden set actually queries by name: error
    codes and instrument serials.
    """
    generated = gen_error_codes().splitlines()[1:]
    gen_ids = {line.split(",", 1)[0] for line in generated if line.strip()}
    clash = gen_ids & real_codes()
    assert not clash, f"generated error codes redefine real ones: {sorted(clash)}"

    real_serials = set()
    for line in (OUT / "lab-inventory.txt").read_text(encoding="utf-8").split():
        if line.startswith("RF-"):
            real_serials.add(line.rstrip(","))
    gen_serials = {w.rstrip(",") for w in gen_inventory().split() if w.startswith("RF-")}
    clash = gen_serials & real_serials
    assert not clash, f"generated serials collide with real ones: {sorted(clash)}"

    print(f"ok: {len(gen_ids)} codes, {len(gen_serials)} serials, no collisions "
          f"with {len(real_codes())} real codes / {len(real_serials)} real serials")


if __name__ == "__main__":
    check()
    for name, fn in FILES.items():
        path = OUT / name
        path.write_text(fn(), encoding="utf-8", newline="\n")
        print(f"{path.relative_to(OUT.parent.parent)}: {len(fn().splitlines())} lines")
