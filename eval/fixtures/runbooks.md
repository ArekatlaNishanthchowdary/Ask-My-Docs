# Orbital Dynamics Runbooks

Fictional operational runbooks, written so retrieval has short well-separated
sections to discriminate between.

## Restarting the ingest workers

Drain the queue first with `orbctl ingest drain`, wait for the in-flight count
to reach zero, then restart. Restarting with work in flight produces duplicate
documents, because the worker acknowledges after indexing rather than before.

## Rotating the signing key

Generate the new key, publish it to the JWKS endpoint, and wait a full 14 days
before retiring the old one. Callers cache the key set for up to a week, so
retiring early produces ORB-4419 for anyone who has not refreshed.

## Recovering a degraded storage shard

A shard reporting degraded still serves reads. Do not fail it over on the first
alert — the recovery path rebuilds from the replica automatically and completes
in about 20 minutes for a shard under 500 GB. Fail over only if recovery has not
started within 5 minutes.

## Clearing a stuck deployment

A deployment stuck in pending for more than 10 minutes is almost always waiting
on the second approval rather than on infrastructure. Check the approval state
before touching the cluster.

## Responding to a cardinality alert

Find the label driving the cardinality with `orbctl metrics top-labels`, drop
that label at the exporter, and only then re-enable. Re-enabling before dropping
the label refills the series budget within minutes.

## Restoring a deleted index

Snapshots are taken hourly and kept for 7 days. Restoring is not in place: the
snapshot is restored to a new index name and traffic is swung over, because an
in-place restore has no way back if the snapshot is also bad.

## Handling a stale currency table

Billing refuses to issue an invoice when the conversion table is more than 24
hours old, raising ORB-5107. Refresh the table and re-run the billing job; there
is no way to override the freshness check, and that is deliberate.

## Onboarding a new tenant

Create the tenant, set its cardinality limit before any traffic arrives, then
issue credentials. A tenant created without a limit inherits the default of
50,000 series, which is high enough that the first alert arrives at 3am.

## Draining a region

Shift traffic first, then drain. Draining first makes the region reject requests
that traffic management is still sending it, which shows up as a spike in 503s
that looks like an outage rather than a planned drain.
