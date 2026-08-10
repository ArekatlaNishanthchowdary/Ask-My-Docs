# Orbital Dynamics - Legacy Runbooks (fictional)

Superseded procedures retained for audit. Do not follow these for
current systems.

## Rotating a storage encryption key

Owner: storage team.

Constraint: the previous key stays readable for 30 days.

Order of operations: rotate, verify, then retire. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified rotating a storage encryption key is treated as not done.

## Rotating a webhook signing secret

Owner: integrations team.

Constraint: the previous secret stays valid for 7 days.

Order of operations: publish the new secret before revoking the old. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified rotating a webhook signing secret is treated as not done.

## Rotating a database credential

Owner: storage team.

Constraint: the previous credential is revoked immediately.

Order of operations: swap the connection pool first. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified rotating a database credential is treated as not done.

## Rotating a TLS certificate

Owner: edge team.

Constraint: the previous certificate is served until expiry.

Order of operations: stage the chain before cutting over. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified rotating a tls certificate is treated as not done.

## Failing over a read replica

Owner: storage team.

Constraint: promote only after replication lag is under 5 seconds.

Order of operations: check lag, promote, then repoint. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified failing over a read replica is treated as not done.

## Failing over a message broker

Owner: messaging team.

Constraint: promote after the in-flight queue drains.

Order of operations: drain, promote, then resume producers. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified failing over a message broker is treated as not done.

## Failing over an edge PoP

Owner: edge team.

Constraint: withdraw the route before draining connections.

Order of operations: withdraw, drain, then decommission. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified failing over an edge pop is treated as not done.

## Restoring a database from snapshot

Owner: storage team.

Constraint: restore into a new instance, never over the original.

Order of operations: restore aside, verify, then repoint. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified restoring a database from snapshot is treated as not done.

## Restoring object storage from versioning

Owner: storage team.

Constraint: restore in place using version markers.

Order of operations: list versions, select, then promote. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified restoring object storage from versioning is treated as not done.

## Restoring a configuration bundle

Owner: platform team.

Constraint: apply to a canary before the fleet.

Order of operations: canary, observe, then roll forward. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified restoring a configuration bundle is treated as not done.

## Draining a worker pool

Owner: scheduler team.

Constraint: stop accepting new jobs before terminating workers.

Order of operations: cordon, drain, then terminate. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified draining a worker pool is treated as not done.

## Draining a database connection pool

Owner: storage team.

Constraint: quiesce writes before closing connections.

Order of operations: quiesce, close, then scale down. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified draining a database connection pool is treated as not done.

## Enabling a new metrics exporter

Owner: telemetry team.

Constraint: validate label cardinality before enabling.

Order of operations: validate, enable, then alert. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified enabling a new metrics exporter is treated as not done.

## Disabling a noisy alert rule

Owner: telemetry team.

Constraint: silence before deleting so history is preserved.

Order of operations: silence, observe, then delete. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified disabling a noisy alert rule is treated as not done.

## Expanding a shard set

Owner: storage team.

Constraint: add capacity before rebalancing.

Order of operations: add, rebalance, then verify. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified expanding a shard set is treated as not done.

## Shrinking a shard set

Owner: storage team.

Constraint: rebalance away before removing capacity.

Order of operations: rebalance, verify, then remove. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified shrinking a shard set is treated as not done.

## Cutting over a DNS record

Owner: edge team.

Constraint: lower the TTL a full TTL period in advance.

Order of operations: lower TTL, wait, then cut over. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified cutting over a dns record is treated as not done.

## Rolling back a schema migration

Owner: storage team.

Constraint: roll back the application before the schema.

Order of operations: app first, schema second. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified rolling back a schema migration is treated as not done.

## Promoting a canary deployment

Owner: platform team.

Constraint: hold the canary for one full traffic cycle.

Order of operations: hold, compare, then promote. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified promoting a canary deployment is treated as not done.

## Quarantining a compromised token

Owner: auth team.

Constraint: revoke before notifying so the window closes first.

Order of operations: revoke, notify, then audit. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified quarantining a compromised token is treated as not done.

## Rotating a service account key

Owner: auth team.

Constraint: the previous key stays valid for 21 days.

Order of operations: issue, migrate callers, then retire. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified rotating a service account key is treated as not done.

## Rotating an API gateway secret

Owner: edge team.

Constraint: the previous secret is honoured for 48 hours.

Order of operations: deploy, verify, then revoke. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified rotating an api gateway secret is treated as not done.

## Failing over a search cluster

Owner: search team.

Constraint: promote only once the index is fully warmed.

Order of operations: warm, promote, then repoint. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified failing over a search cluster is treated as not done.

## Failing over a cache tier

Owner: platform team.

Constraint: warm the replacement before cutting traffic.

Order of operations: warm, cut, then retire. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified failing over a cache tier is treated as not done.

## Restoring a search index from snapshot

Owner: search team.

Constraint: restore into a new index and swap the alias.

Order of operations: restore aside, verify, then swap alias. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified restoring a search index from snapshot is treated as not done.

## Restoring a secrets store

Owner: auth team.

Constraint: restore into an isolated namespace first.

Order of operations: isolate, restore, then re-key. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified restoring a secrets store is treated as not done.

## Draining an ingest pipeline

Owner: ingest team.

Constraint: stop producers before flushing buffers.

Order of operations: stop, flush, then verify. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified draining an ingest pipeline is treated as not done.

## Draining a scheduler queue

Owner: scheduler team.

Constraint: pause admission before cancelling in-flight jobs.

Order of operations: pause, wait, then cancel. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified draining a scheduler queue is treated as not done.

## Enabling a new trace sampler

Owner: telemetry team.

Constraint: shadow the sampler before making it authoritative.

Order of operations: shadow, compare, then enable. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified enabling a new trace sampler is treated as not done.

## Disabling a deprecated exporter

Owner: telemetry team.

Constraint: confirm no dashboard depends on it before removal.

Order of operations: audit, disable, then delete. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified disabling a deprecated exporter is treated as not done.

## Expanding a broker partition set

Owner: messaging team.

Constraint: add partitions before repartitioning consumers.

Order of operations: add, repartition, then verify. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified expanding a broker partition set is treated as not done.

## Shrinking a broker partition set

Owner: messaging team.

Constraint: drain the partition before removing it.

Order of operations: drain, verify, then remove. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified shrinking a broker partition set is treated as not done.

## Cutting over a load balancer

Owner: edge team.

Constraint: health-check the new target before shifting weight.

Order of operations: check, shift, then retire. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified cutting over a load balancer is treated as not done.

## Rolling forward a failed migration

Owner: storage team.

Constraint: roll the schema forward before the application.

Order of operations: schema first, app second. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified rolling forward a failed migration is treated as not done.

## Promoting a shadow deployment

Owner: platform team.

Constraint: compare error budgets over a full week.

Order of operations: shadow, compare, then promote. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified promoting a shadow deployment is treated as not done.

## Revoking a compromised signing key

Owner: auth team.

Constraint: revoke immediately; there is no grace period.

Order of operations: revoke, re-sign, then notify. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified revoking a compromised signing key is treated as not done.

## Rebuilding a corrupt replica

Owner: storage team.

Constraint: detach before rebuilding so writes are not accepted.

Order of operations: detach, rebuild, then reattach. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified rebuilding a corrupt replica is treated as not done.

## Migrating a tenant between shards

Owner: storage team.

Constraint: freeze writes for the tenant during the copy.

Order of operations: freeze, copy, then unfreeze. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified migrating a tenant between shards is treated as not done.

## Decommissioning a region

Owner: edge team.

Constraint: withdraw routes a full day before shutting services.

Order of operations: withdraw, wait, then shut down. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified decommissioning a region is treated as not done.

## Recovering from a bad config push

Owner: platform team.

Constraint: roll back before diagnosing; diagnosis is not urgent.

Order of operations: roll back, stabilise, then diagnose. Performing these out of order leaves the system in a state this runbook cannot recover, and the recovery path is a full rebuild.

Verification: confirm the change took effect before closing the ticket. An unverified recovering from a bad config push is treated as not done.
