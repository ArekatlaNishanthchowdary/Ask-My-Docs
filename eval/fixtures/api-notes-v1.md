# Orbital Dynamics API - v1 Notes (fictional)

The v1 API is frozen. These notes describe its behaviour only.

## Conditional requests

An `If-Match` header that does not match returns 412 and the resource is untouched.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Partial responses

A `fields` parameter narrows the representation; omitted fields are absent, not null.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Bulk endpoints

A bulk call is not atomic: each element reports its own status in the response array.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Long-running operations

A 202 returns an operation handle; polling it does not consume request quota.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Webhook delivery

Delivery is at-least-once with exponential backoff for 24 hours, then the event is dropped.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Webhook ordering

Events are not ordered across resources; use the sequence field within a resource.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Field deprecation

A deprecated field keeps returning values for two minor versions before removal.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Error envelopes

Every 4xx carries a machine-readable code; the human message is not part of the contract.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Sorting

Sort keys must be total; ties are broken by resource id so pages cannot repeat an element.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Filtering

Filters compose with AND only; OR requires separate calls merged client-side.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Time parameters

All timestamps are RFC 3339 in UTC; a naive local time is rejected rather than assumed.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Compression

Responses are gzipped above 1 KB; the client must send an Accept-Encoding header.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Client versioning

The major version is in the path; minor changes are additive and unannounced.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Sandbox environment

Sandbox shares the schema but not the data, and its quotas are a tenth of production.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Request tracing

A client-supplied trace id is echoed on every response, including errors.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Cursor lifetime

A cursor expires 15 minutes after issue; an expired cursor returns 410, not an empty page.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Page size limits

Page size is capped at 200; a larger request is clamped silently rather than rejected.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Concurrent writes

Two writers to one resource are resolved last-writer-wins unless a version is supplied.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Soft deletes

A deleted resource is retrievable by id for 30 days and absent from list endpoints immediately.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Quota accounting

Quota is consumed on request receipt, not on success; a 500 still costs a unit.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Burst allowance

A short burst above the sustained rate is permitted once per hour and is not announced.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Retry semantics

Only 429 and 503 are safe to retry automatically; a 409 means the request will never succeed.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Idempotency scope

An idempotency key is scoped to one endpoint; reusing it elsewhere is a separate operation.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Batch limits

A batch is capped at 50 elements; exceeding it fails the whole batch rather than truncating.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Nested expansion

Expansion is one level deep; a nested expand parameter is ignored without warning.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Null semantics

An explicit null clears a field; omitting the field leaves it unchanged.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## List consistency

List endpoints are eventually consistent; a resource may be absent for up to 5 seconds after creation.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Header casing

Header names are case-insensitive, but custom header values are not normalised.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Content negotiation

Only JSON is served; an Accept header requesting anything else returns 406.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.

## Rate limit headers

Remaining quota is reported per window, and the window resets on a fixed boundary, not a sliding one.

This behaviour is specific to v1 and was changed in v2. Client libraries pinned to v1 keep the semantics described here.
