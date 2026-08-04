# Orbital API Notes

Fictional API reference notes.

## Versioning

The API is versioned in the path, `/v2/…`. A version is supported for 18 months
after its successor ships, and the sunset date is returned in a `Sunset` header
on every response rather than announced only by email.

## Pagination

Cursors are opaque and single-use. A cursor replayed after its page has already
been fetched returns the same page rather than erroring, so a client retrying a
timed-out request cannot silently skip records.

Page size defaults to 100 and caps at 1,000. Asking for more than the cap is not
an error; the response returns the cap and says so in a `X-Page-Size-Clamped`
header.

## Idempotency

Write endpoints accept an `Idempotency-Key`. Keys are retained for 24 hours. A
replayed key returns the original response including its original status code,
which means a replayed create returns 201 rather than 200.

## Rate limits

Limits are per token, not per tenant, so issuing a second token doubles the
budget. This is intentional: it lets a batch job run without starving the
interactive traffic that shares its tenant.

A limited request returns 429 with `Retry-After` in seconds. The value is
computed from the actual window, not a fixed backoff, so honouring it is
strictly better than exponential backoff.

## Webhooks

Deliveries retry for 24 hours with exponential backoff, then stop. A stopped
subscription is disabled rather than deleted, so the endpoint history stays
readable while the owner investigates.

Payloads are signed. The signature covers the raw body, so any middleware that
reformats JSON before verification breaks it.

## Errors

Every error carries a machine-readable `code` matching the ORB-nnnn table, a
human `message`, and a `request_id`. Support asks for the `request_id` first,
because it resolves to the trace and the message alone rarely does.
