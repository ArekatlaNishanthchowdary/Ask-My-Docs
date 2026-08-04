# Orbital Dynamics Internal Handbook

This handbook is fictional. It exists so continuous integration has a corpus it
is allowed to publish, and every fact in it was invented for that purpose.

## Expenses

Staff may book travel without prior approval up to 1,200 EUR per trip. Anything
above that needs a written sign-off from a department head before the booking is
made, not after.

Receipts must be submitted within 30 days. Claims older than 30 days are paid
only at the discretion of the finance lead, and repeated late claims are raised
at the quarterly review.

Meals while travelling are reimbursed at a flat 45 EUR per day. The flat rate
replaces itemised meal receipts entirely, so there is no benefit to keeping
them.

## On-call

The on-call rotation runs weekly, handing over at 10:00 on Wednesdays rather
than at a week boundary, so that a handover never lands on a Monday morning
alongside the weekly planning meeting.

An engineer on call is expected to acknowledge a page within 15 minutes. If a
page goes unacknowledged for 15 minutes it escalates automatically to the
secondary, and after a further 10 minutes to the engineering manager.

Nobody is scheduled on call for two consecutive weeks. The rotation tool refuses
to generate such a schedule rather than warning about it.

## Deployments

Deployments to production are frozen from 16:00 on Thursday until 09:00 on
Monday. The freeze exists because the team is smallest at the weekend, not
because Friday deployments fail more often.

A deployment requires two approvals: one from a reviewer who read the diff, and
one from whoever is currently on call. The on-call approval is about capacity to
respond, not about code quality, so the two cannot come from the same person.

Rollback is expected to take under 4 minutes. If a rollback would take longer
than that, the change must ship behind a feature flag instead.

## Access requests

Access to production data stores is granted for 90 days and then expires. There
is no renewal path — an expired grant is requested again from scratch, which
keeps the list of who has access honest rather than accumulating.

Requests are approved by the data owner named in the service catalogue, never by
the requester's own manager.
