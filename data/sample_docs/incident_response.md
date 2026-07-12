# Incident Response Runbook

## Severity Levels

Incidents are classified into three severities. SEV-1 means full outage or data loss risk affecting all users — page the on-call engineer immediately and notify the incident commander within 5 minutes. SEV-2 means significant degradation affecting a subset of users or features — post in #incidents and assign an owner within 15 minutes. SEV-3 means minor issues with workarounds available — track in the issue tracker, no immediate paging required.

## On-Call Rotation

On-call rotates weekly, Monday to Monday, and is tracked in PagerDuty. The primary on-call must acknowledge pages within 5 minutes; if unacknowledged, PagerDuty escalates to the secondary on-call after 5 minutes, then to the engineering manager after another 10 minutes.

## Incident Commander Role

For any SEV-1, the first responder appoints an Incident Commander (IC) who is not necessarily the person fixing the issue. The IC's job is coordination: maintaining the timeline, communicating status updates every 30 minutes, and deciding when to escalate further or bring in additional teams. The IC does not write code during the incident.

## Communication

All SEV-1 and SEV-2 incidents get a dedicated Slack channel named `#incident-YYYY-MM-DD-short-description`. Status updates go in this channel and are also mirrored to the status page for customer-facing incidents. Do not discuss incident details in DMs — everything must be visible in the incident channel for the postmortem record.

## Postmortems

Every SEV-1 and SEV-2 incident requires a written postmortem within 3 business days. Postmortems are blameless by policy — they focus on systemic causes and process gaps, not individual mistakes. Required sections: timeline, root cause, impact (users affected, duration), and at least 3 concrete action items with owners and due dates.

## Postmortem Action Items

Action items from postmortems are tracked in the `#postmortem-action-items` board and reviewed weekly. Action items open for more than 30 days are escalated to the relevant team's manager automatically.
