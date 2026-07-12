# API Authentication Guide

## Overview

All requests to the Internal Platform API must be authenticated using an API key. Keys are issued per-service and scoped to specific permission levels: read-only, read-write, and admin.

## Generating an API Key

Navigate to Settings > API Keys in the admin console and click "Generate New Key". Each key is shown only once at creation time — store it in your service's secrets manager immediately. Keys cannot be retrieved after the initial display, only revoked and regenerated.

## Authentication Header

Every request must include the key in the `X-API-Key` header:

```
X-API-Key: your_key_here
```

Requests without this header return a 401 Unauthorized response. Requests with an invalid or revoked key also return 401, with an error body specifying `invalid_key` or `revoked_key`.

## Rate Limits

Read-only keys are limited to 1000 requests per minute. Read-write keys are limited to 300 requests per minute. Admin keys are limited to 100 requests per minute, since admin operations are typically bulk or destructive. Exceeding your rate limit returns a 429 status with a `Retry-After` header indicating seconds until the limit resets.

## Key Rotation Policy

All API keys expire automatically after 90 days. The platform sends a warning email 7 days before expiration. Expired keys return a 401 with error code `expired_key`. There is no grace period — rotate keys before expiry to avoid downtime.

## Revoking a Key

Keys can be revoked immediately from the admin console or via the `DELETE /v1/keys/{key_id}` endpoint. Revocation is irreversible and takes effect within 60 seconds across all regions.
