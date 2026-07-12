# Deployment Guide

## Environments

The platform runs three environments: `dev`, `staging`, and `production`. Each environment has its own database, message queue, and config namespace. Deployments always flow dev -> staging -> production; direct-to-production deploys are blocked by CI unless a `hotfix` label is applied to the PR.

## Deployment Pipeline

Every merge to `main` triggers a staging deployment automatically via GitHub Actions. Production deployments require a manual approval step from a team lead, triggered by tagging a release (`v1.2.3` format). The pipeline runs the full test suite, builds a Docker image, pushes it to the internal registry, and performs a rolling update with zero downtime.

## Rollback Procedure

If a production deployment causes errors, run `./scripts/rollback.sh <previous_version_tag>`. This reverts the Kubernetes deployment to the previous image and rolls back any accompanying database migrations that were marked reversible. Migrations not marked reversible require manual intervention — check the migration file for a `# irreversible` comment before assuming rollback is safe.

## Health Checks

Every service exposes a `/healthz` endpoint. Kubernetes liveness probes hit this every 10 seconds; readiness probes hit it every 5 seconds during startup. A service is considered unhealthy after 3 consecutive failures and is automatically restarted.

## Configuration Management

Environment-specific config lives in `config/{env}.yaml`. Secrets are never stored in these files — they are injected at runtime from the secrets manager using the `SECRET_REF:` prefix syntax. Config changes to production require the same PR approval process as code changes.

## Scaling

Services scale horizontally based on CPU utilization, targeting 70% average CPU across pods. The autoscaler has a minimum of 2 replicas and a maximum of 20 replicas per service in production. Staging is capped at 3 replicas to control cost.
