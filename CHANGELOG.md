# Changelog

All notable changes to this distribution. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The wire format, `RemotePolicy` and `python -m zerowam_protocol.serve`, ported from `icil-policy`
  in ICIL-competition-orchestrator. A policy is named by `module:Class` and built with
  `--policy-arg` values; submissions here are weights only, so there is no repository manifest.
- `bundle`: the demonstration bundle written once per unit (`demo.json`, `demo_frames.npz`,
  `private/`), with a sha256 per file and `public_arrays` as the only thing a policy is given.
- `result`: `result.json` with `outcome` and `void_cause`.
- `stubs`: `ZeroPolicy` and `ReplayPolicy`, so a benchmark's chain can be smoke-tested with no model.
