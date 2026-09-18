# Changelog

All notable changes to this distribution. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Protocol 2: a policy may declare `observe_every = N`, returned in the reply to `hello`. The
  benchmark then records an observation after every N-th action of a chunk and sends them, stacked
  oldest first, with the next `act`; a chunk whose length is not a multiple of N is refused.
  `zerowam_protocol.observe` holds the rule, `stack` and `check_chunk`. Under protocol 1 a model
  that conditions on what its own chunk did - Zero-WAM does - could only run blind. N = 0, the
  default, is protocol 1's behaviour. Both ends check the version. (#1)

- The wire format, `RemotePolicy` and `python -m zerowam_protocol.serve`, ported from `icil-policy`
  in ICIL-competition-orchestrator. A policy is named by `module:Class` and built with
  `--policy-arg` values; submissions here are weights only, so there is no repository manifest.
- `bundle`: the demonstration bundle written once per unit (`demo.json`, `demo_frames.npz`,
  `private/`), with a sha256 per file and `public_arrays` as the only thing a policy is given.
- `result`: `result.json` with `outcome` and `void_cause`.
- `stubs`: `ZeroPolicy` and `ReplayPolicy`, so a benchmark's chain can be smoke-tested with no model.
- `serve --max-sessions N`: one server takes N clients one after another and keeps the policy it
  built. A submission is evaluated over many units, and loading tens of gigabytes of weights per
  unit would cost more than the units do.

### Changed

- A bundle's arrays are compressed: an uncompressed click_bell unit was 103 MB, compressed 22 MB.
