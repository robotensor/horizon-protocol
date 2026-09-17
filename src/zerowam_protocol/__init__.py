"""The Zero-WAM competition's protocol: the socket between a benchmark and a model, and the files
they exchange.

A benchmark drives a model it cannot import - the simulator and the model pin different stacks, and
they usually do not even run on the same machine - so the two meet on a socket. This distribution is
both ends of that socket and the two file formats around it, and nothing else: it knows no
benchmark, no channel name, no simulator and no model.

- `Policy`: what a model runtime implements.
- `python -m zerowam_protocol.serve`: serves that policy to one client.
- `zerowam_protocol.client.RemotePolicy`: the client a benchmark drives it with.
- `zerowam_protocol.wire`: the message format between them.
- `zerowam_protocol.bundle`: the demonstration bundle a benchmark writes once per unit.
- `zerowam_protocol.result`: the result a benchmark writes for every evaluated unit.
- `zerowam_protocol.stubs`: `ZeroPolicy` and `ReplayPolicy`, the model-free smoke tests.
"""

from .errors import BundleError, PolicySpecError, PolicyUnavailable, WireError
from .policy import ACTION_TYPES, Policy

__version__ = "0.1.0.dev0"

__all__ = [
    "ACTION_TYPES",
    "BundleError",
    "Policy",
    "PolicySpecError",
    "PolicyUnavailable",
    "WireError",
    "__version__",
]
