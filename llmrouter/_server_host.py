"""Shared host handling for socket binds and emitted server URLs."""

import ipaddress


_INVALID_BRACKETED_HOST = (
    "serve.host must be an IPv6 literal in brackets, such as '[::]' or '[::1]'; "
    "IPv4 addresses and hostnames must be unbracketed"
)


def normalize_bind_host(host):
    """Keep URI brackets out of socket host values while accepting bracketed IPv6 config."""
    if not isinstance(host, str) or not (host.startswith("[") or host.endswith("]")):
        return host

    if not (host.startswith("[") and host.endswith("]")):
        raise ValueError(_INVALID_BRACKETED_HOST)

    address = host[1:-1]
    try:
        ipaddress.IPv6Address(address)
    except ipaddress.AddressValueError as error:
        raise ValueError(_INVALID_BRACKETED_HOST) from error
    return address


def format_url_host(host):
    """Turn a bind host into the host portion of a client-usable server URL."""
    if not isinstance(host, str):
        return host

    try:
        ipaddress.IPv6Address(host)
    except ipaddress.AddressValueError:
        return host

    literal = host.replace("%", "%25", 1) if "%" in host else host
    return f"[{literal}]"
