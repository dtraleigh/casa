"""Tool definitions and dispatch for Mycroft.

Phase 1 has no tools — this module exists so prompt assembly can ask for tool
descriptions without a conditional. Phase 3 (weather) populates it.
"""


def describe_registered_tools() -> str:
    """Return a human-readable description of Mycroft's tools for the system
    prompt. Empty in Phase 1 (no tools registered yet)."""
    return ""
