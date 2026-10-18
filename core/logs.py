"""Helpers for the in-browser log viewer.

The set of viewable files is derived from settings.LOGGING: every handler with
a ``filename`` is a log, plus its rotated ``.1`` .. ``.backupCount`` siblings.
Callers pass a handler key and backup index, never a path, so nothing outside
that whitelist can be read.
"""
import os
import re
from datetime import datetime

from django.conf import settings
from django.utils import timezone

LEVELS = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}

# Matches the 'verbose' formatter: [2026-10-18 09:15:02] ERROR name: message
ENTRY_RE = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] ([A-Z]+) ([^:\s]+): ?(.*)$')


def _file_handlers():
    handlers = settings.LOGGING.get('handlers', {})
    return {key: cfg for key, cfg in handlers.items() if cfg.get('filename')}


def file_info(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {
        'size': st.st_size,
        'mtime': datetime.fromtimestamp(st.st_mtime, tz=timezone.get_current_timezone()),
    }


def available_logs():
    """Every file-backed handler, with the rotated backups that exist on disk."""
    logs = []
    for key, cfg in _file_handlers().items():
        path = str(cfg['filename'])
        backups = []
        for i in range(1, int(cfg.get('backupCount', 0)) + 1):
            info = file_info(f'{path}.{i}')
            if info:
                backups.append({'index': i, 'name': f'{os.path.basename(path)}.{i}', **info})
        logs.append({
            'key': key,
            'label': os.path.basename(path),
            'path': path,
            'info': file_info(path),
            'backups': backups,
        })
    return logs


def resolve_log(key, backup=0):
    """Path for a handler key + backup index, or None if outside the whitelist."""
    cfg = _file_handlers().get(key)
    if cfg is None or not 0 <= backup <= int(cfg.get('backupCount', 0)):
        return None
    path = str(cfg['filename'])
    return f'{path}.{backup}' if backup else path


def tail_lines(path, n, block_size=64 * 1024):
    """Last n lines of a file, reading backwards so large logs aren't loaded whole."""
    chunks = []
    newlines = 0
    with open(path, 'rb') as f:
        pos = f.seek(0, os.SEEK_END)
        # n + 1 newlines guarantees n complete lines even with a trailing newline.
        while pos > 0 and newlines <= n:
            size = min(block_size, pos)
            pos -= size
            f.seek(pos)
            chunk = f.read(size)
            chunks.append(chunk)
            newlines += chunk.count(b'\n')
    data = b''.join(reversed(chunks))
    return data.decode('utf-8', errors='replace').splitlines()[-n:]


def group_entries(lines):
    """Group raw lines into entries; unmatched lines (tracebacks) join the previous one.

    Lines before the first header (e.g. a traceback cut off by the tail) become
    an entry with level None.
    """
    entries = []
    for line in lines:
        m = ENTRY_RE.match(line)
        if m:
            timestamp, level, name, message = m.groups()
            entries.append({'timestamp': timestamp, 'level': level, 'logger': name, 'message': message})
        elif entries:
            entries[-1]['message'] += '\n' + line
        else:
            entries.append({'timestamp': None, 'level': None, 'logger': None, 'message': line})
    return entries


def filter_entries(entries, min_level=None, query=None):
    """Keep entries at or above min_level whose text contains query (case-insensitive)."""
    threshold = LEVELS.get(min_level)
    needle = query.lower() if query else None
    result = []
    for e in entries:
        if threshold is not None and LEVELS.get(e['level'], 0) < threshold:
            continue
        if needle and needle not in f"{e['logger'] or ''} {e['message']}".lower():
            continue
        result.append(e)
    return result
