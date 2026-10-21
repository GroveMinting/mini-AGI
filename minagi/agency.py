"""Sandboxed, permissioned tools with an append-only receipt ledger.

This is a capability sandbox, not a container. A tool can only touch files
under roots named in the policy, and only when its exact name was granted.
There is no shell, no arbitrary Python, and no network tool. ``serve.py`` does
not import this module: tool use is off by default and, for now, is an
operator-invoked path rather than something the model can request.

Every call, including a denied or invalid one, is written to a hash-chained
JSONL ledger before it is attempted. Results are recorded as a digest and a
size, not as file contents, so the ledger is safe to inspect after the fact.
"""

import contextlib
import dataclasses
import datetime
import fcntl
import hashlib
import json
import os
import re
import stat
from pathlib import Path


MAX_PATH_CHARS = 4096
MAX_RECEIPT_ERROR = 500
ZERO_HASH = "0" * 64

TOOL_OPEN, TOOL_CLOSE = "<tool>", "</tool>"
RESULT_OPEN, RESULT_CLOSE = "<result>", "</result>"


class ToolError(Exception):
    """A tool failed under the policy; the call was denied or did not finish."""


class Denied(ToolError):
    """The policy does not grant this call."""


class BadArgs(ToolError):
    """The call's arguments do not match the tool's schema."""


class ReceiptError(Exception):
    """The ledger cannot be read or written without risking its chain."""


@dataclasses.dataclass(frozen=True)
class Arg:
    name: str
    kind: type
    required: bool = True
    default: object = dataclasses.field(default_factory=lambda: _MISSING)
    minimum: int | None = None
    maximum: int | None = None
    max_chars: int | None = None
    choices: tuple = ()


_MISSING = object()


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    capability: str
    args: tuple
    handler: object
    max_result_bytes: int = 65536


@dataclasses.dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


class Registry:
    """The named tools this process is allowed to reason about."""

    def __init__(self):
        self._tools = {}

    def register(self, spec):
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", spec.name):
            raise ValueError(f"bad tool name: {spec.name!r}")
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        if not callable(spec.handler):
            raise ValueError(f"tool {spec.name} has no handler")
        if len({a.name for a in spec.args}) != len(spec.args):
            raise ValueError(f"duplicate argument in {spec.name}")
        if spec.capability not in ("filesystem.read", "filesystem.write"):
            raise ValueError(f"bad capability: {spec.capability!r}")
        self._tools[spec.name] = spec

    def get(self, name):
        return self._tools.get(name)

    def specs(self):
        return tuple(self._tools[n] for n in sorted(self._tools))

    def names(self):
        return tuple(sorted(self._tools))


class Policy:
    """Explicit grants. The empty policy denies every tool."""

    def __init__(self, allowed=(), read_roots=(), write_roots=(),
                 max_file_bytes=1024 * 1024, max_result_bytes=65536):
        self.allowed = frozenset(_as_names(allowed))
        self.read_roots = _roots(read_roots)
        self.write_roots = _roots(write_roots)
        if (isinstance(max_file_bytes, bool)
                or not isinstance(max_file_bytes, int) or max_file_bytes < 1):
            raise ValueError("max_file_bytes must be a positive integer")
        if (isinstance(max_result_bytes, bool)
                or not isinstance(max_result_bytes, int) or max_result_bytes < 1):
            raise ValueError("max_result_bytes must be a positive integer")
        self.max_file_bytes = max_file_bytes
        self.max_result_bytes = max_result_bytes

    def allows(self, name):
        return name in self.allowed

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("policy must be a JSON object")
        allowed_keys = {"allow", "read_roots", "write_roots",
                        "max_file_bytes", "max_result_bytes"}
        extra = set(data) - allowed_keys
        if extra:
            raise ValueError(f"unknown policy keys: {sorted(extra)}")
        allow = data.get("allow", [])
        if not isinstance(allow, list) or not all(
                isinstance(x, str) for x in allow):
            raise ValueError("policy allow must be a list of tool names")
        read = data.get("read_roots", [])
        write = data.get("write_roots", [])
        for name, roots in (("read_roots", read), ("write_roots", write)):
            if not isinstance(roots, list) or not all(
                    isinstance(x, str) for x in roots):
                raise ValueError(f"policy {name} must be a list of paths")
        return cls(allow, read, write,
                   data.get("max_file_bytes", 1024 * 1024),
                   data.get("max_result_bytes", 65536))


@dataclasses.dataclass(frozen=True)
class ToolResult:
    ok: bool
    decision: str
    value: object = None
    error: str = None
    call_seq: int = None
    result_seq: int = None

    def as_dict(self):
        out = {"ok": self.ok, "decision": self.decision}
        if self.value is not None:
            out["value"] = self.value
        if self.error is not None:
            out["error"] = self.error
        if self.call_seq is not None:
            out["call_seq"] = self.call_seq
        if self.result_seq is not None:
            out["result_seq"] = self.result_seq
        return out


class ReceiptLedger:
    """Append-only, hash-chained JSONL. There is deliberately no delete API."""

    def __init__(self, path):
        self.path = os.path.abspath(os.fspath(path))
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, mode=0o700, exist_ok=True)
        if os.path.lexists(self.path):
            st = os.lstat(self.path)
            if stat.S_ISLNK(st.st_mode):
                raise ReceiptError("receipt ledger is a symlink")
            if not stat.S_ISREG(st.st_mode):
                raise ReceiptError("receipt ledger is not a regular file")
            os.chmod(self.path, 0o600)
        else:
            fd = os.open(self.path,
                         os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            os.close(fd)
        self.lock_path = self.path + ".lock"

    @contextlib.contextmanager
    def _locked(self):
        fd = os.open(self.lock_path,
                     os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read(self):
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            chunks = []
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
        finally:
            os.close(fd)
        entries = []
        for n, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                raise ReceiptError(f"blank receipt line {n}")
            try:
                rec = json.loads(line)
            except ValueError as e:
                raise ReceiptError(f"unreadable receipt line {n}: {e}")
            if not isinstance(rec, dict):
                raise ReceiptError(f"receipt line {n} is not an object")
            entries.append(rec)
        return entries

    def _verify(self, entries):
        prev = ZERO_HASH
        for n, rec in enumerate(entries, 1):
            if (not isinstance(rec.get("seq"), int)
                    or isinstance(rec.get("seq"), bool) or rec.get("seq") != n):
                return False, n, f"seq is {rec.get('seq')!r}, expected {n}"
            if rec.get("prev") != prev:
                return False, n, "previous hash does not match"
            claimed = rec.get("hash")
            body = {k: v for k, v in rec.items() if k != "hash"}
            if claimed != _hash(body):
                return False, n, "hash does not match"
            prev = claimed
        return True, len(entries), None

    def verify(self):
        with self._locked():
            entries = self._read()
        ok, count, error = self._verify(entries)
        return {"ok": ok, "entries": count, "error": error}

    def append(self, event, **fields):
        with self._locked():
            entries = self._read()
            ok, _, error = self._verify(entries)
            if not ok:
                raise ReceiptError(f"refusing to extend a broken ledger: {error}")
            rec = dict(fields)
            rec.update({
                "seq": len(entries) + 1,
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="milliseconds"),
                "event": event,
                "prev": entries[-1]["hash"] if entries else ZERO_HASH,
            })
            rec["hash"] = _hash(rec)
            line = (json.dumps(rec, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False) + "\n").encode("utf-8")
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            try:
                _write_all(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
            return rec


def parse_tool_calls(text):
    """Complete, balanced tool calls; incomplete or malformed ones are skipped.

    A half-written call must never run. The body is ordinary JSON with a
    string ``name`` and an object ``args``; anything else is ignored.
    """
    if not isinstance(text, str):
        return []
    calls = []
    at = 0
    while True:
        start = text.find(TOOL_OPEN, at)
        if start < 0:
            break
        body_start = start + len(TOOL_OPEN)
        end = text.find(TOOL_CLOSE, body_start)
        if end < 0:
            break
        body = text[body_start:end].strip()
        at = end + len(TOOL_CLOSE)
        try:
            obj = json.loads(body)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        call_args = obj.get("args", {})
        if not isinstance(name, str) or not isinstance(call_args, dict):
            continue
        calls.append(ToolCall(name, call_args))
    return calls


def result_text(result):
    """The result block the harness injects back into the model's context.

    Every angle bracket in the JSON is escaped, so file text cannot close the
    block and start a second tool call. The markers themselves are added after
    that escaping.
    """
    body = json.dumps(result, ensure_ascii=False, sort_keys=True)
    body = body.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"{RESULT_OPEN}\n{body}\n{RESULT_CLOSE}\n"


def builtin_registry():
    reg = Registry()
    reg.register(ToolSpec(
        "read_text",
        "Read a UTF-8 text file under a granted read root.",
        "filesystem.read",
        (Arg("path", str, max_chars=MAX_PATH_CHARS),
         Arg("max_bytes", int, required=False, default=16384,
             minimum=1, maximum=4 * 1024 * 1024)),
        _read_text))
    reg.register(ToolSpec(
        "list_dir",
        "List entries under a granted read root, without following symlinks.",
        "filesystem.read",
        (Arg("path", str, required=False, default=".",
             max_chars=MAX_PATH_CHARS),
         Arg("depth", int, required=False, default=1,
             minimum=1, maximum=3),
         Arg("max_entries", int, required=False, default=200,
             minimum=1, maximum=1000)),
        _list_dir))
    reg.register(ToolSpec(
        "stat_path",
        "Report type, size, mode, and mtime for a path under a read root.",
        "filesystem.read",
        (Arg("path", str, max_chars=MAX_PATH_CHARS),),
        _stat_path))
    reg.register(ToolSpec(
        "write_text",
        "Write UTF-8 text under an explicitly granted write root.",
        "filesystem.write",
        (Arg("path", str, max_chars=MAX_PATH_CHARS),
         Arg("text", str),
         Arg("append", bool, required=False, default=False)),
        _write_text))
    return reg


def run(registry, name, args, policy, ledger):
    """Run one call. The call receipt is written before any handler runs."""
    spec = registry.get(name)
    clean = None
    if spec is None:
        call = _safe_append(ledger, "call", tool=name, capability=None,
                            args={}, args_sha256=None,
                            decision="deny", reason="unknown tool")
        return ToolResult(False, "deny", error="unknown tool",
                          call_seq=call and call["seq"])

    if not policy.allows(name):
        call = _safe_append(ledger, "call", tool=name,
                            capability=spec.capability,
                            args=_redact(args), args_sha256=_args_hash(args),
                            decision="deny", reason="tool not granted")
        return ToolResult(False, "deny", error="tool not granted",
                          call_seq=call and call["seq"])

    roots = (policy.read_roots if spec.capability == "filesystem.read"
             else policy.write_roots)
    if not roots:
        call = _safe_append(ledger, "call", tool=name,
                            capability=spec.capability,
                            args=_redact(args), args_sha256=_args_hash(args),
                            decision="deny", reason="no root granted")
        return ToolResult(False, "deny", error="no root granted",
                          call_seq=call and call["seq"])

    try:
        clean = _validate(spec.args, args)
    except BadArgs as e:
        call = _safe_append(ledger, "call", tool=name,
                            capability=spec.capability,
                            args=_redact(args), args_sha256=_args_hash(args),
                            decision="deny", reason=str(e))
        return ToolResult(False, "deny", error=str(e),
                          call_seq=call and call["seq"])

    call = _safe_append(ledger, "call", tool=name,
                        capability=spec.capability,
                        args=_redact(clean), args_sha256=_args_hash(clean),
                        decision="allow", reason="granted")
    if call is None:
        return ToolResult(False, "deny",
                          error="receipt ledger unavailable; call not run")
    t0 = datetime.datetime.now(datetime.timezone.utc)
    try:
        value = spec.handler(policy, clean)
        encoded = json.dumps(value, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        limit = min(spec.max_result_bytes, policy.max_result_bytes)
        if len(encoded) > limit:
            raise ToolError(f"result exceeds {limit} bytes")
        result = _safe_append(
            ledger, "result", tool=name, call_seq=call["seq"], ok=True,
            result_sha256=hashlib.sha256(encoded).hexdigest(),
            result_bytes=len(encoded), error=None,
            duration_ms=_elapsed_ms(t0))
        if result is None:
            return ToolResult(False, "error", error="call ran but result receipt failed",
                              call_seq=call["seq"])
        return ToolResult(True, "allow", value=value,
                          call_seq=call["seq"], result_seq=result["seq"])
    except Exception as e:
        message = _clean_error(str(e))
        result = _safe_append(
            ledger, "result", tool=name, call_seq=call["seq"], ok=False,
            result_sha256=None, result_bytes=0, error=message,
            duration_ms=_elapsed_ms(t0))
        return ToolResult(False, "error", error=str(e),
                          call_seq=call["seq"],
                          result_seq=result and result["seq"])


def _read_text(policy, args):
    path, root = _resolve(policy.read_roots, args["path"], must_exist=True)
    if not stat.S_ISREG(os.stat(path, follow_symlinks=False).st_mode):
        raise ToolError("not a regular file")
    cap = min(args["max_bytes"], policy.max_file_bytes)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        data = b""
        while len(data) <= cap:
            chunk = os.read(fd, min(1 << 20, cap + 1 - len(data)))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    truncated = len(data) > cap
    data = data[:cap]
    return {"path": _relative(path, root),
            "text": data.decode("utf-8", errors="replace"),
            "bytes": len(data), "truncated": truncated,
            "sha256": hashlib.sha256(data).hexdigest()}


def _list_dir(policy, args):
    path, root = _resolve(policy.read_roots, args["path"], must_exist=True)
    if not stat.S_ISDIR(os.stat(path, follow_symlinks=False).st_mode):
        raise ToolError("not a directory")
    entries = []
    queue = [(path, 0)]
    while queue and len(entries) < args["max_entries"]:
        here, depth = queue.pop(0)
        try:
            with os.scandir(here) as it:
                items = sorted(it, key=lambda e: e.name)
        except OSError as e:
            raise ToolError(f"cannot list directory: {e.strerror or e}")
        for entry in items:
            if len(entries) >= args["max_entries"]:
                break
            kind = "link" if entry.is_symlink() else (
                "dir" if entry.is_dir(follow_symlinks=False) else
                "file" if entry.is_file(follow_symlinks=False) else "other")
            rel = _relative(Path(entry.path), root)
            entries.append({"path": rel, "type": kind})
            if kind == "dir" and depth + 1 < args["depth"]:
                queue.append((Path(entry.path), depth + 1))
    return {"root": str(root), "entries": entries,
            "truncated": len(entries) >= args["max_entries"]}


def _stat_path(policy, args):
    path, root = _resolve(policy.read_roots, args["path"], must_exist=True)
    st = os.stat(path, follow_symlinks=False)
    if stat.S_ISREG(st.st_mode):
        kind = "file"
    elif stat.S_ISDIR(st.st_mode):
        kind = "dir"
    else:
        kind = "other"
    return {"path": _relative(path, root), "type": kind,
            "size": st.st_size, "mode": stat.S_IMODE(st.st_mode),
            "mtime": st.st_mtime}


def _write_text(policy, args):
    path, root = _resolve(policy.write_roots, args["path"], must_exist=False)
    data = args["text"].encode("utf-8")
    if len(data) > policy.max_file_bytes:
        raise ToolError(f"text exceeds {policy.max_file_bytes} bytes")
    if args["append"] and os.path.exists(path):
        existing = os.stat(path, follow_symlinks=False).st_size
        if existing + len(data) > policy.max_file_bytes:
            raise ToolError(f"appending would exceed {policy.max_file_bytes} bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    flags |= os.O_APPEND if args["append"] else os.O_TRUNC
    try:
        fd = os.open(path, flags, 0o600)
    except FileNotFoundError:
        raise ToolError("parent directory does not exist")
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"path": _relative(path, root), "bytes": len(data),
            "append": args["append"]}


def _validate(spec_args, args):
    if not isinstance(args, dict):
        raise BadArgs("arguments must be a JSON object")
    by_name = {a.name: a for a in spec_args}
    unknown = sorted(set(args) - set(by_name))
    if unknown:
        raise BadArgs(f"unknown argument(s): {', '.join(unknown)}")
    out = {}
    for arg in spec_args:
        if arg.name not in args:
            if arg.required:
                raise BadArgs(f"missing argument: {arg.name}")
            out[arg.name] = arg.default
            continue
        value = args[arg.name]
        if arg.kind is bool:
            if not isinstance(value, bool):
                raise BadArgs(f"{arg.name} must be a boolean")
        elif arg.kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise BadArgs(f"{arg.name} must be an integer")
        elif arg.kind is str:
            if not isinstance(value, str):
                raise BadArgs(f"{arg.name} must be a string")
            if arg.max_chars is not None and len(value) > arg.max_chars:
                raise BadArgs(f"{arg.name} is too long")
        else:
            raise BadArgs(f"{arg.name} has an unsupported type")
        if arg.minimum is not None and value < arg.minimum:
            raise BadArgs(f"{arg.name} is below {arg.minimum}")
        if arg.maximum is not None and value > arg.maximum:
            raise BadArgs(f"{arg.name} is above {arg.maximum}")
        if arg.choices and value not in arg.choices:
            raise BadArgs(f"{arg.name} must be one of {arg.choices}")
        out[arg.name] = value
    return out


def _resolve(roots, raw, must_exist):
    if not isinstance(raw, str) or not raw:
        raise Denied("path is empty")
    if "\x00" in raw or "\n" in raw:
        raise Denied("path contains a forbidden character")
    if len(raw) > MAX_PATH_CHARS:
        raise Denied("path is too long")
    if not roots:
        raise Denied("no root granted")
    p = Path(raw)
    candidate = p if p.is_absolute() else Path(roots[0]) / p
    try:
        if must_exist:
            resolved = candidate.resolve(strict=True)
        else:
            parent = candidate.parent.resolve(strict=True)
            if candidate.name in ("", ".", ".."):
                raise Denied("invalid target name")
            resolved = parent / candidate.name
    except (OSError, RuntimeError):
        raise Denied("path does not resolve")
    root = _root_of(resolved, roots)
    if root is None:
        raise Denied("path is outside the granted roots")
    if must_exist:
        st = os.lstat(resolved)
        if stat.S_ISLNK(st.st_mode):
            raise Denied("path is a symlink")
    else:
        parent = resolved.parent
        if _root_of(parent, roots) is None:
            raise Denied("parent is outside the granted roots")
        if os.path.lexists(resolved):
            st = os.lstat(resolved)
            if stat.S_ISLNK(st.st_mode):
                raise Denied("target is a symlink")
            if not stat.S_ISREG(st.st_mode):
                raise Denied("target is not a regular file")
    return resolved, root


def _root_of(path, roots):
    for root in roots:
        try:
            path.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def _relative(path, root):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def _roots(values):
    out = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("a root must be a non-empty path")
        root = Path(value).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"root is not a directory: {value}")
        if str(root) not in out:
            out.append(str(root))
    return tuple(out)


def _as_names(values):
    if isinstance(values, str):
        values = [values]
    out = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("tool names must be strings")
        for name in value.split(","):
            name = name.strip()
            if name and name not in out:
                out.append(name)
    return tuple(out)


def _args_hash(args):
    if not isinstance(args, dict):
        return None
    raw = json.dumps(args, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _redact(args):
    if not isinstance(args, dict):
        return {}
    out = {}
    for key, value in args.items():
        if key in ("text", "content", "data"):
            out[key] = {"redacted": True, "chars": len(value)
                        if isinstance(value, str) else None}
        else:
            out[key] = value
    return out


def _safe_append(ledger, event, **fields):
    try:
        return ledger.append(event, **fields)
    except Exception:
        return None


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("short write")
        view = view[n:]


def _hash(record):
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _elapsed_ms(t0):
    return round((datetime.datetime.now(datetime.timezone.utc) - t0)
                 .total_seconds() * 1000, 3)


def _clean_error(text):
    text = " ".join(str(text).split())
    return text[:MAX_RECEIPT_ERROR]
