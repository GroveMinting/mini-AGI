#!/usr/bin/env python3
"""Run one sandboxed tool call, or inspect the tool registry and receipts.

Tool use is off by default: a tool needs an explicit --allow (or policy file)
and a granted root. The model cannot call this path; serve.py does not import
the registry.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from minagi.agency import (Policy, ReceiptError, ReceiptLedger,
                           builtin_registry, run)


def _policy(args):
    if args.policy:
        if args.allow or args.read_root or args.write_root:
            raise ValueError("--policy cannot be combined with --allow/--read-root/--write-root")
        return Policy.from_file(args.policy)
    return Policy(args.allow or (), args.read_root or (), args.write_root or (),
                  args.max_file_bytes, args.max_result_bytes)


def _print(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def _list(reg):
    for spec in reg.specs():
        print(f"{spec.name}  [{spec.capability}]")
        print(f"  {spec.description}")
        for arg in spec.args:
            default = "" if arg.required else f" (default {arg.default!r})"
            print(f"  {arg.name}: {arg.kind.__name__}{default}")
    return 0


def _call(args):
    try:
        args_obj = json.loads(args.args_json)
    except ValueError as e:
        print(f"bad --args JSON: {e}", file=sys.stderr)
        return 4
    if not isinstance(args_obj, dict):
        print("--args must be a JSON object", file=sys.stderr)
        return 4
    try:
        policy = _policy(args)
    except (OSError, ValueError) as e:
        print(f"bad policy: {e}", file=sys.stderr)
        return 4
    try:
        ledger = ReceiptLedger(args.receipts)
    except (OSError, ReceiptError) as e:
        print(f"receipts: {e}", file=sys.stderr)
        return 4
    reg = builtin_registry()
    result = run(reg, args.tool, args_obj, policy, ledger)
    _print(result.as_dict())
    if result.ok:
        return 0
    return 2 if result.decision == "deny" else 3


def _verify(args):
    if not os.path.exists(args.receipts):
        print(f"receipts: no ledger at {args.receipts}", file=sys.stderr)
        return 2
    try:
        ledger = ReceiptLedger(args.receipts)
        verdict = ledger.verify()
    except (OSError, ReceiptError) as e:
        print(f"receipts: {e}", file=sys.stderr)
        return 2
    _print(verdict)
    return 0 if verdict["ok"] else 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="show the built-in tools")
    p.set_defaults(func=lambda a: _list(builtin_registry()))

    p = sub.add_parser("call", help="run one call under an explicit policy")
    p.add_argument("tool")
    p.add_argument("--args", dest="args_json", default="{}",
                   help="JSON object of tool arguments")
    p.add_argument("--allow", action="append", default=[],
                   help="tool name to grant; repeat or comma-separate")
    p.add_argument("--read-root", action="append", default=[],
                   help="existing directory reads may touch; repeat")
    p.add_argument("--write-root", action="append", default=[],
                   help="existing directory writes may touch; repeat")
    p.add_argument("--policy", help="JSON policy file (instead of flags)")
    p.add_argument("--receipts", default=os.path.join("runs", "tool_receipts.jsonl"),
                   help="append-only JSONL ledger")
    p.add_argument("--max-file-bytes", type=int, default=1024 * 1024)
    p.add_argument("--max-result-bytes", type=int, default=65536)
    p.set_defaults(func=_call)

    p = sub.add_parser("verify", help="check the receipt hash chain")
    p.add_argument("--receipts", default=os.path.join("runs", "tool_receipts.jsonl"))
    p.set_defaults(func=_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
