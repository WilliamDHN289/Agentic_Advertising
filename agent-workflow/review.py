#!/usr/bin/env python3
"""Local Claude/Codex review gate. Python 3.9+, macOS/Linux, standard library only.

This is a workflow guard, not a security boundary against an agent with write access.
The script never commits, pushes, installs dependencies, or changes CLI permissions.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
STATE = RUNS / "state.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    require(result.returncode == 0, result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def clean():
    require(not git("status", "--porcelain", "--untracked-files=all"),
            "Working tree is not clean. Preserve existing work and commit only intended files before proceeding.")


def validate_schema(value, schema, location="review"):
    kind = schema.get("type")
    expected = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int}
    if kind in expected:
        require(type(value) is expected[kind], location + " has an invalid type")
    if "enum" in schema:
        require(value in schema["enum"], location + " has an invalid enum value")
    if kind == "object":
        required = set(schema.get("required", []))
        require(required <= set(value), location + " has missing fields")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(properties), location + " has unexpected fields")
        for key in value:
            if key in properties:
                validate_schema(value[key], properties[key], location + "." + key)
    elif kind == "array":
        for index, item in enumerate(value):
            validate_schema(item, schema["items"], location + "[" + str(index) + "]")


def gate_decision(review, handoff, expected):
    validate_schema(review, read_json(HERE / "review.schema.json"))
    for key, value in expected.items():
        require(review[key] == value, "Review identity mismatch: " + key)
    criteria_ids = [entry["id"] for entry in handoff["acceptance_criteria"]]
    checked_ids = [entry["criterion_id"] for entry in review["checks"]]
    require(len(checked_ids) == len(set(checked_ids)), "Duplicate criterion in review")
    require(set(criteria_ids) == set(checked_ids), "Review does not account for every submitted criterion")
    if review["decision"] == "APPROVED":
        require(review["criteria_complete"] is True, "Approval omitted original ticket requirements")
        require(all(c["status"] in ("PASS", "NOT_APPLICABLE") and c["evidence"].strip()
                    for c in review["checks"]), "Approval has failed or unverified criteria, or missing evidence")
        require(not any(f["blocking"] for f in review["findings"]), "Approval has unresolved blocking findings")
    return review["decision"]


def begin(args):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.ticket), "Use a simple ticket ID, for example TICKET-001")
    require(args.max_rounds > 0, "max-rounds must be positive")
    clean()
    head = git("rev-parse", "HEAD")
    old = read_json(STATE) if STATE.exists() else None
    if old:
        if old["ticket_id"] == args.ticket:
            print(json.dumps(old, ensure_ascii=False, indent=2))
            return 0
        require(old["decision"] == "APPROVED", "Current ticket has not passed; resume it instead of starting another")
        require(old["approved_head"] == head, "HEAD changed since approval; the old approval cannot authorize new work")
    directory = RUNS / args.ticket
    require(not directory.exists(), "This ticket already has a history; do not overwrite it")
    tickets_path = Path(args.tickets)
    require(not tickets_path.is_absolute() and ".." not in tickets_path.parts, "tickets path must be inside the repository")
    source = git("show", head + ":" + tickets_path.as_posix())
    require(args.ticket in source, "Ticket ID is absent from the committed ticket document")
    directory.mkdir(parents=True)
    (directory / "requirements.md").write_text(source + "\n", encoding="utf-8")
    state = {"ticket_id": args.ticket, "base_commit": head, "tickets_path": tickets_path.as_posix(),
             "max_rounds": args.max_rounds, "round": 0, "decision": "IMPLEMENTING", "approved_head": None}
    write_json(STATE, state)
    write_json(directory / "ticket-state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def run_codex(command, prompt, directory, timeout):
    with (directory / "stdout.log").open("w", encoding="utf-8") as out, (directory / "stderr.log").open("w", encoding="utf-8") as err:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                   text=True, start_new_session=True)
        try:
            process.communicate(prompt, timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise
    require(process.returncode == 0, "Codex failed; inspect stderr.log and stdout.log")


def review(args):
    require(STATE.exists(), "Run begin before review")
    state = read_json(STATE)
    require(state["round"] < state["max_rounds"], "Review round limit reached; human intervention required")
    require(args.timeout > 0, "timeout must be positive")
    clean()
    head = git("rev-parse", "HEAD")
    require(head != state["base_commit"], "No candidate commit exists after this ticket's baseline")
    git("merge-base", "--is-ancestor", state["base_commit"], head)
    require(not git("diff", "--name-only", state["base_commit"], head, "--", HERE.name),
            "Workflow files changed during implementation; ask the user to handle configuration separately")
    require(not git("diff", "--name-only", state["base_commit"], head, "--", state["tickets_path"]),
            "The ticket document changed during implementation; review against the original requirements")
    handoff = read_json(Path(args.handoff).resolve())
    require(handoff.get("ticket_id") == state["ticket_id"], "Handoff ticket ID mismatch")
    criteria = handoff.get("acceptance_criteria")
    require(isinstance(criteria, list) and len(criteria) > 0, "Handoff must list every acceptance criterion")
    ids = []
    for item in criteria:
        require(isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
                and isinstance(item.get("requirement"), str) and item["requirement"].strip(), "Invalid criterion")
        ids.append(item["id"])
    require(len(ids) == len(set(ids)), "Duplicate acceptance criterion ID")
    require(shutil.which("codex"), "codex is missing from PATH; install and sign in first")
    state["round"] += 1
    state["decision"] = "IN_REVIEW"
    state["approved_head"] = None
    directory = RUNS / state["ticket_id"] / ("round-" + str(state["round"]))
    directory.mkdir()
    write_json(directory / "handoff.json", handoff)
    write_json(STATE, state)
    write_json(directory.parent / "ticket-state.json", state)
    identity = {"ticket_id": state["ticket_id"], "base_commit": state["base_commit"], "head_commit": head}
    request = {**identity, "requirements_path": str(directory.parent / "requirements.md"),
               "handoff_path": str(directory / "handoff.json"),
               "previous_review_path": str(directory.parent / ("round-" + str(state["round"] - 1)) / "review.json") if state["round"] > 1 else None}
    prompt = (HERE / "codex-reviewer.md").read_text(encoding="utf-8") + "\n\nCurrent review request:\n" + json.dumps(request, ensure_ascii=False, indent=2)
    (directory / "request.txt").write_text(prompt, encoding="utf-8")
    write_json(directory / "request.json", request)
    output = directory / "review.json"
    command = ["codex", "exec", "--sandbox", "read-only", "--output-schema", str(HERE / "review.schema.json"),
               "--output-last-message", str(output), "-"]
    if args.model:
        command[2:2] = ["--model", args.model]
    print("Running independent Codex review. Logs: " + str(directory), flush=True)
    decision, error = "BLOCKED", None
    try:
        run_codex(command, prompt, directory, args.timeout)
        clean()
        require(git("rev-parse", "HEAD") == head, "HEAD changed during review; approval is invalid")
        decision = gate_decision(read_json(output), handoff, identity)
    except KeyboardInterrupt:
        error = "Review interrupted; no approval recorded"
    except Exception as exc:
        error = str(exc)
    if decision == "CHANGES_REQUESTED" and state["round"] >= state["max_rounds"]:
        decision, error = "BLOCKED", "Review round limit reached; unresolved feedback is in review.json"
    state["decision"] = decision
    state["approved_head"] = head if decision == "APPROVED" else None
    write_json(STATE, state)
    write_json(directory.parent / "ticket-state.json", state)
    gate = {**identity, "round": state["round"], "decision": decision, "error": error,
            "review_path": str(output), "gate_path": str(directory / "gate.json")}
    write_json(directory / "gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return {"APPROVED": 0, "CHANGES_REQUESTED": 2, "BLOCKED": 3}[decision]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("begin")
    start.add_argument("ticket")
    start.add_argument("--tickets", default="tickets.md")
    start.add_argument("--max-rounds", type=int, default=5)
    sub.add_parser("status")
    check = sub.add_parser("review")
    check.add_argument("--handoff", required=True)
    check.add_argument("--timeout", type=int, default=1800)
    check.add_argument("--model", help="Optional model ID available to your Codex account")
    args = parser.parse_args()
    try:
        require(Path(git("rev-parse", "--show-toplevel")).resolve() == ROOT, "Put agent-workflow directly inside the Git repository root")
        require(not git("ls-files", "--", HERE.name + "/runs"), "Runtime files must not be tracked by Git")
        RUNS.mkdir(exist_ok=True)
        with (RUNS / "workflow.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another workflow command is running; wait for it to finish")
            if args.action == "begin":
                return begin(args)
            if args.action == "status":
                print(json.dumps(read_json(STATE) if STATE.exists() else {"decision": "NOT_STARTED"}, ensure_ascii=False, indent=2))
                return 0
            return review(args)
    except Exception as exc:
        print(json.dumps({"decision": "BLOCKED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
