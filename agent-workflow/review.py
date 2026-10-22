#!/usr/bin/env python3
"""Local Claude implementer / Claude reviewer gate. Python 3.9+, macOS/Linux, standard library only.

The reviewer is a separate `claude -p` process signed into a second account through its own
CLAUDE_CONFIG_DIR (default ~/.claude-reviewer, override with REVIEWER_CLAUDE_CONFIG_DIR).

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
REVIEWER_CONFIG = os.path.expanduser(os.environ.get("REVIEWER_CLAUDE_CONFIG_DIR", "~/.claude-reviewer"))


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


def adopt_workflow(args):
    require(STATE.exists(), "No workflow state to update")
    state = read_json(STATE)
    require(state["decision"] != "IN_REVIEW", "A review is in progress")
    clean()
    head = git("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD^")
    changed = git("diff", "--name-only", parent, head).splitlines()
    require(changed and all(path.startswith(HERE.name + "/") for path in changed),
            "HEAD must be a commit that changes only " + HERE.name + "/ files")
    if state["decision"] == "APPROVED":
        require(state["approved_head"] == parent, "HEAD^ is not the approved candidate")
        state["approved_head"] = head
    else:
        git("merge-base", "--is-ancestor", state["base_commit"], parent)
    commits = state.setdefault("workflow_commits", [])
    if head not in commits:
        commits.append(head)
    write_json(STATE, state)
    write_json(RUNS / state["ticket_id"] / "ticket-state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def claude_env(config_dir):
    # Drop the implementer session's variables (nested-session markers, session IDs, tokens) so the
    # child authenticates only through the given config directory.
    env = {key: value for key, value in os.environ.items() if not key.startswith(("CLAUDE", "ANTHROPIC"))}
    if config_dir:
        env["CLAUDE_CONFIG_DIR"] = config_dir
    return env


def auth_status(config_dir):
    result = subprocess.run(["claude", "auth", "status", "--json"], cwd=ROOT, env=claude_env(config_dir),
                            text=True, capture_output=True, timeout=60)
    try:
        status = json.loads(result.stdout)
    except ValueError:
        return {"loggedIn": False}
    return status if isinstance(status, dict) else {"loggedIn": False}


def check_reviewer():
    require(shutil.which("claude"), "claude is missing from PATH; install it first")
    reviewer = auth_status(REVIEWER_CONFIG)
    require(reviewer.get("loggedIn") is True,
            "Reviewer account is not signed in; run: CLAUDE_CONFIG_DIR=" + REVIEWER_CONFIG + " claude auth login")
    implementer = auth_status(os.environ.get("CLAUDE_CONFIG_DIR"))
    require(not reviewer.get("email") or reviewer.get("email") != implementer.get("email"),
            "Reviewer and implementer are signed into the same account; sign " + REVIEWER_CONFIG + " into the second account")
    return {"reviewer_config": REVIEWER_CONFIG, "reviewer_email": reviewer.get("email"),
            "reviewer_subscription": reviewer.get("subscriptionType"), "implementer_email": implementer.get("email")}


def reviewer_command(model):
    # Mirrors the old read-only Codex sandbox: no edit tools, Bash confined by the OS sandbox with the
    # repository write-denied and no network, no unsandboxed retry, no repo settings or MCP servers.
    sandbox = {"sandbox": {"enabled": True, "failIfUnavailable": True, "allowUnsandboxedCommands": False,
                           "filesystem": {"denyWrite": [str(ROOT)]}}}
    command = ["claude", "-p", "--output-format", "stream-json", "--verbose",
               "--json-schema", (HERE / "review.schema.json").read_text(encoding="utf-8"),
               "--tools", "Read,Grep,Glob,Bash", "--permission-mode", "dontAsk",
               "--setting-sources", "user", "--settings", json.dumps(sandbox),
               "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"]
    if model:
        command += ["--model", model]
    return command


def last_result(path):
    result = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    return result


def run_reviewer(command, prompt, directory, timeout):
    with (directory / "stdout.log").open("w", encoding="utf-8") as out, (directory / "stderr.log").open("w", encoding="utf-8") as err:
        process = subprocess.Popen(command, cwd=ROOT, env=claude_env(REVIEWER_CONFIG), stdin=subprocess.PIPE,
                                   stdout=out, stderr=err, text=True, start_new_session=True)
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
    result = last_result(directory / "stdout.log")
    detail = str((result or {}).get("result") or "inspect stderr.log and stdout.log")[:500]
    require(process.returncode == 0 and result is not None and not result.get("is_error"), "Claude reviewer failed: " + detail)
    require(isinstance(result.get("structured_output"), dict), "Claude reviewer returned no structured review; inspect stdout.log")
    return result["structured_output"]


def review(args):
    require(STATE.exists(), "Run begin before review")
    state = read_json(STATE)
    require(state["round"] < state["max_rounds"], "Review round limit reached; human intervention required")
    require(args.timeout > 0, "timeout must be positive")
    clean()
    head = git("rev-parse", "HEAD")
    require(head != state["base_commit"], "No candidate commit exists after this ticket's baseline")
    git("merge-base", "--is-ancestor", state["base_commit"], head)
    workflow_changes = git("rev-list", state["base_commit"] + ".." + head, "--", HERE.name).split()
    require(set(workflow_changes) <= set(state.get("workflow_commits", [])),
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
    check_reviewer()
    state["round"] += 1
    state["decision"] = "IN_REVIEW"
    state["approved_head"] = None
    directory = RUNS / state["ticket_id"] / ("round-" + str(state["round"]))
    directory.mkdir()
    write_json(directory / "handoff.json", handoff)
    write_json(STATE, state)
    write_json(directory.parent / "ticket-state.json", state)
    identity = {"ticket_id": state["ticket_id"], "base_commit": state["base_commit"], "head_commit": head}
    previous = directory.parent / ("round-" + str(state["round"] - 1)) / "review.json"
    request = {**identity, "requirements_path": str(directory.parent / "requirements.md"),
               "handoff_path": str(directory / "handoff.json"),
               "previous_review_path": str(previous) if previous.exists() else None}
    prompt = (HERE / "claude-reviewer.md").read_text(encoding="utf-8") + "\n\nCurrent review request:\n" + json.dumps(request, ensure_ascii=False, indent=2)
    (directory / "request.txt").write_text(prompt, encoding="utf-8")
    write_json(directory / "request.json", request)
    output = directory / "review.json"
    print("Running independent Claude review (" + REVIEWER_CONFIG + "). Logs: " + str(directory), flush=True)
    decision, error = "BLOCKED", None
    try:
        write_json(output, run_reviewer(reviewer_command(args.model), prompt, directory, args.timeout))
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
    sub.add_parser("check-reviewer", help="Verify the reviewer account sign-in without starting a round")
    sub.add_parser("adopt-workflow", help="User only: accept HEAD as a commit that changes only workflow files")
    check = sub.add_parser("review")
    check.add_argument("--handoff", required=True)
    check.add_argument("--timeout", type=int, default=1800)
    check.add_argument("--model", help="Optional model alias or ID available to the reviewer account")
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
            if args.action == "check-reviewer":
                print(json.dumps(check_reviewer(), ensure_ascii=False, indent=2))
                return 0
            if args.action == "adopt-workflow":
                return adopt_workflow(args)
            return review(args)
    except Exception as exc:
        print(json.dumps({"decision": "BLOCKED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
