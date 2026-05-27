# Claude Code Prompt: Implement the Metadata Planner Module

## Context

Read `geode/metadata/DESIGN.md` in full before starting. Then read the following
existing files to understand the conventions already in use:

- `geode/metadata/comparator.py`     — existing API call pattern, retry logic, prompt caching
- `geode/metadata/report.py`         — Finding and ComparisonReport dataclasses
- `geode/metadata/serializers.py`    — bundle_from_db(), StationMetadataBundle, StationInfoRecord
- `geode/metadata/prompts.py`        — how the system prompt and examples are structured
- `geode/dbConnection.py`            — DB connection patterns and query conventions
- `com/StationInfoEdit.py`           — how findings are currently applied (insert_station_info,
                                       update_station_info), TUI structure, and batch mode
- `com/SyncMetadata.py`              — how upsert_audit() is called and how findings are routed

Do NOT start writing code until you have read all of the above files.

---

## Task

Implement a **planner module** at `geode/metadata/planner.py` and integrate it into
`StationInfoEdit.py` only. `SyncMetadata.py` is NOT involved in applying findings —
it only discovers discrepancies and writes audit records. All application of findings
(both batch and interactive) lives in `StationInfoEdit.py`.

The planner's job is: given a specific finding to apply and optional user-provided
plain-language instructions, determine the exact ordered sequence of DB operations
needed to apply it correctly — updating adjacent sessions, avoiding overlaps, and
inserting new records — and return that sequence for execution.

---

## Apply Strategy: SQL-First, Planner-as-Fallback

The apply workflow follows a two-stage approach:

### Stage 1: Direct SQL Apply
For `INSERT` and `UPDATE` findings, first attempt to apply directly using the
existing `insert_station_info` / `update_station_info` functions with the field
values from `file_field_values`. This is fast, cheap, and works for the majority
of cases (e.g. a new session appended at the end of the timeline).

```python
def try_direct_apply(cnn, finding, network_code, station_code) -> bool:
    """
    Attempt direct SQL application of a finding.
    Returns True if successful, False if a conflict was detected.
    Does NOT raise on conflict — caller decides whether to invoke planner.
    """
    try:
        if finding.action == "INSERT":
            insert_station_info(cnn, network_code, station_code,
                                finding.file_field_values)
        elif finding.action == "UPDATE":
            update_station_info(cnn, network_code, station_code,
                                finding.db_record['DateStart'],
                                finding.file_field_values)
        cnn.commit()
        return True
    except (OverlapError, GapError, DatabaseError):
        cnn.rollback()
        return False
```

### Stage 2: Planner (Claude) Apply
If Stage 1 fails due to a conflict, OR if the user has provided plain-language
instructions, invoke the planner. The planner uses Claude to reason about the
full session timeline and produce the correct operation sequence.

**This means Claude is only called when needed** — most straightforward inserts
and updates never touch the API. This keeps costs low and batch mode fast.

```
apply_finding()
    │
    ├─► try_direct_apply()
    │       │
    │       ├─► SUCCESS → mark APPLIED, done
    │       │
    │       └─► CONFLICT or user_instructions provided
    │               │
    │               └─► StationInfoPlanner.plan()
    │                       │
    │                       ├─► show preview to user (TUI)
    │                       │
    │                       └─► on confirm → execute_plan() → mark APPLIED
```

---

## Module: `geode/metadata/planner.py`

### Class: `StationInfoPlanner`

```python
class StationInfoPlanner:
    """
    Given a finding to apply and the current state of the station's DB sessions,
    produces an ordered sequence of DB operations that correctly applies the finding
    without creating gaps or overlaps in the station info timeline.

    Claude is invoked when:
    - Direct SQL apply failed due to a conflict
    - The user has provided plain-language instructions (TUI interactive mode)
    - The finding is REVIEW type (always requires planner — no direct apply)

    Uses temperature=0 for deterministic output.
    """

    def __init__(self, api_key: str | None = None,
                 model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def plan(self,
             finding: Finding,
             all_findings: list[tuple[Finding, str | None]],
             db_sessions: list[StationInfoRecord],
             network_code: str,
             station_code: str,
             user_instructions: str | None = None) -> PlannerResult:
        """
        Produce an ordered sequence of DB operations to apply a single finding.

        Args:
            finding:           The specific finding to apply (any action type,
                               including REVIEW).
            all_findings:      All findings for this station as (Finding, disposition)
                               tuples. Disposition is one of:
                               - None: pending (not yet processed)
                               - "DEFERRED": pending, deferred for later
                               - "APPLIED": already applied to DB
                               - "DISMISSED": deliberately kept as-is
                               - "NO_ACTION": confirmed match, no action needed
                               Claude uses this to reason about full history:
                               pending findings may conflict, applied findings explain
                               current DB state, dismissed findings represent deliberate
                               past decisions that should be flagged if overridden.
            db_sessions:       Current DB sessions for the station (from stationinfo).
            network_code:      3-char network code.
            station_code:      4-char station code.
            user_instructions: Optional plain-language instructions from the user,
                               entered via the TUI. Only available in interactive
                               mode, never in batch mode. Examples:
                               - "Apply the antenna change but keep the existing date"
                               - "Use the file's start date but keep the DB end date"
                               - "Split the session at 2019-03-15 and insert the new one"
                               When provided, Claude must follow these instructions
                               while still ensuring no gaps or overlaps result.

        Returns:
            PlannerResult with operations list, conflicts list, and summary.
        """
```

### Dataclasses

```python
@dataclass
class PlannerOperation:
    """
    A single DB operation in an ordered execution plan.
    Operations must be executed in list order within a single transaction.
    """
    operation:    str           # "INSERT" | "UPDATE" | "DELETE"
    target:       dict | None   # {"NetworkCode": ..., "StationCode": ...,
                                #  "DateStart": "YYYY-MM-DD HH:MM:SS"}
                                # None for INSERT (no existing record to target)
    fields:       dict          # field name -> new value (empty {} for DELETE)
    reason:       str           # human-readable explanation
    finding_hash: int           # hash of the finding this operation belongs to

@dataclass
class PlannerResult:
    operations: list[PlannerOperation]
    conflicts:  list[str]       # empty if no conflicts
    summary:    str             # one-line human-readable summary

class PlannerError(Exception):
    def __init__(self, message: str, raw_response: str | None = None):
        self.message = message
        self.raw_response = raw_response
        super().__init__(message)
```

### Operation Executor

Implement as a standalone function so it can be called from anywhere in
`StationInfoEdit.py` without coupling to the planner class:

```python
def execute_plan(cnn,
                 operations: list[PlannerOperation],
                 network_code: str,
                 station_code: str,
                 dry_run: bool = False) -> list[str]:
    """
    Execute an ordered list of PlannerOperations in a single transaction.

    Args:
        cnn:          GeoDE database connection.
        operations:   Ordered list from StationInfoPlanner.plan().
        network_code: 3-char network code.
        station_code: 4-char station code.
        dry_run:      If True, log operations but do not commit. Used for
                      preview in batch mode and testing.

    Returns:
        List of human-readable strings describing what was done.

    Raises:
        DatabaseError: rolls back the entire transaction on any failure.
                       The finding remains in pending state for retry.
    """
```

---

## Planner System Prompt (`geode/metadata/planner_prompt.py`)

Follow the same structure as `prompts.py`. Apply `cache_control` in the API call.

### Input format

Claude receives a JSON object:

```json
{
  "network_code": "arg",
  "station_code": "srlp",
  "finding_to_apply": { ...single Finding dict with disposition... },
  "all_findings": [ ...all Finding dicts for this station, each with disposition... ],
  "current_db_sessions": [ ...all StationInfoRecord dicts... ],
  "user_instructions": "plain language from user, or null"
}
```

Each finding in `all_findings` includes a `disposition` field:
- `null` or `"DEFERRED"`: Pending — check for conflicts with `finding_to_apply`
- `"APPLIED"`: Already applied to DB — explains current state, use as context
- `"DISMISSED"`: Deliberately kept as-is — flag if new operation would override
- `"NO_ACTION"`: Confirmed match — background context only

### Rules Claude must follow

1. Produce operations in execution order — prerequisites first
2. Never produce a DELETE operation except for ORPHAN_SESSION findings where the
   user has explicitly instructed removal. DELETE must come BEFORE any UPDATE
   that would otherwise create an overlap. A DELETE must always target a specific
   session by (NetworkCode, StationCode, DateStart) and must be accompanied by
   an UPDATE to the preceding or following session to maintain timeline continuity
3. Never create a gap (missing time) between adjacent sessions
4. Never create an overlap between sessions
5. DateEnd of session N must be exactly 1 second before DateStart of session N+1,
   OR null if session N is the last/open-ended session
6. All dates in ISO format "YYYY-MM-DD HH:MM:SS"
7. Every operation must include the `finding_hash` of the finding it belongs to
8. For UPDATE: `target` identifies the session by DateStart
9. For INSERT: `target` must be null
10. Only produce operations for `finding_to_apply` — `all_findings` is context only
11. Use the `disposition` field in `all_findings` to understand context:
    - Pending findings (`null` or `"DEFERRED"`) may conflict — check date boundaries
    - Applied findings explain why the current DB state looks the way it does
    - If `finding_to_apply` would override a `"DISMISSED"` finding's decision,
      flag it in `conflicts` so the user is aware of the potential conflict
12. If a conflict exists with another pending finding, list it in `conflicts`
    and do not produce operations that would conflict
13. When `user_instructions` are provided, they are the ABSOLUTE primary
    directive and completely override the normal implications of the
    finding_type. If the user asks to change only one field, produce
    operations that change only that field — do not add operations for
    date boundaries, adjacent sessions, or other fields even if the
    finding type would normally require them.

    Conflict detection (Rules 11 and 12) must also be scoped to the
    user's actual requested operations only. Do not flag conflicts for
    date boundaries or fields the user explicitly chose not to touch.

    The only exception is timeline integrity: if the user's specific
    instructions would create an invalid timeline (gap or overlap),
    flag it in `conflicts` rather than silently producing an invalid
    sequence
14. For REVIEW findings: Claude must reason about what the user actually wants
    done (based on `user_instructions` and the finding description) and produce
    the minimal set of operations that achieves it safely

### Output format

JSON object only, no preamble:

```json
{
  "operations": [
    {
      "operation": "UPDATE",
      "target": {"NetworkCode": "arg", "StationCode": "srlp",
                 "DateStart": "2017-11-13 00:00:00"},
      "fields": {"DateEnd": "2025-09-01 23:59:59"},
      "reason": "Close preceding session before inserting new firmware session",
      "finding_hash": -34534564
    },
    {
      "operation": "INSERT",
      "target": null,
      "fields": {
        "NetworkCode": "arg", "StationCode": "srlp",
        "DateStart": "2025-09-02 00:00:00",
        "DateEnd": "2026-05-05 23:59:59",
        "AntennaHeight": 0.062,
        "HeightCode": "DHARP",
        "AntennaNorth": -0.011,
        "AntennaEast": -0.026,
        "ReceiverCode": "TRIMBLE NETR9",
        "ReceiverVers": "",
        "ReceiverFirmware": "4.85",
        "ReceiverSerial": "5146K79840",
        "AntennaCode": "TRM57971.00",
        "RadomeCode": "NONE",
        "AntennaSerial": "1441112252",
        "AntennaDAZ": 0.0,
        "Comments": ""
      },
      "reason": "Insert new session with firmware 4.85",
      "finding_hash": -34534564
    }
  ],
  "conflicts": [],
  "summary": "Two operations: close preceding session then insert new firmware session."
}
```

### Examples to include in the planner prompt

Include at least four concrete examples using real SRLP/P415 data from the
comparator prompt for consistency:

**Example A — NEW_SESSION at end of timeline**
Single open-ended DB session. New session appended at the end.
Operations: UPDATE preceding session DateEnd → INSERT new session.

**Example B — NEW_SESSION in middle of timeline**
New session splits an existing DB session (firmware change mid-session).
Operations: UPDATE preceding session DateEnd → INSERT new session →
UPDATE following session DateStart if needed.

**Example C — ORPHAN_SESSION with user instruction to delete**
ORPHAN_SESSION where the user instructs deletion: e.g. "Delete this orphan
session and merge the timeline". Claude produces:
1. DELETE the orphan session (must come first to avoid overlap)
2. UPDATE preceding session DateEnd to cover the gap
Note: DELETE is only allowed for ORPHAN_SESSION with explicit user instruction.

**Example D — ORPHAN_SESSION that user wants to keep**
ORPHAN_SESSION where the Comments field indicates legitimate documentation
(e.g. "Equipment test during installation"). User instructs to keep it.
Claude produces: no operations, summary explains session is retained.

### Implementation requirement

`planner_prompt.py` must be a Python module containing a single string constant
`PLANNER_SYSTEM_PROMPT`. This is the **literal text** sent to Claude as the system
prompt in every `StationInfoPlanner.plan()` call. It must be fully written out —
not a placeholder, not a skeleton, not a reference to this document.

The constant must contain:
1. A role definition opening: "You are a GNSS station info timeline planner..."
2. The input format description (all JSON fields Claude will receive, with
   explanations of what each field contains)
3. All 14 rules listed above, written out verbatim as prompt instructions
4. All four examples (A, B, C, D) with complete JSON input payload and the
   expected complete JSON output — use real SRLP and P415 data from the
   comparator prompt examples for consistency
5. The CRITICAL JSON-only output instruction (same pattern as `prompts.py`):
   no preamble, no markdown fences, start with `{`, end with `}`

The constant is used in `planner.py` exactly like this:

```python
from geode.metadata.planner_prompt import PLANNER_SYSTEM_PROMPT

message = self.client.messages.create(
    model=self.model,
    max_tokens=2048,
    temperature=0,
    system=[{
        "type": "text",
        "text": PLANNER_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{
        "role": "user",
        "content": json.dumps(payload, default=str)
    }]
)
```

When writing `planner_prompt.py`, treat it as you would `prompts.py` — write the
full prompt text as a Python triple-quoted string. The prompt must be complete and
self-contained: Claude will receive only the system prompt and the JSON payload,
nothing else.

---

## Integration: `StationInfoEdit.py`

### Apply flow for INSERT and UPDATE findings

```python
def apply_finding(cnn, finding, all_findings, db_sessions,
                  network_code, station_code,
                  user_instructions: str | None = None):
    """
    Apply a finding using SQL-first, planner-as-fallback strategy.
    user_instructions is only available in interactive TUI mode.
    """
    # If user has provided instructions, skip direct apply and go straight
    # to planner — user instructions imply non-standard handling
    if not user_instructions:
        success = try_direct_apply(cnn, finding, network_code, station_code)
        if success:
            mark_applied(cnn, network_code, station_code, finding.session_hash)
            return

    # Direct apply failed or user has instructions — invoke planner
    planner = StationInfoPlanner()
    result = planner.plan(finding, all_findings, db_sessions,
                          network_code, station_code,
                          user_instructions=user_instructions)

    if result.conflicts:
        show_conflict_dialog(result.conflicts)
        return

    # Show preview before executing
    confirmed = show_plan_preview(result)
    if not confirmed:
        return

    execute_plan(cnn, result.operations, network_code, station_code)
    mark_applied(cnn, network_code, station_code, finding.session_hash)
```

### Apply flow for REVIEW findings

REVIEW findings (ORPHAN_SESSION, MISSING_SESSION) always go through the planner —
never direct SQL apply. The user must provide `user_instructions` explaining
what they want to do. The TUI should show a text input field when the user
selects Apply on a REVIEW finding, with placeholder text like:
"Describe what you want to do with this finding..."

```python
def apply_review_finding(cnn, finding, all_findings, db_sessions,
                         network_code, station_code,
                         user_instructions: str):
    """
    REVIEW findings always require user instructions and always go through
    the planner. Never attempt direct SQL apply for REVIEW findings.
    """
    if not user_instructions.strip():
        show_error("Please provide instructions for how to handle this finding.")
        return

    planner = StationInfoPlanner()
    result = planner.plan(finding, all_findings, db_sessions,
                          network_code, station_code,
                          user_instructions=user_instructions)

    if result.conflicts:
        show_conflict_dialog(result.conflicts)
        return

    confirmed = show_plan_preview(result)
    if not confirmed:
        return

    execute_plan(cnn, result.operations, network_code, station_code)
    mark_applied(cnn, network_code, station_code, finding.session_hash)
```

### Batch mode in `StationInfoEdit.py`

The `--apply-pending` batch mode applies only INSERT findings automatically
(no user confirmation, no user_instructions). It uses the SQL-first strategy
and skips any finding where direct apply fails (logs the failure for human
review). UPDATE and REVIEW findings are never auto-applied in batch mode.

```python
def batch_apply_pending(cnn, network_code, station_code,
                        dry_run: bool = False):
    """
    Auto-apply all pending INSERT findings for a station using SQL-first strategy.
    Findings that fail direct apply are skipped and logged — planner is NOT
    invoked in batch mode (no user to provide instructions or confirm preview).

    Processes findings in DateStart order (earliest first).
    Refreshes db_sessions after each successful apply.
    """
    findings = get_pending_inserts(cnn, network_code, station_code)
    findings.sort(key=lambda f: f.file_field_values.get('DateStart', ''))

    for finding in findings:
        if dry_run:
            logger.info(f"[DRY RUN] Would apply INSERT: {finding.session_hash}")
            continue

        success = try_direct_apply(cnn, finding, network_code, station_code)
        if success:
            mark_applied(cnn, network_code, station_code, finding.session_hash)
            logger.info(f"Applied INSERT {finding.session_hash} for "
                        f"{network_code}.{station_code}")
        else:
            logger.warning(f"INSERT {finding.session_hash} for "
                           f"{network_code}.{station_code} failed direct apply "
                           f"— skipping, requires manual review in TUI")
```

### CLI flags in `StationInfoEdit.py`

```bash
# TUI mode for single station
StationInfoEdit.py arg.unro

# List all pending INSERT findings across network
StationInfoEdit.py --list-inserts
StationInfoEdit.py --list-inserts arg.all
StationInfoEdit.py --list-inserts arg.unro arg.srlp

# Batch apply pending INSERT findings
StationInfoEdit.py --apply-pending all
StationInfoEdit.py --apply-pending arg.all
StationInfoEdit.py --apply-pending arg.unro arg.srlp

# Preview batch apply without committing
StationInfoEdit.py --apply-pending --dry-run arg.unro arg.srlp
```

---

## TUI Changes in `StationInfoEdit.py`

### For INSERT and UPDATE findings

Add a collapsible "Custom Instructions" text area in the Apply dialog. Collapsed
by default (most users won't need it). When expanded, the user can type plain
language like:
- "Apply the antenna change but keep the existing start date"
- "Use the file's end date but keep the DB receiver serial"

If the text area is empty when Apply is clicked: use SQL-first strategy.
If the text area has content: skip SQL-first and go straight to planner.

### For REVIEW findings

Replace the Apply button behavior with a two-step flow:
1. Show a mandatory text input: "Describe what you want to do..."
2. Only enable the Confirm button after the user has typed something
3. Show the plan preview before final execution

The plan preview for all finding types should display each operation as a
human-readable line, e.g.:
```
UPDATE srlp @ 2017-11-13 → set DateEnd = 2025-09-01 23:59:59
INSERT srlp: 2025-09-02 to 2026-05-05 (TRIMBLE NETR9, fw 4.85, TRM57971.00/NONE)
```

---

## Error Handling

- Wrap the Claude API call in the same retry logic as `comparator.py`
  (exponential backoff for 429 and 5xx, up to 5 retries)
- If Claude returns invalid JSON: raise `PlannerError` with raw response
- If an UPDATE targets a session not found in `db_sessions`: raise `PlannerError`
- If `execute_plan` fails mid-sequence: roll back entire transaction
- Never mark a finding as `APPLIED` unless `execute_plan` succeeds without error
- If `conflicts` is non-empty: show to user and do not proceed with execution

---

## Testing

After implementing, test the following scenarios:

1. **Simple INSERT (SQL-first succeeds)**: SRLP new session at end of timeline.
   Verify no Claude call is made.

2. **INSERT with conflict (planner invoked)**: Manually create a session that
   would overlap the insert. Verify SQL-first fails, planner is called, and
   the plan preview is shown.

3. **REVIEW with user_instructions**: Use the 2-minute P415 orphan session.
   Type "Merge this into the preceding session" in the TUI. Verify Claude
   produces an UPDATE on the preceding session's DateEnd.

4. **Batch mode dry-run**: Run `--apply-pending --dry-run arg.srlp`. Verify
   operations are logged but no DB changes are made.

5. **user_instructions bypass**: For a simple INSERT finding, type custom
   instructions in the TUI. Verify the planner is invoked even though direct
   apply would have succeeded.

6. **Conflict detection**: Create two pending INSERT findings with overlapping
   date ranges for the same station. Verify the planner flags the conflict
   and the TUI shows it without executing any operations.
