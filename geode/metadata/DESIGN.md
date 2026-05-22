# GeoDE Metadata Comparator — Design Document

## Purpose

Automate the detection of discrepancies between GNSS station metadata stored in
the GeoDE PostgreSQL database and metadata published by station maintainers
(IGS site logs, `.stninfo` files, or other formats). Claude acts as the reasoning
engine: parsers are already implemented elsewhere in GeoDE; this module handles
serialization, API interaction, and structured reporting.

---

## Module Location

```
pgamit/
└── geode/
    └── metadata/
        ├── __init__.py
        ├── comparator.py        # Main StationMetadataComparator class
        ├── serializers.py       # Normalize DB records and parsed files to shared schema
        ├── prompts.py           # System prompt + domain rules for Claude
        ├── report.py            # Parse Claude JSON response into dataclasses
        └── station_info.py      # StationInfoRecord and StationInfo classes

com/
├── SyncMetadata.py              # Batch synchronization runner (parallel processing)
└── StationInfoEdit.py           # TUI for reviewing and applying audit findings
```

---

## Database Changes Required

Two additions to the schema are needed before this module can run. Both must be
added to the `dbConnections` migrations routine.

### New table: `sources_metadata`

Stores the URL/path structure for metadata files. Fields are identical to
`sources_servers`. Each record in `sources_servers` will have a foreign key
pointing to one record in `sources_metadata`, because the metadata path
conventions (filenames, directory layout) may differ from those of RINEX data
and are defined at the server level, not per station.

```sql
CREATE TABLE sources_metadata (
    id          SERIAL PRIMARY KEY,
    -- same fields as sources_servers (protocol, url, path, etc.)
);

ALTER TABLE sources_servers
    ADD COLUMN metadata_source_id INTEGER REFERENCES sources_metadata(id);
```

### New field: `sources_stations.metadata_hash`

Stores a CRC32 hash of the last downloaded metadata file for each station.
Used to detect changes without re-downloading.

```sql
ALTER TABLE sources_stations
    ADD COLUMN metadata_hash BIGINT;  -- CRC32 signed integer
```

Use CRC32 (`Utils.crc32`) computed over the raw bytes of the downloaded file
before parsing.

### New table: `stationinfo_audit`

Tracks the outcome of every audit finding, keyed by a per-session fingerprint
derived from the external file. This is the mechanism that prevents re-flagging
a finding that a human has already reviewed and dismissed.

```sql
CREATE TABLE public.stationinfo_audit (
    api_id          SERIAL PRIMARY KEY,

    -- which station
    "NetworkCode"   VARCHAR(3)   NOT NULL,
    "StationCode"   VARCHAR(4)   NOT NULL,

    -- CRC32 fingerprint of str(StationInfoRecord) for the session.
    -- This uses the canonical string representation from StationInfoRecord.__str__()
    -- which produces a fixed-width stninfo-format line.
    session_hash    BIGINT       NOT NULL,

    -- what Claude found
    finding_type    VARCHAR(30)  NOT NULL,  -- NEW_SESSION | ORPHAN_SESSION | etc.
    action_required VARCHAR(10)  NOT NULL,  -- INSERT | UPDATE | REVIEW | NO_ACTION

    -- record identifiers and summary
    db_record       JSONB,    -- {"DateStart": "YYYY-MM-DD HH:MM:SS"} for DB record lookup
    claude_summary  TEXT,

    -- structured field values for programmatic updates (JSONB)
    -- Contains {"FieldName": "value"} for each relevant field
    -- Used by StationInfoEdit to apply INSERT/UPDATE without re-parsing
    db_field_values    JSONB,
    file_field_values  JSONB,

    -- human disposition (NULL = not yet reviewed by a human)
    reviewed_by     VARCHAR(80),
    reviewed_at     TIMESTAMP,
    disposition     VARCHAR(10),  -- 'APPLIED' | 'DISMISSED' | 'DEFERRED' | 'NO_ACTION'
    review_notes    TEXT,

    -- audit trail
    created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP    NOT NULL DEFAULT NOW()
);

-- Prevents duplicate audit rows for the same session content
CREATE UNIQUE INDEX stationinfo_audit_unique
    ON public.stationinfo_audit ("NetworkCode", "StationCode", session_hash);
```

The `session_hash` is computed from `str(StationInfoRecord)`, which produces
a canonical fixed-width string representation of the session. This is exposed
as the `session.hash` property on StationInfoRecord.

---

## Workflow Overview

When the metadata comparison program runs for a station, the steps are:

1. Fetch the metadata file from the remote server (or use cached copy).
2. Compute CRC32 of the raw downloaded file bytes.
3. **Layer 1 - File hash check**: Compare against `sources_stations.metadata_hash` in the DB.
   - **No change** (hashes match): skip this station entirely — no parsing or API call needed.
   - **Hash changed or null**: proceed to parsing.
4. Parse the file using the existing GeoDE parser.
5. Build `StationMetadataBundle` objects from both the DB and the parsed file.
6. **Layer 1b - Session hash check**: For each session in the file bundle, compute
   its hash via `session.hash` and check if it exists in `stationinfo_audit`.
   - If ALL sessions already exist in audit: the session data hasn't changed
     (file hash changed due to unrelated metadata like operator name). Update
     `sources_stations.metadata_hash` and skip — no API call needed.
   - If ANY session is new/changed: clear ALL audit records for this station
     (Claude needs full context) and proceed to comparison.
7. **Layer 2 - Bundle equality**: Inside `compare()`, if bundles are semantically
   identical, return `no_action` without calling the API.
8. Call `StationMetadataComparator.compare()` with both bundles.
9. For each finding in the `ComparisonReport`:
   - Write a row to `stationinfo_audit` using `finding.hash` (returned by Claude).
   - Route `INSERT` / `UPDATE` findings to the appropriate queue.
   - Route `REVIEW` findings (always `ORPHAN_SESSION`) to the human review queue.
   - `NO_ACTION` findings are written to the audit table with
     `disposition = 'NO_ACTION'` immediately — no human step needed.
10. Update `sources_stations.metadata_hash` with the new file hash.

**Two-layer fast path — from cheapest to most expensive:**
1. **File hash** (before parsing): unchanged file -> skip entirely.
2. **Session hash check** (after parsing): all sessions already in audit -> skip API call.
3. **Bundle equality** (inside `compare()`): if bundles are semantically identical -> skip API call.

### Why clear ALL audit records?

When session data changes (a new session hash is detected), we clear ALL audit
records for the station rather than filtering individual sessions. This is because
Claude needs full context to reason correctly about session overlaps, gaps, and
chain dependencies. Partial context can lead to incorrect conclusions.

---

## Input Sources

### 1. Database records

Pulled from the GeoDE PostgreSQL database (`stationinfo` table). Each row
represents one session — a time period with specific receiver AND antenna
equipment. A change in either receiver or antenna creates a new row.

### 2. Downloaded files

Parsed using **existing GeoDE parsers** (do not reimplement):
- IGS site log format (`.log`) — `StationInfo.parse_station_info()` auto-detects
  and parses, returning `List[StationInfoRecord]`.
- Station info format (`.stninfo`) — same parser, format auto-detected from content.
- NGL format — same parser, format auto-detected from content.

Both parsers produce combined receiver+antenna sessions with a single date range.

---

## Serialization Schema

Both sources are normalized into `StationMetadataBundle` before being sent to
Claude. This keeps the prompt simple and source-agnostic.

The data model uses **combined sessions** rather than separate receiver/antenna
lists, matching how the database stores records (one row per equipment change)
and how the IGS log parser merges separate receiver/antenna sections.

### Session Data Model

The module uses `StationInfoRecord` from `station_info.py` as the single source of
truth for session data. This avoids duplication and ensures consistency across
parsers, serializers, and comparison logic.

```python
# StationInfoRecord is defined in station_info.py with all receiver/antenna fields.
# Key fields used for comparison:
#   DateStart, DateEnd           - session date range (pyDate.Date objects)
#   ReceiverCode, ReceiverSerial, ReceiverFirmware  - receiver fields
#   AntennaCode, RadomeCode, AntennaSerial          - antenna fields
#   AntennaHeight, AntennaNorth, AntennaEast        - ARP eccentricities
#   source                       - provenance: "database" | "file"
#   hash                         - CRC32 of str(self), computed property

@dataclass
class StationMetadataBundle:
    network_code: str                      # 3-char network code in GeoDE
    station_code: str                      # 4-char
    domes_number: Optional[str]
    sessions: List[StationInfoRecord]
```

### Open-ended dates

Open-ended sessions (no end date) are represented with `DateEnd.year = 9999`,
not `None`. The `pyDate.Date` class handles open-ended dates robustly:

- `Date(stninfo='9999 999 00 00 00')` → year=9999, fyear=9999.0
- `Date(stninfo=None)` → year=9999, fyear=9999.0 (same as above)
- `Date(stninfo='')` → year=9999, fyear=9999.0 (same as above)
- `Date(year=9999, doy=1)` → normalized to year=9999, doy=1, fyear=9999.0

When serializing for comparison, check `year >= 2099` to identify open-ended dates.
The `datetime()` method returns `datetime(9999, 1, 1)` for year 9999 dates.

### Key serialization functions (`serializers.py`)

```python
def bundle_from_db(cnn: Cnn, network_code: str, station_code: str) -> StationMetadataBundle:
    """Query GeoDE DB and return a StationMetadataBundle."""
    ...

def bundle_from_file(path: str | Path, network_code: str,
                     station_code: str) -> StationMetadataBundle:
    """
    Parse any supported file format (auto-detected) and return StationMetadataBundle.
    Filters to only the target station for multi-station files.
    """
    ...

def _bundles_equal(a: StationMetadataBundle, b: StationMetadataBundle) -> bool:
    """
    Return True if both bundles are semantically identical after normalization
    (dates as ISO strings, floats rounded to 4 decimal places).
    Used as a fast-path guard inside compare().
    """
    def _date_to_str(d: Optional[Date]) -> str:
        if d is None or d.year is None or d.year >= 2099:
            return 'open'
        return d.strftime()

    def _norm_session(s: StationInfoRecord) -> tuple:
        return (
            _date_to_str(s.DateStart),
            _date_to_str(s.DateEnd),
            (s.ReceiverCode or '').strip().upper(),
            (s.ReceiverSerial or '').strip().upper(),
            (s.ReceiverVers or '').strip(),
            (s.ReceiverFirmware or '').strip(),
            (s.AntennaCode or '').strip().upper(),
            (s.RadomeCode or '').strip().upper(),
            (s.AntennaSerial or '').strip().upper(),
            round(s.AntennaHeight or 0.0, 4),
            (s.HeightCode or '').strip().upper(),
            round(s.AntennaNorth or 0.0, 4),
            round(s.AntennaEast or 0.0, 4),
            round(s.AntennaDAZ or 0.0, 4),
        )

    norm_a = sorted(_norm_session(s) for s in a.sessions)
    norm_b = sorted(_norm_session(s) for s in b.sessions)
    return norm_a == norm_b

def serialize_for_claude(db: StationMetadataBundle,
                         external: StationMetadataBundle) -> str:
    """Return JSON string with both bundles, ready for the API payload."""
    return json.dumps({
        'network_code': db.network_code,
        'station_code': db.station_code,
        'database_sessions': [s.to_claude_dict() for s in db.sessions],
        'external_sessions': [s.to_claude_dict() for s in external.sessions],
    }, indent=2)
```

### Session hash computation

The session hash is computed as `crc32(str(session))` where `str(session)` produces
the canonical stninfo-format representation. This is exposed as the `hash` property
on `StationInfoRecord`:

```python
# In StationInfoRecord
@property
def hash(self) -> int:
    """CRC32 hash of the canonical string representation."""
    from geode.Utils import crc32
    return crc32(str(self))
```

---

## Comparator Class (`comparator.py`)

```python
class StationMetadataComparator:
    """
    Sends normalized station metadata to Claude and returns a structured
    ComparisonReport. API key is read from ANTHROPIC_API_KEY env var by default.

    Implements:
    - Prompt caching for reduced costs
    - Cost tracking for API calls
    - Retry with exponential backoff for rate limits (429) and server errors (5xx)
    - JSON parse retry with stronger prompt on failure
    """

    def __init__(self,
                 api_key: str | None = None,
                 model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pricing per million tokens
        self._pricing = {
            "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
            "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.00},
            "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of an API call based on token usage."""
        pricing = self._pricing.get(self.model, {"input": 3.00, "output": 15.00})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def compare(self,
                db_bundle: StationMetadataBundle,
                file_bundle: StationMetadataBundle,
                file_source: str = "IGS log") -> ComparisonReport:
        """
        Compare db_bundle against file_bundle.

        NOTE: Audit filtering is handled UPSTREAM in SyncMetadata.
        This method assumes both bundles contain the full session data.

        Fast-path: skip API call if bundles are identical after normalization.

        Returns:
            ComparisonReport with findings, or no_action if fast-path applies
        """
        # Fast path: skip API call if bundles are identical
        if _bundles_equal(db_bundle, file_bundle):
            return ComparisonReport.no_action(db_bundle.network_code,
                                              db_bundle.station_code)

        # Use prompt caching for the system prompt
        cached_system = [
            TextBlockParam(
                type="text",
                text=SYSTEM_PROMPT,
                cache_control={"type": "ephemeral"}
            )
        ]

        payload = serialize_for_claude(db_bundle, file_bundle)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=cached_system,
            messages=[{
                "role": "user",
                "content": (
                    f"Compare the database sessions and the downloaded {file_source} "
                    f"for station {db_bundle.network_code}.{db_bundle.station_code}:\n\n"
                    f"{payload}\n\n"
                    f"IMPORTANT: Output ONLY valid JSON. No explanations or thinking."
                )
            }]
        )

        # Log usage and cost
        usage = message.usage
        cost = self._calculate_cost(usage.input_tokens, usage.output_tokens)

        # Check for cache metrics
        cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
        cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0

        logger.info(
            f"API usage - input: {usage.input_tokens} "
            f"(cached: {cache_read}, created: {cache_creation}), "
            f"output: {usage.output_tokens}, cost: ${cost:.4f}"
        )

        return parse_claude_response(message.content[0].text)
```

---

## Report Dataclasses (`report.py`)

```python
@dataclass
class Finding:
    finding_type: str           # NEW_SESSION, ORPHAN_SESSION, FIRMWARE_UPDATE, etc.
    action: str                 # INSERT | UPDATE | REVIEW | NO_ACTION
    description: str
    affected_fields: list[str]  # ["receiver"], ["antenna"], ["dates"], ["eccentricity"]
    db_record: Optional[dict]   # {"DateStart": "YYYY-MM-DD HH:MM:SS"} or None for INSERT
    db_field_values: Optional[dict] = None      # {"FieldName": "value"} for DB fields
    file_field_values: Optional[dict] = None    # {"FieldName": "value"} for file/recommended values
    hash: int = 0               # session hash passed to and returned by Claude

@dataclass
class ComparisonReport:
    network_code: str
    station_code: str
    summary: str
    findings: list[Finding]

    @property
    def needs_attention(self) -> bool:
        return any(f.action != "NO_ACTION" for f in self.findings)

    def findings_by_action(self, action: str) -> list[Finding]:
        return [f for f in self.findings if f.action == action]

    @classmethod
    def no_action(cls, network_code: str, station_code: str) -> "ComparisonReport":
        return cls(
            network_code=network_code,
            station_code=station_code,
            summary="Sessions match. No action required.",
            findings=[]
        )

def parse_claude_response(raw: str) -> ComparisonReport:
    """Parse Claude's JSON response, extracting hash from each finding."""
    data = json.loads(raw)
    findings = [
        Finding(
            finding_type=f["type"],
            action=f["action"],
            description=f["description"],
            affected_fields=f.get("affected_fields", []),
            db_record=f.get("db_record"),  # {"DateStart": "..."} or None
            db_field_values=f.get("db_field_values"),
            file_field_values=f.get("file_field_values"),
            hash=int(f.get("hash", '0'))
        )
        for f in data.get("findings", [])
    ]
    return ComparisonReport(
        network_code=data["network_code"],
        station_code=data["station_code"],
        summary=data["summary"],
        findings=findings,
    )
```

---

## SyncMetadata Functions

The batch runner (`com/SyncMetadata.py`) implements the upstream audit logic:

### `check_for_new_sessions()`

```python
def check_for_new_sessions(cnn: Cnn,
                           network_code: str,
                           station_code: str,
                           file_bundle: StationMetadataBundle) -> bool:
    """
    Check if file_bundle contains any sessions not already in audit table.

    Used to determine if we need to clear audit and re-analyze with Claude.
    If all session hashes from the file exist in audit, the session data
    hasn't actually changed (even if file hash changed due to unrelated
    metadata like operator name).

    Returns:
        True if there are new/changed sessions requiring analysis
    """
    for session in file_bundle.sessions:
        result = cnn.query_float(f"""
            SELECT 1 FROM stationinfo_audit
            WHERE "NetworkCode" = '{network_code}'
              AND "StationCode" = '{station_code}'
              AND session_hash = {session.hash}
            LIMIT 1
        """, as_dict=True)

        if not result:
            return True  # New session detected

    return False  # All sessions in audit
```

### `clear_station_audit()`

```python
def clear_station_audit(cnn: Cnn,
                        network_code: str,
                        station_code: str) -> int:
    """
    Clear all audit records for a station.

    Called when session data has changed - previous audit conclusions
    may no longer be valid since Claude needs full context for comparison.

    Returns:
        Number of records deleted
    """
    result = cnn.query(f"""
        DELETE FROM stationinfo_audit
        WHERE "NetworkCode" = '{network_code}'
          AND "StationCode" = '{station_code}'
        RETURNING api_id
    """)
    return result.ntuples()
```

### `upsert_audit()`

```python
def upsert_audit(cnn: Cnn, network_code: str, station_code: str,
                 finding: Finding):
    """
    Insert audit entry for a finding.

    Uses finding.hash directly (returned by Claude) rather than
    recomputing, ensuring consistency between what was sent and stored.
    """
    disposition = "'NO_ACTION'" if finding.action == 'NO_ACTION' else 'NULL'

    cnn.query(f"""
        INSERT INTO stationinfo_audit (
            "NetworkCode", "StationCode", session_hash,
            finding_type, action_required,
            db_record, claude_summary,
            db_field_values, file_field_values,
            disposition
        ) VALUES (
            '{network_code}', '{station_code}', {finding.hash},
            '{finding.finding_type}', '{finding.action}',
            $${json.dumps(finding.db_record)}$$::jsonb,
            $${finding.description}$$,
            $${json.dumps(finding.db_field_values)}$$::jsonb,
            $${json.dumps(finding.file_field_values)}$$::jsonb,
            {disposition}
        )
    """)
```

---

## Batch Processing Pattern

```python
from geode.metadata.comparator import StationMetadataComparator
from geode.metadata.serializers import bundle_from_db, bundle_from_file

comparator = StationMetadataComparator()   # reads ANTHROPIC_API_KEY from env

for station in network.stations:
    network_code = station.network_code
    station_code = station.station_code

    # --- Layer 1: file hash check (cheapest) ---
    raw_bytes    = fetch_metadata_file(station)
    new_hash     = compute_file_hash(raw_bytes)
    stored_hash  = get_metadata_hash(cnn, network_code, station_code)

    if new_hash == stored_hash:
        continue                                    # nothing changed

    # --- Parse file ---
    file_path   = save_to_temp(raw_bytes)
    file_bundle = bundle_from_file(file_path, network_code, station_code)

    # --- Layer 1b: session hash check ---
    if not check_for_new_sessions(cnn, network_code, station_code, file_bundle):
        # File changed but session data is the same
        update_metadata_hash(cnn, network_code, station_code, new_hash)
        continue

    # Session data changed - clear audit for full context
    clear_station_audit(cnn, network_code, station_code)

    # --- Compare (Layer 2 bundle equality is inside compare()) ---
    db_bundle = bundle_from_db(cnn, network_code, station_code)
    report = comparator.compare(db_bundle, file_bundle, file_source="metadata file")

    # --- Write all findings to stationinfo_audit ---
    for finding in report.findings:
        upsert_audit(cnn, network_code, station_code, finding)

    # --- Route actionable findings ---
    if report.needs_attention:
        logger.info(f"{network_code}.{station_code}: {report.summary}")

        for finding in report.findings_by_action("INSERT"):
            queue_insert(finding)

        for finding in report.findings_by_action("UPDATE"):
            queue_update(finding)

        for finding in report.findings_by_action("REVIEW"):
            queue_human_review(finding)   # ORPHAN_SESSION always goes here

    # --- Update file hash ---
    update_metadata_hash(cnn, network_code, station_code, new_hash)
```

---

## System Prompt (`prompts.py`)

The prompt encodes geodetic domain rules. Sessions are compared as complete
records (receiver + antenna together), since that is how the database stores
them. Claude identifies which fields differ within matched sessions.

Key features of the prompt:
- Session matching by overlapping date intervals (1-day tolerance)
- Classification rules for each finding type
- JSON output format with `hash` field in each finding
- `db_field_values` and `file_field_values` dicts for structured field data

See `prompts.py` for the full system prompt text.

---

## StationInfoEdit TUI (`com/StationInfoEdit.py`)

A modern Textual-based TUI for reviewing and applying audit findings.

### Features

- **Records Tab**: View all station info records with color-coded audit status
- **Audit Tab**: Review pending findings with diff view and timeline visualization
- **Actions**: Apply, Dismiss, or Defer findings with review notes
- **Batch Mode**: `--apply-pending` flag for automated INSERT application
- **List Mode**: `--list-inserts` flag to list all pending INSERT findings

### Usage

```bash
# TUI mode for single station
StationInfoEdit.py arg.unro

# List all pending INSERT findings
StationInfoEdit.py --list-inserts

# Batch apply pending INSERT findings
StationInfoEdit.py --apply-pending all
StationInfoEdit.py --apply-pending arg.all
StationInfoEdit.py --apply-pending --dry-run arg.unro arg.srlp
```

### Apply Order (Important)

When applying a finding, the operation order is critical:

1. **First**: Attempt to apply the change (`insert_station_info` or `update_station_info`)
2. **Then**: Only if successful, mark the finding as `APPLIED`

This ensures that if an apply fails (e.g., due to overlap conflict), the finding
remains in pending state for retry or manual review.

### Batch Mode Restrictions

Only `INSERT` findings are auto-applied in batch mode. `UPDATE` findings require
manual review in TUI mode because they may involve subjective decisions about
which values to trust.

---

## Design Decisions & Notes

### Session-based comparison (not separate receiver/antenna)

The database stores combined sessions (one row per equipment change, with both
receiver and antenna fields). The IGS log parser (`igslog.py`) already merges
separate receiver/antenna sections into combined sessions. Using the same
combined session model throughout:
- Simplifies session-to-session matching (one-to-one by date overlap)
- Avoids artificial complexity of correlating separate receiver/antenna records
- Aligns with how data is stored and parsed throughout GeoDE

Claude identifies which fields differ (receiver, antenna, or both) within
matched sessions via the `affected_fields` list in findings.

### Trust model

Claude produces findings and recommended actions; **GeoDE code decides whether to
act**. Auto-apply `INSERT` and `UPDATE` only for sources with high trust (e.g.,
official IGN-Ar/RAMSAC logs). Always route `REVIEW` findings (especially
`ORPHAN_SESSION`) to a human queue — never auto-delete DB records.

### Context window cost

A full SRLP-style record (5 sessions, 18-year history) serializes to roughly
500–800 tokens. One station per API call keeps reasoning clean and errors
contained. Do not batch multiple stations into a single call.

### Prompt caching

The system prompt (~2000 tokens) is cached using Anthropic's prompt caching
feature (`cache_control={"type": "ephemeral"}`). This reduces costs when
processing multiple stations in sequence, as the system prompt tokens are
only charged once per cache window (typically 5 minutes).

### API cost tracking

The comparator logs input/output token counts and estimated cost for each
API call. Cache hit/miss metrics are also logged when available.

### Fast-path strategy (two layers)

From cheapest to most expensive:

1. **File hash** (before parsing): CRC32 of raw downloaded bytes vs
   `sources_stations.metadata_hash`. Unchanged file -> skip entirely. This is
   the dominant case for a stable network on any given run.

2. **Session hash check** (after parsing, before Claude): each session's hash
   (`session.hash` = `crc32(str(session))`) is checked against `stationinfo_audit`.
   If ALL sessions exist in audit -> skip API call (session data unchanged, only
   metadata like operator name changed). If ANY session is new -> clear ALL audit
   records and proceed to full comparison.

3. **Bundle equality** (inside `compare()`): if bundles are semantically identical
   after normalization -> skip API call. Secondary guard for edge cases.

### Audit table lifecycle

| Event | `disposition` value | Effect on future runs |
|---|---|---|
| Claude finds NO_ACTION | `NO_ACTION` (set immediately) | Session hash exists in audit |
| Auto-applied INSERT/UPDATE | `APPLIED` (set by queue processor) | Session hash exists in audit |
| Human dismisses REVIEW finding | `DISMISSED` (set by human) | Session hash exists in audit |
| Human defers for later | `DEFERRED` (set by human) | Session hash exists in audit |
| Session data changes | Audit cleared | All sessions re-evaluated |

When session data changes (new hash detected), ALL audit records for the station
are cleared. This ensures Claude always has full context for reasoning.

### ORPHAN_SESSION handling

`ORPHAN_SESSION` findings (DB record with no file counterpart) are identified
by Claude during comparison. They always route to `REVIEW` and require an
explicit human disposition. Never auto-apply a delete to a DB record.

### Format consistency

Input to Claude is JSON (from `serialize_for_claude`). The `db_record` field
in Claude's JSON response is a dict with `{"DateStart": "YYYY-MM-DD HH:MM:SS"}`
to identify the database session being referenced. For INSERT findings (NEW_SESSION),
`db_record` is null since there is no existing DB session.

The structured field data is in `db_field_values` and `file_field_values` dicts,
which contain only the relevant differing fields for programmatic updates.

### Hash consistency

The session hash sent to Claude in the payload is the same hash returned in
the finding and stored in the audit table. This ensures:
- No hash recomputation that could introduce inconsistencies
- The hash stored matches exactly what was evaluated

### Model

Use `claude-sonnet-4-6`. Haiku misses subtle domain conventions; Opus is
unnecessary for structured comparison tasks.

### Environment variable
```
ANTHROPIC_API_KEY=sk-ant-...
```
Set in the shell or in a `.env` file loaded by GeoDE's config layer. Do not
hardcode or commit the key.

### Parallel processing (SyncMetadata)

SyncMetadata supports parallel station processing via `ThreadPoolExecutor`:

```bash
# Default: 4 workers
python SyncMetadata.py all

# Custom worker count
python SyncMetadata.py -w 8 all

# Disable parallel processing
python SyncMetadata.py -np all
```

Each worker uses a thread-local database connection. The download cache is
thread-safe using `threading.Lock()`.

### Rate limit handling (Comparator)

The comparator implements retry with exponential backoff:

- **Rate limits (429)**: Up to 5 retries, base delay 2s, max delay 60s
- **Server errors (5xx)**: Same retry strategy
- **JSON parse failures**: Up to 2 retries with stronger JSON-only prompt

This allows batch processing to continue smoothly even when hitting API limits.

### Future extensions

- Support `.stninfo` multi-station batch files (iterate records, one bundle per
  station code).
- ~~Add a lightweight CLI or web UI for the human review queue~~ — **DONE**:
  `StationInfoEdit.py` provides a full TUI for reviewing findings.
- Store raw Claude JSON responses on the `stationinfo_audit` row (e.g., in a
  `claude_raw` JSONB column) for full reproducibility and debugging.
