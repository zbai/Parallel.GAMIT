#!/usr/bin/env python
"""
Project: Geodesy Database Engine (GeoDE)
Date: May 2026
Author: Demian D. Gomez

Station Information Editor - Modern TUI for managing station metadata.

Features:
- View and edit station info records
- Review Claude audit findings from stationinfo_audit
- Color-coded records based on audit status
- Apply, dismiss, or defer audit findings
- Search and filter records
- Export audit reports

Usage:
    StationInfoEdit.py net.stnm
    StationInfoEdit.py arg.unro
"""

import argparse
import datetime
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List, Dict, Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Header, Footer, Static, DataTable, Button, Input, Label,
    TabbedContent, TabPane, TextArea, Select, Rule, OptionList
)
from textual.widgets.option_list import Option
from textual.message import Message
from textual.suggester import SuggestFromList
from textual import events
from textual.reactive import reactive
from rich.text import Text
from rich.style import Style
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from geode import dbConnection
from geode.metadata.station_info import StationInfo, StationInfoRecord
from geode import pyDate
from geode.pyEvents import Event
from geode.Utils import process_date, add_version_argument, process_stnlist


# =============================================================================
# Searchable Select Widget
# =============================================================================

class SearchableSelect(Vertical):
    """A searchable dropdown widget combining Input with OptionList."""

    DEFAULT_CSS = """
    SearchableSelect {
        height: auto;
        max-height: 12;
    }

    SearchableSelect Input {
        width: 100%;
    }

    SearchableSelect OptionList {
        height: auto;
        max-height: 8;
        border: solid $primary;
        display: none;
    }

    SearchableSelect OptionList.visible {
        display: block;
    }
    """

    class Selected(Message):
        """Posted when a value is selected."""
        def __init__(self, value: str, widget_id: str) -> None:
            self.value = value
            self.widget_id = widget_id
            super().__init__()

    def __init__(self, options: List[str], value: str = "",
                 placeholder: str = "", widget_id: str = "", **kwargs):
        super().__init__(**kwargs)
        self.all_options = options
        self.initial_value = value
        self.placeholder = placeholder
        self.widget_id = widget_id
        self._selected_value = value
        self._initialized = False  # Track if we've finished initial setup
        self._selecting = False  # Track if we're in the middle of a selection

    def compose(self) -> ComposeResult:
        yield Input(
            value=self.initial_value,
            placeholder=self.placeholder,
            id=f"{self.widget_id}_input"
        )
        # Start with empty option list - will be populated on user interaction
        yield OptionList(id=f"{self.widget_id}_list")

    def on_mount(self) -> None:
        """Set initial state - keep dropdown hidden."""
        option_list = self.query_one(f"#{self.widget_id}_list", OptionList)
        option_list.remove_class("visible")
        # Mark as initialized after a short delay to ignore initial input events
        self.set_timer(0.1, self._mark_initialized)

    def _mark_initialized(self) -> None:
        """Mark widget as initialized after mount completes."""
        self._initialized = True

    def _filter_options(self, search: str) -> None:
        """Filter options based on search term."""
        option_list = self.query_one(f"#{self.widget_id}_list", OptionList)
        option_list.clear_options()

        search_lower = search.lower()
        filtered = [opt for opt in self.all_options if search_lower in opt.lower()][:50]

        if filtered:
            for opt in filtered:
                option_list.add_option(Option(opt))

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter options as user types."""
        if event.input.id == f"{self.widget_id}_input":
            # Only show dropdown after initialization (ignore initial value setting)
            if not self._initialized:
                return
            # Don't show dropdown if we just made a selection
            if self._selecting:
                return
            self._filter_options(event.value)
            # Show the option list when typing
            option_list = self.query_one(f"#{self.widget_id}_list", OptionList)
            option_list.add_class("visible")

    def on_focus(self, event: events.Focus) -> None:
        """Show dropdown when input gets focus."""
        # Check if the focus is on our input
        try:
            input_widget = self.query_one(f"#{self.widget_id}_input", Input)
            if self._initialized and input_widget.has_focus:
                self._filter_options(input_widget.value)
                option_list = self.query_one(f"#{self.widget_id}_list", OptionList)
                option_list.add_class("visible")
        except Exception:
            pass

    def on_blur(self, event: events.Blur) -> None:
        """Hide dropdown when input loses focus (with delay for click handling)."""
        self.set_timer(0.2, self._maybe_hide_dropdown)

    def _maybe_hide_dropdown(self) -> None:
        """Hide dropdown if input doesn't have focus."""
        try:
            input_widget = self.query_one(f"#{self.widget_id}_input", Input)
            option_list = self.query_one(f"#{self.widget_id}_list", OptionList)
            if not input_widget.has_focus:
                option_list.remove_class("visible")
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Hide options when input is submitted."""
        if event.input.id == f"{self.widget_id}_input":
            option_list = self.query_one(f"#{self.widget_id}_list", OptionList)
            option_list.remove_class("visible")
            self._selected_value = event.value

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle selection from the option list."""
        if event.option_list.id == f"{self.widget_id}_list":
            selected = str(event.option.prompt)
            self._selecting = True  # Prevent on_input_changed from showing dropdown
            input_widget = self.query_one(f"#{self.widget_id}_input", Input)
            input_widget.value = selected
            self._selected_value = selected
            event.option_list.remove_class("visible")
            self.post_message(self.Selected(selected, self.widget_id))
            # Reset the flag after a short delay
            self.set_timer(0.1, self._clear_selecting)

    def _clear_selecting(self) -> None:
        """Clear the selecting flag."""
        self._selecting = False

    @property
    def value(self) -> str:
        """Get the current value."""
        try:
            input_widget = self.query_one(f"#{self.widget_id}_input", Input)
            return input_widget.value
        except Exception:
            return self._selected_value

    def update_options(self, options: List[str]) -> None:
        """Update the available options (e.g., when antenna changes for radome)."""
        self.all_options = options
        self._filter_options(self.value)


# =============================================================================
# Color Schemes for Audit Status
# =============================================================================

# Colors for finding_type values
AUDIT_COLORS = {
    # Finding types
    'NO_FINDING': 'green',
    'NEW_SESSION': 'red',
    'MISSING_SESSION': 'red',
    'ORPHAN_SESSION': 'magenta',
    'RECEIVER_CHANGE': 'red',
    'ANTENNA_CHANGE': 'red',
    'SERIAL_FIRMWARE_MISMATCH': 'yellow',
    'DATE_MISMATCH': 'orange3',
    'ECCENTRICITY_CHANGE': 'red',
    'GAP_SIMPLIFICATION': 'green',
    'HEIGHT_CODE_CHANGE': 'orange3',
    'OTHER_FINDING': 'cyan',
    None: 'white',  # No audit data
}

DISPOSITION_COLORS = {
    'NO_ACTION': 'green',
    'APPLIED': 'green',
    'DISMISSED': 'dim',
    'DEFERRED': 'orange3',
    None: 'yellow',  # Pending review
}


# =============================================================================
# Database Helper Functions
# =============================================================================

def get_receiver_list(cnn: dbConnection.Cnn) -> List[str]:
    """Get list of all receiver codes from database."""
    result = cnn.query('SELECT DISTINCT "ReceiverCode" FROM receivers ORDER BY "ReceiverCode"')
    return [r['ReceiverCode'] for r in result.dictresult() if r['ReceiverCode']]


def get_antenna_list(cnn: dbConnection.Cnn) -> List[str]:
    """Get list of all distinct antenna codes from database."""
    result = cnn.query('SELECT DISTINCT "AntennaCode" FROM antennas ORDER BY "AntennaCode"')
    return [r['AntennaCode'] for r in result.dictresult() if r['AntennaCode']]


def get_radomes_for_antenna(cnn: dbConnection.Cnn, antenna_code: str) -> List[str]:
    """Get list of valid radome codes for a specific antenna type."""
    if not antenna_code:
        # Return common radomes if no antenna selected
        return ['NONE', 'DOME', 'SCIS', 'SCIT', 'TZGD', 'UNKN']
    result = cnn.query(
        f"SELECT DISTINCT \"RadomeCode\" FROM antennas "
        f"WHERE \"AntennaCode\" = '{antenna_code}' ORDER BY \"RadomeCode\""
    )
    radomes = [r['RadomeCode'] for r in result.dictresult() if r['RadomeCode']]
    return radomes if radomes else ['NONE']


def get_audit_findings(cnn: dbConnection.Cnn,
                       network_code: str,
                       station_code: str) -> List[Dict[str, Any]]:
    """Get all audit findings for a station, ordered by session DateStart."""
    result = cnn.query_float(f'''
        SELECT api_id, session_hash, finding_type, action_required,
               db_record, claude_summary,
               db_field_values, file_field_values,
               reviewed_by, reviewed_at, disposition, review_notes,
               created_at, updated_at
        FROM stationinfo_audit
        WHERE "NetworkCode" = '{network_code}'
          AND "StationCode" = '{station_code}'
        ORDER BY COALESCE(db_record->>'DateStart', file_field_values->>'DateStart', '9999') ASC
    ''', as_dict=True)
    return list(result) if result else []


def get_audit_for_session(cnn: dbConnection.Cnn,
                          network_code: str,
                          station_code: str,
                          date_start: str) -> Optional[Dict[str, Any]]:
    """Get audit finding for a specific session by date start."""
    # Try to find audit record that matches this session's date range
    result = cnn.query_float(f'''
        SELECT finding_type, action_required, disposition
        FROM stationinfo_audit
        WHERE "NetworkCode" = '{network_code}'
          AND "StationCode" = '{station_code}'
          AND db_record LIKE '%{date_start}%'
        ORDER BY created_at DESC
        LIMIT 1
    ''', as_dict=True)
    return result[0] if result else None


def update_audit_disposition(cnn: dbConnection.Cnn,
                             api_id: int,
                             disposition: str,
                             reviewed_by: str,
                             review_notes: str = None):
    """Update the disposition of an audit finding."""
    notes_sql = f", review_notes = $${review_notes}$$" if review_notes else ""
    cnn.query(f'''
        UPDATE stationinfo_audit
        SET disposition = '{disposition}',
            reviewed_by = '{reviewed_by}',
            reviewed_at = NOW(){notes_sql},
            updated_at = NOW()
        WHERE api_id = {api_id}
    ''')


# =============================================================================
# Modal Screens
# =============================================================================

class ConfirmDialog(ModalScreen[bool]):
    """A modal dialog for confirmation."""

    def __init__(self, title: str, message: str):
        super().__init__()
        self.title = title
        self.message = message

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self.title, classes="dialog-title"),
            Static(self.message, classes="dialog-message"),
            Horizontal(
                Button("Yes", variant="success", id="yes"),
                Button("No", variant="error", id="no"),
                classes="dialog-buttons"
            ),
            classes="dialog"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class EditRecordScreen(ModalScreen[Optional[Dict[str, str]]]):
    """Modal screen for editing a station info record with searchable equipment fields."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    # Fields that use SearchableSelect dropdowns
    SEARCHABLE_FIELDS = {'ReceiverCode', 'AntennaCode', 'RadomeCode', 'HeightCode'}

    def __init__(self, record: Optional[Dict[str, str]] = None,
                 title: str = "Edit Record",
                 cnn: dbConnection.Cnn = None):
        super().__init__()
        self.record = record or {}
        self.title_text = title
        self.cnn = cnn
        self.fields = OrderedDict([
            ('DateStart', ''),
            ('DateEnd', ''),
            ('ReceiverCode', ''),
            ('ReceiverSerial', ''),
            ('ReceiverVers', ''),
            ('ReceiverFirmware', ''),
            ('AntennaCode', ''),
            ('RadomeCode', ''),
            ('AntennaSerial', ''),
            ('AntennaHeight', '0.000'),
            ('AntennaNorth', '0.000'),
            ('AntennaEast', '0.000'),
            ('AntennaDAZ', '0.0'),
            ('HeightCode', 'DHARP'),
            ('Comments', ''),
        ])
        # Populate with existing values
        for key in self.fields:
            if key in self.record:
                self.fields[key] = str(self.record.get(key, '') or '')

        # Load equipment lists from database
        self.receivers = get_receiver_list(cnn) if cnn else []
        self.antennas = get_antenna_list(cnn) if cnn else []
        # Get initial radomes based on current antenna
        current_antenna = self.fields.get('AntennaCode', '')
        self.radomes = get_radomes_for_antenna(cnn, current_antenna) if cnn else ['NONE']
        # Standard height codes
        self.height_codes = [
            'DHARP', 'DHBCR', 'DHBGP', 'DHPAB', 'DHTGP',
            'SLBCE', 'SLBCR', 'SLBDN', 'SLBGN', 'SLBGP', 'SLHGP', 'SLSGP', 'SLTGN'
        ]

    def _create_field_widget(self, field: str, value: str):
        """Create the appropriate widget for a field."""
        if field == 'ReceiverCode' and self.receivers:
            return SearchableSelect(
                options=self.receivers,
                value=value,
                placeholder="Type or select receiver...",
                widget_id=f"select_{field}",
                classes="field-select"
            )
        elif field == 'AntennaCode' and self.antennas:
            return SearchableSelect(
                options=self.antennas,
                value=value,
                placeholder="Type or select antenna...",
                widget_id=f"select_{field}",
                classes="field-select"
            )
        elif field == 'RadomeCode':
            return SearchableSelect(
                options=self.radomes,
                value=value,
                placeholder="Type or select radome...",
                widget_id=f"select_{field}",
                classes="field-select"
            )
        elif field == 'HeightCode':
            return SearchableSelect(
                options=self.height_codes,
                value=value,
                placeholder="Select height code...",
                widget_id=f"select_{field}",
                classes="field-select"
            )
        else:
            return Input(value=value, id=f"input_{field}", classes="field-input")

    def _field_group(self, field: str) -> Horizontal:
        """Create a label + widget group for a field."""
        value = self.fields.get(field, '')
        css_class = "field-group-select" if field in self.SEARCHABLE_FIELDS else "field-group"
        return Horizontal(
            Label(f"{field}:", classes="field-label-compact"),
            self._create_field_widget(field, value),
            classes=css_class
        )

    def compose(self) -> ComposeResult:
        with Vertical(classes="edit-dialog"):
            yield Static(self.title_text, classes="dialog-title")
            with Vertical(classes="edit-form"):
                # Row 1: DateStart, DateEnd (2 cols)
                yield Horizontal(
                    self._field_group('DateStart'),
                    self._field_group('DateEnd'),
                    classes="field-row-multi"
                )
                # Row 2: ReceiverCode, ReceiverSerial (2 cols)
                yield Horizontal(
                    self._field_group('ReceiverCode'),
                    self._field_group('ReceiverSerial'),
                    classes="field-row-multi"
                )
                # Row 3: ReceiverVers, ReceiverFirmware (2 cols)
                yield Horizontal(
                    self._field_group('ReceiverVers'),
                    self._field_group('ReceiverFirmware'),
                    classes="field-row-multi"
                )
                # Row 4: AntennaCode, RadomeCode, AntennaSerial (3 cols)
                yield Horizontal(
                    self._field_group('AntennaCode'),
                    self._field_group('RadomeCode'),
                    self._field_group('AntennaSerial'),
                    classes="field-row-multi"
                )
                # Row 5: AntennaHeight, AntennaNorth, AntennaEast (3 cols)
                yield Horizontal(
                    self._field_group('AntennaHeight'),
                    self._field_group('AntennaNorth'),
                    self._field_group('AntennaEast'),
                    classes="field-row-multi"
                )
                # Row 6: AntennaDAZ, HeightCode (2 cols)
                yield Horizontal(
                    self._field_group('AntennaDAZ'),
                    self._field_group('HeightCode'),
                    classes="field-row-multi"
                )
                # Row 7: Comments (full width)
                yield Horizontal(
                    Label("Comments:", classes="field-label-compact"),
                    Input(value=self.fields.get('Comments', ''), id="input_Comments", classes="field-input-wide"),
                    classes="field-row-multi"
                )
            yield Static("[Ctrl+S] Save  [Esc] Cancel", classes="dialog-help")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save", variant="success", id="save")
                yield Button("Cancel", variant="error", id="cancel")

    def on_searchable_select_selected(self, event: SearchableSelect.Selected) -> None:
        """Handle selection from searchable selects, update radomes when antenna changes."""
        if event.widget_id == "select_AntennaCode":
            antenna_code = event.value.strip()
            # Update radomes list for the new antenna
            self.radomes = get_radomes_for_antenna(self.cnn, antenna_code)
            # Update the radome SearchableSelect options
            try:
                radome_select = self.query_one(".field-select", SearchableSelect)
                for widget in self.query(SearchableSelect):
                    if widget.widget_id == "select_RadomeCode":
                        widget.update_options(self.radomes)
                        break
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "save":
            self._save()
        elif event.button.id == "cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        self._save()

    def _get_field_value(self, field: str) -> str:
        """Get value from either Input or SearchableSelect widget."""
        if field in self.SEARCHABLE_FIELDS:
            # Try to find SearchableSelect
            for widget in self.query(SearchableSelect):
                if widget.widget_id == f"select_{field}":
                    return widget.value.strip()
            return ""
        else:
            # Regular Input
            try:
                input_widget = self.query_one(f"#input_{field}", Input)
                return input_widget.value.strip()
            except Exception:
                return ""

    def _save(self) -> None:
        """Validate and save the record."""
        result = {}
        for field in self.fields:
            value = self._get_field_value(field)

            # Validation
            valid, value, error = self._validate_field(field, value)
            if not valid:
                self.notify(f"{field}: {error}", severity="error")
                return

            result[field] = value

        self.dismiss(result)

    def _validate_field(self, field: str, value: str) -> tuple:
        """
        Validate a field value. Returns (valid, processed_value, error_msg).

        Field constraints from database schema:
        - ReceiverCode: varchar(22), FK to receivers table
        - ReceiverSerial: varchar(22)
        - ReceiverFirmware: varchar(10)
        - ReceiverVers: varchar(22)
        - AntennaCode: varchar(22), FK to antennas table
        - AntennaSerial: varchar(20)
        - RadomeCode: varchar(7), must exist in antennas table for given antenna
        - AntennaHeight: numeric(6,4) - max 99.9999
        - AntennaNorth/East: numeric(12,4)
        - HeightCode: must be valid height code
        """
        # --- Date fields ---
        if field in ('DateStart', 'DateEnd'):
            try:
                if value.strip() == '' and field == 'DateEnd':
                    return True, str(pyDate.Date(stninfo=None)), None
                elif value.strip() == '':
                    return False, value, "Date is required"
                else:
                    if ' ' in value:
                        # DOY format: YYYY DDD HH MM SS
                        return True, str(pyDate.Date(stninfo=value)), None
                    else:
                        # Try process_date for other formats
                        return True, str(process_date([value])[0]), None
            except Exception:
                return False, value, "Invalid date format (use YYYY DDD HH MM SS)"

        # --- Receiver fields ---
        elif field == 'ReceiverCode':
            if not value:
                return False, value, "Receiver code required"
            if len(value) > 22:
                return False, value, "Max length is 22 characters"
            rs = self.cnn.query(f"SELECT * FROM receivers WHERE \"ReceiverCode\" = '{value.upper()}'")
            if rs.ntuples() == 0:
                return False, value, "Receiver code not found in database"
            return True, rs.dictresult()[0]['ReceiverCode'], None

        elif field == 'ReceiverSerial':
            if len(value) > 22:
                return False, value, "Max length is 22 characters"
            return True, value.strip() if value else value, None

        elif field == 'ReceiverVers':
            if len(value) > 22:
                return False, value, "Max length is 22 characters"
            return True, value.strip() if value else value, None

        elif field == 'ReceiverFirmware':
            if len(value) > 10:
                return False, value, "Max length is 10 characters"
            # Accept numeric formats, dashes, or empty
            if value and not (re.findall(r'^\d+[.]?\d*[DEde+-]?\d*$', value)
                    or value in ('-', '--', '---', '----', '-----', '')):
                return False, value, "Invalid firmware format"
            if value in ('-', '--', '---', '----'):
                value = '-----'
            return True, value.upper() if value else value, None

        # --- Antenna fields ---
        elif field == 'AntennaCode':
            if not value:
                return False, value, "Antenna code required"
            if len(value) > 22:
                return False, value, "Max length is 22 characters"
            rs = self.cnn.query(f"SELECT * FROM antennas WHERE \"AntennaCode\" = '{value.upper()}'")
            if rs.ntuples() == 0:
                return False, value, "Antenna code not found in database"
            return True, rs.dictresult()[0]['AntennaCode'], None

        elif field == 'RadomeCode':
            if not value:
                return False, value, "Radome code required"
            if len(value) > 7:
                return False, value, "Max length is 7 characters"
            value = value.upper().strip()
            # Get the current antenna code to validate the radome
            antenna_code = self._get_field_value('AntennaCode')
            if antenna_code:
                rs = self.cnn.query(
                    f"SELECT * FROM antennas WHERE \"AntennaCode\" = '{antenna_code.upper()}' "
                    f"AND \"RadomeCode\" = '{value}'"
                )
                if rs.ntuples() == 0:
                    return False, value, f"Radome '{value}' not valid for antenna '{antenna_code}'"
            return True, value, None

        elif field == 'AntennaSerial':
            if len(value) > 20:
                return False, value, "Max length is 20 characters"
            return True, value.strip() if value else value, None

        # --- Eccentricity fields ---
        elif field == 'AntennaHeight':
            try:
                val = float(value) if value else 0.0
                # numeric(6,4) means max value is 99.9999
                if val < -99.9999 or val > 99.9999:
                    return False, value, "Value out of range (-99.9999 to 99.9999)"
                return True, value, None
            except ValueError:
                return False, value, "Must be a number"

        elif field in ('AntennaNorth', 'AntennaEast'):
            try:
                val = float(value) if value else 0.0
                # numeric(12,4) allows large values
                if val < -99999999.9999 or val > 99999999.9999:
                    return False, value, "Value out of range"
                return True, value, None
            except ValueError:
                return False, value, "Must be a number"

        elif field == 'AntennaDAZ':
            try:
                val = float(value) if value else 0.0
                if val < -360.0 or val > 360.0:
                    return False, value, "Value must be between -360 and 360"
                return True, value, None
            except ValueError:
                return False, value, "Must be a number"

        # --- Height code ---
        elif field == 'HeightCode':
            valid_codes = ('DHTGP', 'DHPAB', 'SLBDN', 'SLBCR', 'SLTEP',
                          'DHBCR', 'SLHGP', 'SLTGN', 'DHARP', 'SLBCE',
                          'DHBGP', 'SLBGN', 'SLBGP', 'SLSGP')
            if not value:
                return False, value, "Height code required"
            if value.upper() not in valid_codes:
                return False, value, f"Must be one of: {', '.join(sorted(valid_codes))}"
            return True, value.upper(), None

        # --- Comments (no validation needed) ---
        elif field == 'Comments':
            return True, value, None

        # Default: accept as-is, uppercase if non-empty
        return True, value.upper() if value else value, None

class DiffViewScreen(ModalScreen[None]):
    """Modal screen showing diff between DB and file records."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def __init__(self, finding: Dict[str, Any], stn_info: Optional[StationInfo] = None):
        super().__init__()
        self.finding = finding
        self.stn_info = stn_info

    def _find_db_record(self) -> Optional[StationInfoRecord]:
        """Find the actual DB record using DateStart from finding."""
        if not self.stn_info:
            return None

        db_record_info = self.finding.get('db_record')
        if not db_record_info or not isinstance(db_record_info, dict):
            return None

        date_start_str = db_record_info.get('DateStart')
        if not date_start_str:
            return None

        # Find the record in stn_info.records using DateStart
        for record in self.stn_info.records:
            if record.DateStart:
                record_date_str = record.DateStart.strftime()
                if record_date_str == date_start_str or record_date_str[:10] == date_start_str[:10]:
                    return record
        return None

    def _format_field_changes(self) -> str:
        """Format structured field changes for display."""
        db_vals = self.finding.get('db_field_values') or {}
        file_vals = self.finding.get('file_field_values') or {}

        lines = []
        # Get all field names from both dicts
        all_fields = set(db_vals.keys()) | set(file_vals.keys())

        # Define field order: DateStart and DateEnd first, then sorted rest
        priority_fields = ['DateStart', 'DateEnd']
        ordered_fields = [f for f in priority_fields if f in all_fields]
        ordered_fields += sorted(f for f in all_fields if f not in priority_fields)

        for field in ordered_fields:
            db_val = db_vals.get(field, '(none)')
            file_val = file_vals.get(field, '(none)')
            lines.append(f"  {field}:")
            lines.append(f"    DB:   {db_val}")
            lines.append(f"    File: {file_val}")

        return "\n".join(lines) if lines else "(no field changes)"

    def _format_db_record(self, record: Optional[StationInfoRecord]) -> str:
        """Format DB record as field: value pairs in two columns."""
        if not record:
            db_record_info = self.finding.get('db_record')
            if db_record_info and isinstance(db_record_info, dict):
                return f"(DB record not found for DateStart: {db_record_info.get('DateStart', 'unknown')})"
            return '(no DB record - NEW_SESSION)'

        # Get all fields from the record using to_claude_dict
        fields = record.to_claude_dict()
        fields.pop('hash', None)  # Remove hash field

        # Trim StationName to 16 chars (1 less than column width) to avoid column shift
        if 'StationName' in fields and fields['StationName']:
            fields['StationName'] = fields['StationName'][:16]

        # Extract Comments for separate display at bottom
        comments = fields.pop('Comments', '') or ''

        # Format as two columns (excluding Comments)
        lines = []
        field_list = list(fields.items())
        mid = (len(field_list) + 1) // 2

        for i in range(mid):
            left_key, left_val = field_list[i]
            left_str = f"  {left_key}: {left_val or ''}"
            if i + mid < len(field_list):
                right_key, right_val = field_list[i + mid]
                right_str = f"{right_key}: {right_val or ''}"
                lines.append(f"{left_str:<45} {right_str}")
            else:
                lines.append(left_str)

        # Add Comments at the bottom as single column
        lines.append(f"  Comments: {comments}")

        return "\n".join(lines)

    def compose(self) -> ComposeResult:
        # Get the actual DB record using DateStart
        db_record = self._find_db_record()
        db_record_str = self._format_db_record(db_record)

        summary = self.finding.get('claude_summary') or '(no summary)'
        field_changes = self._format_field_changes()

        yield Container(
            Static(f"Diff View - {self.finding['finding_type']}", classes="dialog-title"),
            ScrollableContainer(
                Static("Claude's Analysis:", classes="section-title"),
                Static(summary, classes="summary-text"),
                Rule(),
                Static("Field Changes:", classes="section-title"),
                Static(field_changes, classes="summary-text"),
                Rule(),
                Static("Database Record:", classes="section-title"),
                Static(db_record_str, classes="record-text db-record"),
                classes="diff-scroll"
            ),
            Button("Close", variant="primary", id="close"),
            classes="diff-dialog"
        )

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)


class AuditActionScreen(ModalScreen[Optional[tuple]]):
    """Modal screen for taking action on an audit finding."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, finding: Dict[str, Any]):
        super().__init__()
        self.finding = finding

    def compose(self) -> ComposeResult:
        yield Container(
            Static(f"Action: {self.finding['finding_type']}", classes="dialog-title"),
            ScrollableContainer(
                Static(self.finding.get('claude_summary', ''), classes="summary-text"),
                classes="summary-scroll"
            ),
            Rule(),
            Label("Review Notes (optional):"),
            TextArea(id="notes", classes="notes-input"),
            Horizontal(
                Button("Apply", variant="success", id="apply"),
                Button("Dismiss", variant="warning", id="dismiss"),
                Button("Defer", variant="default", id="defer"),
                Button("Cancel", variant="error", id="cancel"),
                classes="dialog-buttons"
            ),
            classes="action-dialog"
        )

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        else:
            notes = self.query_one("#notes", TextArea).text
            action = event.button.id.upper()
            if action == "APPLY":
                action = "APPLIED"
            elif action == "DISMISS":
                action = "DISMISSED"
            elif action == "DEFER":
                action = "DEFERRED"
            self.dismiss((action, notes))


class TimelineViewScreen(ModalScreen[None]):
    """Modal screen showing session timeline visualization."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def __init__(self, findings: List[Dict[str, Any]], station_id: str,
                 stn_info: Optional[StationInfo] = None):
        super().__init__()
        self.findings = findings
        self.station_id = station_id
        self.stn_info = stn_info

    def _find_db_record_by_date(self, date_start_str: str) -> Optional[StationInfoRecord]:
        """Find DB record by DateStart string."""
        if not self.stn_info or not date_start_str:
            return None
        for record in self.stn_info.records:
            if record.DateStart:
                record_date_str = record.DateStart.strftime()
                if record_date_str == date_start_str or record_date_str[:10] == date_start_str[:10]:
                    return record
        return None

    def _generate_timeline(self) -> str:
        """Generate timeline using actual DB records."""
        lines = []
        lines.append(f"[bold]Session Timeline for {self.station_id}[/bold]")
        lines.append("")
        lines.append("[dim]Legend:[/dim] [green]●[/green] match  [yellow]●[/yellow] mismatch  [red]●[/red] missing/new")
        lines.append("")
        lines.append("[bold]  #  Type              DB Start           DB End             File Start         File End           Action[/bold]")
        lines.append("─" * 115)

        # Build rows with actual DB record data
        rows = []
        for f in self.findings:
            finding_type = f.get('finding_type', '')
            action = f.get('action_required', '')
            db_record_info = f.get('db_record')
            file_vals = f.get('file_field_values') or {}

            # Get actual DB record dates
            db_start = ''
            db_end = ''
            if db_record_info and isinstance(db_record_info, dict):
                date_start_str = db_record_info.get('DateStart', '')
                actual_record = self._find_db_record_by_date(date_start_str)
                if actual_record:
                    db_start = str(actual_record.DateStart) if actual_record.DateStart else ''
                    if actual_record.DateEnd and actual_record.DateEnd.year < 9999:
                        db_end = str(actual_record.DateEnd)
                    else:
                        db_end = '(open)'
                else:
                    db_start = date_start_str  # Fallback to stored value

            # Get file dates
            file_start = file_vals.get('DateStart', '')
            file_end = file_vals.get('DateEnd', '') or '(open)'

            rows.append({
                'type': finding_type,
                'action': action,
                'db_start': db_start or '—',
                'db_end': db_end or '—',
                'file_start': file_start or '—',
                'file_end': file_end,
            })

        # Sort by date (prefer DB start, then file start)
        def sort_key(row):
            for ds in (row['db_start'], row['file_start']):
                if ds and ds not in ('—', ''):
                    # Parse YYYY DDD HH MM SS format
                    try:
                        parts = ds.split()
                        if len(parts) >= 2:
                            return (int(parts[0]), int(parts[1]))
                    except (ValueError, IndexError):
                        pass
            return (9999, 999)

        rows.sort(key=sort_key)

        # Format rows
        for i, row in enumerate(rows, 1):
            # Color coding based on match status
            start_match = row['db_start'] == row['file_start'] or (
                row['db_start'][:10] == row['file_start'][:10] if len(row['db_start']) >= 10 and len(row['file_start']) >= 10 else False
            )
            end_match = row['db_end'] == row['file_end'] or (
                row['db_end'] == '(open)' and row['file_end'] == '(open)'
            )

            # Format with colors
            def fmt_date(val, is_match, is_missing):
                if is_missing or val == '—':
                    return f"[red]{val:17}[/red]"
                elif is_match:
                    return f"[green]{val:17}[/green]"
                else:
                    return f"[yellow]{val:17}[/yellow]"

            db_start_fmt = fmt_date(row['db_start'], start_match, row['db_start'] == '—')
            db_end_fmt = fmt_date(row['db_end'], end_match, row['db_end'] == '—')
            file_start_fmt = fmt_date(row['file_start'], start_match, row['file_start'] == '—')
            file_end_fmt = fmt_date(row['file_end'], end_match, row['file_end'] == '—')

            # Action indicator
            if row['action'] == 'INSERT':
                action_fmt = "[red]INSERT[/red]"
            elif row['action'] == 'UPDATE':
                action_fmt = "[yellow]UPDATE[/yellow]"
            else:
                action_fmt = f"[dim]{row['action'][:10]}[/dim]"

            type_short = row['type'][:15]
            lines.append(f" {i:2}  {type_short:15} {db_start_fmt} {db_end_fmt} {file_start_fmt} {file_end_fmt} {action_fmt}")

        return "\n".join(lines)

    def compose(self) -> ComposeResult:
        timeline = self._generate_timeline()

        yield Container(
            Static("Session Timeline", classes="dialog-title"),
            ScrollableContainer(
                Static(timeline, classes="timeline-text"),
                classes="timeline-scroll"
            ),
            Button("Close", variant="primary", id="close"),
            classes="timeline-dialog"
        )

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)


# =============================================================================
# Main Application
# =============================================================================

class StationInfoEditApp(App):
    """Modern TUI for editing station information records."""

    CSS = """
    Screen {
        background: $surface;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        height: 100%;
        padding: 0;
    }

    ContentSwitcher {
        height: 100%;
    }

    .dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: solid $primary;
    }

    .edit-dialog {
        width: 130;
        height: auto;
        max-height: 36;
        padding: 1 2;
        background: $panel;
        border: solid $primary;
    }

    .field-row-select {
        height: auto;
        min-height: 3;
        padding: 0;
    }

    .field-select {
        width: 1fr;
    }

    .field-row-multi {
        height: auto;
        min-height: 3;
        width: 100%;
        padding: 0;
    }

    .field-group {
        width: 1fr;
        height: 3;
    }

    .field-group-select {
        width: 1fr;
        height: auto;
        min-height: 3;
    }

    .field-label-compact {
        width: 18;
        padding: 1 1 0 0;
    }

    .field-input-wide {
        width: 1fr;
    }

    .diff-dialog {
        width: 100;
        height: auto;
        max-height: 40;
        padding: 1 2;
        background: $panel;
        border: solid $primary;
    }

    .diff-dialog .diff-scroll {
        height: auto;
        max-height: 30;
        border: solid $secondary;
        margin: 0 0 1 0;
    }

    .action-dialog {
        width: 80;
        height: auto;
        max-height: 35;
        padding: 1 2;
        background: $panel;
        border: solid $primary;
    }

    .action-dialog .summary-scroll {
        height: auto;
        max-height: 15;
        border: solid $secondary;
        margin: 0 0 1 0;
    }

    .dialog-title {
        text-align: center;
        text-style: bold;
        padding: 0 1;
        color: $text;
        background: $primary;
        height: 1;
    }

    .dialog-message {
        padding: 1;
    }

    .dialog-buttons Button {
        margin: 0 1;
    }

    .section-title {
        text-style: bold;
        color: $secondary;
        padding: 0 0 0 0;
    }

    .summary-text {
        padding: 1;
        background: $surface;
    }

    .record-text {
        padding: 1;
        background: $surface;
        overflow-x: auto;
    }

    .db-record {
        color: $text;
    }

    .file-record {
        color: $success;
    }

    .edit-form {
        height: auto;
        padding: 0;
    }

    .field-row {
        height: 3;
        padding: 0;
    }

    .field-label {
        width: 20;
        padding: 1 1 0 0;
    }

    .field-input {
        width: 1fr;
    }

    .dialog-buttons {
        height: 3;
        width: 100%;
        align: center middle;
    }

    .dialog-help {
        height: 1;
        width: 100%;
        text-align: center;
        color: $text-muted;
    }

    .notes-input {
        height: 5;
        margin: 1 0;
    }

    .timeline-dialog {
        width: 130;
        height: 35;
        padding: 1 2;
        background: $panel;
        border: solid $primary;
    }

    .timeline-scroll {
        height: 28;
        border: solid $secondary;
    }

    .timeline-text {
        padding: 1;
        background: $surface;
    }

    #search-box {
        dock: top;
        height: 3;
        padding: 0 1;
    }

    #search-box.hidden {
        display: none;
    }

    #search-input {
        width: 50;
    }

    #records-table {
        height: 100%;
    }

    #audit-table {
        height: 100%;
    }

    .stats-bar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $primary;
    }

    DataTable > .datatable--header {
        text-style: bold;
        background: $primary;
    }

    DataTable > .datatable--cursor {
        background: $secondary;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("n", "new_record", "New Record"),
        Binding("e", "edit_record", "Edit"),
        Binding("enter", "edit_record", "Edit", show=False),
        Binding("d", "delete_record", "Delete"),
        Binding("f", "focus_search", "Search"),
        Binding("/", "focus_search", "Search"),
        Binding("escape", "clear_search", "Clear Search"),
        Binding("x", "export", "Export"),
        Binding("t", "show_timeline", "Timeline"),
        Binding("tab", "next_tab", "Next Tab"),
        Binding("shift+tab", "prev_tab", "Prev Tab"),
    ]

    def __init__(self, cnn: dbConnection.Cnn, network_code: str, station_code: str):
        super().__init__()
        self.cnn = cnn
        self.network_code = network_code
        self.station_code = station_code
        self.stn_info: Optional[StationInfo] = None
        self.audit_findings: List[Dict[str, Any]] = []
        self.filtered_records: List[int] = []
        self.filtered_findings: List[int] = []
        self.search_term = ""
        # State for modal screen callbacks
        self._editing_record: Optional[StationInfoRecord] = None
        self._editing_idx: Optional[int] = None
        self._deleting_record: Optional[StationInfoRecord] = None
        self._selected_finding: Optional[Dict[str, Any]] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Records", id="records-tab"):
                yield Horizontal(
                    Label("Search: "),
                    Input(placeholder="Filter records...", id="search-input"),
                    Button("Clear", id="clear-search"),
                    id="search-box",
                    classes="hidden"
                )
                yield DataTable(id="records-table")
                yield Static("", id="records-stats", classes="stats-bar")
            with TabPane("Audit Findings", id="audit-tab"):
                yield DataTable(id="audit-table")
                yield Static("", id="audit-stats", classes="stats-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app after mounting."""
        self.title = f"Station Info Editor - {self.network_code}.{self.station_code}"
        self._load_data()
        self._setup_tables()
        # Focus the records table by default
        self.query_one("#records-table", DataTable).focus()

    def _load_data(self) -> None:
        """Load station info and audit findings."""
        self.stn_info = StationInfo(
            self.cnn, self.network_code, self.station_code, allow_empty=True
        )
        self.audit_findings = get_audit_findings(
            self.cnn, self.network_code, self.station_code
        )
        self.filtered_records = list(range(len(self.stn_info.records)))
        self.filtered_findings = list(range(len(self.audit_findings)))

    def _setup_tables(self) -> None:
        """Setup the data tables with columns."""
        # Setup records table
        records_table = self.query_one("#records-table", DataTable)
        records_table.cursor_type = "row"
        records_table.add_columns("Idx", "Status", "Start Date", "End Date",
                                  "Receiver", "Antenna", "Radome", "Height")

        # Setup audit table
        audit_table = self.query_one("#audit-table", DataTable)
        audit_table.cursor_type = "row"
        audit_table.add_columns("ID", "Type", "Action", "Disposition", "Summary", "Created")

        # Populate tables
        self._refresh_records_table()
        self._refresh_audit_table()

    def _get_record_audit_status(self, record: StationInfoRecord) -> Optional[Dict]:
        """Get audit status for a record based on date matching."""
        date_str = record.DateStart.strftime() if record.DateStart else ""
        for finding in self.audit_findings:
            db_rec = finding.get('db_record', '')
            if db_rec and date_str and date_str[:10] in db_rec:
                return finding
        return None

    def _refresh_records_table(self) -> None:
        """Refresh the records table with current data."""
        table = self.query_one("#records-table", DataTable)
        table.clear()

        for idx in self.filtered_records:
            if idx >= len(self.stn_info.records):
                continue
            record = self.stn_info.records[idx]

            # Get audit status for coloring
            audit = self._get_record_audit_status(record)
            if audit:
                finding_type = audit.get('finding_type', '')
                disposition = audit.get('disposition')
                if disposition:
                    color = DISPOSITION_COLORS.get(disposition, 'white')
                else:
                    color = AUDIT_COLORS.get(finding_type, 'white')
                status = disposition or finding_type or ''
            else:
                color = 'white'
                status = ''

            # Format dates as DOY (YYYY DDD HH MM SS)
            start_date = str(record.DateStart) if record.DateStart else ''
            if record.DateEnd and record.DateEnd.year and record.DateEnd.year < 2099:
                end_date = str(record.DateEnd)
            else:
                end_date = '9999 999 00 00 00'

            # Create styled row
            row = [
                Text(str(idx + 1), style=color),
                Text(status[:15], style=color),
                Text(start_date, style=color),
                Text(end_date, style=color),
                Text(str(record.ReceiverCode or '')[:20], style=color),
                Text(str(record.AntennaCode or '')[:15], style=color),
                Text(str(record.RadomeCode or '')[:5], style=color),
                Text(f"{record.AntennaHeight:.4f}" if record.AntennaHeight else '0.0000', style=color),
            ]
            table.add_row(*row, key=str(idx))

        # Update stats (escape brackets with backslash for Rich markup)
        stats = self.query_one("#records-stats", Static)
        total = len(self.stn_info.records)
        shown = len(self.filtered_records)
        stats.update(f"Records: {shown}/{total} | \\[n]ew \\[e]dit \\[d]elete \\[f]ind \\[x]export")

    def _refresh_audit_table(self) -> None:
        """Refresh the audit findings table."""
        table = self.query_one("#audit-table", DataTable)
        table.clear()

        for idx in self.filtered_findings:
            if idx >= len(self.audit_findings):
                continue
            finding = self.audit_findings[idx]

            finding_type = finding.get('finding_type', '')
            action = finding.get('action_required', '')
            disposition = finding.get('disposition')
            summary = (finding.get('claude_summary') or '')[:50]
            created = str(finding.get('created_at', ''))[:10]

            # Color based on disposition or finding type
            if disposition:
                color = DISPOSITION_COLORS.get(disposition, 'white')
            else:
                color = AUDIT_COLORS.get(finding_type, 'white')

            row = [
                Text(str(finding.get('api_id', '')), style=color),
                Text(finding_type[:20], style=color),
                Text(action[:10], style=color),
                Text(disposition or 'PENDING', style=color),
                Text(summary, style=color),
                Text(created, style=color),
            ]
            table.add_row(*row, key=str(idx))

        # Update stats (escape brackets with backslash for Rich markup)
        stats = self.query_one("#audit-stats", Static)
        total = len(self.audit_findings)
        pending = sum(1 for f in self.audit_findings if not f.get('disposition'))
        stats.update(f"Findings: {total} | Pending: {pending} | \\[Enter] View/Action \\[t] Timeline \\[x] Export")

    def _filter_records(self, term: str) -> None:
        """Filter records by search term."""
        self.search_term = term.lower()
        if not self.search_term:
            self.filtered_records = list(range(len(self.stn_info.records)))
        else:
            self.filtered_records = []
            for idx, record in enumerate(self.stn_info.records):
                record_str = str(record).lower()
                if self.search_term in record_str:
                    self.filtered_records.append(idx)
        self._refresh_records_table()

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_refresh(self) -> None:
        """Reload data from database."""
        self._load_data()
        self._refresh_records_table()
        self._refresh_audit_table()
        self.notify("Data refreshed")

    def action_focus_search(self) -> None:
        """Show and focus the search input."""
        try:
            search_box = self.query_one("#search-box")
            search_box.remove_class("hidden")
            search = self.query_one("#search-input", Input)
            search.focus()
        except Exception:
            pass

    def action_clear_search(self) -> None:
        """Clear search filter and hide search box."""
        try:
            search = self.query_one("#search-input", Input)
            search.value = ""
            self._filter_records("")
            # Hide the search box and focus the table
            search_box = self.query_one("#search-box")
            search_box.add_class("hidden")
            self.query_one("#records-table", DataTable).focus()
        except Exception:
            pass

    def action_new_record(self) -> None:
        """Create a new station info record."""
        # Pre-populate with last record values if available
        # Default DateStart to current date, DateEnd to open (9999 999 00 00 00)
        initial = {
            'DateStart': str(pyDate.Date(datetime=datetime.datetime.now())),
            'DateEnd': '',  # Empty string will be converted to 9999 999 00 00 00
        }
        if self.stn_info.records:
            last = self.stn_info.records[-1]
            initial.update({
                'AntennaHeight': str(last.AntennaHeight),
                'HeightCode': last.HeightCode,
                'AntennaNorth': str(last.AntennaNorth),
                'AntennaEast': str(last.AntennaEast),
                'ReceiverCode': last.ReceiverCode,
                'ReceiverVers': last.ReceiverVers or '',
                'ReceiverFirmware': last.ReceiverFirmware or '',
                'ReceiverSerial': last.ReceiverSerial or '',
                'AntennaCode': last.AntennaCode,
                'RadomeCode': last.RadomeCode,
                'AntennaSerial': last.AntennaSerial or '',
                'AntennaDAZ': str(last.AntennaDAZ or 0.0),
            })

        self.push_screen(
            EditRecordScreen(initial, "New Station Info Record", self.cnn),
            self._on_new_record_result
        )

    def _on_new_record_result(self, result: Optional[Dict]) -> None:
        """Handle result from new record screen."""
        if result:
            try:
                record = StationInfoRecord(
                    self.network_code, self.station_code, _record=result
                )
                self.stn_info.insert_station_info(record)
                self._load_data()
                self._refresh_records_table()
                self.notify("Record created successfully", severity="information")
            except Exception as e:
                self.notify(f"Error: {e}", severity="error")

    def action_edit_record(self) -> None:
        """Edit the selected record."""
        table = self.query_one("#records-table", DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self.filtered_records):
            return

        idx = self.filtered_records[table.cursor_row]
        record = self.stn_info.records[idx]

        # Store the original record for the callback
        self._editing_record = record
        self._editing_idx = idx

        # Convert record to dict
        record_dict = {
            'DateStart': str(record.DateStart) if record.DateStart else '',
            'DateEnd': str(record.DateEnd) if record.DateEnd and record.DateEnd.year and record.DateEnd.year < 2099 else '',
            'AntennaHeight': str(record.AntennaHeight or 0.0),
            'HeightCode': record.HeightCode or 'DHARP',
            'AntennaNorth': str(record.AntennaNorth or 0.0),
            'AntennaEast': str(record.AntennaEast or 0.0),
            'ReceiverCode': record.ReceiverCode or '',
            'ReceiverVers': record.ReceiverVers or '',
            'ReceiverFirmware': record.ReceiverFirmware or '',
            'ReceiverSerial': record.ReceiverSerial or '',
            'AntennaCode': record.AntennaCode or '',
            'RadomeCode': record.RadomeCode or '',
            'AntennaSerial': record.AntennaSerial or '',
            'AntennaDAZ': str(record.AntennaDAZ or 0.0),
            'Comments': record.Comments or '',
        }

        self.push_screen(
            EditRecordScreen(record_dict, f"Edit Record {idx + 1}", self.cnn),
            self._on_edit_record_result
        )

    def _on_edit_record_result(self, result: Optional[Dict]) -> None:
        """Handle result from edit record screen."""
        if result:
            try:
                new_record = StationInfoRecord(
                    self.network_code, self.station_code, _record=result
                )
                self.stn_info.update_station_info(self._editing_record, new_record)
                self._load_data()
                self._refresh_records_table()
                self.notify("Record updated successfully", severity="information")
            except Exception as e:
                self.notify(f"Error: {e}", severity="error")
        self._editing_record = None
        self._editing_idx = None

    def action_delete_record(self) -> None:
        """Delete the selected record."""
        table = self.query_one("#records-table", DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self.filtered_records):
            return

        idx = self.filtered_records[table.cursor_row]
        record = self.stn_info.records[idx]

        # Store the record for the callback
        self._deleting_record = record

        self.push_screen(
            ConfirmDialog("Delete Record", f"Delete record starting {record.DateStart}?"),
            self._on_delete_confirm
        )

    def _on_delete_confirm(self, confirmed: bool) -> None:
        """Handle delete confirmation result."""
        if confirmed:
            try:
                self.stn_info.delete_station_info(self._deleting_record)
                self._load_data()
                self._refresh_records_table()
                self.notify("Record deleted", severity="warning")
            except Exception as e:
                self.notify(f"Error: {e}", severity="error")
        self._deleting_record = None

    def action_show_timeline(self) -> None:
        """Show session timeline visualization for all findings."""
        if not self.audit_findings:
            self.notify("No audit findings to display", severity="warning")
            return

        station_id = f"{self.network_code}.{self.station_code}"
        self.push_screen(TimelineViewScreen(self.audit_findings, station_id, self.stn_info))

    async def action_export(self) -> None:
        """Export audit report to JSON."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_report_{self.network_code}_{self.station_code}_{timestamp}.json"

        report = {
            'network_code': self.network_code,
            'station_code': self.station_code,
            'exported_at': datetime.datetime.now().isoformat(),
            'records': [
                {
                    'date_start': str(r.DateStart) if r.DateStart else None,
                    'date_end': str(r.DateEnd) if r.DateEnd else None,
                    'receiver': r.ReceiverCode,
                    'antenna': r.AntennaCode,
                    'radome': r.RadomeCode,
                }
                for r in self.stn_info.records
            ],
            'audit_findings': [
                {
                    'api_id': f.get('api_id'),
                    'finding_type': f.get('finding_type'),
                    'action_required': f.get('action_required'),
                    'disposition': f.get('disposition'),
                    'summary': f.get('claude_summary'),
                    'db_record': f.get('db_record'),
                }
                for f in self.audit_findings
            ]
        }

        with open(filename, 'w') as fh:
            json.dump(report, fh, indent=2)

        self.notify(f"Exported to {filename}", severity="information")

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "search-input":
            self._filter_records(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "clear-search":
            self.action_clear_search()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in tables (ENTER key on a row)."""
        table_id = event.data_table.id

        if table_id == "records-table":
            # ENTER on records table triggers edit
            self.action_edit_record()

        elif table_id == "audit-table":
            # Show audit action dialog
            if event.cursor_row >= len(self.filtered_findings):
                return

            idx = self.filtered_findings[event.cursor_row]
            finding = self.audit_findings[idx]

            # Store finding for callbacks
            self._selected_finding = finding

            # First show the diff view (pass stn_info to look up actual DB record)
            self.push_screen(DiffViewScreen(finding, self.stn_info), self._on_diff_view_dismissed)

    def _on_diff_view_dismissed(self, _result: Any) -> None:
        """Handle diff view dismissal, show action dialog if needed."""
        finding = self._selected_finding
        if not finding:
            return

        # If not already settled, show action dialog
        if not finding.get('disposition') or finding.get('disposition') == 'DEFERRED':
            self.push_screen(AuditActionScreen(finding), self._on_audit_action_result)
        else:
            self._selected_finding = None

    def _on_audit_action_result(self, result: Optional[tuple]) -> None:
        """Handle audit action result."""
        finding = self._selected_finding
        self._selected_finding = None

        if result:
            action, notes = result
            try:
                import getpass
                user = getpass.getuser()

                # If applying, try to update/insert the station info record FIRST
                # Only mark as APPLIED if the apply succeeds
                if action == 'APPLIED':
                    self._apply_audit_finding(finding)

                # Now that the action succeeded, update the disposition
                update_audit_disposition(
                    self.cnn,
                    finding['api_id'],
                    action,
                    user,
                    notes
                )

                self._load_data()
                self._refresh_audit_table()
                self._refresh_records_table()
                self.notify(f"Finding marked as {action}", severity="information")
            except Exception as e:
                self.notify(f"Error: {e}", severity="error")

    def _apply_audit_finding(self, finding: Dict[str, Any]) -> None:
        """
        Apply an audit finding using structured field values.

        Delegates to apply_finding() which handles INSERT and UPDATE actions
        using the existing logic in StationInfo.insert_station_info() and
        StationInfo.update_station_info().

        Args:
            finding: Audit finding dict with file_field_values JSONB data
        """
        apply_finding(self.cnn, self.stn_info, finding)


# =============================================================================
# Batch Apply Functions
# =============================================================================

def get_pending_findings(cnn: dbConnection.Cnn,
                         network_code: str,
                         station_code: str) -> List[Dict[str, Any]]:
    """
    Get pending audit findings that can be auto-applied.

    Returns findings where:
    - disposition is NULL (not yet reviewed)
    - action_required is INSERT only (UPDATE requires human review)

    Args:
        cnn: Database connection
        network_code: Station network code
        station_code: Station code

    Returns:
        List of finding dicts ready for application
    """
    result = cnn.query_float(f'''
        SELECT api_id, session_hash, finding_type, action_required,
               db_record, claude_summary,
               db_field_values, file_field_values
        FROM stationinfo_audit
        WHERE "NetworkCode" = '{network_code}'
          AND "StationCode" = '{station_code}'
          AND disposition IS NULL
          AND action_required = 'INSERT'
        ORDER BY created_at ASC
    ''', as_dict=True)
    return list(result) if result else []


def list_insert_findings(cnn: dbConnection.Cnn) -> None:
    """
    List all pending INSERT findings across all stations.

    Displays a formatted list to stdout with station, date, finding type,
    and truncated summary.
    """
    result = cnn.query_float('''
        SELECT "NetworkCode", "StationCode", finding_type,
               file_field_values, claude_summary, created_at
        FROM stationinfo_audit
        WHERE disposition IS NULL
          AND action_required = 'INSERT'
        ORDER BY "NetworkCode", "StationCode",
                 COALESCE(file_field_values->>'DateStart', '9999') ASC
    ''', as_dict=True)

    findings = list(result) if result else []

    if not findings:
        print("No pending INSERT findings found.")
        return

    print(f"\nPending INSERT findings: {len(findings)}")
    print("=" * 100)
    print(f"{'Station':<12} {'DateStart':<12} {'Finding Type':<20} {'Summary':<50}")
    print("-" * 100)

    for f in findings:
        station_id = f"{f['NetworkCode']}.{f['StationCode']}"
        file_vals = f.get('file_field_values') or {}
        date_start = file_vals.get('DateStart', '')[:10] if file_vals.get('DateStart') else ''
        finding_type = f.get('finding_type', '')[:20]
        summary = f.get('claude_summary', '') or ''
        summary = (summary[:47] + '...') if len(summary) > 50 else summary

        print(f"{station_id:<12} {date_start:<12} {finding_type:<20} {summary:<50}")

    print("-" * 100)
    print(f"Total: {len(findings)} pending INSERT finding(s)\n")


def apply_finding(cnn: dbConnection.Cnn,
                  stn_info: StationInfo,
                  finding: Dict[str, Any]) -> None:
    """
    Apply a single audit finding to the stationinfo table.

    For INSERT: Creates new record and updates preceding session's end date.
    For UPDATE: Uses db_record["DateStart"] to find the actual DB record,
                then applies file_field_values to update it.

    Note: Batch mode (--apply-pending) only processes INSERT findings.
    UPDATE findings require manual review in TUI mode.

    Args:
        cnn: Database connection
        stn_info: StationInfo instance for the station
        finding: Audit finding dict with file_field_values

    Raises:
        ValueError: If finding cannot be applied
    """
    action = finding.get('action_required')
    file_field_values = finding.get('file_field_values')
    network_code = stn_info.NetworkCode
    station_code = stn_info.StationCode

    if not file_field_values:
        raise ValueError(f"Finding {finding.get('api_id')} has no file_field_values")

    if action == 'INSERT':
        # Create new record from file_field_values
        record_dict = _field_values_to_record_dict(file_field_values)
        new_record = StationInfoRecord(network_code, station_code, _record=record_dict)

        # Validate that DateStart was properly set
        if new_record.DateStart is None:
            raise ValueError(
                f"Finding {finding.get('api_id')}: DateStart is None after parsing. "
                f"file_field_values: {file_field_values}"
            )

        # insert_station_info() handles overlap detection and updates preceding
        # session's end date automatically via _handle_insert_overlaps()
        stn_info.insert_station_info(new_record)

    elif action == 'UPDATE':
        # db_record is now a dict with {"DateStart": "YYYY-MM-DD HH:MM:SS"}
        db_record_info = finding.get('db_record')
        if not db_record_info or not isinstance(db_record_info, dict):
            raise ValueError(f"Finding {finding.get('api_id')} has no db_record for UPDATE")

        date_start_str = db_record_info.get('DateStart')
        if not date_start_str:
            raise ValueError(f"Finding {finding.get('api_id')} db_record missing DateStart")

        # Find the actual record in stn_info.records using DateStart
        db_record = None
        for record in stn_info.records:
            if record.DateStart:
                # Compare date strings
                record_date_str = record.DateStart.strftime()
                if record_date_str == date_start_str or record_date_str[:10] == date_start_str[:10]:
                    db_record = record
                    break

        if not db_record:
            raise ValueError(
                f"Finding {finding.get('api_id')}: Cannot find DB record with "
                f"DateStart={date_start_str}"
            )

        # Create updated record by starting with actual DB record and applying file_field_values
        updated_dict = _record_to_dict(db_record)
        updated_dict.update(_field_values_to_record_dict(file_field_values))
        new_record = StationInfoRecord(network_code, station_code, _record=updated_dict)

        # Update the record
        stn_info.update_station_info(db_record, new_record)

    else:
        raise ValueError(f"Cannot apply finding with action '{action}'")


def _field_values_to_record_dict(field_values: Dict[str, Any]) -> Dict[str, str]:
    """Convert JSONB field values to a record dict for StationInfoRecord.

    Handles date format conversion: Claude returns ISO format (YYYY-MM-DD HH:MM:SS)
    but StationInfoRecord expects station info format (YYYY DDD HH MM SS).
    """
    from geode.pyDate import Date
    result = {}
    for key, value in field_values.items():
        if value is not None:
            str_value = str(value)
            # Convert ISO date format to station info format for DateStart/DateEnd
            if key in ('DateStart', 'DateEnd') and '-' in str_value:
                try:
                    # Parse ISO format: "2024-01-15 10:30:00" or "2024-01-15"
                    if ' ' in str_value:
                        dt = datetime.datetime.strptime(str_value, '%Y-%m-%d %H:%M:%S')
                    else:
                        dt = datetime.datetime.strptime(str_value, '%Y-%m-%d')
                    # Convert to station info format via pyDate.Date
                    str_value = str(Date(datetime=dt))
                except (ValueError, IndexError) as e:
                    raise ValueError(
                        f"Failed to parse {key}='{value}' as date: {e}"
                    )
            result[key] = str_value
    return result


def _record_to_dict(record: StationInfoRecord) -> Dict[str, str]:
    """Convert a StationInfoRecord to a dict for modification."""
    # Use to_claude_dict() which includes all fields including StationName
    result = record.to_claude_dict()
    # Remove hash field (not needed for record reconstruction)
    result.pop('hash', None)
    # Convert None values to empty strings for consistency
    for key, value in result.items():
        if value is None:
            result[key] = ''
        elif not isinstance(value, str):
            result[key] = str(value)
    return result


def mark_finding_applied(cnn: dbConnection.Cnn, api_id: int, user: str) -> None:
    """Mark a finding as APPLIED in the audit table."""
    cnn.query(f'''
        UPDATE stationinfo_audit
        SET disposition = 'APPLIED',
            reviewed_by = '{user}',
            reviewed_at = NOW(),
            updated_at = NOW()
        WHERE api_id = {api_id}
    ''')


def apply_pending_for_station(cnn: dbConnection.Cnn,
                               network_code: str,
                               station_code: str,
                               dry_run: bool = False,
                               verbose: bool = False) -> Dict[str, Any]:
    """
    Apply all pending findings for a single station.

    Wraps all changes in a transaction - rolls back on any error.

    Args:
        cnn: Database connection
        network_code: Station network code
        station_code: Station code
        dry_run: If True, don't commit changes
        verbose: If True, print detailed output

    Returns:
        Dict with results: applied, skipped, errors
    """
    import getpass
    user = getpass.getuser()
    station_id = f"{network_code}.{station_code}"

    result = {
        'station': station_id,
        'applied': 0,
        'skipped': 0,
        'errors': [],
    }

    # Get pending findings
    findings = get_pending_findings(cnn, network_code, station_code)

    if not findings:
        if verbose:
            print(f"  [{station_id}] No pending findings")
        return result

    if verbose:
        print(f"  [{station_id}] Found {len(findings)} pending finding(s)")

    # Load station info
    try:
        stn_info = StationInfo(cnn, network_code, station_code, allow_empty=True)
    except Exception as e:
        result['errors'].append(f"Failed to load station info: {e}")
        return result

    # Begin transaction
    cnn.begin_transac()

    try:
        for finding in findings:
            api_id = finding.get('api_id')
            finding_type = finding.get('finding_type')
            action = finding.get('action_required')

            if verbose:
                print(f"    - Applying {finding_type} ({action})...", end=' ')

            try:
                if not dry_run:
                    apply_finding(cnn, stn_info, finding)
                    mark_finding_applied(cnn, api_id, user)
                result['applied'] += 1
                if verbose:
                    print("OK")
            except Exception as e:
                result['errors'].append(f"Finding {api_id}: {e}")
                if verbose:
                    print(f"FAILED: {e}")
                raise  # Re-raise to trigger rollback

        # Commit transaction
        if not dry_run:
            cnn.commit_transac()
        else:
            cnn.rollback_transac()
            if verbose:
                print(f"  [{station_id}] Dry run - changes rolled back")

    except Exception as e:
        # Rollback on any error
        cnn.rollback_transac()
        if verbose:
            print(f"  [{station_id}] Transaction rolled back due to error")

    return result


def batch_apply_pending(cnn: dbConnection.Cnn,
                        stations: List[Dict[str, str]],
                        dry_run: bool = False,
                        verbose: bool = False) -> None:
    """
    Apply pending findings for multiple stations.

    Args:
        cnn: Database connection
        stations: List of station dicts with NetworkCode, StationCode
        dry_run: If True, don't commit changes
        verbose: If True, print detailed output
    """
    from tqdm import tqdm

    stats = {
        'total_stations': len(stations),
        'stations_with_findings': 0,
        'total_applied': 0,
        'total_errors': 0,
    }

    print(f"\nProcessing {len(stations)} station(s)...")
    if dry_run:
        print("[DRY RUN MODE - no changes will be committed]\n")

    for stn in tqdm(stations, desc='Applying', disable=verbose):
        network_code = stn['NetworkCode']
        station_code = stn['StationCode']

        result = apply_pending_for_station(
            cnn, network_code, station_code,
            dry_run=dry_run, verbose=verbose
        )

        if result['applied'] > 0 or result['errors']:
            stats['stations_with_findings'] += 1
        stats['total_applied'] += result['applied']
        stats['total_errors'] += len(result['errors'])

        # Print errors
        for error in result['errors']:
            print(f"  ERROR [{result['station']}]: {error}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Stations processed:      {stats['total_stations']}")
    print(f"  Stations with findings:  {stats['stations_with_findings']}")
    print(f"  Findings applied:        {stats['total_applied']}")
    print(f"  Errors:                  {stats['total_errors']}")
    if dry_run:
        print("\n[DRY RUN] No changes were committed")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Edit station info records with modern TUI or batch apply pending INSERT findings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    %(prog)s arg.unro              # TUI mode for station UNRO
    %(prog)s igs.algo              # TUI mode for station ALGO

    %(prog)s --list-inserts        # List all pending INSERT findings
    %(prog)s --apply-pending all   # Batch apply pending INSERT findings
    %(prog)s --apply-pending arg.all   # Apply for all ARG stations
    %(prog)s --apply-pending arg.unro arg.srlp  # Apply for specific stations
    %(prog)s --apply-pending --dry-run all      # Preview without changes

Note: Only INSERT findings are auto-applied. UPDATE and REVIEW findings
require manual review in TUI mode.

Keyboard shortcuts (TUI mode):
    n       New record
    e       Edit selected record
    d       Delete selected record
    f, /    Focus search box
    Esc     Clear search
    r       Refresh data
    x       Export audit report
    Tab     Switch tabs
    q       Quit
        '''
    )

    parser.add_argument('stn', type=str, nargs='*',
                       help="Station(s) in net.stnm format, or 'all'")

    parser.add_argument('--list-inserts', action='store_true',
                       help="List all pending INSERT findings across all stations")

    parser.add_argument('--apply-pending', action='store_true',
                       help="Batch mode: apply all pending INSERT findings (UPDATE requires manual review)")

    parser.add_argument('--dry-run', action='store_true',
                       help="Preview changes without committing (use with --apply-pending)")

    parser.add_argument('--verbose', action='store_true',
                       help="Verbose output (use with --apply-pending)")

    add_version_argument(parser)

    args = parser.parse_args()

    # Connect to database
    cnn = dbConnection.Cnn('gnss_data.cfg')

    # List INSERT findings mode
    if args.list_inserts:
        list_insert_findings(cnn)
        return 0

    # Batch apply mode
    if args.apply_pending:
        if not args.stn:
            print("ERROR: Station(s) required with --apply-pending")
            return 1
        # Process station list
        stations = process_stnlist(cnn, args.stn,
                                   print_summary=not args.verbose,
                                   summary_title='Stations to process:')

        if not stations:
            print("No stations found matching criteria")
            return 1

        batch_apply_pending(cnn, stations,
                           dry_run=args.dry_run,
                           verbose=args.verbose)
        return 0

    # TUI mode - single station only
    if not args.stn:
        print("ERROR: Station required. Use --list-inserts to see pending findings.")
        parser.print_usage()
        return 1

    if len(args.stn) > 1:
        print("ERROR: TUI mode supports only one station. Use --apply-pending for multiple stations.")
        return 1

    stn = args.stn[0]

    if stn.lower() == 'all':
        print("ERROR: 'all' is only valid with --apply-pending flag")
        return 1

    # Parse station identifier
    if '.' in stn:
        network_code, station_code = stn.split('.', 1)
    else:
        network_code = None
        station_code = stn

    # Query station
    if network_code:
        rs = cnn.query(
            f"SELECT * FROM stations WHERE \"NetworkCode\" = '{network_code}' "
            f"AND \"StationCode\" = '{station_code}'"
        )
    else:
        rs = cnn.query(
            f"SELECT * FROM stations WHERE \"StationCode\" = '{station_code}'"
        )

    if rs.ntuples() == 0:
        print(f"ERROR: Station '{stn}' not found")
        return 1
    elif rs.ntuples() > 1:
        print("ERROR: Multiple stations found. Use net.stnm format:")
        for r in rs.dictresult():
            print(f"  {r['NetworkCode']}.{r['StationCode']}")
        return 1

    station = rs.dictresult()[0]
    network_code = station['NetworkCode']
    station_code = station['StationCode']

    # Run the TUI app
    app = StationInfoEditApp(cnn, network_code, station_code)
    app.run()

    return 0


if __name__ == '__main__':
    exit(main())
