from __future__ import annotations

from pathlib import Path

from thesis_rag.corpus.context_groups import build_section_context_groups
from thesis_rag.corpus.html_parser import ParsedPage, parse_html_page
from thesis_rag.schema import validate_record_shape


CORPUS_HASH = "sha256:" + ("6" * 64)


def _parse(tmp_path: Path, html: str, *, intent: str) -> ParsedPage:
    path = tmp_path / intent / "content.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return parse_html_page(
        content_path=path,
        repo_root=tmp_path,
        corpus_id="fixture",
        corpus_hash=CORPUS_HASH,
        intent=intent,
        language="de",
    )


def _groups(parsed: ParsedPage):
    return build_section_context_groups(
        sections=parsed.sections,
        content_blocks=parsed.content_blocks,
        config={
            "output_path": "groups.jsonl",
            "micro_max_effective_chars": 200,
            "target_group_effective_chars": 400,
            "max_group_effective_chars": 1600,
            "max_focus_sections": 4,
            "lead_in_max_chars": 120,
        },
        corpus_hash=CORPUS_HASH,
    )


def test_groups_adjacent_micro_siblings_under_structural_parent(tmp_path: Path):
    parsed = _parse(
        tmp_path,
        """
        <main>
          <h1>Study information</h1>
          <h2>First</h2><p>Short first fact.</p>
          <h2>Second</h2><p>Short second fact.</p>
          <h2>Third</h2><p>Short third fact.</p>
        </main>
        """,
        intent="micro-run",
    )

    groups = _groups(parsed)
    focus_group = next(group for group in groups if group.focus_section_ids)
    root = parsed.sections[0]
    assert focus_group.context_section_ids == [root.section_id]
    assert focus_group.focus_section_ids == [
        section.section_id for section in parsed.sections[1:]
    ]
    assert "adjacent_micro_sections" in focus_group.reason_codes
    assert "structural_ancestor_context" in focus_group.reason_codes
    assert all(group.group_index == index for index, group in enumerate(groups))
    for group in groups:
        validate_record_shape(group.to_dict())


def test_isolated_micro_section_remains_a_singleton(tmp_path: Path):
    long_body = "Long substantive fact. " * 20
    parsed = _parse(
        tmp_path,
        f"""
        <main>
          <h1>Building</h1>
          <h2>Arrival</h2><p>{long_body}</p>
          <h2>Parking</h2><p>There is one accessible parking space.</p>
          <h2>Elevators</h2><p>{long_body}</p>
        </main>
        """,
        intent="isolated-micro",
    )

    groups = _groups(parsed)
    parking = parsed.sections[2]
    parking_group = next(
        group for group in groups if parking.section_id in group.focus_section_ids
    )
    assert parking_group.focus_section_ids == [parking.section_id]
    assert "isolated_micro_section" in parking_group.reason_codes


def test_short_colon_lead_in_is_context_for_child_groups(tmp_path: Path):
    parsed = _parse(
        tmp_path,
        """
        <main>
          <h1>Application</h1><p>The following requirements apply:</p>
          <h2>Certificate</h2><p>Submit the certificate.</p>
          <h2>Form</h2><p>Submit the signed form.</p>
        </main>
        """,
        intent="lead-in",
    )

    groups = _groups(parsed)
    lead_in = parsed.sections[0]
    assert all(lead_in.section_id not in group.focus_section_ids for group in groups)
    child_group = next(group for group in groups if group.focus_section_ids)
    assert lead_in.section_id in child_group.context_section_ids
    assert "lead_in_context" in child_group.reason_codes
    assert "structural_ancestor_context" in child_group.reason_codes


def test_related_sibling_lead_in_is_context_for_following_section(tmp_path: Path):
    parsed = _parse(
        tmp_path,
        """
        <main>
          <h1>Personnel</h1>
          <div id="shared-grid" class="grid">
            <div id="lead-frame" class="frame frame--type-text">
              <h2>SAP HCM support</h2><p>For SAP HCM questions, contact:</p>
            </div>
            <div id="table-frame" class="frame frame--type-table">
              <h2>Contacts</h2><p>The SAP HCM team can be reached by email.</p>
            </div>
          </div>
        </main>
        """,
        intent="related-sibling-lead-in",
    )

    groups = _groups(parsed)
    lead_in = parsed.sections[1]
    contacts = parsed.sections[2]
    contacts_group = next(
        group for group in groups if contacts.section_id in group.focus_section_ids
    )

    assert lead_in.section_id in contacts_group.context_section_ids
    assert "lead_in_context" in contacts_group.reason_codes


def test_unrelated_sibling_after_colon_remains_separate(tmp_path: Path):
    parsed = _parse(
        tmp_path,
        """
        <main>
          <h1>Information</h1>
          <div id="status-grid" class="grid">
            <h2>Technical issue?</h2><p>Status details are on the following page:</p>
          </div>
          <div id="services-grid" class="grid">
            <h2>Our services</h2><p>This independent description of general
            university offerings is deliberately long enough to remain its own
            substantive section and does not describe the technical status.</p>
          </div>
        </main>
        """,
        intent="unrelated-sibling-after-colon",
    )

    groups = _groups(parsed)
    technical = parsed.sections[1]
    services = parsed.sections[2]
    services_group = next(
        group for group in groups if services.section_id in group.focus_section_ids
    )

    assert technical.section_id not in services_group.context_section_ids
    assert "lead_in_context" not in services_group.reason_codes


def test_unused_heading_only_leaf_is_explicit_orphan(tmp_path: Path):
    parsed = _parse(
        tmp_path,
        "<main><h1>Root</h1><h2>Empty leaf</h2></main>",
        intent="orphan",
    )

    groups = _groups(parsed)
    assert all(not group.focus_section_ids for group in groups)
    assert {group.reason_codes[0] for group in groups} == {"orphan_structural"}
    assert {
        section_id
        for group in groups
        for section_id in group.context_section_ids
    } == {section.section_id for section in parsed.sections}


def test_micro_run_respects_maximum_focus_count(tmp_path: Path):
    child_html = "".join(
        f"<h2>Part {index}</h2><p>Fact {index}.</p>" for index in range(5)
    )
    parsed = _parse(
        tmp_path,
        f"<main><h1>Root</h1>{child_html}</main>",
        intent="focus-limit",
    )

    focus_groups = [group for group in _groups(parsed) if group.focus_section_ids]
    assert [len(group.focus_section_ids) for group in focus_groups] == [4, 1]
    assert max(len(group.focus_section_ids) for group in focus_groups) == 4
