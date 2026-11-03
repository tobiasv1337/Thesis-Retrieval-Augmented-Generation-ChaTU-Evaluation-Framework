from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from thesis_rag.corpus.html_parser import (
    extract_page_html_metadata,
    parse_html_page,
    resolve_citation_url,
    table_to_text,
)
from thesis_rag.corpus.metadata import TopicMetadata
from thesis_rag.schema import validate_record_shape


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _topic_metadata() -> TopicMetadata:
    row = {
        "manual_title_deutsch": "Manueller Titel",
        "generated_title_deutsch": "Generierter Titel",
        "extracted_title_deutsch": "Extrahierter Titel",
        "manual_title_english": "Manual title",
        "generated_title_english": "Generated title",
        "extracted_title_english": "Extracted title",
    }
    return TopicMetadata(
        intent="topic",
        depth=1,
        parent="root",
        etag="etag",
        link="https://www.tu.berlin/topic-from-mapping",
        mapping="https://www.tu.berlin/en/topic-from-mapping",
        row=row,
        manual_utterance_count=2,
    )


def test_parser_extracts_page_sections_tables_contact_media_and_offsets(tmp_path):
    repo_root = tmp_path
    content_path = _write(
        repo_root / "data/az/scraper/topics/topic/content.html",
        """
        <div class="content clr">
          <div class="grid" id="c100">
            <header><h1>Top Heading</h1></header>
            <div class="frame frame--type-text" id="c101">
              <p>First paragraph&nbsp;with   spaces.</p>
              <ul><li>List entry</li></ul>
              <h2>Fees</h2>
              <table>
                <caption>Semester fees</caption>
                <tr><th>Amount</th><th>Kind</th></tr>
                <tr><td>50 EUR</td><td>Admin fee</td></tr>
              </table>
            </div>
            <div class="frame frame--type-tubstaff_contacts" id="c102">
              <p class="name">Student Office</p>
              <p class="mail">office(at)tu-berlin.de</p>
            </div>
            <figure>
              <img src="image.jpg" alt="Students at a desk" title="Desk image" />
              <small class="copyright">TU Berlin</small>
              <figcaption>Advising situation</figcaption>
            </figure>
            <div class="gallery-source">
              <span class="data"
                data-alternative="Gallery alternative"
                data-title="Gallery title"
                data-description="Gallery description"
                data-big="gallery.jpg"
                data-copyright="Photographer"></span>
            </div>
            <figure class="video">
              <video width="400"><source src="movie.mp4" type="video/mp4" /></video>
              <figcaption>Intro video</figcaption>
            </figure>
            <div class="teaser"><p>Teaser copy</p></div>
            <nav><ul><li>Navigation copy</li></ul></nav>
          </div>
        </div>
        """,
    )
    page_path = _write(
        repo_root / "data/az/scraper/topics/topic/page.html",
        """
        <html lang="de">
          <head>
            <title>HTML Page Title - TU Berlin</title>
            <meta name="description" content="A compact description." />
            <link rel="canonical" href="https://www.tu.berlin/canonical-topic" />
            <link rel="alternate" hreflang="en"
              href="https://www.tu.berlin/en/canonical-topic" />
            <script type="application/ld+json">
              [{
                "@context": "https://schema.org",
                "@type": "BreadcrumbList",
                "itemListElement": [
                  {
                    "@type": "ListItem",
                    "position": 1,
                    "item": {"@id": "https://www.tu.berlin", "name": "TU"}
                  },
                  {
                    "@type": "ListItem",
                    "position": 2,
                    "item": {
                      "@id": "https://www.tu.berlin/topic",
                      "name": "Topic"
                    }
                  }
                ]
              }]
            </script>
          </head>
          <body></body>
        </html>
        """,
    )

    parsed = parse_html_page(
        content_path=content_path,
        page_path=page_path,
        repo_root=repo_root,
        corpus_id="fixture",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        intent="topic",
        language="de",
        topic_metadata=_topic_metadata(),
    )

    assert parsed.page.page_id.startswith("page:")
    assert parsed.page.title == "Manueller Titel"
    assert parsed.page.url == "https://www.tu.berlin/canonical-topic"
    assert parsed.page.canonical_url == "https://www.tu.berlin/canonical-topic"
    assert parsed.page.source_url == "https://www.tu.berlin/topic-from-mapping"
    assert parsed.page.meta_description == "A compact description."
    assert parsed.page.page_html_language == "de"
    assert parsed.page.hreflang == {
        "en": "https://www.tu.berlin/en/canonical-topic",
    }
    assert parsed.page.breadcrumbs[-1]["name"] == "Topic"
    assert parsed.page.source_path == "data/az/scraper/topics/topic/content.html"
    assert parsed.page.page_source_path == "data/az/scraper/topics/topic/page.html"
    assert {
        "prose",
        "list",
        "table",
        "contact",
        "image_or_media",
        "teaser",
        "navigation_heavy",
    } <= set(parsed.page.content_flags)

    assert [section.heading_path for section in parsed.sections] == [
        ["Top Heading"],
        ["Top Heading", "Fees"],
    ]
    assert parsed.sections[0].heading_level == 1
    assert parsed.sections[0].parent_section_id is None
    assert parsed.sections[1].heading_level == 2
    assert parsed.sections[1].parent_section_id == parsed.sections[0].section_id
    assert parsed.sections[1].ancestor_section_ids == [
        parsed.sections[0].section_id,
    ]
    assert parsed.sections[0].body_text.startswith("First paragraph")
    assert parsed.sections[1].body_text.startswith("[Table: Semester fees]")
    assert "First paragraph with spaces." in parsed.sections[0].text
    assert "Student Office" in parsed.sections[1].text
    assert "Header" not in parsed.sections[1].text
    assert "Columns: Amount | Kind" in parsed.sections[1].text
    assert "Row 1: Amount = 50 EUR ; Kind = Admin fee" in parsed.sections[1].text
    assert "contact" in parsed.sections[1].content_flags
    assert parsed.sections[0].frame_ids == ["c100", "c101"]
    for section in parsed.sections:
        assert parsed.page.text[section.char_start : section.char_end] == section.text
        if section.heading_text:
            assert (
                parsed.page.text[
                    section.heading_char_start : section.heading_char_end
                ]
                == section.heading_text
            )
        if section.body_text:
            assert (
                parsed.page.text[section.body_char_start : section.body_char_end]
                == section.body_text
            )
        validate_record_shape(section.to_dict())

    assert [block.block_index for block in parsed.content_blocks] == list(
        range(len(parsed.content_blocks)),
    )
    assert {block.kind for block in parsed.content_blocks} >= {
        "heading",
        "text",
        "table",
        "contact",
    }
    for block in parsed.content_blocks:
        assert parsed.page.text[block.char_start : block.char_end] == block.text
        section = next(
            item for item in parsed.sections if item.section_id == block.section_id
        )
        assert (
            section.text[block.section_char_start : block.section_char_end]
            == block.text
        )
        validate_record_shape(block.to_dict())
    first_body_block = next(
        block for block in parsed.content_blocks if block.text.startswith("First paragraph")
    )
    assert first_body_block.container_path[-1].element_id == "c101"
    assert "frame--type-text" in first_body_block.container_path[-1].container_types

    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table.section_id == parsed.sections[1].section_id
    assert table.caption == "Semester fees"
    assert table.text.startswith("[Table: Semester fees]")
    assert parsed.page.text[table.char_start : table.char_end] == table.text

    assert len(parsed.images) == 3
    assert {image.media_type for image in parsed.images} == {
        "gallery_image",
        "image",
        "video",
    }
    assert all(image.page_id == parsed.page.page_id for image in parsed.images)
    assert all(image.section_id for image in parsed.images)

    validate_record_shape(parsed.page.to_dict())
    validate_record_shape(table.to_dict())
    for image in parsed.images:
        validate_record_shape(image.to_dict())


def test_parser_preserves_source_fragment_on_matching_canonical_base(tmp_path):
    repo_root = tmp_path
    content_path = _write(
        repo_root / "data/az/scraper/topics/topic/content.html",
        "<main><h1>Fragment section</h1><p>Anchored text.</p></main>",
    )
    page_path = _write(
        repo_root / "data/az/scraper/topics/topic/page.html",
        """
        <html lang="de">
          <head>
            <title>Fragment Page - TU Berlin</title>
            <link rel="canonical" href="https://www.tu.berlin/topic" />
          </head>
        </html>
        """,
    )
    metadata = _topic_metadata()
    metadata = TopicMetadata(
        intent=metadata.intent,
        depth=metadata.depth,
        parent=metadata.parent,
        etag=metadata.etag,
        link="https://www.tu.berlin/topic#c1640947",
        mapping=metadata.mapping,
        row=metadata.row,
        manual_utterance_count=metadata.manual_utterance_count,
    )

    parsed = parse_html_page(
        content_path=content_path,
        page_path=page_path,
        repo_root=repo_root,
        corpus_id="fixture",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        intent="topic",
        language="de",
        topic_metadata=metadata,
    )

    assert parsed.page.url == "https://www.tu.berlin/topic#c1640947"
    assert parsed.page.canonical_url == "https://www.tu.berlin/topic"
    assert parsed.page.source_url == "https://www.tu.berlin/topic#c1640947"
    assert {section.url for section in parsed.sections} == {
        "https://www.tu.berlin/topic#c1640947",
    }
    validate_record_shape(parsed.page.to_dict())
    for section in parsed.sections:
        validate_record_shape(section.to_dict())


def test_citation_url_resolver_allows_safe_base_variants_only():
    assert resolve_citation_url(
        canonical_url="https://www.tu.berlin/topic/",
        source_url="http://tu.berlin/topic#c1234",
    ) == "https://www.tu.berlin/topic/#c1234"

    assert resolve_citation_url(
        canonical_url="/topic/page",
        source_url="https://www.tu.berlin/topic/page#c1234",
    ) == "https://www.tu.berlin/topic/page#c1234"

    assert resolve_citation_url(
        canonical_url="https://www.tu.berlin/topic?view=full",
        source_url="https://www.tu.berlin/topic?view=print#c1234",
    ) == "https://www.tu.berlin/topic?view=full"

    assert resolve_citation_url(
        canonical_url="https://www.tu.berlin/topic",
        source_url="https://sub.tu.berlin/topic#c1234",
    ) == "https://www.tu.berlin/topic"

    assert resolve_citation_url(
        canonical_url="https://www.tu.berlin/Topic/Page",
        source_url="https://www.tu.berlin/topic/page#c1234",
    ) == "https://www.tu.berlin/Topic/Page"


def test_parser_prefers_authoritative_html_title_over_extracted_mapping(
    tmp_path,
):
    repo_root = tmp_path
    content_path = _write(
        repo_root / "data/az/scraper/topics/management/content_english.html",
        "<div><h2>Prof. Dr.-Ing. Sebastian Möller</h2></div>",
    )
    page_path = _write(
        repo_root / "data/az/scraper/topics/management/page_english.html",
        """
        <html lang="en">
          <head>
            <title>Management - TU Berlin</title>
            <link rel="canonical"
              href="https://www.tu.berlin/en/qu/about-us/management" />
          </head>
        </html>
        """,
    )
    topic = TopicMetadata(
        intent="leitung-uber-uns3",
        depth=1,
        parent=None,
        etag="etag",
        link="https://www.tu.berlin/qu/ueber-uns/leitung",
        mapping="https://www.tu.berlin/en/qu/about-us/management",
        row={
            "manual_title_deutsch": "",
            "extracted_title_deutsch": "Leitung",
            "manual_title_english": "",
            "extracted_title_english": "Line",
        },
    )

    parsed = parse_html_page(
        content_path=content_path,
        page_path=page_path,
        repo_root=repo_root,
        corpus_id="fixture",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        intent=topic.intent,
        language="en",
        topic_metadata=topic,
    )

    assert parsed.page.title == "Management"
    assert parsed.sections[0].title == "Management"


def test_parser_creates_fallback_sections_for_no_heading_and_empty_pages(tmp_path):
    repo_root = tmp_path
    no_heading = _write(
        repo_root / "data/az/scraper/topics/no-heading/content.html",
        "<div><p>Only a paragraph.</p></div>",
    )
    parsed = parse_html_page(
        content_path=no_heading,
        repo_root=repo_root,
        corpus_id="fixture",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        intent="no-heading",
        language="de",
    )

    assert len(parsed.sections) == 1
    assert parsed.sections[0].heading_path == ["__page__"]
    assert parsed.sections[0].text == "Only a paragraph."
    assert parsed.sections[0].heading_text is None
    assert parsed.sections[0].body_text == "Only a paragraph."
    assert parsed.sections[0].structure_quality_flags == [
        "body_only",
        "synthetic_root",
    ]
    assert parsed.page.text == "Only a paragraph."

    empty = _write(
        repo_root / "data/az/scraper/topics/empty/content.html",
        "<div></div>",
    )
    parsed_empty = parse_html_page(
        content_path=empty,
        repo_root=repo_root,
        corpus_id="fixture",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        intent="empty",
        language="de",
    )

    assert len(parsed_empty.sections) == 1
    assert parsed_empty.sections[0].heading_path == ["__page__"]
    assert parsed_empty.sections[0].text == ""
    assert parsed_empty.sections[0].structure_quality_flags == [
        "empty_section",
        "synthetic_root",
    ]
    assert parsed_empty.page.text == ""
    validate_record_shape(parsed_empty.page.to_dict())
    validate_record_shape(parsed_empty.sections[0].to_dict())


def test_parser_offsets_ignore_empty_leading_media_section(tmp_path):
    repo_root = tmp_path
    content_path = _write(
        repo_root / "data/az/scraper/topics/leading-media/content.html",
        """
        <div>
          <img src="decorative.jpg" />
          <h2>First visible section</h2>
          <p>Visible body.</p>
        </div>
        """,
    )

    parsed = parse_html_page(
        content_path=content_path,
        repo_root=repo_root,
        corpus_id="fixture",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        intent="leading-media",
        language="en",
    )

    assert [section.text for section in parsed.sections] == [
        "",
        "First visible section\nVisible body.",
    ]
    assert parsed.sections[1].char_start == 0
    for section in parsed.sections:
        assert parsed.page.text[section.char_start : section.char_end] == section.text


def test_duplicate_text_produces_stable_distinct_section_ids(tmp_path):
    repo_root = tmp_path
    content_path = _write(
        repo_root / "data/az/scraper/topics/dup/content.html",
        """
        <div>
          <h2>Duplicate</h2><p>Same text.</p>
          <h2>Duplicate</h2><p>Same text.</p>
        </div>
        """,
    )

    first = parse_html_page(
        content_path=content_path,
        repo_root=repo_root,
        corpus_id="fixture",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        intent="dup",
        language="de",
    )
    second = parse_html_page(
        content_path=content_path,
        repo_root=repo_root,
        corpus_id="fixture",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        intent="dup",
        language="de",
    )

    first_ids = [section.section_id for section in first.sections]
    second_ids = [section.section_id for section in second.sections]
    assert first_ids == second_ids
    assert len(set(first_ids)) == 2


def test_parser_persists_heading_tree_and_structure_quality_flags(tmp_path):
    content_path = _write(
        tmp_path / "data/az/scraper/topics/tree/content.html",
        """
        <main>
          <h1>Root</h1>
          <h3>Skipped level</h3><p>Nested fact.</p>
          <h2>Sibling</h2>
          <h3>Leaf</h3><p>Leaf fact.</p>
        </main>
        """,
    )

    parsed = parse_html_page(
        content_path=content_path,
        repo_root=tmp_path,
        corpus_id="fixture",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        intent="tree",
        language="de",
    )

    root, skipped, sibling, leaf = parsed.sections
    assert root.structure_quality_flags == ["heading_only"]
    assert skipped.parent_section_id == root.section_id
    assert skipped.ancestor_section_ids == [root.section_id]
    assert skipped.structure_quality_flags == ["heading_level_gap"]
    assert sibling.parent_section_id == root.section_id
    assert sibling.structure_quality_flags == ["heading_only"]
    assert leaf.parent_section_id == sibling.section_id
    assert leaf.ancestor_section_ids == [root.section_id, sibling.section_id]
    assert leaf.heading_path == ["Root", "Sibling", "Leaf"]


def test_page_html_metadata_and_table_flattening_helpers(tmp_path):
    page_path = _write(
        tmp_path / "page.html",
        """
        <html lang="en">
          <head>
            <title>English Title</title>
            <link rel="canonical" href="https://example.test/en" />
          </head>
        </html>
        """,
    )
    metadata = extract_page_html_metadata(page_path)
    assert metadata.title == "English Title"
    assert metadata.language == "en"
    assert metadata.canonical_url == "https://example.test/en"

    soup = BeautifulSoup(
        """
        <table>
          <tr><td>10 EUR</td><td>Fee</td><td>Law</td></tr>
          <tr><td>Key</td><td>Value</td></tr>
          <tr><td>A</td><td>B</td><td>C</td><td>D</td></tr>
        </table>
        """,
        "html.parser",
    )
    table_text, caption = table_to_text(soup.table)
    assert caption is None
    assert table_text.startswith("[Table]")
    assert (
        "Row 1: Column 1 = 10 EUR ; Column 2 = Fee ; Column 3 = Law"
        in table_text
    )
    assert "Row 2: Column 1 = Key ; Column 2 = Value" in table_text
    assert (
        "Row 3: Column 1 = A ; Column 2 = B ; "
        "Column 3 = C ; Column 4 = D"
    ) in table_text


def test_table_flattening_preserves_blank_header_columns_and_all_cell_text():
    soup = BeautifulSoup(
        """
        <table class="table-headerposition-1">
          <thead>
            <tr><th>Honorarprofessur</th><th></th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Prof. Michael Wagner</td>
              <td>michael.wagner[a]canberra.edu.au</td>
            </tr>
          </tbody>
        </table>
        """,
        "html.parser",
    )

    table_text, _ = table_to_text(soup.table)

    assert "Columns: Honorarprofessur | Column 2" in table_text
    assert (
        "Row 1: Honorarprofessur = Prof. Michael Wagner ; "
        "Column 2 = michael.wagner[a]canberra.edu.au"
    ) in table_text
    for value in (
        "Honorarprofessur",
        "Prof. Michael Wagner",
        "michael.wagner[a]canberra.edu.au",
    ):
        assert value in table_text


def test_table_flattening_handles_row_headers_group_rows_and_colspans():
    row_header_soup = BeautifulSoup(
        """
        <table class="table-headerposition-2">
          <tbody>
            <tr>
              <th>Kategorie</th>
              <td>Preis für vorbildliche Lehre</td>
              <td>Sonderpreis</td>
            </tr>
            <tr><td>Weitere Angaben</td><td></td><td></td></tr>
          </tbody>
        </table>
        """,
        "html.parser",
    )
    colspan_soup = BeautifulSoup(
        """
        <table>
          <thead>
            <tr><th colspan="2">Schedule</th><th>Room</th></tr>
          </thead>
          <tbody>
            <tr><td>Monday</td><td>10:00</td><td>H 101</td></tr>
          </tbody>
        </table>
        """,
        "html.parser",
    )

    row_header_text, _ = table_to_text(row_header_soup.table)
    colspan_text, _ = table_to_text(colspan_soup.table)

    assert (
        "Row 1 (Kategorie): Column 2 = Preis für vorbildliche Lehre ; "
        "Column 3 = Sonderpreis"
    ) in row_header_text
    assert "Row 2: Weitere Angaben" in row_header_text
    assert (
        "Columns: Schedule [column 1] | Schedule [column 2] | Room"
    ) in colspan_text
    assert (
        "Row 1: Schedule [column 1] = Monday ; "
        "Schedule [column 2] = 10:00 ; Room = H 101"
    ) in colspan_text


def test_parser_extracts_semantic_list_titles_without_generic_inline_noise(tmp_path):
    repo_root = tmp_path
    content_path = _write(
        repo_root / "data/az/scraper/topics/topic/content.html",
        """
        <div class="frame frame--type-menu_teaser_sitemap" id="c2184937">
          <header><h2>Weitere Wissenschaftsbereiche</h2></header>
          <div class="inner-wrap layout-1">
            <div class="solr__listItem">
              <a class="solr__listItemLink" href="/news/bildung">
                <div class="h3 solr__listItemTitle">Bildungswissenschaften</div>
                <div class="solr__listItemContent">
                  <p>Informationen zu den Bildungswissenschaften.</p>
                </div>
              </a>
            </div>
            <div class="solr__listItem">
              <a class="solr__listItemLink" href="/news/geistes">
                <span class="icon icon--arrow"></span>
                Geisteswissenschaften
              </a>
            </div>
          </div>
          <div class="generic-controls">
            © TU Berlin
            <a href="/more">mehr</a>
            <button>Alles ausklappen</button>
          </div>
          <script>console.log("ignored script code");</script>
          <style>.ignored-css { color: red; }</style>
          <p>
            Sichtbarer Absatz.
            <noscript>JavaScript muss aktiviert werden.</noscript>
          </p>
        </div>
        """,
    )
    page_path = _write(
        repo_root / "data/az/scraper/topics/topic/page.html",
        """
        <html lang="de">
          <head><title>Test - TU Berlin</title></head>
          <body></body>
        </html>
        """,
    )
    parsed = parse_html_page(
        content_path=content_path,
        page_path=page_path,
        repo_root=repo_root,
        corpus_id="test_corpus",
        corpus_hash="sha256:d04b98f48e8f8bcc15c6ae5ac050801cd6dcfd428fb5f9e65c4e16e7807340fa",
        intent="topic",
        language="de",
        topic_metadata=_topic_metadata(),
    )
    assert any(
        "Weitere Wissenschaftsbereiche" in sec.heading_path
        for sec in parsed.sections
    )
    assert any("Bildungswissenschaften" in sec.text for sec in parsed.sections)
    assert any("Geisteswissenschaften" in sec.text for sec in parsed.sections)
    assert parsed.page.text.splitlines().count("Bildungswissenschaften") == 1
    assert "Sichtbarer Absatz." in parsed.page.text
    assert "© TU Berlin" not in parsed.page.text
    assert "\nmehr\n" not in f"\n{parsed.page.text}\n"
    assert "Alles ausklappen" not in parsed.page.text
    assert "ignored script code" not in parsed.page.text
    assert "ignored-css" not in parsed.page.text
    assert "JavaScript muss aktiviert werden" not in parsed.page.text


def test_parser_keeps_none_placeholder_source_empty(tmp_path):
    repo_root = tmp_path
    content_path = _write(
        repo_root / "data/az/scraper/topics/topic/content.html",
        "None",
    )
    page_path = _write(
        repo_root / "data/az/scraper/topics/topic/page.html",
        "<html lang='de'><head><title>Placeholder</title></head></html>",
    )
    parsed = parse_html_page(
        content_path=content_path,
        page_path=page_path,
        repo_root=repo_root,
        corpus_id="test_corpus",
        corpus_hash="sha256:d04b98f48e8f8bcc15c6ae5ac050801cd6dcfd428fb5f9e65c4e16e7807340fa",
        intent="topic",
        language="de",
        topic_metadata=_topic_metadata(),
    )

    assert parsed.page.text == ""
    assert len(parsed.sections) == 1
    assert parsed.sections[0].text == ""
