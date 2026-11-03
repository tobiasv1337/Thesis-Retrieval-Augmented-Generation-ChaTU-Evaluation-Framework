"""ChaTU source panels with literal evidence and safe source links."""

import csv
import html
from pathlib import Path
import re
from urllib.parse import quote, urlsplit


with (Path(__file__).resolve().parents[2] / "templates/responses/sekretariat.csv").open(
    encoding="utf-8-sig", newline="",
) as stream:
    _TEMPLATES = {row["name"]: row for row in csv.DictReader(stream)}


def response_text(name, language):
    return _TEMPLATES[name]["deutsch" if language == "de" else "english"]


def format_answer(answer, language):
    quoted = "> " + answer.replace("\n\n", "\r>\r> ").replace("\n", "\n> ")
    return (response_text("generated", language) + "\n" + quoted
            + "\r\r" + response_text("generated_hint", language))


def short_text(text, limit):
    """Compact a display-only excerpt, marking any cut explicitly."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    prefix = text[:limit - 1]
    if not text[limit - 1].isspace() and " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.rstrip() + "…"


def markdown_text(text, formatting=False):
    # Answers retain emphasis/lists but cannot create links, images or raw HTML.
    # The legacy product prompt already forbids answer links; source URLs belong
    # to the deterministic panels. Escape backslashes first to prevent bypasses.
    chars = "\\[]" if formatting else "\\`*_{}[]()#+-.!|"
    value = str(text)
    if formatting:
        value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub("([" + re.escape(chars) + "])", r"\\\1", value)
    value = html.escape(value, quote=False)
    return re.sub(r"(?m)^([ \t]{0,3})&gt;", r"\1>", value) if formatting else value


def safe_url(value):
    if not isinstance(value, str) or any(ord(c) < 33 for c in value) or "\\" in value:
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
            return None
    except ValueError:
        return None
    return quote(value, safe=":/?#@!$&'*+,;=%-._~")


def source_cards(result, language):
    """Use the backend's shared page order for panels and topic buttons."""
    context = set(result["context_target_ids"])
    cards = {}
    for item in sorted(result["items"], key=lambda item: item["rank"]):
        page = item["page_id"]
        titles = result.get("display_titles", {}).get(page, {})
        card = cards.setdefault(page, {
            "page_id": page, "title": item.get("title") or page,
            "topic_titles": dict(titles.get("pills", {})),
            "panel_titles": dict(titles.get("panels", {})),
            "link": safe_url(item.get("url")), "excerpts": [],
        })
        card["excerpts"].append({
            "rank": item["rank"], "text": item["text"], "target_id": item["target_id"],
            "in_context": item["target_id"] in context,
            "heading_path": list(item.get("heading_path") or []),
        })
    order = list(dict.fromkeys([*result.get("page_order", []), *cards]))
    return [cards[page] for page in order if page in cards]


def panel(cards, language, question):
    de = language == "de"
    rows = []
    for card in cards:
        preview = []
        # Keep the full ranked evidence in the dialog/API; only its UI preview
        # is shortened. Older tracker entries may not contain heading paths.
        if card["excerpts"]:
            excerpt = card["excerpts"][0]
            title = " ".join(card["title"].split()).casefold()
            headings = []
            for heading in excerpt.get("heading_path", []):
                heading = " ".join(heading.split())
                if (heading and heading.casefold() != title
                        and (not headings or heading.casefold() != headings[-1].casefold())):
                    headings.append(heading)
            if headings:
                preview.append("**" + markdown_text(short_text(" › ".join(headings[-2:]), 120)) + "**")
            preview.append(markdown_text(short_text(excerpt["text"], 300)))
            if not any(item["in_context"] for item in card["excerpts"]):
                preview.append("_Weiterer Suchtreffer_" if de else "_Additional search result_")
        display_title = card.get("panel_titles", {}).get(language) or card["title"]
        rows.append({"page_id": card["page_id"], "title": markdown_text(display_title),
                     "link": card["link"], "preview": "\n\n".join(preview)})
    return {
        "results_table": rows,
        "dialog_state": {
            "message": question,
            "result_count": len(cards),
        },
    }


def sidebar(custom, language):
    """Map the established result table directly; no global debug files."""
    title = "Sekretariat: Quellen" if language == "de" else "Sekretariat: sources"
    message_label = "Nachricht" if language == "de" else "Message"
    count_label = "Anzahl der Ergebnisse" if language == "de" else "Result Count"
    state = custom["dialog_state"]
    columns = ["Titel", "Vorschau", "Link"] if language == "de" else ["Title", "Preview", "Link"]
    return {
        "header": ("## " + title + "\n\n" + message_label + ": " + markdown_text(state["message"])
                   + "\n\n" + count_label + ": " + str(state["result_count"])),
        "column_order": columns,
        "content": [{
            "img_url": None, "url": row["link"],
            "card_description": "### " + row["title"] + "\n\n" + row["preview"],
            # The frontend escapes pipes again when joining Markdown table cells.
            "table": {columns[0]: row["title"].replace(r"\|", "&#124;"),
                      columns[1]: row["preview"].replace(r"\|", "&#124;"),
                      columns[2]: row["link"] or ""},
        } for row in custom["results_table"]],
    }
