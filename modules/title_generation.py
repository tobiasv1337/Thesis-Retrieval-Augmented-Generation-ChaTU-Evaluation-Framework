import os, re, requests, collections, ast
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timezone

from .config import Config

OPEN, CLOSE = "⟦", "⟧"

PROMPT = {
    "deutsch": (
        "Rolle: UX-Texter für Universitäts-Studienseiten.\n"
        "Aufgabe: Erstelle den BESTEN Button-Titel, um diese Seite unter ~4000 Seiten zu unterscheiden.\n\n"
        "WICHTIG - ZUERST PRÜFEN:\n"
        "- Übersichts-/Umbrella-Seite erkannt wenn: Seite führt mehrere verwandte Themen ein, organisiert sie in Abschnitten, listet sie auf, oder fasst sie zusammen\n"
        "- Bei Übersichtsseiten: Verwende IMMER den HAUPTKATEGORIE-/ÜBERBEGRIFF, NICHT ein Unterthema\n"
        "  * Wenn extrahierte Themen vorhanden: Erstes Thema ist oft der Überbegriff (z.B. 'Studienstart'), verwende diesen\n"
        "  * Analysiere trotzdem den Seitentext um sicherzustellen, dass es eine Übersichtsseite ist\n"
        "  Beispiel: Seite über 'Studienstart' mit Unterthemen Visum, Wohnung, Anmeldung → Titel: 'Studienstart' (NICHT 'Visum' oder 'Visum und Aufenthalt')\n"
        "  Weitere Beispiele: 'Prüfungen' (nicht 'Anmeldung'), 'Campusleben' (nicht 'Sport'), 'IT-Dienste' (nicht 'Account')\n"
        "- Wenn Seite hauptsächlich EIN spezifisches Thema behandelt (andere nur erwähnt): Wähle dieses HAUPTTHEMA\n"
        "- Extrahierte Themen sind nur HINWEISE - analysiere immer den Seitentext primär\n\n"
        "Anforderungen:\n"
        "- Länge: 1-5 Wörter, idealerweise 2-3 Wörter. Sei extrem präzise - entferne alle unnötigen Wörter. Fokus auf das Kernthema/Aktion.\n"
        "- Unterscheidungskraft: Der Titel muss diese Seite eindeutig unter 4000+ ähnlichen Seiten identifizieren.\n"
        "- Erstelle KEINE Liste oder mehrere Titel - NUR EIN Titel\n"
        "- Priorisiere wichtige Informationen:\n"
        "  * Primäre Aktion/Prozess (z.B. 'Rückmeldung', 'Einschreibung')\n"
        "  * Zielgruppe wenn relevant (z.B. 'für Master', 'für Bachelor')\n"
        "  * Spezifisches Semester wenn relevant (z.B. 'SS24', 'WS23')\n"
        "  * Spezifische Themenkategorie falls nötig zur Unterscheidung\n"
        "  * Eindeutige Identifikatoren (Formulare, Fristen, Ämter) wenn nötig\n"
        "- VERBOTEN:\n"
        "  * Generische Begriffe wie 'Information', 'Details', 'Seite', 'Über', 'Mehr', 'Übersicht'\n"
        "  * 'TU Berlin', 'Technische Universität' oder Universitätsnamen erwähnen (unnötiger Kontext)\n"
        "  * Unnötige beschreibende Wörter wie 'für Studierende', 'an der TU Berlin', 'Quiet Areas', 'Overview'\n"
        "  * Unnötige Zahlen, Daten, Jahre, IDs oder technische Codes (Semester-Codes wie SS24/WS23 sind erlaubt)\n"
        "  * Unvollständige oder sinnlose Wörter\n"
        "  * Vage Beschreibungen - bevorzuge spezifische Substantive/Aktionen\n"
        "  * Mehrere Titel oder Listen - NUR EIN Titel\n"
        "- Kontext beachten:\n"
        "  * Wenn Seite für spezifischen Abschluss (Master/Bachelor), einbeziehen\n"
        "  * Wenn Seite für spezifisches Semester (z.B. SS24, WS23), einbeziehen\n"
        "  * Wenn Seite ein Unterthema ist, unterscheidenden Eltern-Kontext einbeziehen\n"
        "  * Fokus auf das, was diese Seite von ähnlichen unterscheidet\n\n"
        f"Antworte NUR mit EINEM Titel in Markern {OPEN}Titel{CLOSE}. Keine Listen, keine mehreren Titel."
    ),
    "english": (
        "Role: UX copywriter for university study pages.\n"
        "Task: Create the BEST button title to distinguish this page among ~4000 pages.\n\n"
        "CRITICAL - CHECK FIRST:\n"
        "- Overview/Umbrella page detected if: Page introduces multiple related topics, organizes them in sections, lists them, or summarizes them\n"
        "- For overview pages: ALWAYS use the MAIN CATEGORY/TOPIC AREA name, NOT a subtopic\n"
        "  * If extracted topics are provided: First topic is often the umbrella term (e.g., 'Start of Studies'), use this\n"
        "  * Still analyze the page text to confirm it's an overview page\n"
        "  Example: Page about 'Start of Studies' with subtopics Visa, Housing, Registration → Title: 'Start of Studies' (NOT 'Visa' or 'Residence and Housing')\n"
        "  More examples: 'Exams' (not 'Registration'), 'Campus Life' (not 'Sports'), 'IT Services' (not 'Account')\n"
        "- If page is primarily about ONE specific topic (others just mentioned): Pick that MAIN topic\n"
        "- Extracted topics are only HINTS - always analyze the page text primarily\n\n"
        "Requirements:\n"
        "- Length: 1-5 words, ideally 2-3 words. Be extremely concise - remove all unnecessary words. Focus on the core topic/action only.\n"
        "- Distinctiveness: The title must uniquely identify this page among 4000+ similar pages.\n"
        "- Do NOT create a list or multiple titles - ONLY ONE title\n"
        "- Prioritize key information:\n"
        "  * Primary action/process (e.g., 'Registration', 'Enrollment')\n"
        "  * Target audience when relevant (e.g., 'for Master', 'for Bachelor')\n"
        "  * Specific semester when relevant (e.g., 'SS24', 'WS23')\n"
        "  * Specific topic category if needed for distinction\n"
        "  * Unique identifiers (forms, deadlines, offices) when necessary\n"
        "- FORBIDDEN:\n"
        "  * Generic terms like 'Information', 'Details', 'Page', 'About', 'More', 'Overview'\n"
        "  * Mentioning 'TU Berlin', 'Technische Universität', or any university name (unnecessary context)\n"
        "  * Unnecessary descriptive words like 'for students', 'at TU Berlin', 'Quiet Areas', 'Overview'\n"
        "  * Unnecessary numbers, dates, years, IDs, or technical codes (semester codes like SS24/WS23 are allowed)\n"
        "  * Incomplete or meaningless words\n"
        "  * Vague descriptions - prefer specific nouns/actions\n"
        "  * Multiple titles or lists - ONLY ONE title\n"
        "- Consider context:\n"
        "  * If page is for specific degree (Master/Bachelor), include it\n"
        "  * If page is for specific semester (e.g., SS24, WS23), include it\n"
        "  * If page is a subtopic, include distinguishing parent context\n"
        "  * Focus on what makes this page different from similar ones\n\n"
        f"Reply ONLY with ONE title in markers {OPEN}title{CLOSE}. No lists, no multiple titles."
    ),
}


def resolve_llm_url() -> str:
    url = (
        Config().get("qa_llm_url")
        or os.getenv("CHATU_QA_LLM_URL")
        or "http://127.0.0.1:5001/generate"
    )
    if url.startswith("http://0.0.0.0:"):
        url = "http://127.0.0.1:" + url.split("http://0.0.0.0:", 1)[1]
    return url.rstrip("/")


def call_title_llm(messages):
    url = resolve_llm_url()
    timeout_s = float(
        Config().get(
            "qa_llm_timeout_seconds",
            os.getenv("CHATU_QA_LLM_TIMEOUT_SECONDS", "30"),
        )
    )

    if url.endswith("/v1") or "/v1/" in (url + "/"):
        r = requests.post(
            f"{url}/chat/completions",
            json={
                "model": Config().get("qa_llm_model", "meta-llama-3.1-8b-instruct"),
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 32,
                "stop": [CLOSE, "\n"],
            },
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        )
        r.raise_for_status()
        choices = r.json().get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("message") or {}).get("content", ""))

    r = requests.post(
        url,
        json={
            "messages": messages,
            "max_length": 32,
            "do_sample": False,
            "temperature": 0.0,
            "stop": [CLOSE, "\n"],
        },
        headers={"Content-Type": "application/json"},
        timeout=timeout_s,
    )
    r.raise_for_status()
    return str(r.json().get("generate", ""))


def get_button_title(text: str, language: str = "deutsch", topics: list = None) -> str:
    lang = "deutsch" if language.lower().startswith("deutsch") else "english"

    # Format topics hint if provided
    topics_hint = ""
    if topics:
        # Handle both list and string formats
        if isinstance(topics, str):
            # Try to parse if it looks like a list string
            try:
                topics = ast.literal_eval(topics)
            except (ValueError, SyntaxError):
                # If parsing fails, treat as comma-separated string
                topics = [t.strip() for t in topics.split(",")]
        elif not isinstance(topics, list):
            topics = [str(topics)]

        if len(topics) > 0:
            topics_list = ", ".join([str(t) for t in topics[:5]])  # Limit to first 5 topics
            if lang == "deutsch":
                topics_hint = f"\n\nHinweis - Extrahierte Themen (nur als Referenz, nicht immer perfekt): {topics_list}\n"
                topics_hint += "Bei Übersichtsseiten: Erstes Thema in der Liste ist oft der Überbegriff. Analysiere trotzdem den Seitentext."
            else:
                topics_hint = f"\n\nHint - Extracted topics (reference only, not always perfect): {topics_list}\n"
                topics_hint += "For overview pages: First topic in list is often the umbrella term. Still analyze the page text."

    messages = [
        {"role": "system", "content": PROMPT[lang]},
        {"role": "user",
         "content": ("Seitentext (Plaintext):\n\n" if lang == "deutsch" else "Page text (plaintext):\n\n")
                    + text.strip()
                    + topics_hint
                    + (
                        "\n\nAntworte NUR mit dem Titel in Markern." if lang == "deutsch" else "\n\nReply ONLY with the title in markers.")
         },
    ]

    raw = call_title_llm(messages)

    # Extract title from markers if present
    # Try to find content between markers (handle both ⟦⟧ and ⟦⟩ cases)
    title = raw.strip()

    # Remove opening marker if present
    if OPEN in title:
        start_idx = title.find(OPEN) + len(OPEN)
        title = title[start_idx:]

    # Remove closing marker if present (try both ⟧ and ⟩)
    if CLOSE in title:
        end_idx = title.find(CLOSE)
        title = title[:end_idx]
    elif "⟩" in title:
        end_idx = title.find("⟩")
        title = title[:end_idx]

    # Remove all non-alphanumeric characters from beginning and end
    # This handles cases like ⟦Title⟩ or other weird markers
    title = title.strip()

    # Remove non-alphanumeric from start
    while title and not (title[0].isalnum() or title[0] in ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß', '-', ' ']):
        title = title[1:]

    # Remove non-alphanumeric from end
    while title and not (title[-1].isalnum() or title[-1] in ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß', '-', ' ']):
        title = title[:-1]

    return title.strip()


def generate_titles_for_mapping(
    az_extractor,
    path: str,
    *,
    generate_llm_titles: bool = True,
):
    """
    Generate button titles for all pages in mapping.csv using data from AZKeywordExtractor.

    Args:
        az_extractor: AZKeywordExtractor instance with loaded data
        path: Path to the directory containing mapping.csv
    """
    logging.info("Start: Generate Titles for Mapping")

    # Load mapping.csv
    mapping_file = Path(path) / "mapping.csv"
    if not mapping_file.exists():
        logging.warning(f"mapping.csv not found at {mapping_file}")
        return

    mapping = pd.read_csv(mapping_file)
    #mapping_file = Path(path) / "mapping_new.csv"

    # Add columns if they don't exist
    for col in ["extracted_title_deutsch", "extracted_title_english", "generated_title_deutsch",
                "generated_title_english", "manual_title_deutsch", "manual_title_english"]:
        if col not in mapping.columns:
            mapping[col] = None

    # Check if az_extractor has data loaded
    if not hasattr(az_extractor, 'data') or az_extractor.data is None or len(az_extractor.data) == 0:
        logging.warning("AZKeywordExtractor has no data loaded. Skipping title generation.")
        return

    # Get today's date for timestamp comparison
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    today_end = today_start + 86400  # 24 hours in seconds

    # Iterate through loaded data and generate titles only for intents that need it
    total_rows = len(az_extractor.data)
    error_count = 0
    skipped_count = 0
    batch_size = 50  # Save to CSV every N titles generated
    titles_since_last_save = 0
    total_updated = 0

    logging.info(f"Processing {total_rows} pages...")

    for idx, row in az_extractor.data.iterrows():
        if idx % 100 == 0 and idx > 0:
            logging.info(
                f"Progress: {idx}/{total_rows} ({idx * 100 / total_rows:.1f}%). Updated: {total_updated}, Skipped: {skipped_count}, Errors: {error_count}")

        intent = row.get("intent")
        language = row.get("language", "deutsch")
        text = row.get("text", "")
        topics = row.get("topics")

        if not intent or not text:
            error_count += 1
            continue

        # Extract first topic as extracted_title
        mask = mapping["intent"] == intent
        if mask.any():
            # Save extracted_title for this language
            if len(topics) > 0:
                extracted_column = f"extracted_title_{language}"
                if extracted_column in mapping.columns:
                    mapping.loc[mask, extracted_column] = topics[0] if topics and len(topics) > 0 else None

        if not generate_llm_titles:
            continue

        # Check if this (intent, language) combination needs title generation
        # Check if title is empty OR if timestamp is from today for this specific language
        column_name = f"generated_title_{language}"
        generated_title_existing = None
        mask = mapping["intent"] == intent
        if mask.any() and column_name in mapping.columns:
            generated_title_existing = mapping.loc[mask, column_name].iloc[0]

        title_empty = pd.isna(generated_title_existing) or (
                    isinstance(generated_title_existing, str) and generated_title_existing.strip() == "")

        # Check timestamp for this specific language if title exists
        timestamp = row.get("timestamp") if not title_empty else None

        # Skip if title exists and timestamp is not from today
        if not title_empty and (not timestamp or not (today_start <= timestamp < today_end)):
            skipped_count += 1
            continue

        try:
            # Generate title
            generated_title = get_button_title(text, language=language, topics=topics)

            # Update mapping dataframe immediately
            mask = mapping["intent"] == intent
            if mask.any():

                # Update the appropriate language column for generated title
                column_name = f"generated_title_{language}"
                if column_name in mapping.columns:
                    mapping.loc[mask, column_name] = generated_title
                    titles_since_last_save += 1
                    total_updated += mask.sum()

            # Batch save every batch_size titles
            if titles_since_last_save >= batch_size:
                mapping.to_csv(mapping_file, index=False)
                logging.info(
                    f"Batch saved: {titles_since_last_save} titles saved to mapping.csv (total updated: {total_updated})")
                titles_since_last_save = 0

        except Exception as e:
            logging.warning(f"Error generating title for intent '{intent}' ({language}): {e}")
            error_count += 1
            continue

    logging.info(f"Generated titles. Skipped: {skipped_count}, Errors: {error_count}")

    # Final save: save any remaining updates
    logging.info(f"Final save: {titles_since_last_save} titles saved to mapping.csv")
    columns = ["depth", "intent", "extracted_title_deutsch", "extracted_title_english", "generated_title_deutsch",
               "generated_title_english", "manual_title_deutsch", "manual_title_english", "link", "mapping", "parent",
               "etag"]

    # Save all columns to preserve everything
    mapping[columns].to_csv(mapping_file, index=False)

    logging.info(f"Total: {total_updated} rows updated in mapping.csv with generated titles")
    logging.info("Done: Generate Titles for Mapping")


# Beispiel
if __name__ == "__main__":
    from .keywords import AZKeywordExtractor
    from .config import Config

    path = Config().get("az_data_dir")

    az = AZKeywordExtractor()
    az.data = az.load_website_data()
    generate_titles_for_mapping(az, path)
    print(az.data)
