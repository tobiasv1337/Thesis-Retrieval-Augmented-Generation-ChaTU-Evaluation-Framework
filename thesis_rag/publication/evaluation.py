"""Aggregate synthetic benchmark composition without exporting question text."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from thesis_rag.publication.data import Sources


SPLITS = (
    ("questions_dev", "development", "Development", "Dev."),
    ("questions_validation", "validation", "Validation", "Val."),
    ("questions_test", "test", "Test", "Test"),
)
GROUPS = (
    ("Language", (("de", "German"), ("en", "English"))),
    ("Query form", (
        ("complete_question", "Complete question"),
        ("conversational_short", "Short conversational"),
        ("telegraphic_keywords", "Keyword query"),
    )),
    ("Evidence scope", (("localized", "Localized"), ("multi_section", "Several sections"))),
    ("User roles", (
        ("student", "Students"), ("student_only", "Students only"),
        ("staff", "Staff"), ("researcher", "Researchers"),
        ("public", "Public"), ("visitor", "Visitors"),
    )),
    ("Wording and framing", (
        ("paraphrase", "Natural paraphrases"), ("situational", "Situational requests"),
    )),
)


def _question_counts(questions: list[dict]) -> Counter:
    counts = Counter()
    for question in questions:
        metadata = question["metadata"]
        language = question["language"]
        form = metadata.get("surface_form")
        roles = metadata.get("user_roles")
        traits = metadata.get("traits")
        framing = metadata.get("query_framing")
        paraphrase = metadata.get("natural_paraphrase_present")
        if language not in ("de", "en"):
            raise ValueError("Unexpected query language in synthetic benchmark")
        if form not in ("complete_question", "conversational_short", "telegraphic_keywords"):
            raise ValueError("Missing or invalid query form in synthetic benchmark")
        if (not isinstance(roles, list) or not roles
                or any(role not in ("student", "staff", "researcher", "public", "visitor")
                       for role in roles) or len(roles) != len(set(roles))):
            raise ValueError("Missing or invalid reviewed user roles in synthetic benchmark")
        if not isinstance(traits, list) or any(not isinstance(trait, str) for trait in traits):
            raise ValueError("Missing or invalid question traits in synthetic benchmark")
        if framing not in ("situational_request", "direct_lookup"):
            raise ValueError("Missing or invalid reviewed framing in synthetic benchmark")
        if not isinstance(paraphrase, bool):
            raise ValueError("Missing or invalid reviewed paraphrase label in synthetic benchmark")
        counts[language] += 1
        counts[form] += 1
        # Generation context can cover several sections even for localized evidence.
        counts["multi_section" if "same_page_multi_section" in traits else "localized"] += 1
        # Role percentages use questions as denominator, including multi-role questions.
        counts.update(roles)
        counts["student_only"] += roles == ["student"]
        counts["paraphrase"] += paraphrase
        counts["situational"] += framing == "situational_request"
    return counts


def question_composition_spec(data: Sources) -> dict:
    manifest = data.json("datasets", "base_question_set_manifest.json")
    by_split = {}
    seen = set()
    for artifact, split, label, _ in SPLITS:
        path = data.verify(manifest["artifact_paths"][artifact], manifest["artifact_hashes"][artifact])
        with path.open() as handle:
            questions = [json.loads(line) for line in handle]
        total = len(questions)
        if total != manifest["artifact_counts"][artifact] or total == 0:
            raise ValueError(f"Question count differs from sealed manifest: {artifact}")
        for question in questions:
            if question["metadata"]["split"] != split:
                raise ValueError(f"Unexpected split in {artifact}")
            if question["query_id"] in seen:
                raise ValueError("Repeated query identity in synthetic splits")
            seen.add(question["query_id"])
        by_split[split] = (label, total, _question_counts(questions))

    rows = []
    table_rows = []
    for group, categories in GROUPS:
        for key, characteristic in categories:
            table_row = [f"{group}: {characteristic}"]
            for split, (label, total, counts) in by_split.items():
                share = counts[key] / total
                rows.append(dict(group=group, characteristic=characteristic, key=key,
                                 split=split, label=label, count=counts[key], total=total,
                                 share=share))
                table_row.append(f"{100 * share:.1f}")
            table_rows.append(table_row)
    spec = dict(
        topic="01_evaluation", name="synthetic_question_composition",
        title="Synthetic benchmark composition", kind="question_composition",
        rows=rows, placement="main candidate", inputs=dict(data.asset_used),
        options={
            "table_headers": ["Characteristic"] + [
                f"{label} (%)\n(n = {total:,})" for label, total, _ in by_split.values()
            ],
            "table_rows": table_rows,
        },
    )
    data.asset_used.clear()
    return spec


def draw_question_composition(asset, plt, width_mm):
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(width_mm / 25.4, 74 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 150), ylim=(0, 74))
    ax.axis("off")
    cmap = LinearSegmentedColormap.from_list(
        "share", ["#f0f5fa", "#bdd7ea", "#6baed6", "#2171b5", "#084594"]
    )
    rows = {(row["key"], row["split"]): row for row in asset.rows}
    cell_width, label_width = 11.6, 36.0
    for x in (1, 78):
        for index, (_, split, _, short) in enumerate(SPLITS):
            center = x + label_width + index * cell_width + (cell_width - .5) / 2
            total = rows[("de", split)]["total"]
            ax.text(center, 70, short, ha="center", va="center", fontsize=8, fontweight="bold")
            ax.text(center, 66, f"n={total:,}", ha="center", va="center", fontsize=6.8, color="#555555")
    layout = (
        (1, 60.5, [56, 51.5]),
        (1, 45.5, [41, 36.5, 32]),
        (1, 26, [21.5, 17]),
        (78, 60.5, [56, 51.5, 47, 42.5, 38, 33.5]),
        (78, 27.5, [23, 18.5]),
    )
    for (group, categories), (x, heading_y, positions) in zip(GROUPS, layout, strict=True):
        heading = "User roles (overlapping)" if group == "User roles" else group
        ax.text(x, heading_y, heading, fontsize=8.2, fontweight="bold", va="center", color="#222222")
        for (key, label), y in zip(categories, positions, strict=True):
            ax.text(x, y, label, fontsize=7.4, va="center", color="#222222")
            for index, (_, split, _, _) in enumerate(SPLITS):
                share = rows[(key, split)]["share"]
                left = x + label_width + index * cell_width
                ax.add_patch(Rectangle((left, y - 2.05), cell_width - .5, 4.1,
                                       facecolor=cmap(share), edgecolor="none"))
                ax.text(left + (cell_width - .5) / 2, y, f"{100 * share:.1f}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if share >= .63 else "#162632")
    ax.text(1, 8.2, "Share of questions within each split (%)", fontsize=7.2, va="center", color="#444444")
    for index in range(120):
        ax.add_patch(Rectangle((111 + index * .29, 6.3), .3, 3.8,
                               facecolor=cmap(index / 119), edgecolor="none"))
    for value in (0, 50, 100):
        ax.text(111 + value * .348, 3.5, str(value), ha="center", va="center", fontsize=6.7, color="#444444")
    return fig


def main(argv=None):
    """Export composition independently of retrieval-result publication bundles."""
    from thesis_rag.publication.export import Asset, configure_style, write_asset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("configs/thesis/publication/catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/thesis/publication"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    catalog = json.loads((root / args.catalog).read_text())
    data = Sources(root, catalog)
    asset = Asset(**question_composition_spec(data))
    output = root / args.output
    record = write_asset(asset, output, configure_style(), catalog)
    provenance = output / asset.topic / "data" / (asset.name + ".provenance.json")
    provenance.write_text(json.dumps({"inputs": asset.inputs, "asset": record}, indent=2) + "\n")
    print(f"Exported {asset.name}: {output / asset.topic}")


if __name__ == "__main__":
    main()
