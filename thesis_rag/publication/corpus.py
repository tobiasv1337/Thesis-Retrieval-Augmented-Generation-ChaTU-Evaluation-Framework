"""Descriptive corpus figures from the frozen scraper and retrieval inventories."""

from collections import Counter, defaultdict
import csv
import json
from urllib.parse import urlsplit


BLUE = "#235A81"
LIGHT_BLUE = "#89B6D3"
GREY = "#DEE3E7"
INK = "#263746"
ROOT_NAMES = {
    "studyguide": "Study guide", "az": "A–Z directory",
    "studierendenservice": "Student services", "campusleben": "Campus life",
    "studienstart": "International welcome", "bewerben-und-einschreiben": "Applications and enrolment",
    "prufungen": "Examinations", "studienorganisation": "Study organisation",
    "lehrraume": "Learning spaces", "itdienste": "IT services", "studienangebot": "Degree programmes",
}


def crawl_paths(mapping):
    """Retain the recorded forest, rejecting ambiguous, missing or cyclic parents."""
    links = {r["link"]: r for r in mapping}
    if len(links) != len(mapping) or len({r["intent"] for r in mapping}) != len(mapping):
        raise ValueError("Duplicate crawl identity")
    paths = {}
    for row in mapping:
        path, seen, current = [], set(), row
        while current:
            if current["intent"] in seen:
                raise ValueError("Cyclic crawl parent")
            seen.add(current["intent"]); path.append(current["intent"])
            parent = current["parent"]
            if parent and parent not in links:
                raise ValueError("Missing crawl parent")
            current = links.get(parent)
        paths[row["intent"]] = tuple(reversed(path))
    return paths


def priority_membership(page_id, resolved, roots, expanded):
    """Project page membership onto its raw aliases and contained fragments."""
    targets = set(resolved[page_id])
    if targets & expanded and not targets <= expanded:
        raise ValueError("Ambiguous raw-record Priority membership")
    return "manual" if targets & roots else "descendant" if targets & expanded else "other"


def corpus_specs(data):
    """Full census only: no sampling, model calls or inferential statistics."""
    import numpy as np
    snapshot = data.json("datasets", "snapshot_manifest.json")

    def artifact(name):
        path = data.verify(snapshot["artifact_paths"][name], snapshot["artifact_hashes"][name])
        return [json.loads(line) for line in path.open()]

    pages = artifact("pages"); quality = artifact("page_quality")
    sources = [json.loads(l) for l in data.verify(snapshot["artifact_paths"]["source_files"], snapshot["source_manifest_hash"]).open()]
    mapping_ref = next(r for r in sources if r["role"] == "mapping")
    with data.verify(mapping_ref["source_path"], mapping_ref["file_hash"]).open() as f:
        mapping = list(csv.DictReader(f))
    paths = crawl_paths(mapping)
    if set(paths) != {r["intent"] for r in pages}:
        raise ValueError("Scraper topics and corpus inventory differ")
    q = {r["page_id"]: r for r in quality}
    if len(q) != len(pages) or len({r["page_id"] for r in pages}) != len(pages):
        raise ValueError("Invalid corpus inventory")
    if any(r["corpus_hash"] != snapshot["corpus_hash"] for r in pages + quality):
        raise ValueError("Mixed corpus snapshots")
    catalogue = data.json("corpus_priority" if "corpus_priority" in data.catalog["sources"] else "page_scope", "priority_catalogue.json")
    bm = json.loads(data.verify(catalogue["catalogue_manifest_path"], catalogue["catalogue_manifest_hash"]).read_text())
    if bm["corpus_hash"] != snapshot["corpus_hash"]:
        raise ValueError("Priority catalogue belongs to another corpus")
    audit = json.loads(data.verify(bm["artifact_paths"]["audit"], bm["artifact_hashes"]["audit"]).read_text())
    data.verify(bm["input_paths"]["manual_csv"], bm["input_hashes"]["manual_csv"])
    roots, expanded = set(catalogue["root_page_ids"]), set(catalogue["page_ids"])
    if roots != set(audit["restricted_page_ids"]) or not roots <= expanded <= set(q):
        raise ValueError("Invalid Priority catalogue membership")
    if any(not q[p]["eligible_standard_index"] for p in expanded):
        raise ValueError("Priority catalogue contains ineligible targets")
    index = data.json("corpus", "index_manifest.json")["source"]
    if index["corpus_hash"] != snapshot["corpus_hash"] or index["eligibility_policy"] != "eligible_standard_index_v1":
        raise ValueError("Incompatible corpus index")
    data.verify(index["chunk_path"], index["chunk_hash"])
    by_intent = {r["intent"]: r for r in mapping}
    friendly = {
        "sich-weiterbilden-studium":"Continuing education",
        "studieren-unter-besonderen-bedingungen-studium":"Special study conditions",
        "zum-familienburo-studieren-unter-besonderen-bedingungen":"Family office",
        "uber-procience-uber-uns":"About proScience",
        "beratung-information-studienberatung":"Information and advice",
        "zuganglichkeit-und-erreichbarkeit-studieren-mit-beeintrachtigungen":"Accessibility",
        "orientieren-einsteigen-studienberatung":"Orientation and getting started",
        "typo3handbuch-dienste-fur-neuberufene":"TYPO3 manual",
        "webbaukasten-typo3handbuch":"Website toolkit",
        "wege-zum-beruf-studyguide":"Career paths",
        "studienstart-studyguide":"Starting your studies",
        "prufungen-studyguide":"Examinations",
        "beratungsangebote-zum-familienburo":"Advice services",
        "zu-arbeitsangelegenheiten-beratungsangebote":"Employment matters",
        "elternzeit-mutterschutz-zu-arbeitsangelegenheiten":"Parental leave and maternity",
        "newsaktuelles-startup-grunden":"News",
        "howtos-itsupport":"How-to guides",
        "kultur-sport-campusleben":"Culture and sport",
        "erfahrungen-sammeln-orientieren-einsteigen":"Gaining experience",
        "mathematik-und-modellierung-forschungsschwerpunkte":"Mathematics and modelling",
        "studierende-campusit":"Students",
        "projekte-innocampus":"Projects",
        "qio-uber-procience":"QIO",
        "zems-kultur-sport":"ZEMS",
        "organisatorisches-und-vorkurse-studienstart":"Organisation and preparatory courses",
        "studieren-mit-beeintrachtigungen-beratung-information":"Studying with disabilities",
        "wissenschaftlich-arbeiten-abschlussarbeit":"Academic work",
        "ins-ausland-gehen-studium":"Going abroad",
        "angebote-fur-schulerinnen-und-lehrkrafte-erfahrungen-sammeln":"School outreach",
        "dienste-fur-neuberufene-ticketsystem":"New appointees",
        "zum-newsroom-grune-technologien-und-produktion":"Newsroom",
        "forderprogramme-beispiel":"Funding programmes",
        "tu-startups-startup-grunden":"TU Startups",
        "karriereseminar-orientierungsveranstaltungen":"Career seminar",
        "transfer-support-newsaktuelles":"Transfer support",
        "nidi-projekte":"NIDI",
        "angebote-dein-labor":"Activities",
    }
    records = []
    for page in pages:
        path = paths[page["intent"]]
        original = by_intent[page["intent"]]
        title = ROOT_NAMES.get(page["intent"], friendly.get(page["intent"],
            original["manual_title_english"] or original["extracted_title_english"] or ""))
        title = title[:1].upper() + title[1:]
        records.append(dict(page_id=page["page_id"], intent=page["intent"], language=page["language"],
            node_title=title, parent_intent=paths[page["intent"]][-2] if len(path)>1 else "",
            crawl_root=path[0], crawl_branch=path[1] if len(path)>1 else path[0], crawl_path=" / ".join(path),
            source_url=page["url"], fragment=bool(urlsplit(page["url"]).fragment),
            eligible=q[page["page_id"]]["eligible_standard_index"], text_chars=len(page["text"]),
            priority=priority_membership(page["page_id"], audit["canonical_page_ids"], roots, expanded),
            exact_priority_member=page["page_id"] in expanded,
            resolved_page_ids=";".join(audit["canonical_page_ids"][page["page_id"]])))
    records.sort(key=lambda r: (r["crawl_path"], r["language"], r["page_id"]))
    eligible = [r for r in records if r["eligible"]]
    if len(records) != snapshot["page_count"] or len(eligible) != 4509:
        raise ValueError("Unexpected frozen corpus size")
    counts = Counter(r["priority"] for r in records)
    inputs = dict(data.asset_used); data.asset_used.clear()
    specs = []

    def add(name, title, rows, kind, **options):
        specs.append(dict(topic="00_corpus", name=name, title=title, rows=rows,
                          kind=kind, options=options, inputs=inputs))

    summary = []
    for root in sorted({r["crawl_root"] for r in records}, key=lambda root: -sum(r["crawl_root"]==root for r in records)):
        rr = [r for r in records if r["crawl_root"] == root]
        summary.append([ROOT_NAMES[root], f"{len({r['intent'] for r in rr}):,}",
                        f"{sum(r['language']=='de' for r in rr):,}", f"{sum(r['language']=='en' for r in rr):,}",
                        f"{sum(r['priority']!='other' for r in rr):,}"])
    summary.append(["Total",f"{len(mapping):,}",f"{sum(r['language']=='de' for r in records):,}",
                    f"{sum(r['language']=='en' for r in records):,}",f"{counts['manual']+counts['descendant']:,}"])
    options = dict(table_headers=["Crawl entry point", "Topics", "DE", "EN", "Priority\nrecords"], table_rows=summary)
    add("corpus_overview", "Scraped corpus by crawl entry point", records, "corpus_treemap", **options)
    add("corpus_priority_overlay", "Priority sources within the complete scraper corpus", records, "corpus_treemap", highlight=True, **options)
    topics = []
    for intent in sorted(paths, key=lambda i: paths[i]):
        rr = [r for r in records if r["intent"]==intent]
        topics.append({**rr[0], "language":"topic", "page_id":intent, "language_variants":len(rr)})
    add("scraper_topic_hierarchy", "Original scraper topics", topics, "corpus_treemap", topic_units=True, **options)

    language_rows = []
    for label, cohort in [("Source records", records),("Indexable pages", eligible)]:
        for language in ("de", "en"):
            n = sum(r["language"] == language for r in cohort)
            language_rows.append(dict(cohort=label, language=language, count=n, total=len(cohort), share=n/len(cohort)))
    add("corpus_languages", "Corpus languages before and after preparation",
        language_rows, "corpus_languages", table_headers=["Population", "Language", "Count", "Share (%)"],
        table_rows=[[r["cohort"],"German" if r["language"]=="de" else "English",f"{r['count']:,}",f"{100*r['share']:.1f}"] for r in language_rows])

    length_summary = []
    for label, cohort in [("Source records", records),("Indexable pages", eligible)]:
        for language in ("de", "en"):
            values = [r["text_chars"] for r in cohort if r["language"]==language]
            length_summary.append([label,"DE" if language=="de" else "EN",f"{len(values):,}",
                f"{sum(v==0 for v in values):,}",f"{np.median(values):,.0f}",f"{np.percentile(values,95):,.0f}",f"{max(values):,}"])
    add("corpus_page_lengths", "Extracted text lengths before and after preparation",
        records,"corpus_lengths",
        table_headers=["Population", "Lang.", "Records", "Empty", "Median", "P95", "Max."], table_rows=length_summary)

    inventory = [
        {"Unit":"Scraper topics", "Count":f"{len(mapping):,}", "Definition":"Unfiltered mapping entries"},
        {"Unit":"Topics with URL fragments", "Count":f"{sum(bool(urlsplit(r['link']).fragment) for r in mapping):,}", "Definition":"Included in scraper topics"},
        {"Unit":"Language-specific source records", "Count":f"{len(records):,}", "Definition":"DE and EN counted separately"},
        {"Unit":"Eligible pages", "Count":f"{len(eligible):,}", "Definition":"Standard-index quality rules"},
        {"Unit":"Indexed chunks", "Count":f"{index['document_count']:,}", "Definition":"Section-preferred 800/100"},
        {"Unit":"Direct Priority pages", "Count":f"{len(roots):,}", "Definition":"Manual sources and language counterparts"},
        {"Unit":"Expanded Priority pages", "Count":f"{len(expanded):,}", "Definition":"Includes direct pages and descendants to depth 5"},
        {"Unit":"Priority-associated raw records", "Count":f"{counts['manual']+counts['descendant']:,}", "Definition":"Includes recorded fragments and aliases"},
    ]
    add("corpus_inventory", "Corpus inventory and counting units",
        inventory,"table",column_format="@{}XrX@{}")
    reasons = Counter(reason for row in quality for reason in row["exclusion_reasons"])
    labels = {"empty":"Empty extracted text", "error_template":"Error template", "noncanonical_exact_duplicate":"Noncanonical exact duplicate", "contained_fragment_subrecord":"Contained fragment subrecord"}
    add("corpus_quality_flags", "Corpus-quality exclusions",
        [{"Exclusion reason":labels[k],"Records":f"{v:,}"} for k,v in sorted(reasons.items())],
        "table",column_format="@{}Xr@{}")
    return specs


def treemap_nodes(rows):
    """One frame per scraper topic; own records and descendants counted once.

    The stored source inventory stays at language-record granularity. A topic is
    marked direct if either language resolves to a direct manual source, otherwise
    descendant if either resolves to the expanded catalogue. Colour is local to
    this topic and is never inherited from a Priority parent.
    """
    nodes = {}
    for row in rows:
        node = nodes.setdefault(row["intent"], dict(
            id="topic:"+row["intent"], rows=[], children=[], label=row["node_title"],
            parent="topic:"+row["parent_intent"] if row["parent_intent"] else "",
            root=row["crawl_root"], depth=row["crawl_path"].count(" / ")))
        node["rows"].append(row)
    by_id = {n["id"]: n for n in nodes.values()}
    roots = []
    for node in nodes.values():
        if node["parent"]:
            by_id[node["parent"]]["children"].append(node["id"])
        else:
            roots.append(node["id"])
        node["value"] = len(node["rows"])
        memberships = {r["priority"] for r in node["rows"]}
        node["priority"] = "manual" if "manual" in memberships else "descendant" if "descendant" in memberships else "other"

    def count(node_id, path):
        if node_id in path:
            raise ValueError("Cyclic treemap parent")
        node = by_id[node_id]
        node["subtree_value"] = node["value"] + sum(count(c, path | {node_id}) for c in node["children"])
        return node["subtree_value"]

    if sum(count(root, set()) for root in roots) != len(rows):
        raise ValueError("Disconnected treemap hierarchy")
    if len({r["page_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate treemap record")
    return list(nodes.values())


def draw_treemap(asset, width_mm=150):
    """Original Plotly nesting, with short English labels and explicit weights.

    Use a detailed logical canvas and scale the complete vector artwork to the
    requested print width. Enlarging every node's header at print resolution
    would collapse the deepest subtrees. Labels shrink locally as in the original.
    """
    import html
    import plotly.graph_objects as go
    nodes = treemap_nodes(asset.rows)
    palette = dict(zip(ROOT_NAMES, ["#D97768", "#43A58A", "#747FB9", "#A47DB7", "#D6A366",
                                   "#57AABC", "#CE7698", "#8D9BAD", "#C3B35E", "#96B56A", "#BC87BC"], strict=True))
    colors = []
    for node in nodes:
        if asset.options.get("highlight"):
            color = {"manual": BLUE, "descendant": LIGHT_BLUE, "other": "#CBD1D7"}[node["priority"]]
        else:
            rgb = [int(palette[node["root"]][i:i+2], 16) for i in (1, 3, 5)]
            light = .45 if node["depth"] == 0 else .26 if node["children"] else 0
            color = "#"+"".join(f"{round(v+(255-v)*light):02x}" for v in rgb)
        colors.append(color)
    fig = go.Figure(go.Treemap(
        ids=[n["id"] for n in nodes], parents=[n["parent"] for n in nodes],
        labels=[html.escape(n["label"]) for n in nodes], values=[n["value"] for n in nodes],
        branchvalues="remainder", sort=True, maxdepth=-1,
        tiling=dict(packing="squarify", pad=2),
        marker=dict(colors=colors, line=dict(color="white", width=.75), pad=dict(t=26, l=3, r=3, b=3)),
        textinfo="label", textfont=dict(family="DejaVu Sans", size=16,
            color=["white" if asset.options.get("highlight") and n["priority"] == "manual" else INK for n in nodes]),
        pathbar=dict(visible=False), root=dict(color="white"), hoverinfo="skip"))
    fig.update_layout(width=1200, height=860, margin=dict(l=2, r=2, t=2, b=82), paper_bgcolor="white",
                      font=dict(family="DejaVu Sans", size=16, color=INK))
    if asset.options.get("highlight"):
        for x, label, color in [(.04, "Direct Priority pages", BLUE),
                (.39, "Descendants (depth ≤ 5)", LIGHT_BLUE), (.78, "Other pages", "#CBD1D7")]:
            fig.add_shape(type="rect", xref="paper", yref="paper", x0=x, x1=x+.018,
                          y0=-.045, y1=-.021, line_width=0, fillcolor=color)
            fig.add_annotation(xref="paper", yref="paper", x=x+.027, y=-.033,
                text=label, showarrow=False, xanchor="left", yanchor="middle", font=dict(size=19))
        topics = sum(n["priority"] != "other" for n in nodes)
        records = sum(r["priority"] != "other" for r in asset.rows)
        footer = f"{topics:,} Priority-associated topics · {records:,} of {len(asset.rows):,} source records ({100*records/len(asset.rows):.1f}%)"
    else:
        unit = "topics" if asset.options.get("topic_units") else "language-specific source records"
        footer = f"{len(asset.rows):,} {unit} · unfiltered scraper corpus"
    fig.add_annotation(xref="paper", yref="paper", x=.5, y=-.075 if asset.options.get("highlight") else -.045,
        text=footer, showarrow=False, xanchor="center", yanchor="middle", font=dict(size=19))
    return fig


def write_treemap(asset, paths, width_mm):
    import os
    from pathlib import Path
    import shutil
    import plotly.io as pio
    # Reuse an installed browser; this exporter never downloads one implicitly.
    if not os.environ.get("BROWSER_PATH") and not any(shutil.which(n) for n in ("chromium", "chromium-browser", "google-chrome", "chrome")):
        cached = sorted((Path.home()/".cache/ms-playwright").glob("chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"))
        if cached:
            os.environ["BROWSER_PATH"] = str(cached[-1])
    pio.defaults.mathjax = None
    fig = draw_treemap(asset, width_mm)
    print_scale = width_mm/25.4*96/fig.layout.width
    pio.write_images([fig]*len(paths), file=[str(p) for p in paths], format=[p.suffix[1:] for p in paths],
        width=fig.layout.width, height=fig.layout.height, scale=[2 if p.suffix == ".png" else print_scale for p in paths])


def draw_corpus(asset, plt, width_mm):
    import numpy as np
    from matplotlib.ticker import PercentFormatter, LogLocator, NullFormatter
    width = width_mm / 25.4
    if asset.kind == "corpus_languages":
        fig,axes=plt.subplots(1,2,figsize=(width,44/25.4));fig.subplots_adjust(left=.035,right=.965,top=.83,bottom=.04,wspace=.3)
        for i,(ax,label) in enumerate(zip(axes,["Source records","Indexable pages"],strict=True)):
            rows=[r for r in asset.rows if r["cohort"]==label]
            wedges,_=ax.pie([r["count"] for r in rows],colors=[BLUE,"#BCC9D3"],startangle=90,counterclock=False,
                            wedgeprops=dict(width=.30,edgecolor="white",linewidth=1.5))
            ax.text(0,.07,f"{rows[0]['total']:,}",ha="center",va="center",fontsize=14,color=INK)
            ax.text(0,-.18,"records" if i==0 else "pages",ha="center",fontsize=8,color=INK)
            for wedge,row in zip(wedges,rows,strict=True):
                angle=np.deg2rad((wedge.theta1+wedge.theta2)/2)
                side = 1 if np.cos(angle) > 0 else -1
                ax.text(side*1.09,1.12*np.sin(angle),
                        ("German" if row["language"]=="de" else "English")+f"\n{100*row['share']:.1f}%",
                        ha="left" if side > 0 else "right",va="center",fontsize=8,color=INK)
            ax.set(xlim=(-1.95,1.95),ylim=(-1.1,1.1))
            ax.text(.5,1.06,chr(97+i)+") "+label,transform=ax.transAxes,ha="center",fontsize=9,color=INK)
        return fig
    if asset.kind == "corpus_lengths":
        fig,axes=plt.subplots(1,2,figsize=(width,80/25.4),sharey=True)
        fig.subplots_adjust(left=.11,right=.98,top=.84,bottom=.26,wspace=.15)
        for i,ax in enumerate(axes):
            rr=[r for r in asset.rows if r["text_chars"]>0 and (not i or r["eligible"])]
            for language,color,style in [("de",BLUE,"-"),("en","#6F8495","--")]:
                values=np.sort([r["text_chars"] for r in rr if r["language"]==language])
                ax.step(values,np.arange(1,len(values)+1)/len(values),where="post",color=color,linestyle=style,
                        label="German" if language=="de" else "English",lw=1.3)
            ax.set_xscale("log");ax.set_xlim(1,200000);ax.set_ylim(0,1.01)
            ax.set_xticks([10,100,1000,10000,100000],["10","100","1k","10k","100k"])
            ax.xaxis.set_minor_locator(LogLocator(base=10,subs=[]));ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis="both")
            ax.set_xlabel("Text length (characters)")
            title="Non-empty source records" if i==0 else "Indexable pages"
            ax.text(0,1.07,f"{chr(97+i)}) {title}\n(n = {len(rr):,})",transform=ax.transAxes,fontsize=8.5,color=INK)
        axes[0].set_ylabel("Cumulative share")
        fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
        return fig
    raise ValueError(f"Unknown corpus plot {asset.kind}")
