"""Export thesis figures and tables from verified experiment results."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import os
import re
import subprocess
import textwrap
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from thesis_rag.publication.data import Sources, exactly_one, number, select


@dataclass
class Asset:
    topic: str
    name: str
    title: str
    rows: list[dict]
    kind: str = "points"
    placement: str = "main candidate"
    options: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)


def translated_source_availability_rows(data: Sources) -> list[dict]:
    """Use view-filtered slice metrics, not a report's preformatted primary view."""
    rows = []
    for category, label in [("de_en_counterpart", "DE/EN counterparts"),
                            ("de_en_unpaired", "DE/EN without counterpart links"),
                            ("de_only", "German only"), ("en_only", "English only")]:
        scores = [exactly_one(data.table("language_translated", "slice_metrics.csv"),
                  judgment_view=data.catalog["sources"]["language_translated"]["view"],
                  experiment_id=eid, stream="items", target_level="page", cutoff=5,
                  metric="counterpart_family_ndcg", slice_dimension="source_language_coverage",
                  slice_value=category) for eid in ["language_query_complete_hybrid",
                      "language_cross_lingual_counterpart_dedup_hybrid"]]
        if scores[0]["query_count"] != scores[1]["query_count"]:
            raise ValueError("Language-policy subgroup denominators differ")
        rows.append({"Source availability": label, "Eligible queries": scores[0]["query_count"],
                     "Query complete": f"{float(scores[0]['estimate']):.3f}",
                     "Deduplication": f"{float(scores[1]['estimate']):.3f}"})
    return rows


def final_view_sensitivity_rows(data: Sources) -> list[dict]:
    """Compare the final synthetic systems using the same runs and two QREL views."""
    rows = []
    for eid in data.catalog["variants"]["final_synthetic"]:
        label = data.style("final_synthetic", eid)["label"]
        row = {"System": "NLU/Priority ZeRank-2" if eid == "nlu_priority_zerank" else label}
        for metric, heading, all_queries in [("counterpart_family_ndcg", "nDCG@20", False),
                                             ("hit_at_grade_2", "Hit@20 (%)", True),
                                             ("strict_hit_at_grade_3", "Strict Hit@20 (%)", True)]:
            for view, label in [("judged_only", "Judged-only"), ("hybrid", "Hybrid")]:
                score = exactly_one(data.table("final_synthetic", "performance_by_cutoff.csv"),
                                    experiment_id=eid, judgment_view="gemma4_31b:" + view,
                                    stream="items", target_level="page", metric=metric, cutoff=20)
                value = number(score.get("all_queries_estimate" if all_queries else "estimate"))
                if value is None:
                    raise ValueError("Missing final judgment-view sensitivity score")
                row[heading + "\n" + label] = f"{100 * value:.1f}" if all_queries else f"{value:.3f}"
        rows.append(row)
    return rows


def baseline_cohort_rows(data: Sources) -> list[dict]:
    """Compare independent cohorts using the same Baseline and final chunk stream."""
    cohorts=[('baseline_synthetic','Synthetic','#0072B2','o'),
             ('baseline_logs','User logs','#D55E00','s')]
    definitions=[data.catalog['sources'][source] for source, *_ in cohorts]
    if ({d['view'] for d in definitions}!={'gemma4_31b:judged_only'}
            or len({data.catalog['variants'][source]['legacy_refreshed_nlu']['config_hash']
                    for source, *_ in cohorts})!=1):
        raise ValueError('Baseline cohort comparison requires the same configuration and judgment view')
    rows=[]
    for source,label,color,marker in cohorts:
        raw=select(data.table(source,'stage_summary__hit_rates.csv'),
                   experiment_id='legacy_refreshed_nlu',judgment_view=data.catalog['sources'][source]['view'],
                   split='development',stream='items',population='all_queries')
        n=data.catalog['sources'][source]['n']
        for cutoff in [1,3,5,10,15]:
            for metric,panel in [('hit_at_grade_2','Grade ≥2'),('strict_hit_at_grade_3','Grade 3')]:
                r=exactly_one(raw,cutoff=cutoff,metric=metric)
                if (r['status']!='ok' or any(int(r[k])!=n for k in ('query_count','denominator','complete_queries'))
                        or r['target_space_id']!='chunk:legacy_recursive_character_800_100_min200'):
                    raise ValueError('Incomplete or incompatible Baseline final-ranking cohort')
                rows.append({**r,'source':source,'label':label,'panel':panel,'x':cutoff,
                    'estimate':float(r['estimate']),'ci_lower':float(r['ci_lower']),
                    'ci_upper':float(r['ci_upper']),'scale':100,'color':color,'marker':marker,'linestyle':'-'})
    return rows


def build_assets(data: Sources) -> list[Asset]:
    """Select data only. This function does not draw or fit statistics."""
    from thesis_rag.publication.corpus import corpus_specs
    from thesis_rag.publication.evaluation import question_composition_spec
    assets: list[Asset] = [Asset(**spec) for spec in corpus_specs(data)]
    assets.append(Asset(**question_composition_spec(data)))

    def add(topic, name, title, rows, *, kind="points", appendix=False, **options):
        if not rows:
            raise ValueError(f"Empty publication asset: {name}")
        assets.append(Asset(topic, name, title, rows, kind,
                            "appendix candidate" if appendix else "main candidate",
                            options, dict(data.asset_used)))
        data.asset_used.clear()

    def styled(source, experiment, row, **extra):
        return {**row, **data.style(source, experiment), **extra}

    def quality(source, topic, name, title, cutoff, *, experiments=None, target="page",
                appendix=False, rank_metric=None, ranking_panel=False):
        ids = experiments or list(data.catalog["variants"][source])
        rank_metric = rank_metric or ("counterpart_family_ndcg" if target == "page" else "ndcg")
        unit = "Page" if target == "page" else "Chunk"
        metrics = [("hit_at_grade_2", f"{unit} Hit@{cutoff}\n(grade ≥2, %)", True),
                   ("strict_hit_at_grade_3", f"{unit} Hit@{cutoff}\n(grade 3, %)", True),
                   (rank_metric, f"{'Page-family' if rank_metric == 'counterpart_family_ndcg' else unit}\nnDCG@{cutoff}", False)]
        rows = []
        for eid in ids:
            for metric, panel, all_queries in metrics:
                row = data.metric(source, eid, target, metric, cutoff, all_queries=all_queries)
                rows.append(styled(source, eid, row, panel=panel, scale=100 if all_queries else 1))
        add(topic, name, title, rows, appendix=appendix)
        if ranking_panel:
            assets.append(replace(assets[-1], name=name + "_ranking", title=title + " — ranking",
                                  rows=select(rows, metric=rank_metric), options={"width_mm": 75}))

    def latency(source, topic, name, *, experiments=None, appendix=True):
        ids = experiments or list(data.catalog["variants"][source])
        rows = []
        for eid in ids:
            r = exactly_one(data.table(source, "efficiency_summary.csv"), experiment_id=eid)
            rows.append(styled(source, eid, dict(source=source, experiment_id=eid,
                panel="Retrieval latency (s)", estimate=float(r["retrieval_median_ms"])/1000,
                ci_lower=None, ci_upper=None, p95=float(r["retrieval_p95_ms"])/1000,
                mean=float(r["retrieval_mean_ms"])/1000, scale=1)))
        add(topic, name, "Retrieval latency", rows, kind="latency", appendix=appendix)

    def paired(source, topic, name, *, comparisons=None, cutoff=20, target="page",
               metric="counterpart_family_ndcg", appendix=False):
        d = data.catalog["sources"][source]
        candidates = select(data.table(source, "paired_differences.csv"), judgment_view=d["view"],
                            stream="items", target_level=target, cutoff=cutoff, metric=metric)
        rows = []
        seen = set()
        for r in candidates:
            if comparisons and r["comparison_id"] not in comparisons:
                continue
            key = (r["left_experiment_id"], r["right_experiment_id"])
            if key in seen:
                continue
            seen.add(key)
            if number(r["ci_lower"]) is None:
                raise ValueError(f"Paired interval missing in {source}/{key}")
            left, right = (data.style(source, eid) for eid in key)
            rows.append({**right, "label": right["label"] + " − " + left["label"],
                         "source": source, "left": key[0], "right": key[1],
                         "panel": f"Δ {'page-family' if target == 'page' else target} nDCG@{cutoff}",
                         "estimate": number(r["mean_delta"]), "ci_lower": number(r["ci_lower"]),
                         "ci_upper": number(r["ci_upper"]), "n": int(r["paired_query_count"]),
                         "scale": 1, "judgment_view": d["view"], "resampling_unit": "paired_query"})
        add(topic, name, "Paired quality differences", rows, kind="differences", appendix=appendix)
        if name == "metadata_effect":
            for row in rows:
                row["label"] = data.style(source, row["left"])["label"]

    def tradeoff(source, topic, name, experiments, *, target="chunk", metric="ndcg", cutoff=20, appendix=False,
                 focus_experiments=None):
        rows=[]
        for i,eid in enumerate(experiments,1):
            row=data.metric(source,eid,target,metric,cutoff)
            timing=exactly_one(data.table(source,"efficiency_summary.csv"),experiment_id=eid)
            rows.append(styled(source,eid,row,point_id=i,x=float(timing["retrieval_p95_ms"])/1000,
                median_s=float(timing["retrieval_median_ms"])/1000,scale=1,
                panel=f"{'Page-family' if metric=='counterpart_family_ndcg' else target.capitalize()} nDCG@{cutoff}"))
            if focus_experiments is not None:
                rows[-1]['focused']=eid in focus_experiments
                if not rows[-1]['focused']:
                    rows[-1]['color']='#9AA0A6'
        add(topic,name,"Retrieval quality–latency trade-off",
            rows,kind="tradeoff",appendix=appendix,focus_labels=focus_experiments is not None)

    def ranking_views(source,topic,name,experiments,cutoff,*,appendix=True):
        rows=[]
        for eid in experiments:
            for target,metric,panel in [("chunk","ndcg",f"Chunk\nnDCG@{cutoff}"),
                                        ("page","counterpart_family_ndcg",f"Page-family\nnDCG@{cutoff}")]:
                row=data.metric(source,eid,target,metric,cutoff)
                rows.append(styled(source,eid,row,panel=panel,scale=1))
        add(topic,name,"Chunk and page-family ranking quality",
            rows,appendix=appendix)

    # Model calibration: a separate human-labelled sample, not retrieval evaluation.
    import yaml
    jd = data.catalog["sources"]["judge"]
    decision = yaml.safe_load(data.verify(jd["selection_path"], jd["selection_sha256"]).read_text())
    if decision["judge_analysis_id"] != data.manifest("judge")["judge_analysis_id"]:
        raise ValueError("Judge selection belongs to a different calibration analysis")
    judge_order = ["gemma4_31b", "qwen3_8_27b", "gemma4_26b_a4b", "qwen3_6_27b",
                   "qwen3_6_35b_a3b", "llama3_3_70b", "gpt_oss_20b", "gemma4_e4b", "llama3_1_8b"]
    rows = []
    for jid in judge_order:
        for metric, panel in [("quadratic_weighted_kappa", "Weighted κ"), ("exact_agreement", "Exact\nagreement"),
                              ("within_one_grade_agreement", "Within-one\nagreement")]:
            r = exactly_one(data.table("judge", "bootstrap_intervals.csv"), judge_id=jid, metric=metric)
            rows.append(dict(source="judge", judge_id=jid, label=r["judge"], panel=panel,
                             estimate=number(r["estimate"]), ci_lower=number(r["ci_lower"]),
                             ci_upper=number(r["ci_upper"]), scale=1,
                             color="#0072B2" if jid == "gemma4_31b" else "#88939D", marker="o"))
            rows[-1]["selected"] = jid == decision["selected_default_judge_id"]
    add("01_evaluation", "judge_calibration", "Judge calibration against human grades",
        rows)
    rows=[]
    for jid in judge_order:
        for metric,panel,direction in [("binary_f1","Binary F1\n(grade ≥2)","max"),
                ("grade_3_f1","Grade-3 F1","max"),("mean_absolute_grade_delta","Mean absolute\ngrade error","min")]:
            r=exactly_one(data.table("judge","bootstrap_intervals.csv"),judge_id=jid,metric=metric)
            rows.append(dict(source="judge",judge_id=jid,label=r["judge"],panel=panel,metric=metric,
                estimate=float(r["estimate"]),ci_lower=float(r["ci_lower"]),ci_upper=float(r["ci_upper"]),
                scale=1,direction=direction,bounded=metric!="mean_absolute_grade_delta",color="#0072B2" if jid=="gemma4_31b" else "#88939D",
                marker="o",selected=jid==decision["selected_default_judge_id"]))
    add("01_evaluation","judge_thresholds_and_error","Judge relevance decisions and grade error",
        rows,appendix=True)
    rows=[]
    for jid in judge_order:
        for r in select(data.table("judge","per_grade_metrics.csv"),judge_id=jid):
            rows.append(dict(source="judge",label=r["judge"],panel="Grade "+r["grade"],estimate=float(r["f1"]),
                n=int(r["reference_support"]),precision=float(r["precision"]),recall=float(r["recall"]),
                scale=1,selected=jid==decision["selected_default_judge_id"]))
    add("01_evaluation","judge_per_grade_f1","Judge performance by relevance grade",
        rows,kind="heatmap",appendix=True)
    rows=[]
    for r in select(data.table("judge","paired_model_differences.csv"),left_judge_id="gemma4_31b",right_judge_id="qwen3_8_27b"):
        reverse=r["direction"]=="lower_is_better"
        label={"quadratic_weighted_kappa":"Weighted κ","exact_agreement":"Exact agreement",
            "within_one_grade_agreement":"Within-one agreement","binary_f1":"Binary F1",
            "grade_3_f1":"Grade-3 F1","mean_absolute_grade_delta":"Mean absolute error (reversed)"}[r["metric"]]
        rows.append(dict(source="judge",label=label,panel="Difference favouring Gemma 4 31B",scale=1,
            estimate=float(r["estimate_left_minus_right"])*(-1 if reverse else 1),
            ci_lower=float(r["ci_upper"])*-1 if reverse else float(r["ci_lower"]),
            ci_upper=float(r["ci_lower"])*-1 if reverse else float(r["ci_upper"]),color="#0072B2",marker="o"))
    add("01_evaluation","judge_top_two_paired","Paired comparison of the leading judges",
        rows,kind="differences",appendix=True)
    rows=[]
    for jid in judge_order:
        rr=select(data.table("judge","confusion_matrices.csv"),judge_id=jid)
        groups=[("≥2 lower",lambda d:d<=-2,"#46799A"),("1 lower",lambda d:d==-1,"#98B9CD"),
                ("Exact",lambda d:d==0,"#B2B9BE"),("1 higher",lambda d:d==1,"#DAAF78"),
                ("≥2 higher",lambda d:d>=2,"#B36F50")]
        for label,predicate,color in groups:
            count=sum(int(r["count"]) for r in rr if predicate(int(r["judge_grade"])-int(r["human_grade"])))
            rows.append(dict(source="judge",label=rr[0]["judge"],panel=label,estimate=count/432,
                count=count,n=432,color=color,scale=100,selected=jid==decision["selected_default_judge_id"]))
    add("01_evaluation","judge_error_direction","Direction and magnitude of judge errors",
        rows,kind="stacked",appendix=True)
    rows = select(data.table("judge", "confusion_matrices.csv"), judge_id="gemma4_31b")
    add("01_evaluation", "judge_confusion", "Selected judge: grade confusion",
        rows, kind="confusion", appendix=True)

    # A compact, source-backed dataset table, without exposing question text.
    base = data.json("datasets", "base_question_set_manifest.json")
    audit = data.json("datasets", "verification.json")
    data.json("datasets", "manifest.json")
    dataset_rows = []
    for split, n in [("Development", 1000), ("Validation", 500), ("Test", 1000)]:
        dataset_rows.append({"Dataset": "Synthetic", "Split": split, "Queries": n, "Role": "Retrieval evaluation"})
    for split, n in audit["new_split_counts"].items():
        dataset_rows.append({"Dataset": "Logs, corrected export", "Split": split.capitalize(),
                             "Queries": n, "Role": "Development / final evaluation"})
    dataset_rows += [dict(Dataset="Manual-page benchmark", Split="Diagnostic", Queries=2924, Role="Known-target navigation"),
                     dict(Dataset="Priority cohort", Split="Development", Queries=250, Role="115 synthetic + 135 logs")]
    # Verify synthetic counts against the recorded manifest instead of silently trusting prose.
    if [base["artifact_counts"][key] for key in ["questions_dev", "questions_validation", "questions_test"]] != [1000, 500, 1000]:
        raise ValueError("Synthetic dataset counts changed")
    add("01_evaluation", "evaluation_datasets", "Evaluation datasets",
        dataset_rows, kind="table")

    quality("chunking", "03_chunking", "chunking_quality", "Chunking strategies and chunk length", 5, appendix=True)
    quality("chunking", "03_chunking", "chunking_algorithms", "Chunking algorithms at 800/100 characters", 5,
            experiments=["chunker_fixed_hybrid", "chunker_recursive_hybrid", "chunker_section_strict_hybrid", "chunker_section_preferred_800_hybrid"],
            ranking_panel=True)
    quality("chunking", "03_chunking", "chunking_length", "Section-preferred chunk length", 5,
            experiments=["chunker_section_preferred_600_hybrid", "chunker_section_preferred_800_hybrid", "chunker_section_preferred_1000_hybrid"],
            ranking_panel=True)
    latency("chunking", "03_chunking", "chunking_latency")
    chunk_rows=[]
    for eid in data.catalog["variants"]["chunking"]:
        r=exactly_one(data.table("chunking","efficiency_summary.csv"),experiment_id=eid)
        chunk_rows.append({"Variant":data.style("chunking",eid)["label"],
                          "Indexed chunks":f"{int(r['indexed_document_count']):,}",
                          "Index (MiB)":f"{float(r['index_byte_size'])/1024**2:.1f}",
                          "Build (s)":f"{float(r['index_build_duration_ms'])/1000:.1f}",
                          "p95 (ms)":f"{float(r['retrieval_p95_ms']):.1f}"})
    add("03_chunking","chunking_resources","Chunking: index and runtime costs",
        chunk_rows,kind="table",appendix=True,column_format="@{}Xrrrr@{}")
    for source, name, title in [("language", "language_policies", "Language policy: original queries"),
                                ("language_translated", "translated_queries", "Language policy: translated queries")]:
        quality(source, "04_language", name, title, 5, appendix=source.endswith("translated"))
    rows=[]
    for eid in data.catalog["variants"]["language"]:
        for metric,panel in [("ndcg","Raw page\nnDCG@5"),("counterpart_family_ndcg","Page-family\nnDCG@5")]:
            rows.append(styled("language",eid,data.metric("language",eid,"page",metric,5),panel=panel,scale=1))
        r=exactly_one(data.table("language","counterpart_family_summary.csv"),experiment_id=eid,
                      judgment_view=data.catalog["sources"]["language"]["view"],stream="items",cutoff=5)
        rows.append(styled("language",eid,dict(source="language",experiment_id=eid,
            estimate=float(r["counterpart_duplicate_query_rate"]),ci_lower=None,ci_upper=None,
            panel="Queries with\nduplicates (%)",scale=100,direction="min",n=int(r["query_count"]))))
    add("04_language","counterpart_deduplication","Language counterparts: quality and duplication",
        rows)
    rows = []
    policies = [("query_complete", "Query complete"), ("cross_lingual", "Cross-lingual"),
                ("cross_lingual_dedup", "Cross-lingual + dedup.")]
    signals = [("correct", "Correct"), ("always_de", "Always DE"), ("always_en", "Always EN"), ("swapped", "Swapped")]
    for prefix, label in policies:
        for x, (signal, category) in enumerate(signals):
            eid = prefix + "_" + signal
            r = data.metric("language_signal", eid, "page", "counterpart_family_ndcg", 5)
            rows.append(styled("language_signal", eid, r, label=label, x=x, category=category,
                               panel="Page-family nDCG@5", scale=1))
    add("04_language", "language_signal_robustness", "Robustness to the supplied language signal",
        rows, kind="categorical")
    quality("rrf_dedup20", "04_language", "rrf_dedup_top20", "Reciprocal-rank fusion constant", 20, appendix=True)
    ranking_views("rrf_dedup20","04_language","rrf_selection_metric",
                  list(data.catalog["variants"]["rrf_dedup20"]),20,appendix=False)

    quality("architecture", "05_retrieval", "sparse_dense_hybrid", "Baseline, sparse, dense and hybrid retrieval", 15,
            experiments=["legacy_refreshed_nlu", "modern_sparse", "modern_dense", "modern_hybrid_k10"])
    quality("architecture", "02_baseline", "baseline_routing_ablation", "Baseline routing diagnostics", 15,
            experiments=["legacy_refreshed_nlu", "legacy_no_routing", "legacy_gold_intent_diagnostic"], appendix=True)
    latency("architecture", "05_retrieval", "architecture_latency", experiments=["legacy_refreshed_nlu", "modern_sparse", "modern_dense", "modern_hybrid_k10"])

    plain = [eid for eid in data.catalog["variants"]["rerankers"] if not (eid.endswith("_metadata") or eid.endswith("_tu"))]
    quality("rerankers", "06_reranking", "reranker_models", "Reranker model comparison", 20,
            experiments=plain, target="chunk", appendix=True)
    latency("rerankers", "06_reranking", "reranker_latency", experiments=plain)
    quality("rerankers", "06_reranking", "reranker_all_variants", "Rerankers, metadata and domain instructions", 20,
            target="chunk", appendix=True)
    paired("rerankers", "06_reranking", "metadata_effect", target="chunk", metric="ndcg",
           comparisons=["bge_metadata", "qwen06b_metadata", "qwen4b_metadata", "qwen8b_metadata", "nemotron_metadata", "zerank_metadata"])
    metadata_asset=assets[-1]
    for row in metadata_asset.rows:
        row["panel"]="Δ chunk\nnDCG@20"
    extra=[]
    for row in metadata_asset.rows:
        r=exactly_one(data.table("rerankers","paired_differences.csv"),left_experiment_id=row["left"],
            right_experiment_id=row["right"],judgment_view=data.catalog["sources"]["rerankers"]["view"],
            stream="items",target_level="page",metric="counterpart_family_ndcg",cutoff=20)
        extra.append({**row,"panel":"Δ page-family\nnDCG@20","estimate":float(r["mean_delta"]),
                      "ci_lower":float(r["ci_lower"]),"ci_upper":float(r["ci_upper"]),"n":int(r["paired_query_count"])})
    metadata_asset.rows.extend(extra)
    metadata_asset.inputs.update(data.asset_used);data.asset_used.clear()
    retained=["reranker_none", "reranker_nemotron_1b", "reranker_nemotron_1b_metadata",
              "reranker_zerank_2", "reranker_zerank_2_metadata"]
    tradeoff("rerankers","06_reranking","reranker_selection_tradeoff",
             list(data.catalog["variants"]["rerankers"]),focus_experiments=retained)
    tradeoff("rerankers","06_reranking","reranker_all_tradeoffs",list(data.catalog["variants"]["rerankers"]),appendix=True)
    ranking_views("rerankers","06_reranking","retained_reranker_quality",retained,20)
    rows=[]
    for eid in retained:
        for k in [1,3,5,10,20]:
            row=data.metric("rerankers",eid,"chunk","ndcg",k)
            rows.append(styled("rerankers",eid,row,x=k,panel="Chunk nDCG@k",scale=1))
    add("06_reranking","reranker_early_ranks","Selected rerankers across ranking cutoffs",
        rows,kind="curves",xlabel="Retrieval cutoff k",appendix=True)
    quality("validation", "06_reranking", "validation_transfer", "Transfer to synthetic validation", 20,
            ranking_panel=True)

    rows = []
    for size in ("06b", "4b", "8b"):
        left = f"reranker_qwen3_{size}"
        for target, metric, panel in [("chunk", "ndcg", "Δ chunk\nnDCG@20"),
                ("page", "counterpart_family_ndcg", "Δ page-family\nnDCG@20")]:
            r = exactly_one(data.table("rerankers", "paired_differences.csv"),
                comparison_id="qwen_instructions", left_experiment_id=left,
                right_experiment_id=left+"_tu", judgment_view=data.catalog["sources"]["rerankers"]["view"],
                stream="items", target_level=target, cutoff=20, metric=metric)
            rows.append(styled("rerankers", left, dict(source="rerankers", panel=panel,
                estimate=float(r["mean_delta"]), ci_lower=float(r["ci_lower"]),
                ci_upper=float(r["ci_upper"]), n=int(r["paired_query_count"]), scale=1)))
    add("06_reranking", "qwen_domain_instruction", "Effect of the TU-specific reranking instruction",
        rows, kind="differences", appendix=True)

    # Known-target navigation reports descriptive estimates for the full cohorts.
    from thesis_rag.publication.navigation import descendant_specs
    assets.extend(Asset(**spec) for spec in descendant_specs(data))

    def known_navigation(source, ids, comparison, topic, name, title, appendix=False, extra=()):
        rows = []
        for sid, eid in [(source, eid) for eid in ids] + list(extra):
            for k in [1, 5, 20]:
                r = exactly_one(data.table(sid, "metrics.csv"), comparison_id=comparison,
                                experiment_id=eid, scope="all", target_view="exact_page", cutoff=k, metric="hit")
                if int(r["queries"]) != data.catalog["sources"][sid]["n"]:
                    raise ValueError("Navigation cohort is not the full registered cohort")
                rows.append(styled(sid, eid, dict(source=sid, experiment_id=eid, scope="all",
                    target_view="exact_page", comparison=comparison, panel=f"Target Hit@{k} (%)",
                    estimate=number(r["estimate"]), ci_lower=None, ci_upper=None,
                    n=int(r["queries"]), scale=100)))
        add(topic, name, title, rows, appendix=appendix)
    for suffix, comparison, appendix in [("full", "full_corpus", False), ("restricted", "restricted_corpus", True)]:
        known_navigation("navigation_manual", ["backup_"+suffix, "modern_bm25_"+suffix,
                         "modern_dense_"+suffix, "modern_"+suffix], comparison, "07_navigation",
                         "manual_pages_"+suffix, "Backup NLU and retrieval for manual-page navigation", appendix,
                         extra=[("navigation_tables", "page_metadata_"+suffix), ("navigation_tables", "page_summary_"+suffix)])
    known_navigation("navigation_priority", ["main_nlu", "backup_full", "modern_full", "modern_nemotron_full",
                     "page_metadata_full", "page_summary_full", "group_metadata_full", "group_summary_full"],
                     "full_corpus", "07_navigation", "priority_known_targets", "Known-target navigation on the Priority cohort")

    for source, name, title, ids in [
        ("navigation_manual", "backup_nlu_hit_curves", "Backup NLU versus sparse, dense and hybrid retrieval",
         ["backup", "modern_dense", "modern_bm25", "modern"]),
        ("navigation_tables", "manual_page_table_hit_curves", "Page-table retrieval on the curated navigation benchmark",
         ["backup", "modern", "page_metadata", "page_summary"]),
        ("navigation_priority", "main_nlu_hit_curves", "Main NLU versus global retrieval and source tables",
         ["main_nlu", "backup", "modern_nemotron", "page_summary", "group_summary"]),
        ("navigation_priority", "page_table_hit_curves", "Page and group tables with and without summaries",
         ["main_nlu", "modern_nemotron", "page_metadata", "page_summary", "group_metadata", "group_summary"]),
    ]:
        rows = []
        panel_notes = {}
        if name == "backup_nlu_hit_curves":
            source_definition = data.catalog["sources"][source]
            suite = json.loads(data.verify(source_definition["suite_manifest"], source_definition["suite_sha256"]).read_text())
            question_set = suite["question_set"]
            benchmark = json.loads(data.verify(question_set["manifest_path"], question_set["manifest_hash"]).read_text())
            for suffix, panel in [("full", "Full corpus"), ("restricted", "Restricted catalogue")]:
                run_entry = exactly_one(suite["experiment_runs"], experiment_id="modern_"+suffix)
                run = json.loads(data.verify(run_entry["run_manifest_path"], run_entry["run_manifest_hash"]).read_text())
                components = run["identity"]["runtime_identity"]["metadata"]["components"]
                language_view = components["dense_global"]["metadata"]["query_language_view"]
                page_count = sum(language_view["native_page_counts"].values())
                intent_count = benchmark["counts"]["backup_"+suffix+"_labels"]
                panel_notes[panel] = f"{intent_count:,} intents / {page_count:,} pages"
        for suffix, comparison, panel in [("full", "full_corpus", "Full corpus"),
                ("restricted", "restricted_corpus", "Restricted catalogue")]:
            for stem in ids:
                eid = "main_nlu" if stem == "main_nlu" else stem+"_"+suffix
                rr = select(data.table(source, "metrics.csv"), comparison_id=comparison, experiment_id=eid,
                            scope="all", target_view="exact_page", metric="hit")
                if len(rr) != 20:
                    raise ValueError(f"Expected complete navigation curve: {source}/{comparison}/{eid}")
                for r in rr:
                    if int(r["queries"]) != data.catalog["sources"][source]["n"]:
                        raise ValueError(f"Navigation curves use different cohort sizes: {source}/{eid}")
                    rows.append(styled(source, eid, dict(source=source, experiment_id=eid,
                        panel=panel, x=int(r["cutoff"]), estimate=float(r["estimate"]),
                        ci_lower=None, ci_upper=None, scale=100,
                        n=int(r["queries"]), scope="all", target_view="exact_page"),
                        label="Main NLU: direct" if eid == "main_nlu" else
                              "Page metadata" if source == "navigation_tables" and stem == "page_metadata" else
                              "Page summaries" if source == "navigation_tables" and stem == "page_summary" else
                              data.style(source, eid)["label"].replace(" (catalogue)", "")))
        add("07_navigation", name, title,
            rows, kind="curves", appendix=name in {"page_table_hit_curves", "manual_page_table_hit_curves"}, xlabel="Page-ranking depth k",
            ylabel="Target-page Hit@k", full_percent=True, shared_y=True,
            marker_cutoffs=[1,3,5,10,20], line_cutoffs=[1,3,5,10,20], show_intervals=False,
            xticks=[1,5,10,20], table_cutoffs=[1,3,5,10,20],
            panel_n=None if name == "backup_nlu_hit_curves" else data.catalog['sources'][source]['n'],
            panel_notes=panel_notes)
    quality("priority_navigation", "07_navigation", "priority_graded_pages", "Graded page relevance on the Priority cohort", 20,
            experiments=["main_nlu", "backup_full", "modern_nemotron_full", "page_metadata_full",
                         "page_summary_full", "group_metadata_full", "group_summary_full"])
    quality("priority_navigation", "07_navigation", "navigation_all_variants", "Navigation scope and reranker ablations", 20, appendix=True)

    # Restore the graded-page comparison, keeping it separate from reference-page hits.
    scope_report = data.manifest("page_scope")
    data.json("page_scope", "priority_catalogue.json")
    scope_rows = data.table("page_scope", "summary.csv")
    overview = scope_report["priority_config"]["overview_experiment_ids"]
    scope_labels = {
        "modern_nemotron_full": "Global Nemotron",
        "modern_nemotron_restricted": "Priority roots + Nemotron",
        "modern_nemotron_priority_depth5": "Priority retrieval + Nemotron",
        "page_summary_restricted": "Page table: summaries",
        "group_metadata_restricted": "Group table: metadata",
        "group_summary_restricted": "Group table: summaries",
    }
    rows = []
    for eid in sorted(overview, key=lambda eid: eid != "main_nlu"):
        for depth in ([0, 5] if eid == "main_nlu" else [0]):
            values = select(scope_rows, experiment_id=eid, depth=depth, cohort="all",
                            judgment_view="gemma4_31b:judged_only", threshold=3, policy="cached_only")
            if len(values) != 5 or {int(r["root_cutoff"]) for r in values} != {1, 3, 5, 10, 20}:
                raise ValueError(f"Incomplete graded navigation curve: {eid}/depth {depth}")
            style = data.style("priority_navigation", eid)
            if eid == "main_nlu":
                style = {**style, "label": "Main NLU: direct" if depth == 0 else "Main NLU + descendants ≤5",
                         "linestyle": "-" if depth == 0 else "--"}
            else:
                style = {**style, "label": scope_labels[eid]}
                if eid == "modern_nemotron_restricted":
                    style["linestyle"] = "--"
                elif eid == "modern_nemotron_priority_depth5":
                    style["linestyle"] = "-"
            for r in sorted(values, key=lambda r: int(r["root_cutoff"])):
                n = int(r["queries"])
                if n != data.catalog["sources"]["page_scope"]["n"]:
                    raise ValueError("Graded navigation must retain the full query cohort")
                for field, count, panel in [
                    ("known_source_hit_rate", "known_hits", "All sources"),
                    ("priority_known_source_hit_rate", "priority_known_hits", "Priority sources"),
                ]:
                    if abs(float(r[field]) - int(r[count]) / n) > 1e-12:
                        raise ValueError("Graded navigation rate does not match its all-query denominator")
                    rows.append({**r, **style, "source": "page_scope", "x": int(r["root_cutoff"]),
                                 "panel": panel, "estimate": float(r[field]), "hits": int(r[count]),
                                 "n": n, "scale": 100, "ci_lower": None, "ci_upper": None})
    scope_options = dict(kind="curves", xlabel="Initial page-ranking cutoff k",
        ylabel="Known hit rate (grade 3)", full_percent=True, shared_y=True,
        xticks=[1, 3, 5, 10, 20], legend_columns=2,
        panel_note="Only the dashed NLU curve expands roots")
    focus = {"main_nlu", "modern_nemotron_full", "modern_nemotron_priority_depth5"}
    scope_inputs = dict(data.asset_used)
    add("07_navigation", "main_nlu_descendant_hits", "Page retrieval and Main-NLU descendant expansion",
        [r for r in rows if r["experiment_id"] in focus], **scope_options)
    data.asset_used.update(scope_inputs)
    add("07_navigation", "main_nlu_descendant_hits_all", "Page retrieval and Main-NLU expansion: all variants",
        [{**r, "color": "#8B9097"} if r["experiment_id"] not in focus else dict(r) for r in rows],
        appendix=True, **scope_options)

    data.asset_used.update(scope_inputs)
    size_rows = [{**r, "estimate": float(r["mean_scope_pages"]), "scale": 1,
                  "panel": "Mean scope size (pages)", "bounded": False}
                 for r in rows if r["panel"] == "All sources"]
    add("07_navigation", "main_nlu_scope_size", "Pages considered before and after NLU expansion",
        size_rows, kind="scope_size", xlabel="Initial page-ranking cutoff k")

    # The scope diagnostic is descriptive: unjudged descendants are not negative evidence.
    rows = []
    for eid in ["main_nlu", "backup_full", "modern_nemotron_full", "page_summary_full"]:
        for depth in range(6):
            matches = select(data.table("page_scope", "summary.csv"), cohort="all", depth=depth, experiment_id=eid,
                             judgment_view="gemma4_31b:judged_only", threshold=2, root_cutoff=5, policy="cached_only")
            r = exactly_one(matches)
            if int(r["queries"]) != 250:
                raise ValueError("The expansion diagnostic requires the complete 250-query cohort")
            for key, panel, scale in [("known_source_hit_rate", "Known relevant hit (%)", 100), ("mean_scope_pages", "Mean scope size (pages)", 1)]:
                rows.append(styled("priority_navigation", eid, dict(r, source="page_scope", experiment_id=eid,
                    x=depth, panel=panel, estimate=float(r[key]), ci_lower=None, ci_upper=None, scale=scale,
                    queries=int(r["queries"]), unresolved_queries=int(r["unresolved_queries"]),
                    mean_unknown_scope_pages=float(r["mean_unknown_scope_pages"])), linestyle="-"))
    add("07_navigation", "descendant_scope", "Descendant expansion: coverage and scope size",
        rows, kind="curves", xlabel="Descendant depth (0 = roots only)", appendix=True,
        panel_titles={"Known relevant hit (%)": "Evidence coverage", "Mean scope size (pages)": "Search-space size"},
        panel_ylabels={"Known relevant hit (%)": "Known hit rate (grade ≥2)", "Mean scope size (pages)": "Pages per query (mean)"},
        panel_note="Top 5 roots; all methods expanded",
        percent_panels=["Known relevant hit (%)"],
        panel_limits={"Mean scope size (pages)": [0, 250]},
        panel_ticks={"Mean scope size (pages)": [0, 50, 100, 150, 200, 250]})

    quality("parent_routing", "08_routing", "page_group_downstream", "Downstream retrieval after page and group routing", 20,
            ranking_panel=True)
    latency("parent_routing", "08_routing", "routing_latency")
    quality("main_nlu", "08_routing", "main_nlu_routing", "Hard and additive Main-NLU routing", 20)
    quality("priority_downstream", "08_routing", "priority_downstream", "Downstream retrieval on the Priority cohort", 20)
    quality("priority_union", "09_priority", "candidate_union", "Priority candidate union and fusion", 20)
    quality("log_development", "08_routing", "original_log_development", "Retrieval and routing on user-log development queries", 20,
            ranking_panel=True)

    for source, name, title, appendix in [
        ("priority_reservation", "reserved_slots", "Reserved evidence and fallback policies", True),
        ("priority_synthetic", "synthetic_generalization", "Priority transfer to synthetic development", True),
        ("priority_logs", "log_generalization", "Priority transfer to log development", False),
    ]:
        quality(source, "09_priority", name, title, 20, target="chunk", appendix=appendix)
        rows = []
        for eid in ["global_hybrid", "fixed_children_depth5_3", "nlu_priority_nemotron", "nlu_priority_zerank"]:
            for criterion, panel in [("general", "All sources"), ("priority", "Priority sources")]:
                for k in [1, 3, 5, 10, 20]:
                    r = exactly_one(data.table(source, "priority__relevance.csv"), scope="all", denominator="all_queries",
                                    experiment_id=eid, stream="items", target="chunk", cutoff=k, threshold=3, criterion=criterion)
                    rows.append(styled(source, eid, dict(source=source, experiment_id=eid, panel=panel,
                        estimate=float(r["hit_rate"]), ci_lower=None, ci_upper=None, x=k,
                        hits=int(r["hits"]), n=int(r["queries"]), scale=100, criterion=criterion)))
        add("09_priority", name+"_priority_hits", "Early hits on general and Priority evidence",
            rows, kind="curves", xlabel="Retrieval cutoff k", ylabel="Chunk Hit@k (grade 3, %)", appendix=appendix)

    # Corrected replay changes ordering only; its top-20 membership must stay invariant.
    rows = []
    for source, panel in [("priority_synthetic", "Synthetic development"), ("priority_logs", "Log development")]:
        totals = set()
        for ordering, label, color, marker in [("protected_prefix", "All reserved first", "#666666", "s"),
            ("nlu_prefix_3", "NLU prefix (3)", "#009E73", "^"), ("reranker", "Reranker order", "#0072B2", "o")]:
            for k in [1, 3, 5, 10, 20]:
                r = exactly_one(data.table(source, "priority_diagnostic__ordering.csv"), source_experiment="nlu_priority_nemotron",
                                ordering=ordering, branch="all", grade=3, criterion="general", cutoff=k)
                if k == 20:
                    totals.add(r["hits"])
                rows.append(dict(source=source, label=label, ordering=ordering, panel=panel, color=color, marker=marker,
                                 x=k, estimate=float(r["hit_rate"]), ci_lower=None, ci_upper=None, n=int(r["queries"]), scale=100))
        if len(totals) != 1:
            raise ValueError("Ordering replay changes top-20 membership")
    add("09_priority", "reserved_evidence_ordering", "Ordering of reserved evidence",
        rows, kind="curves", xlabel="Retrieval cutoff k", ylabel="Chunk Hit@k (grade 3, %)")
    supplement = data.table("priority_supplement", "paired_hit_differences.csv")
    rows = []
    for r in supplement:
        if r["cutoff"] not in ["1", "5", "20"] or r["grade"] != "3":
            continue
        source = {"priority":"priority_reservation", "synthetic":"priority_synthetic", "logs":"priority_logs"}.get(r["cohort"])
        if not source:
            raise ValueError(f"Unknown supplement cohort: {r['cohort']}")
        rows.append({"Cohort": r["cohort"], "Comparison": data.style(source,r["right"])["label"] + " − " + data.style(source,r["left"])["label"],
                     "Source": r["criterion"], "k":r["cutoff"], "Δ (pp)":f"{100*float(r['delta']):+.1f}",
                     "95% CI (pp)":f"[{100*float(r['ci_low']):+.1f}, {100*float(r['ci_high']):+.1f}]"})
    # Split long supplement tables into readable cohort-specific tables.
    for cohort in dict.fromkeys(r["Cohort"] for r in rows):
        data.table("priority_supplement", "paired_hit_differences.csv")
        add("09_priority", "paired_priority_hits_"+cohort, "Paired Priority hit-rate differences",
            [{k:v for k,v in r.items() if k != "Cohort"} for r in rows if r["Cohort"] == cohort],
            kind="table", appendix=True, column_format="@{}Xlrrr@{}")

    # Final test results use the frozen synthetic and corrected 1,000-query log cohorts.
    for source, topic, title in [("final_synthetic", "10_final_synthetic", "Final comparison on synthetic test queries"),
                                 ("final_logs_corrected_1000", "11_final_user_logs", "Final comparison on corrected log test queries (n = 1,000)")]:
        quality(source, topic, "system_quality", title, 20)
        latency(source, topic, "system_latency", appendix=False)
        paired(source, topic, "paired_system_differences")
        context_rows = []
        for eid in data.catalog["variants"][source]:
            for metric, panel in [("hit_at_grade_2", "Context Hit@20\n(grade ≥2, %)"), ("strict_hit_at_grade_3", "Context Hit@20\n(grade 3, %)")]:
                row = data.metric(source, eid, "context_native_mixed", metric, 20,
                                  stream="context_items", all_queries=True)
                context_rows.append(styled(source, eid, row, panel=panel, scale=100))
        add(topic, "constructed_context_evidence", "Evidence in the constructed context",
            context_rows, appendix=True)
        rows = []
        for eid in data.catalog["variants"][source]:
            for k in [1, 3, 5, 10, 20]:
                for metric, panel in [("hit_at_grade_2", "Page Hit@k (grade ≥2, %)"), ("strict_hit_at_grade_3", "Page Hit@k (grade 3, %)")]:
                    row = data.metric(source, eid, "page", metric, k, all_queries=True)
                    rows.append(styled(source, eid, row, panel=panel, x=k, scale=100))
        add(topic, "early_evidence_hits", "Evidence coverage by retrieval cutoff",
            rows, kind="curves", xlabel="Retrieval cutoff k", appendix=True)
        rows = data.table(source, "direct_context__legacy_refreshed_nlu__grades.csv")
        total = sum(int(r["Queries"]) for r in rows)
        normalized = [dict(Grade=r["Grade"], Queries=int(r["Queries"]), Share=100*float(r["Share"])) for r in rows]
        add(topic, "baseline_direct_page_grades", "Relevance of Baseline direct-page evidence",
            normalized, kind="grades", appendix=True)

    # Mixed context contains complete native units and judged rendered fragments.
    rows = []
    for eid in ["global_hybrid", "hybrid_page_summary", "hybrid_group_summary", "hierarchical"]:
        for budget in [1000, 2000, 4000, 8000, 16000]:
            row = data.metric("parent_routing", eid, "context_native_mixed",
                              "strict_hit_at_grade_3", 20, stream=f"context_items@chars={budget}")
            rows.append(styled("parent_routing", eid, row, x=budget/1000, panel="Context Hit (grade 3, %)", scale=100))
    add("12_context", "context_character_budget", "Evidence under a context character budget",
        rows, kind="curves", xlabel="Context budget (1,000 characters)", appendix=True,
        small_multiples=True, reference="Global Nemotron", xscale="log2")

    # Frozen settings are read from the actual resolved configurations, not inferred from names.
    import yaml
    source = "final_synthetic"
    data.manifest(source)
    suite = json.loads((data.root/data.catalog["sources"][source]["suite_manifest"]).read_text())
    configuration_rows = []
    for e in suite["experiments"]:
        path = data.verify(e["resolved_system_config_path"], e["resolved_system_config_file_hash"])
        settings = yaml.safe_load(path.read_text())["settings"]
        pipeline = settings.get("pipeline")
        if pipeline:
            routing = pipeline["page_routing"]
            configuration_rows.append({"Variant":data.style(source,e["experiment_id"])["label"],
                "Candidates":str(pipeline["candidate_k"]),
                "Priority union":str(routing.get("candidate_union",{}).get("priority_k",0)),
                "Priority slots":str(pipeline.get("protected_priority_slots",0)),
                "NLU slots":str(pipeline.get("main_nlu_priority_slots",0)),
                "Scope depth":str(routing.get("descendant_depth","--")),
                "Ordering":pipeline.get("priority_ordering","reranker").replace("_"," ")})
    add("10_final_synthetic", "frozen_retrieval_settings", "Frozen modern retrieval settings",
        configuration_rows,kind="table")

    # Diagnostics answer different questions from the aggregate quality overview.
    rows=[]
    for language in ["de","en"]:
        for prefix,label in policies:
            for x,(signal,category) in enumerate(signals):
                eid=prefix+"_"+signal
                r=exactly_one(data.table("language_signal","slice_metrics.csv"),experiment_id=eid,
                    judgment_view=data.catalog["sources"]["language_signal"]["view"],stream="items",
                    target_level="page",metric="counterpart_family_ndcg",cutoff=5,slice_dimension="language",slice_value=language)
                rows.append(styled("language_signal",eid,dict(source="language_signal",estimate=float(r["estimate"]),
                    n=int(r["query_count"]),ci_lower=None,ci_upper=None,scale=1),label=label,x=x,category=category,
                    panel="German queries" if language=="de" else "English queries"))
    add("04_language","language_signal_by_language","Language-signal sensitivity by query language",
        rows,kind="categorical",ylabel="Page-family nDCG@5",appendix=True)
    add("04_language", "translated_source_availability", "Translated-query page-family nDCG@5 by source availability",
        translated_source_availability_rows(data), kind="table", appendix=True,
        column_format="@{}Xrrr@{}")
    paired("rrf_dedup20","04_language","rrf_paired_selection",target="chunk",metric="ndcg",appendix=True)
    tradeoff("architecture","05_retrieval","architecture_tradeoff",
        ["legacy_refreshed_nlu","modern_sparse","modern_dense","modern_hybrid_k10"],
        target="page",metric="counterpart_family_ndcg",cutoff=15,appendix=True)
    rows=[]
    for eid in ["legacy_refreshed_nlu","modern_sparse","modern_dense","modern_hybrid_k10"]:
        for k in [1,3,5,10,15]:
            for metric,panel in [("hit_at_grade_2","Chunk Hit@k (grade ≥2, %)"),("strict_hit_at_grade_3","Chunk Hit@k (grade 3, %)")]:
                rows.append(styled("architecture",eid,data.metric("architecture",eid,"chunk",metric,k,all_queries=True),x=k,panel=panel,scale=100))
    add("05_retrieval","architecture_early_evidence","Evidence depth: Baseline, sparse, dense and hybrid",
        rows,kind="curves",xlabel="Retrieval cutoff k",appendix=True)
    paired("parent_routing","08_routing","routing_paired_effects",comparisons=["routing_methods","page_summary_effect","group_summary_effect"],appendix=True)
    for source,name,bindings in [
        ("navigation_manual","manual_navigation_ranking",[("navigation_manual",e) for e in ["backup_full","modern_bm25_full","modern_dense_full","modern_full"]]+[("navigation_tables",e) for e in ["page_metadata_full","page_summary_full"]]),
        ("navigation_priority","priority_navigation_ranking",[("navigation_priority",e) for e in ["main_nlu","backup_full","modern_nemotron_full","page_metadata_full","page_summary_full","group_metadata_full","group_summary_full"]])]:
        rows=[]
        for sid,eid in bindings:
            r=exactly_one(data.table(sid,"metrics.csv"),comparison_id="full_corpus",experiment_id=eid,scope="all",target_view="exact_page",cutoff=20,metric="mrr")
            rows.append(styled(sid,eid,dict(source=sid,estimate=float(r["estimate"]),ci_lower=None,ci_upper=None,
                scale=1,panel="Known-target MRR@20",n=int(r["queries"]))))
        add("07_navigation",name,"Rank of the first known target",
            rows,appendix=True)
    # Hit gains/losses use the existing query-paired diagnostic, including its intervals.
    for source,topic in [("main_nlu","08_routing"),("log_development","08_routing"),
                         ("final_synthetic","10_final_synthetic"),("final_logs_corrected_1000","11_final_user_logs")]:
        rows=[];counts=[]
        for r in select(data.table(source,"nlu_diagnostic__paired_summary.csv"),target="chunk",k=20,cohort="all_evaluated"):
            eid=r["experiment_id"]
            if eid not in data.catalog["variants"][source]:continue
            row=styled(source,eid,dict(source=source,panel=f"Δ chunk Hit@20\n(grade {'≥2' if r['grade']=='2' else '3'}, pp)",
                estimate=float(r["hit_delta"]),ci_lower=float(r["hit_ci_lower"]),ci_upper=float(r["hit_ci_upper"]),
                scale=100,n=int(r["n"]),gained=int(r["hit_gains"]),lost=int(r["hit_losses"]),
                judgment_view=r["judgment_view"],baseline=r["baseline"]))
            rows.append(row)
            counts.append({"Variant":row["label"],"Grade":r["grade"],"n":r["n"],
                "Gained":r["hit_gains"],"Lost":r["hit_losses"],"Δ (pp)":f"{100*float(r['hit_delta']):+.1f}"})
        add(topic,source+"_paired_hit_changes","Paired hit gains and losses against global retrieval",
            rows,kind="differences",appendix=True)
        data.table(source,"nlu_diagnostic__paired_summary.csv")
        add(topic,source+"_hit_change_counts","Queries gaining and losing an evidence hit",
            counts,kind="table",appendix=True,column_format="@{}Xrrrrr@{}")
    # Priority quality must be assessed on relevant Priority evidence, not just page membership.
    for source,topic,ids in [
        ("priority_downstream","09_priority",["global_hybrid","fixed_topic_branch","fixed_topic_only","priority_page_summary"]),
        ("priority_union","09_priority",list(data.catalog["variants"]["priority_union"])),
        ("priority_navigation","07_navigation",["main_nlu","backup_full","modern_nemotron_full","page_summary_full","group_summary_full"]),
        ("final_synthetic","10_final_synthetic",list(data.catalog["variants"]["final_synthetic"])),
        ("final_logs_corrected_1000","11_final_user_logs",list(data.catalog["variants"]["final_logs_corrected_1000"]))]:
        rows=[];target="page" if source=="priority_navigation" else "chunk"
        for eid in ids:
            for criterion,panel in [("general","All sources"),("priority","Priority sources")]:
                for k in [1,3,5,10,20]:
                    r=exactly_one(data.table(source,"priority__relevance.csv"),scope="all",denominator="all_queries",
                        experiment_id=eid,stream="items",target=target,cutoff=k,threshold=3,criterion=criterion)
                    rows.append(styled(source,eid,dict(source=source,estimate=float(r["hit_rate"]),ci_lower=None,ci_upper=None,
                        hits=int(r["hits"]),n=int(r["queries"]),pool_positive_queries=int(r["pool_positive_queries"]),
                        x=k,panel=panel,scale=100)))
        add(topic,source+"_priority_evidence","General evidence and relevant Priority evidence",
            rows,kind="curves",
            xlabel="Retrieval cutoff k",ylabel=f"{target.capitalize()} Hit@k (grade 3, %)",appendix=True)
    rows=[];branch_rows=[]
    for source,cohort in [("priority_synthetic","Synthetic dev."),("priority_logs","Log dev.")]:
        for branch,label in [("main_nlu","Main NLU"),("priority_fallback","Priority fallback")]:
            r=exactly_one(data.table(source,"priority_diagnostic__protected_evidence.csv"),experiment="nlu_priority_nemotron",branch=branch)
            n=int(r["branch_queries"]);grade2=int(r["queries_with_grade2"]);grade3=int(r["queries_with_grade3"])
            if not 0<=grade3<=grade2<=n:raise ValueError("Invalid protected evidence counts")
            for panel,count,color in [("No grade ≥2",n-grade2,"#B2B9BE"),("Grade 2 only",grade2-grade3,"#56B4E9"),("Any grade 3",grade3,"#719D84")]:
                rows.append(dict(source=source,label=cohort+": "+label,panel=panel,estimate=count/n,count=count,n=n,scale=100,color=color))
            for grade in [2,3]:
                q=exactly_one(data.table(source,"priority_diagnostic__branch_effects.csv"),experiment="nlu_priority_nemotron",branch=branch,grade=grade,criterion="priority",cutoff=20)
                branch_rows.append({"Cohort / branch":cohort+": "+label,"Grade":grade,"n":q["queries"],"Global hits":q["baseline_hits"],"Hits":q["candidate_hits"],"Gained":q["gained"],"Lost":q["lost"]})
    add("09_priority","protected_evidence_quality","Relevance of reserved evidence by routing branch",
        rows,kind="stacked",xlabel="Questions in routing branch (%)",appendix=True)
    for s in ["priority_synthetic","priority_logs"]:data.table(s,"priority_diagnostic__branch_effects.csv")
    add("09_priority","priority_branch_gains_losses","Priority hit changes by routing branch",
        branch_rows,kind="table",appendix=True,column_format="@{}Xrrrrrr@{}")
    direct_rows=[];rescue_rows=[]
    for source,cohort in [("log_development","Log dev."),("final_synthetic","Synthetic test"),("final_logs_corrected_1000","Log test (1,000)")]:
        for r in data.table(source,"direct_context__legacy_refreshed_nlu__incremental.csv"):
            direct_rows.append({"Cohort":cohort,"Grade":r["Threshold"],"n":r["Queries"],"Without":r["Without page"],"With":r["With page"],"Gained":r["Gained"],"Lost":r["Lost"]})
        for r in select(data.table(source,"nlu_diagnostic__direct_cohorts.csv"),cohort="legacy_rescued",k=20):
            if r["experiment_id"] not in ["global_hybrid","nlu_priority_nemotron","nlu_priority_zerank"]:continue
            rescue_rows.append({"Cohort":cohort,"Variant":data.style(source,r["experiment_id"])["label"],"Grade":r["grade"],"n":r["n"],"Chunk hits":int(float(r['chunk_grade_'+r['grade']])),"Page hits":int(float(r['page_grade_'+r['grade']]))})
    add("02_baseline","direct_page_incremental","Incremental contribution of direct-page injection",
        direct_rows,kind="table",appendix=True,column_format="@{}Xrrrrrr@{}")
    for s in ["log_development","final_synthetic","final_logs_corrected_1000"]:data.table(s,"nlu_diagnostic__direct_cohorts.csv")
    add("02_baseline","direct_page_rescue_coverage","Modern retrieval on Baseline direct-page rescue cases",
        rescue_rows,kind="table",appendix=True,column_format="@{}XXrrrr@{}")
    for source,topic in [("final_synthetic","10_final_synthetic"),("final_logs_corrected_1000","11_final_user_logs")]:
        tradeoff(source,topic,"system_tradeoff",list(data.catalog["variants"][source]),target="page",metric="counterpart_family_ndcg",appendix=True)
        rows=[]
        for eid in data.catalog["variants"][source]:
            for k in [1,3,5,10,20]:
                rows.append(styled(source,eid,data.metric(source,eid,"page","counterpart_family_ndcg",k),x=k,panel="Page-family nDCG@k",scale=1))
        add(topic,"system_ranking_by_cutoff","Page-family ranking at early cutoffs",
            rows,kind="curves",xlabel="Retrieval cutoff k",appendix=True)
        rows=[]
        for eid in data.catalog["variants"][source]:
            for language in ["de","en"]:
                r=exactly_one(data.table(source,"slice_metrics.csv"),experiment_id=eid,
                    judgment_view=data.catalog["sources"][source]["view"],stream="items",target_level="page",
                    metric="counterpart_family_ndcg",cutoff=20,slice_dimension="language",slice_value=language)
                rows.append(styled(source,eid,dict(source=source,panel=language.upper(),estimate=float(r["estimate"]),
                    ci_lower=None,ci_upper=None,scale=1,n=int(r["query_count"]))))
        add(topic,"language_subgroups","Page-family ranking by query language",
            rows,kind="heatmap",appendix=True)
        rows=[]
        for eid in data.catalog["variants"][source]:
            if eid=="legacy_refreshed_nlu":continue
            candidates=select(data.table(source,"paired_differences.csv"),left_experiment_id="legacy_refreshed_nlu",
                right_experiment_id=eid,judgment_view=data.catalog["sources"][source]["view"],stream="items",
                target_level="page",metric="counterpart_family_ndcg",cutoff=20)
            r=exactly_one(candidates);n=int(r["paired_query_count"])
            for key,panel,color in [("right_wins","Higher nDCG","#78A9C5"),("ties","Equal nDCG","#D0D5D9"),("right_losses","Lower nDCG","#D9A578")]:
                rows.append(styled(source,eid,dict(source=source,panel=panel,estimate=int(r[key])/n,
                    count=int(r[key]),n=n,scale=100),color=color))
        add(topic,"ranking_wins_ties_losses","Query-level ranking changes against the Baseline",
            rows,kind="stacked",xlabel="Paired questions (%)",appendix=True)

    add("10_final_synthetic", "judgment_view_sensitivity", "Final synthetic results with and without source anchors",
        final_view_sensitivity_rows(data), kind="table", appendix=True,
        column_format="@{}Xrrrrrr@{}",
        column_groups=[("nDCG@20", 2), ("Hit@20 (%)", 2), ("Strict Hit@20 (%)", 2)])
    rows=[]
    for jid in judge_order:
        r=exactly_one(data.table("judge","judge_summary.csv"),judge_id=jid)
        rows.append({"Judge":r["judge"],"Quantization":r["quantization_level"],"Context":f"{int(r['context_length']):,}",
                     "Temperature":f"{float(r['temperature']):.1f}","Token limit":f"{int(r['max_tokens']):,}"})
    add("01_evaluation","judge_runtime_settings","Judge calibration runtime settings",
        rows,kind="table",appendix=True,column_format="@{}Xlrrr@{}")
    for source,topic,cutoff in [("language","04_language",5),("final_synthetic","10_final_synthetic",20),
                                ("final_logs_corrected_1000","11_final_user_logs",20)]:
        rows=[]
        for eid in data.catalog["variants"][source]:
            r=exactly_one(data.table(source,"language_alignment.csv"),experiment_id=eid,
                judgment_view=data.catalog["sources"][source]["view"],stream="items",target_level="chunk",cutoff=cutoff)
            n=float(r["retrieved_target_observations"])
            same=float(r["same_language_target_count"]);cross=float(r["cross_language_target_count"])
            same_rel=float(r["same_language_relevant_count_at_grade_2"]);cross_rel=float(r["cross_language_relevant_count_at_grade_2"])
            for panel,count,color,hatch in [("Same language, grade ≥2",same_rel,"#0072B2",""),
                ("Same language, grade <2",same-same_rel,"#A5D0E4","//"),
                ("Other language, grade ≥2",cross_rel,"#E69F00",""),
                ("Other language, grade <2",cross-cross_rel,"#F3D6A1","//"),
                ("Unknown language",float(r["unknown_language_target_count"]),"#BBBBBB","")]:
                rows.append(styled(source,eid,dict(source=source,panel=panel,estimate=count/n,count=int(count),n=int(n),scale=100),color=color,hatch=hatch))
        add(topic,"language_and_relevance","Language and relevance of retrieved chunks",
            rows,kind="stacked",xlabel="Retrieved chunks",count_axis=True,appendix=True)
    rows=[]
    for eid in data.catalog["variants"]["final_synthetic"]:
        original=exactly_one(data.table("final_synthetic","efficiency_summary.csv"),experiment_id=eid)["experiment"]
        r=exactly_one(data.table("final_synthetic","comparison_summary__reference_page_coverage.csv"),Experiment=original,k=20)
        for key,color in [("Exact Page","#78A9C5"),("Counterpart only","#D9A578"),("Not found","#D0D5D9"),("No reference","#EEEEEE")]:
            rows.append(styled("final_synthetic",eid,dict(source="final_synthetic",panel=key,estimate=int(r[key])/int(r["Queries"]),
                count=int(r[key]),n=int(r["Queries"]),scale=100),color=color))
    add("10_final_synthetic","reference_page_recovery","Recovery of the synthetic source page",
        rows,kind="stacked",xlabel="Synthetic test questions (%)",appendix=True)
    tradeoff("main_nlu","08_routing","main_nlu_tradeoff",list(data.catalog["variants"]["main_nlu"]),
        target="page",metric="counterpart_family_ndcg",appendix=True)
    tradeoff("log_development","08_routing","log_development_tradeoff",
        ["legacy_refreshed_nlu","hybrid_without_reranker","global_hybrid","hybrid_zerank"],
        target="page",metric="counterpart_family_ndcg",appendix=True)
    for source,topic,cutoff in [("chunking","03_chunking",5),("language","04_language",5),
                              ("architecture","05_retrieval",15),("validation","06_reranking",20)]:
        rows=[];seen=set()
        for r in select(data.table(source,"paired_differences.csv"),judgment_view=data.catalog["sources"][source]["view"],
                        stream="items",target_level="page",metric="counterpart_family_ndcg",cutoff=cutoff):
            pair=(r["left_experiment_id"],r["right_experiment_id"])
            if pair in seen:continue
            seen.add(pair)
            rows.append({"Comparison":data.style(source,pair[1])["label"]+" − "+data.style(source,pair[0])["label"],
                "Δ nDCG":f"{float(r['mean_delta']):+.3f}","95% CI":f"[{float(r['ci_lower']):+.3f}, {float(r['ci_upper']):+.3f}]",
                "Win/tie/loss":r["right_wins"]+"/"+r["ties"]+"/"+r["right_losses"]})
        add(topic,source+"_paired_comparisons","Paired development comparisons",
            rows,kind="table",appendix=True,column_format="@{}Xrrr@{}")

    # Reports determine each cohort's common cutoff; unavailable values remain gaps.
    def baseline_stages(stage_source, suffix, cohort):
        stage_labels={'stage:vector_search':'Vector search', 'stage:intent_filter':'Intent filter',
                      'stage:post_filters@current':'Keyword + length filter', 'items':'Final retrieval'}
        def stage_table(name):
            rows=select(data.table(stage_source, 'stage_summary__'+name+'.csv'),
                        experiment_id='legacy_refreshed_nlu', judgment_view=data.catalog['sources'][stage_source]['view'])
            if {r['target_space_id'] for r in rows}!={'chunk:legacy_recursive_character_800_100_min200'}:
                raise ValueError('Unexpected Baseline stage target space')
            return rows

        n=data.catalog['sources'][stage_source]['n']
        cutoffs={int(r['cutoff']) for r in stage_table('quality')}
        if len(cutoffs)!=1:
            raise ValueError('Expected one report-supported common stage cutoff')
        common_cutoff=cutoffs.pop()

        rows=[]
        for stream,label in [('stage:vector_search','Vector search'),('stage:intent_filter','Intent filter'),
                ('stage:post_filters@current','Keyword + length filter'),('items','Final retrieval'),
                ('context_items','Chunks in context')]:
            for metric,panel,all_queries in [('hit_at_grade_2',f'Chunk Hit@{common_cutoff}\n(grade ≥2, %)',True),
                    ('strict_hit_at_grade_3',f'Chunk Hit@{common_cutoff}\n(grade 3, %)',True),('ndcg',f'Chunk\nnDCG@{common_cutoff}',False)]:
                if stream=='context_items':
                    r=data.metric(stage_source,'legacy_refreshed_nlu','chunk',metric,common_cutoff,
                                  stream=stream,all_queries=all_queries)
                else:
                    r=exactly_one(stage_table('hit_rates'), stream=stream, metric=metric, cutoff=common_cutoff,
                                  population='all_queries') if all_queries else exactly_one(stage_table('quality'),stream=stream,cutoff=common_cutoff)
                    if all_queries and (r['status']!='ok' or int(r['complete_queries'])!=n or int(r['denominator'])!=n):
                        raise ValueError('Incomplete Baseline stage evaluation')
                    r={**r, 'source':stage_source, 'estimate':float(r['estimate'] if all_queries else r['ndcg']),
                       'ci_lower':float(r['ci_lower']), 'ci_upper':float(r['ci_upper']),
                       'n':int(r['denominator'] if all_queries else r['eligible_queries']), 'statistics':'stored_analysis'}
                rows.append(styled(stage_source,'legacy_refreshed_nlu',r,label=label,panel=panel,scale=100 if all_queries else 1))
        add('02_baseline','baseline_stage_quality'+suffix,'Evidence across the Baseline retrieval stages — '+cohort,
            rows)

        rows=[]
        cutoff_styles=[('#0072B2','o'),('#E69F00','s'),('#009E73','D'),('#D55E00','^'),('#CC79A7','v')]
        stage_ticks=['Vector\nsearch','Intent\nfilter','Keyword +\nlength','Final\nretrieval']
        missing=[]
        for cutoff,(color,marker) in zip([1,3,5,10,15],cutoff_styles,strict=True):
            for r in select(stage_table('hit_rates'),cutoff=cutoff,population='all_queries'):
                if int(r['denominator'])!=n or int(r['query_count'])!=n:
                    raise ValueError('Unexpected Baseline checkpoint cohort')
                valid=r['status']=='ok'
                if valid and int(r['complete_queries'])!=n:
                    raise ValueError('Incomplete Baseline checkpoint evaluation')
                if not valid:
                    if any(r[key] for key in ('estimate','ci_lower','ci_upper')):
                        raise ValueError('Invalid checkpoint has numeric estimates')
                    missing.append(f"k = {cutoff} at {stage_labels[r['stream']]}")
                position=int(r['stage_position'])
                if stage_labels[r['stream']]!=list(stage_labels.values())[position-1]:
                    raise ValueError('Unexpected Baseline checkpoint order')
                rows.append({**r,'source':stage_source,'label':str(cutoff),
                    'panel':'Grade ≥2' if r['metric']=='hit_at_grade_2' else 'Grade 3',
                    'x':position,'category':stage_ticks[position-1],
                    'estimate':float(r['estimate']) if valid else None,'ci_lower':float(r['ci_lower']) if valid else None,
                    'ci_upper':float(r['ci_upper']) if valid else None,'scale':100,'color':color,'marker':marker,'linestyle':'-'})
        add('02_baseline','baseline_stage_hit_curves'+suffix,'Evidence across the Baseline retrieval checkpoints — '+cohort,
            rows,kind='curves',appendix=True,
            shared_y=True,full_percent=True,panel_n=n,xlabel='Retrieval checkpoint',ylabel='Chunk Hit@k',
            interval_style='band',tick_labels=stage_ticks,legend_columns=5,legend_title='Rank cutoff k',
            table_row_header='Cutoff / grade')

        rows=[]
        for r in stage_table('hit_changes'):
            if int(r['paired_queries'])!=n or int(r['cutoff'])!=common_cutoff:
                raise ValueError('Unexpected Baseline paired cohort or cutoff')
            counts={key:int(r[key]) for key in ('gained','lost','retained_hit','retained_miss')}
            if sum(counts.values())!=n or abs((counts['gained']-counts['lost'])/n-float(r['mean_delta']))>1e-10:
                raise ValueError('Inconsistent Baseline transition counts')
            rows.append({**r,**counts,'source':stage_source,
                'label':stage_labels[r['left_stream']]+' → '+stage_labels[r['right_stream']],
                'panel':'Grade ≥2' if r['metric']=='hit_at_grade_2' else 'Grade 3',
                'estimate':float(r['mean_delta']),'ci_lower':float(r['ci_lower']),'ci_upper':float(r['ci_upper'])})
        add('02_baseline','baseline_stage_hit_changes'+suffix,'Queries gaining or losing evidence between stages — '+cohort,
            rows,kind='stage_changes',appendix=True,cohort=cohort,query_count=n,cutoff=common_cutoff)

        raw=data.table(stage_source,'direct_context__legacy_refreshed_nlu__grades.csv')
        total=sum(int(r['Queries']) for r in raw)
        add('02_baseline','baseline_direct_page_grades'+suffix,'Relevance of directly injected Baseline pages — '+cohort,
            [dict(Grade=r['Grade'],Queries=int(r['Queries']),Share=100*float(r['Share'])) for r in raw],kind='grades',appendix=True,compact=True)
        # Verify that both tables describe the same injection cases.
        total=sum(int(r['Queries']) for r in data.table(stage_source,'direct_context__legacy_refreshed_nlu__grades.csv'))
        raw=data.table(stage_source,'direct_context__legacy_refreshed_nlu__incremental.csv')
        if any(int(r['Queries'])!=total for r in raw):
            raise ValueError('Inconsistent direct-page injection cohort')
        for r in raw:
            before,after,gained,lost=(int(r[k]) for k in ('Without page','With page','Gained','Lost'))
            if not (0<=before<=total and 0<=after<=total and 0<=gained<=total-before
                    and 0<=lost<=before and after-before==gained-lost):
                raise ValueError('Inconsistent direct-page contribution counts')
        add('02_baseline','baseline_direct_page_contribution'+suffix,'Additional evidence from direct-page injection — '+cohort,
            [{'Threshold':'Grade ≥2' if r['Threshold']=='2' else 'Grade 3',
              'Without\npage':r['Without page'],'With\npage':r['With page'],'Gained':r['Gained'],'Lost':r['Lost']}
             for r in raw],kind='direct_contribution',query_count=total,appendix=True)

        matrix=data.table(stage_source,'direct_context__legacy_refreshed_nlu__cooccurrence.csv')
        counts={r['Direct page']:[int(r['Retrieved: no hit']),int(r['Retrieved: hit'])] for r in matrix}
        useful=exactly_one(raw,Threshold='2')
        if (set(counts)!={'Grade 0–1','Grade 2–3'}
                or any(v<0 for row in counts.values() for v in row)
                or sum(sum(row) for row in counts.values())!=total
                or sum(row[1] for row in counts.values())!=int(useful['Without page'])
                or counts['Grade 2–3'][0]!=int(useful['Gained'])):
            raise ValueError('Direct-page relevance matrix disagrees with contribution counts')
        # Include the incremental table in the matrix's provenance and cross-check.
        data.table(stage_source,'direct_context__legacy_refreshed_nlu__incremental.csv')
        add('02_baseline','baseline_direct_page_cooccurrence'+suffix,'Direct-page and retrieved-chunk relevance — '+cohort,
            matrix,kind='direct_cooccurrence',query_count=total,appendix=True)

    baseline_stages('baseline_synthetic', '', 'synthetic development')
    baseline_stages('baseline_logs', '_logs', 'corrected user-log development')

    add('02_baseline','baseline_synthetic_vs_logs','Baseline retrieval on synthetic and user-log queries',
        baseline_cohort_rows(data),kind='curves',shared_y=True,full_percent=True,panel_n=250,
        xlabel='Retrieval cutoff k',ylabel='Chunk Hit@k',interval_style='band',legend_columns=2,
        endpoint_labels=True,table_row_header='Dataset / grade')

    # Remove duplicate presentations, retaining the source CSVs through the retained comparisons.
    redundant = {
        "chunking_quality", "architecture_latency", "reranker_latency", "priority_known_targets",
        "rrf_dedup_top20", "routing_latency", "synthetic_generalization", "log_generalization",
    }
    assets = [a for a in assets if a.name not in redundant]
    for asset in assets:
        # Full variant grids remain available as tables; paired plots show the mechanism more clearly.
        if asset.name in {"reranker_all_variants", "navigation_all_variants", "manual_pages_full",
                          "manual_pages_restricted", "retained_reranker_quality", "candidate_union",
                          "reserved_slots", "priority_graded_pages", "priority_downstream",
                          "priority_navigation_ranking", "manual_navigation_ranking"}:
            asset.options["table_only"] = True
        if asset.name == 'priority_union_priority_evidence':
            asset.title='Priority candidate counts and evidence retrieval'
        if asset.name == "language_and_relevance":
            for r in asset.rows:
                r["panel"] = r["panel"].replace("grade <2", "other")

    from thesis_rag.publication.comparisons import comparison_assets
    assets.extend(comparison_assets(assets))

    # Main-text candidates form a compact narrative; the complete library remains searchable.
    main_names={"corpus_overview","corpus_priority_overlay","corpus_languages","corpus_page_lengths","corpus_inventory",
        "judge_calibration","synthetic_question_composition","chunking_algorithms","chunking_length","language_signal_robustness",
        "counterpart_deduplication","sparse_dense_hybrid","reranker_selection_tradeoff","metadata_effect",
        "validation_transfer","backup_nlu_hit_curves","main_nlu_hit_curves","main_nlu_descendant_hits","main_nlu_scope_size","page_group_downstream",
        "main_nlu_routing","original_log_development","priority_development_comparison","reserved_evidence_ordering",
        "final_quality_comparison","system_latency","evaluation_datasets","frozen_retrieval_settings","baseline_stage_hit_curves","baseline_stage_hit_curves_logs",
        "baseline_synthetic_vs_logs","baseline_direct_page_contribution","baseline_direct_page_contribution_logs",
        "backup_nlu_target_depth_full", "chunking_algorithms_ranking", "chunking_length_ranking",
        "validation_transfer_ranking", "page_group_downstream_ranking", "original_log_development_ranking",
        "final_priority_comparison"}
    for asset in assets:
        asset.placement="main candidate" if asset.name in main_names else "appendix candidate"
    return assets


def configure_style():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis-publication-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#9B9B9B", "axes.linewidth": .5, "axes.axisbelow": True,
        "grid.color": "#E2E5E8", "grid.linewidth": .5, "lines.linewidth": 1.1,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "chatu-thesis-publication-v1", "savefig.facecolor": "white",
    })
    return plt


def metric_axis(axis, low, high, *, scale=1, bins=4, bounded=True):
    """Enclose the full data range with labelled ticks, including near 0/100%."""
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    margin = max((high-low)*.08, scale*.018)
    locator = MaxNLocator(bins, steps=[1, 2, 2.5, 5, 10], min_n_ticks=1)
    ticks = list(locator.tick_values(low-margin, high+margin))
    if bounded:
        ticks = sorted({max(0, min(scale, float(t))) for t in ticks})
    if len(ticks) < 2:
        ticks = [0, scale]
    pad = (ticks[-1]-ticks[0])*.04
    axis.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:.3g}'))
    axis.axes.set_xlim(ticks[0]-pad, ticks[-1]+pad) if axis.axis_name == "x" else axis.axes.set_ylim(ticks[0]-pad, ticks[-1]+pad)


def draw(asset: Asset, plt, *, width_mm=150):
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D

    if asset.kind in ("cohort_quality", "cohort_priority"):
        from thesis_rag.publication.comparisons import draw_comparison
        return draw_comparison(asset, plt, width_mm)

    if asset.kind == "question_composition":
        from thesis_rag.publication.evaluation import draw_question_composition
        return draw_question_composition(asset, plt, width_mm)

    if asset.kind.startswith("corpus_"):
        from thesis_rag.publication.corpus import draw_corpus
        return draw_corpus(asset, plt, width_mm)

    width = width_mm/25.4
    panels = list(dict.fromkeys(r.get("panel", "") for r in asset.rows))
    if asset.kind in ("points", "latency", "differences"):
        labels = list(dict.fromkeys(r["label"] for r in asset.rows))
        compact = width_mm < 100 and len(panels) == 1
        wrap = 31 if len(panels) == 1 and not compact else 25
        wrapped = [textwrap.fill(label, wrap, break_long_words=False) for label in labels]
        line_counts = [label.count("\n")+1 for label in wrapped]
        y = np.cumsum([0]+[max(.9, n*.55) for n in line_counts[:-1]])
        height_mm = max(52, 28 + sum(max(6.5, n*4.1) for n in line_counts))
        fig, axes = plt.subplots(1, len(panels), figsize=(width, height_mm/25.4), squeeze=False, sharey=True)
        left = .42 if len(panels) == 1 else .36
        fig.subplots_adjust(left=left, right=.95 if compact else .975, top=.97,
                            bottom=max(.13, 15/height_mm), wspace=.30)
        for i, (ax, panel) in enumerate(zip(axes[0], panels)):
            values = select(asset.rows, panel=panel)
            for row in values:
                yy = y[labels.index(row["label"])]; factor=row.get("scale",1)
                point = row["estimate"]*factor
                low, high = row.get("ci_lower"), row.get("ci_upper")
                if low is not None:
                    ax.plot([low*factor, high*factor], [yy,yy], color=row["color"], lw=1.15, solid_capstyle="round")
                ax.plot(point, yy, marker=row["marker"], color=row["color"], linestyle="none", markersize=4.3, markeredgewidth=.7)
                if asset.kind == "latency":
                    ax.plot([point,row["p95"]], [yy,yy], color=row["color"], lw=.8)
                    ax.plot(row["p95"], yy, marker=row["marker"], color=row["color"], markerfacecolor="white", markersize=4.3)
            ax.set_yticks(y, wrapped if i == 0 else [])
            if i:
                ax.tick_params(labelleft=False)
            ax.tick_params(axis="y", length=0, pad=7)
            ax.set_xlabel(panel if "\n" in panel else panel.replace(" (%)", "\n(%)") if len(panels)>1 else panel, labelpad=6)
            ax.xaxis.set_major_locator(MaxNLocator(3 if len(panels)>1 else 5))
            ax.grid(axis="x")
            if asset.kind == "differences":
                ax.axvline(0, color="#707070", lw=.8, linestyle="--")
                scale = values[0].get("scale",1)
                metric_axis(ax.xaxis, min(0,min(r['ci_lower'] for r in values)*scale),
                            max(0,max(r['ci_upper'] for r in values)*scale), scale=scale,
                            bins=3 if len(panels)>1 else 5, bounded=False)
            elif asset.kind == "latency":
                ax.set_xlim(left=0); ax.margins(x=.1)
            else:
                scale = values[0].get("scale",1)
                lo = min(r["ci_lower"] if r.get("ci_lower") is not None else r["estimate"] for r in values)*scale
                hi = max(r["ci_upper"] if r.get("ci_upper") is not None else r["estimate"] for r in values)*scale
                metric_axis(ax.xaxis, lo, hi, scale=scale, bins=2 if compact or len(panels)>1 else 5,
                            bounded=values[0].get("bounded",True))
            ax.set_ylim(y[-1]+.65,-.65)
        # Shared axes must retain the labels assigned to the first panel.
        axes[0,0].set_yticks(y, wrapped)
        selected={r["label"] for r in asset.rows if r.get("selected")}
        for tick,label in zip(axes[0,0].get_yticklabels(),labels):
            tick.set_fontweight("bold" if label in selected else "normal")
        if compact:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            label_left = min(tick.get_window_extent(renderer).x0 for tick in axes[0,0].get_yticklabels())
            fig.subplots_adjust(left=left + (2/25.4 - label_left/fig.dpi)/width)
    elif asset.kind == "curves" and asset.options.get("small_multiples"):
        from matplotlib.ticker import PercentFormatter, ScalarFormatter
        labels = list(dict.fromkeys(r["label"] for r in asset.rows))
        fig, axes = plt.subplots(2, 2, figsize=(width,112/25.4), sharex=True, sharey=True)
        fig.subplots_adjust(left=.13,right=.98,bottom=.16,top=.93,hspace=.46,wspace=.22)
        low=min(r['ci_lower']*100 for r in asset.rows); high=max(r['ci_upper']*100 for r in asset.rows)
        reference=sorted(select(asset.rows,label=asset.options["reference"]),key=lambda r:r["x"])
        for ax,label in zip(axes.flat, labels, strict=True):
            rr=sorted(select(asset.rows,label=label),key=lambda r:r["x"]);style=rr[0]
            x=[r['x'] for r in rr];y=[r['estimate']*100 for r in rr]
            ax.fill_between(x,[r['ci_lower']*100 for r in rr],[r['ci_upper']*100 for r in rr],color=style['color'],alpha=.12,lw=0)
            ax.plot(x,y,color=style['color'],marker=style['marker'],markersize=3.8)
            if label!=asset.options['reference']:
                ax.plot(x,[r['estimate']*100 for r in reference],color=reference[0]['color'],ls='--',lw=.9)
            ax.set_xscale('log',base=2);ax.set_xticks(x);ax.xaxis.set_major_formatter(ScalarFormatter())
            metric_axis(ax.yaxis,low,high,scale=100,bins=4)
            ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.grid(axis='y')
            ax.set_title(textwrap.fill(label,25),fontsize=8.5,fontweight='bold' if style.get('selected') else 'normal',pad=8)
        fig.supxlabel(asset.options['xlabel'],fontsize=9,y=.03)
        fig.supylabel('Context Hit (grade 3)',fontsize=9,x=.005)
    elif asset.kind in ("categorical", "curves"):
        identities={r["label"]:(r["color"],r["marker"],r.get("linestyle","-")) for r in asset.rows}
        if len(set(identities.values()))!=len(identities):
            raise ValueError(f"Indistinguishable series styles: {asset.name}")
        labels=list(dict.fromkeys(r["label"] for r in asset.rows))
        height_mm=82 if len(panels)==1 else 88
        legend_columns=asset.options.get('legend_columns',min(2,len(labels)))
        legend_rows=(len(labels)+legend_columns-1)//legend_columns
        height_mm += max(0,legend_rows-2)*5
        fig, axes = plt.subplots(1,len(panels),figsize=(width,height_mm/25.4),squeeze=False,
                                 sharey=asset.options.get('shared_y',False))
        bottom_mm=17+legend_rows*5+(8 if asset.options.get('tick_labels') and not asset.options.get('compact_ticks') else 0)+(4 if asset.options.get('legend_title') else 0)
        has_panel_note = bool(asset.options.get('panel_note') or asset.options.get('panel_notes'))
        top=.82 if has_panel_note and asset.options.get('panel_n') else .87 if asset.options.get('panel_n') or has_panel_note else .94
        fig.subplots_adjust(left=.12,right=.975,bottom=bottom_mm/height_mm,top=top,
                            wspace=.13 if asset.options.get('shared_y') else .4)
        handles=[]
        for i,(ax,panel) in enumerate(zip(axes[0],panels)):
            for j,label in enumerate(labels):
                rr=sorted(select(asset.rows,panel=panel,label=label),key=lambda r:r["x"])
                if not rr: continue
                style=rr[0]; factor=style.get("scale",1)
                x=np.array([r["x"] for r in rr],dtype=float)
                y=np.array([r["estimate"]*factor if r['estimate'] is not None else np.nan for r in rr])
                lo=[r.get("ci_lower") for r in rr]; hi=[r.get("ci_upper") for r in rr]
                if asset.kind=="categorical":
                    x+=(j-(len(labels)-1)/2)*.18
                    errors=np.maximum(0,np.array([y-np.array(lo)*factor,np.array(hi)*factor-y])) if all(v is not None for v in lo+hi) else None
                    ax.errorbar(x,y,yerr=errors,fmt=style["marker"],color=style["color"],
                                linestyle="none",capsize=2,markersize=4.4,elinewidth=1)
                    categories=[r["category"].replace("Always ","Always\n") if len(panels)>1 else r["category"] for r in rr]
                    ax.set_xticks(range(len(rr)),categories);ax.set_xlim(-.45,len(rr)-.55)
                else:
                    line_indices=[z for z,r in enumerate(rr) if r['x'] in asset.options.get('line_cutoffs',[v['x'] for v in rr])]
                    mark_indices=[z for z,idx in enumerate(line_indices) if rr[idx]['x'] in asset.options.get('marker_cutoffs',[v['x'] for v in rr])]
                    ax.plot(x[line_indices],y[line_indices],color=style["color"],marker=style["marker"],linestyle=style.get("linestyle","-"),markersize=3.7,markevery=mark_indices,
                            markerfacecolor=style.get('markerfacecolor',style['color']))
                    if asset.options.get('show_intervals', True) and any(a is not None and b is not None for a,b in zip(lo,hi)):
                        idx=[z for z,r in enumerate(rr) if lo[z] is not None and hi[z] is not None
                             and r['x'] in asset.options.get('interval_cutoffs',[v['x'] for v in rr])]
                        # Draw interval endpoints directly: a cluster percentile interval need not contain its estimate.
                        if asset.options.get('interval_style')=='band':
                            # Keep absent checkpoints in the path so bands cannot bridge missing evidence.
                            lower=np.array([lo[z] if z in idx else np.nan for z in range(len(rr))],dtype=float)
                            upper=np.array([hi[z] if z in idx else np.nan for z in range(len(rr))],dtype=float)
                            ax.fill_between(x,lower*factor,upper*factor,
                                            color=style['color'],alpha=.07,linewidth=0)
                        else:
                            ax.vlines(x[idx],np.array(lo)[idx].astype(float)*factor,np.array(hi)[idx].astype(float)*factor,color=style['color'],lw=.75,alpha=.8)
                    ticks=asset.options.get('xticks',sorted(set(r["x"] for r in select(asset.rows,panel=panel))))
                    ax.set_xticks(ticks,asset.options['tick_labels']) if asset.options.get('tick_labels') else ax.set_xticks(ticks)
                    if asset.options.get('endpoint_labels'):
                        last=next(r for r in reversed(rr) if r['estimate'] is not None)
                        ax.annotate(f"{last['estimate']*factor:.1f}%",(last['x'],last['estimate']*factor),
                                    xytext=(5,0),textcoords='offset points',va='center',fontsize=8,color=style['color'])
                        span=max(ticks)-min(ticks)
                        ax.set_xlim(min(ticks)-span*.05,max(ticks)+span*.35)
                if i==0:
                    handles.append(Line2D([],[],color=style["color"],marker=style["marker"],
                        linestyle="none" if asset.kind=="categorical" else style.get("linestyle","-"),markersize=4,label=label,
                        markerfacecolor=style.get('markerfacecolor',style['color'])))
            ax.grid(axis="y");ax.yaxis.set_major_locator(MaxNLocator(4))
            if asset.options.get('full_percent') or panel in asset.options.get('percent_panels',[]):
                from matplotlib.ticker import PercentFormatter
                ax.set_ylim(-1,102);ax.set_yticks(range(0,101,20));ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
            else:
                rr=[r for r in select(asset.rows,panel=panel) if r['estimate'] is not None];scale=rr[0].get('scale',1)
                metric_axis(ax.yaxis,min(r.get('ci_lower') if r.get('ci_lower') is not None else r['estimate'] for r in rr)*scale,
                    max(r.get('ci_upper') if r.get('ci_upper') is not None else r['estimate'] for r in rr)*scale,scale=scale,bins=4,
                    bounded=all(r.get('bounded',True) for r in rr) and panel!='Mean scope size (pages)')
                if panel=='Mean scope size (pages)':
                    ax.set_ylim(bottom=0)
            if limits := asset.options.get('panel_limits',{}).get(panel):
                ax.set_ylim(*limits)
            if ticks := asset.options.get('panel_ticks',{}).get(panel):
                ax.set_yticks(ticks)
            ax.set_xlabel('' if asset.options.get('shared_xlabel') else asset.options.get("xlabel", "Language signal"))
            ylabel=asset.options.get('panel_ylabels',{}).get(panel, panel if len(panels)==1 else asset.options.get("ylabel",panel))
            ax.set_ylabel('' if i and asset.options.get('shared_y') else ylabel)
            if len(panels)>1:
                title=asset.options.get('panel_titles',{}).get(panel,panel if asset.options.get("ylabel") else '')
                panel_label=chr(97+i)+")"+(" "+title if title else '')
                if asset.options.get('panel_n'):
                    panel_label+=f"\n(n = {asset.options['panel_n']:,}"
                    if asset.options.get('panel_clusters'):panel_label+=f"; {asset.options['panel_clusters']} target groups"
                    panel_label+=')'
                if panel_note := asset.options.get('panel_notes',{}).get(panel, asset.options.get('panel_note')):
                    panel_label+='\n'+panel_note
                ax.text(0,1.04,panel_label,transform=ax.transAxes,fontsize=8)
        if asset.options.get('shared_xlabel'):
            fig.supxlabel(asset.options['xlabel'], fontsize=9, y=.19)
        legend=fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),frameon=False,
                   ncol=legend_columns,handlelength=1.5,columnspacing=1.1,title=asset.options.get('legend_title'),title_fontsize=8)
        selected={r["label"] for r in asset.rows if r.get("selected")}
        for label in legend.get_texts():
            if label.get_text() in selected:
                label.set_fontweight("bold")
    elif asset.kind=="scope_size":
        # Merge only genuinely identical direct-page counts; retain every variant in the table.
        direct = select(asset.rows, depth=0)
        expanded = select(asset.rows, depth=5)
        if not direct or not expanded or len({r["label"] for r in expanded}) != 1:
            raise ValueError("Scope-size plot requires direct methods and one expanded NLU series")
        cutoffs = sorted({r["x"] for r in asset.rows})
        labels = list(dict.fromkeys(r["label"] for r in direct))
        counts = []
        for k in cutoffs:
            values = [exactly_one(direct, label=label, x=k)["estimate"] for label in labels]
            if len(set(values)) != 1:
                raise ValueError("Cannot combine unequal direct-page counts into one curve")
            counts.append(values[0])
        expanded = [exactly_one(expanded, x=k) for k in cutoffs]
        fig, ax = plt.subplots(figsize=(width, 80/25.4))
        fig.subplots_adjust(left=.14, right=.975, bottom=.24, top=.96)
        ax.plot(cutoffs, counts, color="#666666", marker="o", markersize=3.7,
                label="Direct pages (all methods)")
        style = expanded[0]
        ax.plot(cutoffs, [r["estimate"] for r in expanded], color=style["color"],
                marker=style["marker"], linestyle="--", markersize=3.7, label=style["label"])
        for r in expanded:
            ax.annotate(f"{r['estimate']:.0f}", (r["x"], r["estimate"]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8, color=style["color"])
        ax.annotate(f"{counts[-1]:.0f}", (cutoffs[-1], counts[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#666666")
        ax.set_xticks(cutoffs)
        ax.set_xlim(0, max(cutoffs)*1.08)
        ax.set_ylim(0, max(r["estimate"] for r in expanded)*1.14)
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.set_xlabel(asset.options["xlabel"])
        ax.set_ylabel("Distinct pages per query (mean)")
        ax.grid(axis="y")
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5,.02),
                   frameon=False, ncol=2, handlelength=1.6, columnspacing=1.2)
    elif asset.kind=="tradeoff":
        from matplotlib.transforms import Bbox
        focus_labels=asset.options.get('focus_labels',False)
        indexed=len(asset.rows)>6 and not focus_labels
        fig,ax=plt.subplots(figsize=(width,(165 if indexed else 128 if focus_labels else 92)/25.4))
        fig.subplots_adjust(left=.14,right=.975,bottom=.43 if indexed else 48/128 if focus_labels else .17,top=.96)
        low=min(r["ci_lower"] for r in asset.rows);high=max(r["ci_upper"] for r in asset.rows)
        span=max(high-low,.05)
        ax.set_xlim(0,max(r["x"] for r in asset.rows)*1.24+.15)
        metric_axis(ax.yaxis,low,high,scale=1,bins=5)
        ax.set_xlabel("Retrieval p95 (s)");ax.set_ylabel(asset.rows[0]["panel"])
        ax.grid();ax.xaxis.set_major_locator(MaxNLocator(6))
        for r in asset.rows:
            background=focus_labels and not r['focused']
            ax.errorbar(r["x"],r["estimate"],yerr=[[r["estimate"]-r["ci_lower"]],[r["ci_upper"]-r["estimate"]]],
                fmt=r["marker"],color=r["color"],capsize=2,markersize=4 if background else 5,elinewidth=1,
                zorder=1 if background else 2)
        if indexed or focus_labels:
            def legend_label(r):
                label=r['label']
                if focus_labels and not r['focused']:
                    label=label.replace(' + TU + titles/headings',' + TU + T/H').replace(' + titles/headings',' + T/H').replace(' + TU instruction',' + TU')
                elif not focus_labels:
                    label=label.replace(' + titles/headings',' + metadata').replace(' instruction','')
                return f"{r['point_id']:02d}  "+label
            handles=[Line2D([],[],color=r['color'],marker=r['marker'],linestyle='none',
                label=legend_label(r)) for r in asset.rows]
            legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.052 if focus_labels else .01),
                ncol=3 if focus_labels else 2,fontsize=7.3 if focus_labels else 7.4,frameon=False,
                columnspacing=.9 if focus_labels else 1,handletextpad=.45 if focus_labels else .5,
                handlelength=1 if focus_labels else 2,labelspacing=.2 if focus_labels else .55)
            for text,r in zip(legend.get_texts(),asset.rows,strict=True):
                emphasis=r['focused'] and r['experiment_id']!='reranker_none' if focus_labels else r.get('selected')
                text.set_fontweight('bold' if emphasis else 'normal')
                if focus_labels:
                    text.set_color('#27333D' if r['focused'] else '#60666C')
            if focus_labels:
                fig.text(.52,.018,'T/H: titles and headings. TU: domain instruction',ha='center',fontsize=7,color='#60666C')
        fig.canvas.draw();renderer=fig.canvas.get_renderer();occupied=[]
        label_order=sorted(asset.rows,key=lambda r:(focus_labels and not r.get('focused',False),
                                                  not r.get("selected",False),r["point_id"]))
        for r in label_order:
            xy=ax.transData.transform((r["x"],r["estimate"]))
            occupied.append(Bbox.from_bounds(xy[0]-4,xy[1]-4,8,8))
        for r in label_order:
            background=focus_labels and not r['focused']
            label=str(r["point_id"]) if indexed or background else textwrap.fill(r["label"],24,break_long_words=False)
            chosen=None
            for radius in [8,18,30,45,65,90]:
                for dx,dy in [(radius,radius),(radius,-radius),(-radius,radius),(-radius,-radius),(0,radius),(0,-radius)]:
                    annotation=ax.annotate(label,(r["x"],r["estimate"]),xytext=(dx,dy),textcoords="offset points",
                        ha="right" if dx<0 else "left",va="bottom" if dy>=0 else "top",fontsize=7.5 if background else 8,
                        fontweight="bold" if r.get("selected") else "normal",color="#60666C" if background else "#27333D")
                    box=annotation.get_window_extent(renderer).expanded(1.06,1.12)
                    inside=ax.bbox.contains(box.x0,box.y0) and ax.bbox.contains(box.x1,box.y1)
                    if inside and not any(box.overlaps(other) for other in occupied):
                        occupied.append(box);chosen=(dx,dy);break
                    annotation.remove()
                if chosen:break
            if chosen is None:
                raise ValueError(f"Cannot place trade-off label without overlap: {asset.name}/{label}")
            ax.annotate("",(r["x"],r["estimate"]),xytext=chosen,textcoords="offset points",
                arrowprops=dict(arrowstyle="-",color=r["color"],lw=.6,shrinkA=3,shrinkB=4))
    elif asset.kind in ("heatmap","stacked"):
        labels=list(dict.fromkeys(r["label"] for r in asset.rows))
        panels=[p for p in panels if any(r['estimate'] for r in select(asset.rows,panel=p))] if asset.kind=='stacked' else panels
        height_mm=max(66,36+len(labels)*8)
        fig,ax=plt.subplots(figsize=(width,height_mm/25.4))
        bottom=(22 if asset.options.get('count_axis') else 30 if len(panels)>3 else 22)/height_mm if asset.kind=="stacked" else .12
        fig.subplots_adjust(left=.35,right=.95 if asset.options.get('count_axis') else .98,bottom=bottom,top=.97)
        if asset.kind=="heatmap":
            values=np.array([[exactly_one(asset.rows,label=label,panel=panel)["estimate"] for panel in panels] for label in labels])
            ax.imshow(values,cmap="Blues",vmin=0,vmax=asset.options.get("vmax",1),aspect="auto")
            for i,label in enumerate(labels):
                for j,panel in enumerate(panels):
                    ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",fontsize=8,
                            color="white" if values[i,j]>.65*asset.options.get("vmax",1) else "#27333D")
            ax.set_xticks(range(len(panels)),panels)
        else:
            left=np.zeros(len(labels));count_axis=asset.options.get('count_axis',False)
            maximum=max(r.get('n',100) for r in asset.rows) if count_axis else 100
            for panel in panels:
                rr=[exactly_one(asset.rows,label=label,panel=panel) for label in labels]
                values=np.array([r['count'] if count_axis else r["estimate"]*100 for r in rr])
                ax.barh(range(len(labels)),values,left=left,color=rr[0]["color"],label=panel,height=.65,
                        hatch=rr[0].get('hatch',''),edgecolor='white',linewidth=.45)
                for i,value in enumerate(values):
                    if value >= maximum*.075:
                        color='white' if rr[0]['color'] in ('#0072B2','#009E73') and not rr[0].get('hatch') else '#1B2630'
                        ax.text(left[i]+value/2,i,f"{value:,.0f}" if count_axis else f"{value:.1f}",
                                ha="center",va="center",fontsize=7.5,color=color)
                left+=values
            expected=[exactly_one(asset.rows,label=label,panel=panels[0])['n'] for label in labels] if count_axis else 100
            if not np.allclose(left,expected,atol=.01):raise ValueError(f"Incomplete composition: {asset.name}")
            ax.set_xlim(0,maximum);ax.set_xlabel(asset.options.get("xlabel","Query-target pairs (%)"));ax.invert_yaxis()
            if count_axis:
                from matplotlib.ticker import StrMethodFormatter
                ax.xaxis.set_major_locator(MaxNLocator(4,integer=True));ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
                ax.grid(axis='x')
            fig.legend(loc="lower center",bbox_to_anchor=(.58,.005),ncol=2 if count_axis else 3,frameon=False,columnspacing=1,fontsize=7.5)
        ax.set_yticks(range(len(labels)),[textwrap.fill(s,26,break_long_words=False) for s in labels])
        ax.tick_params(axis="both",length=0)
        selected={r["label"] for r in asset.rows if r.get("selected")}
        for tick,label in zip(ax.get_yticklabels(),labels):tick.set_fontweight("bold" if label in selected else "normal")
    elif asset.kind=="confusion":
        fig,ax=plt.subplots(figsize=(width,85/25.4));fig.subplots_adjust(left=.24,right=.82,bottom=.18,top=.96)
        counts=np.zeros((4,4));proportions=np.zeros((4,4))
        for r in asset.rows:
            i,j=int(r["human_grade"]),int(r["judge_grade"]);counts[i,j]=int(r["count"]);proportions[i,j]=float(r["row_proportion"])
        ax.imshow(proportions,cmap="Blues",vmin=0,vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j,i,str(int(counts[i,j])),ha="center",va="center",color="white" if proportions[i,j]>.55 else "#333333")
        ax.set_xticks(range(4));ax.set_yticks(range(4));ax.set_xlabel("Judge grade");ax.set_ylabel("Human grade")
    elif asset.kind=='stage_changes':
        from matplotlib.patches import Patch
        labels=list(dict.fromkeys(r['label'] for r in asset.rows))
        fig,axes=plt.subplots(1,len(panels),figsize=(width,76/25.4),sharey=True,squeeze=False)
        fig.subplots_adjust(left=.37,right=.98,top=.89,bottom=.28,wspace=.2)
        limit=max(max(r['gained'],r['lost']) for r in asset.rows)+8
        for i,(ax,panel) in enumerate(zip(axes[0],panels,strict=True)):
            for j,label in enumerate(labels):
                r=exactly_one(asset.rows,label=label,panel=panel)
                for key,sign,color in [('gained',1,'#009E73'),('lost',-1,'#D62728')]:
                    value=r[key]
                    if value:
                        ax.barh(j,sign*value,height=.5,color=color)
                        ax.text(sign*(value+1),j,f'{sign*value:+d}',va='center',ha='left' if sign>0 else 'right',fontsize=8)
                if not (r['gained'] or r['lost']):
                    ax.plot(0,j,'o',color='#666666',ms=3)
                    ax.text(2,j,'0',va='center',fontsize=8)
            ax.axvline(0,color='#777777',lw=.6)
            ticks=MaxNLocator(3,integer=True,symmetric=True).tick_values(-limit,limit)
            ax.set_xticks([tick for tick in ticks if -limit<=tick<=limit])
            ax.set_xlim(-limit,limit);ax.grid(axis='x')
            ax.set_yticks(range(len(labels)),[s.replace(' → ','\n→ ').replace('Keyword + length filter','Keyword + length') for s in labels])
            ax.tick_params(axis='y',length=0,labelleft=i==0)
            ax.set_xlabel('Questions');ax.text(0,1.08,chr(97+i)+') '+panel,transform=ax.transAxes,fontsize=8.5)
        axes[0,0].invert_yaxis()
        fig.legend(handles=[Patch(color='#009E73',label='Gained hit'),Patch(color='#D62728',label='Lost hit')],
                   loc='lower center',bbox_to_anchor=(.63,.01),frameon=False,ncol=2)
    elif asset.kind=='direct_contribution':
        # Third-page panels retain readable labels in a three-part diagnostic.
        fig,ax=plt.subplots(figsize=(min(width_mm,50)/25.4,40/25.4))
        fig.subplots_adjust(left=.31,right=.98,bottom=.41,top=.97)
        total=asset.options['query_count']
        for offset,key,label,color in [(-.19,'Without\npage','Chunks only','#88939D'),
                                       (.19,'With\npage','Chunks + page','#0072B2')]:
            values=[int(r[key]) for r in asset.rows]
            ax.barh(np.arange(len(values))+offset,values,height=.34,color=color,label=label)
            for y,value in enumerate(values):
                ax.text(value+total*.025,y+offset,str(value),va='center',fontsize=8)
        ax.set_yticks(range(len(asset.rows)),[r['Threshold'] for r in asset.rows])
        ax.tick_params(axis='y',labelsize=7)
        ax.invert_yaxis();ax.tick_params(axis='y',length=0)
        ax.set_xlim(0,total*1.12);ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
        ax.set_xlabel('Queries with a hit',fontsize=8)
        ax.grid(axis='x')
        fig.legend(loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False,
                   fontsize=7,columnspacing=1,handlelength=1.3)
    elif asset.kind=='direct_cooccurrence':
        fig,ax=plt.subplots(figsize=(min(width_mm,50)/25.4,40/25.4))
        fig.subplots_adjust(left=.26,right=.98,bottom=.28,top=.98)
        counts=np.array([[int(r['Retrieved: no hit']),int(r['Retrieved: hit'])] for r in asset.rows])
        shares=counts/asset.options['query_count']
        ax.imshow(shares,cmap='Blues',vmin=0,vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(counts[i,j]),ha='center',va='center',fontsize=9,
                        color='white' if shares[i,j]>.55 else '#333333')
        ax.set_xticks([0,1],['No hit','Hit'])
        ax.set_yticks([0,1],[r['Direct page'].removeprefix('Grade ') for r in asset.rows])
        ax.set_xlabel('Retrieved chunks',fontsize=8)
        ax.set_ylabel('Injected page grade',fontsize=8)
        ax.set_xticks([.5],minor=True);ax.set_yticks([.5],minor=True)
        ax.grid(which='minor',color='white',linewidth=2)
        ax.tick_params(which='both',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    elif asset.kind=="grades":
        compact=asset.options.get('compact',False)
        fig,ax=plt.subplots(figsize=((min(width_mm,50)/25.4 if compact else width),(40 if compact else 57)/25.4))
        fig.subplots_adjust(left=.25 if compact else .13,right=.99 if compact else .97,bottom=.29 if compact else .23,top=.96 if compact else .94)
        colors=["#AAAAAA","#D0D5D9","#56B4E9","#009E73"]
        for r in asset.rows:
            grade=int(r["Grade"]); share=r["Share"]
            ax.bar(grade,share,color=colors[grade],width=.58)
            ax.text(grade,share+1.5,f"{share:.1f}%\n(n={r['Queries']})",ha="center",fontsize=7 if compact else 8)
        ax.set_xticks(range(4));ax.set_xlabel("Relevance grade");ax.set_ylabel("Injection cases (%)");ax.set_ylim(0,max(r['Share'] for r in asset.rows)*1.3);ax.grid(axis="y")
        if compact:
            ax.xaxis.label.set_fontsize(8);ax.yaxis.label.set_fontsize(8)
    else:
        raise ValueError(f"Unknown figure type {asset.kind}")
    return fig


def tex_escape(value):
    return "".join({"&":r"\&", "%":r"\%", "$":r"\$", "#":r"\#", "_":r"\_",
                    "{":r"\{", "}":r"\}", "~":r"\textasciitilde{}", "^":r"\textasciicircum{}",
                    "\\":r"\textbackslash{}", "≥":r"$\geq$", "≤":r"$\leq$", "κ":r"$\kappa$", "Δ":r"$\Delta$",
                    "−":"--", "–":"--", "→":r"$\rightarrow$", "∞":r"$\infty$"}.get(c,c) for c in str(value))


def table_cells(asset):
    if asset.kind in ("cohort_quality", "cohort_priority"):
        rows = []
        for cohort in asset.options["cohorts"]:
            headers, cells = table_cells(replace(
                asset, kind="points" if asset.kind == "cohort_quality" else "curves",
                rows=select(asset.rows, cohort=cohort)))
            rows.extend([[tex_escape(cohort) + " / " + row[0], *row[1:]] for row in cells])
        return ["Dataset / " + headers[0].lower(), *headers[1:]], rows
    if "table_rows" in asset.options:
        return asset.options["table_headers"], [[tex_escape(v) for v in row] for row in asset.options["table_rows"]]
    if asset.kind=='stage_changes':
        return ['Transition','Grade','Gained','Lost','Δ (pp)','95% CI'],[
            [tex_escape(r['label']),tex_escape('≥2' if r['panel']=='Grade ≥2' else '3'),str(r['gained']),str(r['lost']),
             f"{100*r['estimate']:+.1f}", f"[{100*r['ci_lower']:+.1f}, {100*r['ci_upper']:+.1f}]"] for r in asset.rows]
    if asset.kind=='stacked' and asset.options.get('count_axis'):
        panels=[p for p in dict.fromkeys(r['panel'] for r in asset.rows)
                if any(r['count'] for r in select(asset.rows,panel=p))]
        headers=['Variant']+[p.replace(', ', '\n') for p in panels]+['Total']
        rows=[]
        for label in dict.fromkeys(r['label'] for r in asset.rows):
            rr=select(asset.rows,label=label)
            name=tex_escape(label)
            if any(r.get('selected') for r in rr):name=r'\textbf{'+name+'}'
            rows.append([name]+[f"{exactly_one(rr,panel=p)['count']:,}" for p in panels]+[f"{rr[0]['n']:,}"])
        return headers,rows
    if asset.kind=="tradeoff":
        rows=[]
        best_quality=max(round(r["estimate"],3) for r in asset.rows)
        best_p95=min(round(r["x"],3) for r in asset.rows)
        for r in asset.rows:
            label=tex_escape(r["label"])
            if r.get("selected"):label=r"\textbf{"+label+"}"
            value=f"{r['estimate']:.3f}";timing=f"{r['x']:.3f}"
            if round(r["estimate"],3)==best_quality:value=r"\textbf{"+value+"}"
            if round(r["x"],3)==best_p95:timing=r"\textbf{"+timing+"}"
            interval=f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            rows.append([label,str(r["point_id"]),r"\shortstack{"+value+r"\\{\scriptsize "+interval+"}}",timing])
        return ["Variant","ID",asset.rows[0]["panel"].replace(" nDCG","\nnDCG"),"p95 (s)"],rows
    if asset.kind in ("points", "differences", "latency", "heatmap", "stacked"):
        panels=list(dict.fromkeys(r["panel"] for r in asset.rows));labels=list(dict.fromkeys(r["label"] for r in asset.rows))
        headers=["Variant"] + panels
        if asset.kind=="latency": headers=["Variant","Median (s)","P95 (s)","Mean (s)"]
        rows=[]
        selected={r["label"] for r in asset.rows if r.get("selected")}
        for label in labels:
            cells=[r"\textbf{"+tex_escape(label)+"}" if label in selected else tex_escape(label)]
            if asset.kind=="latency":
                r=exactly_one(asset.rows,label=label)
                for key in ["estimate","p95","mean"]:
                    value=f"{r[key]:.3f}"
                    best=min(float(f"{item[key]:.3f}") for item in asset.rows)
                    cells.append(r"\textbf{"+value+"}" if float(value)==best else value)
            else:
                for panel in panels:
                    r=exactly_one(asset.rows,label=label,panel=panel);scale=r.get("scale",1);digits=1 if scale==100 else 3
                    point=f"{r['estimate']*scale:.{digits}f}"
                    if asset.kind in ("points","heatmap"):
                        extremum=min if r.get("direction")=="min" else max
                        best=extremum(float(f"{item['estimate']*scale:.{digits}f}") for item in select(asset.rows,panel=panel))
                        if float(point)==best:
                            point=r"\textbf{"+point+"}"
                    if r.get("ci_lower") is not None:
                        interval=f"[{r['ci_lower']*scale:.{digits}f}, {r['ci_upper']*scale:.{digits}f}]"
                        point=r"\shortstack{"+point+r"\\{\scriptsize "+interval+"}}"
                    cells.append(point)
            rows.append(cells)
        return headers,rows
    if asset.kind in ("table", "direct_contribution", "direct_cooccurrence"):
        return list(asset.rows[0]), [[tex_escape(v) for v in row.values()] for row in asset.rows]
    if asset.kind=="grades":
        return ["Grade","Queries","Share (%)"],[[str(r["Grade"]),str(r["Queries"]),f"{r['Share']:.1f}"] for r in asset.rows]
    if asset.kind=="confusion":
        return ["Human grade"]+[f"Judge {i}" for i in range(4)],[[str(i)]+[str(exactly_one(asset.rows,human_grade=i,judge_grade=j)["count"]) for j in range(4)] for i in range(4)]
    # Curve data are exported in a compact wide table per panel.
    panels=list(dict.fromkeys(r["panel"] for r in asset.rows))
    xs=asset.options.get('table_cutoffs',sorted({r["x"] for r in asset.rows}))
    headers=[asset.options.get('table_row_header',"Variant / measure")]+[str(next((r["category"] for r in asset.rows if r["x"]==x and "category" in r),x)) for x in xs]
    rows=[]
    for panel in panels:
        for label in dict.fromkeys(r["label"] for r in select(asset.rows,panel=panel)):
            values=select(asset.rows,panel=panel,label=label)
            name=tex_escape(label+(" / "+panel if len(panels)>1 else ""))
            row=[r"\textbf{"+name+"}" if any(r.get("selected") for r in values) else name]
            for x in xs:
                r=exactly_one(values,x=x);scale=r.get("scale",1)
                if r['estimate'] is None:
                    row.append('---')
                    continue
                digits=1 if scale==100 or asset.kind=="scope_size" else 3
                value=f"{r['estimate']*scale:.{digits}f}"
                # Scope size has no universally preferable direction.
                if panel!="Mean scope size (pages)":
                    best=max(float(f"{item['estimate']*scale:.{digits}f}") for item in select(asset.rows,panel=panel,x=x)
                             if item['estimate'] is not None)
                    if float(value)==best:
                        value=r"\textbf{"+value+"}"
                if asset.options.get('table_intervals'):
                    if r.get('ci_lower') is None or r.get('ci_upper') is None:
                        raise ValueError(f'Missing tabulated interval: {asset.name}/{label}/{x}')
                    interval=f"[{r['ci_lower']*scale:.{digits}f}, {r['ci_upper']*scale:.{digits}f}]"
                    value=r"\shortstack{"+value+r"\\{\scriptsize "+interval+"}}"
                row.append(value)
            rows.append(row)
    return headers,rows



def write_asset(asset, output, plt, catalog):
    topic=output/asset.topic
    for folder in ["figures","tables","data"]:
        (topic/folder).mkdir(parents=True,exist_ok=True)
    written=[]
    def write(relative,content):
        path=topic/relative;path.write_text(content);written.append(path);return path
    keys=list(dict.fromkeys(k for r in asset.rows for k in r))
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=keys);writer.writeheader();writer.writerows(asset.rows)
    write("data/"+asset.name+".csv",buf.getvalue())
    headers,rows=table_cells(asset)
    # tabularx wraps descriptive labels without scaling away the chosen type size.
    formats="@{}X"+"r"*(len(headers)-1)+"@{}"
    if asset.kind=="table": formats=asset.options.get("column_format", "@{}"+"X"*len(headers)+"@{}")
    header_cells=[r"\shortstack{"+r"\\".join(tex_escape(line) for line in h.split("\n"))+"}" if "\n" in h else tex_escape(h) for h in headers]
    header_tex=" & ".join(header_cells)+r" \\"+"\n"
    if groups := asset.options.get("column_groups"):
        if sum(span for _, span in groups) != len(headers) - 1:
            raise ValueError(f"Column groups do not match table columns: {asset.name}")
        group_cells=[""]; rules=[]; first=2
        for label, span in groups:
            group_cells.append(r"\multicolumn{"+str(span)+"}{c}{"+tex_escape(label)+"}")
            rules.append(r"\cmidrule(lr){"+str(first)+"-"+str(first+span-1)+"}")
            first += span
        labels=[tex_escape(headers[0]), *[tex_escape(h.rsplit("\n", 1)[-1]) for h in headers[1:]]]
        header_tex=(" & ".join(group_cells)+r" \\"+"\n"+" ".join(rules)+"\n"
                    +" & ".join(labels)+r" \\"+"\n")
    tex=(r"\begingroup"+"\n"+r"\small"+"\n"
         +r"\setlength{\tabcolsep}{3pt}"+"\n"+r"\renewcommand{\arraystretch}{1.15}"+"\n"
         +r"\noindent\begin{tabularx}{\linewidth}{"+formats+"}\n"+r"\toprule"+"\n"
         +header_tex+r"\midrule"+"\n"
         +"\n".join(" & ".join(row)+r" \\" for row in rows)+"\n"+r"\bottomrule"+"\n"
         +r"\end{tabularx}\par"+"\n"+r"\endgroup"+"\n")
    write("tables/"+asset.name+".tex",tex)
    if asset.kind!="table" and not asset.options.get('table_only'):
        if asset.kind == "corpus_treemap":
            from thesis_rag.publication.corpus import write_treemap
            paths = [topic/"figures"/(asset.name+"."+ext) for ext in ("pdf", "png", "svg")]
            write_treemap(asset, paths, catalog["width_mm"])
            written.extend(paths)
        else:
            fig=draw(asset,plt,width_mm=asset.options.get("width_mm", catalog["width_mm"]))
            fig.canvas.draw()
            bounds=fig.get_tightbbox(fig.canvas.get_renderer())
            if bounds.x0 < -.01 or bounds.y0 < -.01 or bounds.x1 > fig.get_figwidth()+.01 or bounds.y1 > fig.get_figheight()+.01:
                raise ValueError(f"Figure content outside canvas: {asset.name}: {bounds.bounds}")
            renderer=fig.canvas.get_renderer()
            for ax in fig.axes:
                ticks=[tick.get_window_extent(renderer) for tick in ax.get_xticklabels() if tick.get_visible() and tick.get_text()]
                if any(left.overlaps(right) for left,right in zip(ticks,ticks[1:])):
                    raise ValueError(f"Overlapping tick labels: {asset.name}")
                for legend in fig.legends:
                    if legend.get_window_extent(renderer).overlaps(ax.xaxis.label.get_window_extent(renderer)):
                        raise ValueError(f"Legend overlaps axis label: {asset.name}")
            for ext in ["pdf","png","svg"]:
                path=topic/"figures"/(asset.name+"."+ext)
                metadata={"Creator":"ChaTU thesis publication exporter", "CreationDate":None,"ModDate":None} if ext=="pdf" else ({"Date":None} if ext=="svg" else {})
                fig.savefig(path,dpi=240,metadata=metadata)
                written.append(path)
            plt.close(fig)
    return {"topic":asset.topic,"name":asset.name,"title":asset.title,
            "kind":"table" if asset.options.get('table_only') else asset.kind,
            "placement":asset.placement,"files":[str(p.relative_to(output)) for p in written]}


def write_coverage_audit(output, assets, catalog, root):
    """Map source report figures to thesis exports and record omitted figures."""
    families={
        "metric_overview":("points","categorical"),"counterpart_aware_quality_overview":("points","categorical"),
        "quality_efficiency":("tradeoff",),"hit_latency":("latency","curves","tradeoff"),
        "performance_by_cutoff":("curves",),"paired_primary_metric_differences":("differences","paired_comparisons","translated_source_availability","paired_priority_hits"),
        "paired_win_tie_loss":("ranking_wins_ties_losses","differences","paired_comparisons"),
        "raw_vs_counterpart_aware_effects":("metadata_effect","counterpart_deduplication","ranking_views","points"),
        "counterpart_redundancy":("counterpart_deduplication",),
        "language_alignment":("language_and_relevance",),"language_relevance_composition":("language_and_relevance",),
        "language_signal_robustness":("language_signal_robustness","language_signal_by_language"),
        "language_swap_cohort_effects":("translated_source_availability",),
        "context_budget":("context_character_budget","constructed_context_evidence"),
        "direct_context_contribution":("direct_page_incremental","direct_page_rescue_coverage","baseline_direct_page_contribution","baseline_direct_page_cooccurrence"),
        "direct_context_grades":("baseline_direct_page_grades","direct_page_incremental"),
        "nlu_hit_changes":("paired_hit_changes","hit_change_counts"),"nlu_target_changes":("hit_change_counts",),
        "nlu_rescue_cases":("direct_page_rescue_coverage",),"nlu_rescue_coverage":("direct_page_rescue_coverage",),
        "priority_hit_curves":("priority_hits","priority_evidence"),"priority_relevance":("priority_hits","priority_evidence"),
        "priority_branch_ablation":("reserved_slots","generalization","priority_evidence"),
        "priority_branch_effects":("priority_branch_gains_losses",),
        "priority_ordering_replay":("reserved_evidence_ordering",),
        "priority_protected_evidence":("protected_evidence_quality",),
        "reference_page_coverage":("reference_page_recovery",),"slice_robustness":("language_subgroups",),
        "judgment_view_robustness":("judgment_view_sensitivity",),
        "hit_by_k":("hit_curves","points"),"mrr_by_k":("navigation_ranking",),"paired_hit_differences":("points",),
        "stage_progression":("baseline_stage_quality",),
        "stage_hit_rates":("baseline_stage_hit_curves",),
        "stage_hit_changes":("baseline_stage_hit_changes",),
        "confusion_matrices_counts":("judge_confusion",),"confusion_matrices_row_normalized":("judge_confusion",),
        "confusion_matrices_top_judges":("judge_confusion","judge_top_two_paired"),
        "grade_distributions":("judge_error_direction","judge_per_grade_f1"),
        "per_grade_f1":("judge_per_grade_f1",),"paired_top_judge_differences":("judge_top_two_paired",),
        "signed_grade_differences":("judge_error_direction",),
    }
    deliberately_not_repeated={
        "pairwise_agreement_heatmap":"Selection is evaluated against human labels; inter-model agreement adds no independent validity evidence.",
        "per_query_error_heatmap":"Query-ID heatmap is exploratory; calibration intervals and the paired comparison retain query-cluster uncertainty.",
        "slice_robustness":"Retain final language subgroups; repeated exploratory heatmaps for every tuning run are not separate decisions.",
        "reference_page_coverage":"Known-source recovery is consolidated in navigation and the synthetic test, separately from graded relevance.",
        "language_alignment":"Language/relevance composition is consolidated in the language experiment and both test cohorts.",
        "language_relevance_composition":"Language/relevance composition is consolidated in the language experiment and both test cohorts.",
        "counterpart_redundancy":"The duplicate mechanism is shown once in the language-policy diagnostic.",
        "context_budget":"Budget evidence is consolidated in the mixed-context diagnostic; rendered-fragment-only series are not whole-context hits.",
        "priority_ordering_replay":"The fixed-membership replay is shown on both development cohorts; repeated test replays are not new runs.",
        "priority_protected_evidence":"Branch-specific reserved-evidence quality is consolidated across both development cohorts.",
        "judgment_view_robustness":"Source-anchor sensitivity is tabulated for the final synthetic test. Main comparisons use judged-only relevance.",
        "nlu_target_changes":"Target-count changes are not query hit-rate changes; query-level paired gains/losses are the selected diagnostic.",
        "quality_efficiency":"Direct trade-off figures are retained where they explain a choice (architecture, rerankers, NLU routing, logs and test); other sweeps retain their quality comparison.",
        "performance_by_cutoff":"Early-rank curves are retained for architecture, rerankers, Priority and test. Fixed-cutoff tuning comparisons are shown at their decision cutoff.",
        "paired_primary_metric_differences":"Paired uncertainty is consolidated in selection contrasts; full pair grids for language-signal scenarios are omitted.",
        "paired_win_tie_loss":"Per-query direction is consolidated in paired comparison tables and test figures; repetitive per-contrast bars are omitted.",
        "raw_vs_counterpart_aware_effects":"Raw/family effects are shown for counterpart deduplication and metadata; repeated language-signal contrasts use the family metric.",
        "direct_context_contribution":"Direct-page contribution is consolidated across log development and both test cohorts.",
        "direct_context_grades":"Injection relevance is shown for both test cohorts; early exploratory repetitions stay in the original reports.",
        "nlu_hit_changes":"Query gains and losses are retained for NLU routing, log development and both tests; Priority development has paired hit tables.",
        "priority_branch_effects":"Branch contributions are tabulated across both development cohorts; test figures retain general and Priority evidence curves.",
        "hit_latency":"Quality and latency are combined in trade-off figures where selection depends on both; other reports retain the relevant quality comparison.",
    }
    rows=[]
    for source,definition in catalog["sources"].items():
        if "manifest" not in definition:continue
        directory=Path(definition["manifest"]).parent
        related=[a for a in assets if any(str(p).startswith(str(directory)+"/") for p in a.inputs)]
        for file in sorted((root/directory).rglob("*.png")):
            family=file.stem.split("__")[0]
            tokens=families.get(family,("main_nlu_descendant_hits",) if family.startswith("navigation_hits")
                               else ("descendant_scope",) if family.startswith("scope_") else ())
            matches=[a for a in related if a.kind in tokens or any(token in a.name for token in tokens)]
            if definition.get("publication_status")=="superseded":
                status="superseded";reason=definition["exclusion_reason"]
                matches=[a for a in assets if any(str(p).startswith(str(Path(catalog['sources'][definition['superseded_by']]['manifest']).parent)+"/") for p in a.inputs)]
            elif "no_backup_training_overlap" in file.stem:
                status="excluded sensitivity";reason="Use the full cohort, including training overlaps, as requested."
                matches=[]
            elif family in deliberately_not_repeated and (not matches or family=="nlu_target_changes"):
                status="consolidated diagnostic";reason=deliberately_not_repeated[family]
            elif matches:
                status="covered";reason="Scientific comparison retained in the linked exports; duplicate contrasts, views and denominators are consolidated."
            else:
                status="source retained";reason="Available in the original report; no separate thesis asset."
            rows.append(dict(source=source,family=family,old_figure=str(file.relative_to(root)),status=status,
                thesis_assets=";".join(a.topic+"/"+a.name for a in matches),reason=reason))
    audit=output/"review";audit.mkdir(exist_ok=True)
    with (audit/"report_plot_coverage.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups={}
    for r in rows:
        key=(r["source"],r["family"],r["status"])
        groups.setdefault(key,[]).append(r)
    body=[]
    for (source,family,status),group in groups.items():
        row=group[0]
        old=os.path.relpath(root/row["old_figure"],audit)
        links=" · ".join(f'<a href="../index.html#{html.escape(name.replace("/","--"))}">{html.escape(name.split("/")[-1])}</a>' for name in row["thesis_assets"].split(";") if name)
        body.append(f'<tr><td>{html.escape(source)}</td><td><a href="{old}">{html.escape(family)}</a> ({len(group)})</td><td>{status}</td><td>{html.escape(row["reason"])}<br>{links}</td></tr>')
    (audit/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Report-to-thesis coverage</title>'
        '<style>body{font:14px/1.5 system-ui;margin:30px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd;vertical-align:top}a{color:#006da7}th{position:sticky;top:0;background:white}</style>'
        f'<h1>Report-to-thesis coverage</h1><p>{len(rows):,} PNG artifacts in the pinned reports; {len(groups)} source/family/status groups. '
        'The review compares plot families and the resulting scientific claims, not every duplicate contrast individually. '
        'Old figures remain in their original report directories. Links in this audit require the implementation repository.</p>'
        '<p><a href="../index.html">Thesis exports</a> · <a href="report_plot_coverage.csv">Complete artifact inventory (CSV)</a></p>'
        '<table><thead><tr><th>Experiment</th><th>Old figure family</th><th>Disposition</th><th>Reason / thesis assets</th></tr></thead><tbody>'
        +"".join(body)+'</tbody></table></html>')


def write_index(output, records, catalog):
    topics=list(dict.fromkeys(r["topic"] for r in records))
    nav=" · ".join(f'<a href="#{topic}">{html.escape(topic.replace("_"," "))}</a>' for topic in sorted(topics))
    sections=[]
    for topic in sorted(topics):
        cards=[]
        for r in records:
            if r["topic"]!=topic:continue
            base=f"{topic}/figures/{r['name']}"
            image=f'<a href="{base}.pdf"><img loading="lazy" src="{base}.png" alt="{html.escape(r["title"])}"></a>' if r["kind"]!="table" else ""
            links=[]
            for file in r["files"]:
                label=("LaTeX table" if file.endswith(".tex") else Path(file).suffix[1:].upper())
                links.append(f'<a href="{file}">{label}</a>')
            cards.append(f'<article id="{topic}--{r["name"]}" data-placement="{r["placement"]}"><h3>{html.escape(r["title"])}</h3><small>{html.escape(r["name"])} · {r["placement"]}</small>{image}'
                         '<p class="links">'+" · ".join(links)+"</p></article>")
        sections.append(f'<section id="{topic}"><h2>{html.escape(topic.replace("_"," "))}</h2>'+"".join(cards)+"</section>")
    pending="".join(f'<li><b>{html.escape(p["cohort"])}</b>: {html.escape(p["reason"])}</li>' for p in catalog["pending"])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChaTU thesis figures and tables</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#25313c;background:#f4f5f6;max-width:1080px;margin:auto;padding:32px}
h1,h2,h3{font-weight:600;line-height:1.25}h2{margin-top:50px}article{background:white;padding:24px;margin:22px 0;border:1px solid #dfe3e7;border-radius:5px}
img{display:block;width:100%;max-width:720px;height:auto;margin:20px auto}a{color:#006da7}small{color:#66717b}.links{font-size:14px}code,pre{background:#e9edf0;padding:3px 5px}pre{overflow:auto;padding:16px}nav{line-height:2}
</style><h1>Thesis figures and tables</h1><p>English labels · vector figures · embedded fonts · source-pinned tables.</p>
<p><a href="review/index.html">Comparison with the original reports and excluded superseded experiments</a></p>
<p><label>Show <select id="placement"><option value="">All exports</option><option value="main candidate">Main-text candidates</option><option value="appendix candidate">Supplementary exports</option></select></label>
<label> Search <input id="search" placeholder="Judge, latency, Priority…"></label> <span id="count"></span></p>
<details><summary>Use in LaTeX</summary><p>Copy this directory into the thesis project, for example as <code>publication/</code>. Add to the preamble:</p>
<pre>\\usepackage{graphicx,booktabs,tabularx}</pre>
<p>Include <code>figures/*.pdf</code> with <code>\\includegraphics</code> and <code>tables/*.tex</code> with <code>\\input</code>. Write the figure/table environment, caption and label in the thesis. Exported tables contain only the table body and local formatting. Figures are sized for 150 mm width, 75 mm for standalone ranking panels, or 50 mm for three-part direct-injection diagnostics. Tables adapt to the text width.</p>
<p>Plots have no headline inside the artwork. Quality intervals are 95% bootstrap intervals; runtime ranges are percentiles. Main/appendix placements are suggestions. Final log-test assets use the corrected 1,000-query cohort; the full 2,093-query evaluation is listed separately.</p>
<p>Bold table names identify retained variants; bold estimates identify column optima and displayed ties, not statistical significance.</p>
<p>Regenerate from the implementation repository:</p><pre>uv run --group analysis python scripts/thesis/export_publication.py --catalog PATH_TO_CATALOG --output OUTPUT_DIRECTORY</pre>
<p>Use the pinned <code>catalog.json</code> copied into this bundle, or the matching publication catalog in the implementation repository. On a fresh output directory, add <code>--prepare-statistics</code> once. This prepares missing summaries from stored query metrics, without retrieval or judging. New reports must be bound explicitly.</p></details>
<h2>Pending results</h2><ul>'''+pending+'</ul><nav>'+nav+'</nav>'+"".join(sections)+'''<script>
function filter(){let n=0;const q=document.querySelector('#search').value.toLowerCase(),p=document.querySelector('#placement').value;
document.querySelectorAll('article').forEach(a=>{a.hidden=Boolean((p&&a.dataset.placement!==p)||(q&&!a.textContent.toLowerCase().includes(q)));if(!a.hidden)n++;});
document.querySelectorAll('section').forEach(s=>s.hidden=!Array.from(s.querySelectorAll('article')).some(a=>!a.hidden));document.querySelector('#count').textContent=n+' assets';}
document.querySelector('#placement').addEventListener('change',filter);document.querySelector('#search').addEventListener('input',filter);filter();
</script></html>'''
    (output/"index.html").write_text(page)
    (output/"index.json").write_text(json.dumps({"schema_version":"thesis_export_v2","assets":records,"pending":catalog["pending"]},indent=2)+"\n")


def write_portable(output, records, assets):
    """Create an HTML gallery with embedded assets and PDF catalogues."""
    by_key={(a.topic,a.name):a for a in assets}
    page=(output/'index.html').read_text()
    page=page.replace('<p><a href="review/index.html">Comparison with the original reports and excluded superseded experiments</a></p>',
                      '<p>Offline edition: figures, tables and download links are embedded in this file.</p>')
    def embedded(match):
        attribute,relative=match.groups()
        path=output/relative
        if not path.is_file():
            return match.group(0)
        mime={'.png':'image/png','.pdf':'application/pdf','.svg':'image/svg+xml',
              '.csv':'text/csv','.tex':'text/plain','.json':'application/json'}.get(path.suffix,'application/octet-stream')
        data=base64.b64encode(path.read_bytes()).decode()
        return f'{attribute}="data:{mime};base64,{data}"'+(f' download="{path.name}"' if attribute=='href' else '')
    page=re.sub(r'(src|href)="([^"#][^"]*)"',embedded,page)
    def cell(value):
        value=str(value)
        value=value.replace(r'\shortstack{','').replace(r'{\scriptsize ','').replace(r'\\','<br>')
        value=re.sub(r'\\textbf\{([^{}]*)\}',r'<b>\1</b>',value)
        value=value.replace(r'$\geq$','≥').replace(r'$\leq$','≤').replace(r'$\Delta$','Δ').replace(r'$\kappa$','κ').replace(r'$\rightarrow$','→')
        value=value.replace(r'$\infty$','∞')
        value=value.replace(r'\%','%').replace(r'\&','&amp;').replace(r'\_','_').replace('{','').replace('}','')
        return value
    for r in records:
        asset=by_key[r['topic'],r['name']];headers,rows=table_cells(asset)
        table='<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h).replace('\n','<br>')+'</th>' for h in headers)+'</tr></thead><tbody>'
        table+=''.join('<tr>'+''.join('<td>'+cell(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
        table='<details><summary>Numerical table</summary>'+table+'</details>' if r['kind']!='table' else table
        marker=f'<article id="{r["topic"]}--{r["name"]}"'
        start=page.index(marker);end=page.index('</article>',start)
        page=page[:end]+table+page[end:]
    page=page.replace('</style>','table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:7px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}.table-scroll{overflow:auto}</style>')
    (output/'preview.html').write_text(page)
    # Reading copies preserve figure widths; thesis captions are written separately.
    preamble=(r'\documentclass[10pt]{article}'+'\n'+r'\usepackage[paper=a4paper,textwidth=150mm,top=16mm,bottom=18mm]{geometry}'+'\n'
        +r'\usepackage[T1]{fontenc}\usepackage{graphicx,booktabs,tabularx}'+'\n'
        +r'\usepackage{lmodern}\renewcommand{\familydefault}{\sfdefault}\begin{document}'+'\n')
    for name, chosen in [('catalog',records),('main_figures',[r for r in records if r['placement']=='main candidate']),
                         ('tables',records)]:
        parts=[preamble]
        for r in chosen:
            table=name=='tables' or r['kind']=='table'
            if table:
                content=(output/r['topic']/'tables'/(r['name']+'.tex')).read_text()
            else:
                width_mm=by_key[r['topic'],r['name']].options.get('width_mm')
                width=f'{width_mm}mm' if width_mm else r'\linewidth'
                content=r'\noindent\includegraphics[width='+width+']{'+r['topic']+'/figures/'+r['name']+'.pdf}'
            parts += [r'\clearpage\noindent{\small '+tex_escape(r['topic'].replace('_',' ')+' / '+r['title'])+r'}\par\medskip'+'\n',
                      content+'\n']
        parts.append(r'\end{document}'+'\n')
        (output/(name+'.tex')).write_text(''.join(parts))
        result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=output,
                              capture_output=True,text=True)
        if result.returncode:
            raise ValueError(f"PDF catalogue failed: {name}: {result.stdout[-3000:]}")


def write_bundle(output):
    """Package the portable reading copies and the individual thesis assets."""
    with zipfile.ZipFile(output/'publication.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(output.rglob('*')):
            relative=path.relative_to(output)
            if (not path.is_file() or path.name=='publication.zip' or path.suffix in ('.aux','.log','.out')
                    or path.name.endswith('.bbox.html') or relative==Path('review/index.html')):
                continue
            if relative==Path('index.html'):
                page=path.read_text().replace('<p><a href="review/index.html">Comparison with the original reports and excluded superseded experiments</a></p>',
                    '<p><a href="review/report_plot_coverage.csv">Original-report coverage inventory (CSV)</a></p>')
                archive.writestr('publication/'+str(relative),page)
            else:
                archive.write(path,'publication/'+str(relative))
        archive.writestr('publication/START_HERE.txt',
            'Open preview.html for the offline gallery with figures and numerical tables.\n'
            'main_figures.pdf: main-text candidates; catalog.pdf: complete reading copy; tables.pdf: all tables.\n'
            'Each experiment folder contains vector figures, LaTeX tables and CSV values.\n'
            'LaTeX inclusion instructions are in the gallery. Original experiment reports remain in the implementation repository.\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",type=Path,default=Path("configs/thesis/publication/catalog.json"))
    parser.add_argument("--output",type=Path,default=Path("outputs/thesis/publication"))
    parser.add_argument("--prepare-statistics",action="store_true",help="Explicitly fill missing summaries via the existing analysis bootstrap before rendering.")
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[2];catalog_path=root/args.catalog;output=root/args.output
    catalog=json.loads(catalog_path.read_text())
    if catalog["schema_version"]!="thesis_publication_v1" or catalog["navigation_scope"]!="all":
        raise ValueError("Unsupported catalog or navigation cohort")
    output.mkdir(parents=True,exist_ok=True)
    (output/"catalog.json").write_text(json.dumps(catalog,indent=2)+"\n")
    statistics_path=output/"statistics.json"
    supplement=json.loads(statistics_path.read_text()) if statistics_path.exists() else None
    if args.prepare_statistics:
        preparing=Sources(root,catalog,prepare=True,supplement=supplement)
        print("Preparing missing analysis summaries from stored query metrics…",flush=True)
        build_assets(preparing)
        supplement=preparing.supplement
        statistics_path.write_text(json.dumps(supplement,indent=2)+"\n")
        print(f"Prepared {len(supplement['statistics'])} summaries; no retrieval or judging.",flush=True)
    elif supplement is None:
        raise ValueError("Run once with --prepare-statistics to prepare missing historical summaries.")
    data=Sources(root,catalog,supplement=supplement)
    assets=build_assets(data)
    plt=configure_style();records=[]
    for asset in assets:
        print(f"Exporting {asset.topic}/{asset.name}",flush=True)
        records.append(write_asset(asset,output,plt,catalog))
    records.sort(key=lambda record: (record['topic'],record['placement']!='main candidate'))
    write_coverage_audit(output,assets,catalog,root)
    write_index(output,records,catalog)
    write_portable(output,records,assets)
    write_bundle(output)
    print(f"Exported {sum(r['kind']!='table' for r in records)} figures and {len(assets)} tables: {output/'preview.html'}")


if __name__=="__main__":
    main()
