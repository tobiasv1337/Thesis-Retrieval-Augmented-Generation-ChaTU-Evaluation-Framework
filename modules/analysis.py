from pathlib import Path
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import os
import logging

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import zscore

def get_mapping_stats(df, path="data/az/", json_out="mapping_stats.json"):

    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["#children"] = pd.to_numeric(df["#children"], errors="coerce")
    df["#descendants"] = pd.to_numeric(df["#descendants"], errors="coerce")

    topics_per_depth = df["depth"].value_counts().sort_index().to_dict()
    topics_per_depth_cumulative = {}
    total = 0
    for depth in sorted(topics_per_depth):
        total += topics_per_depth[depth]
        topics_per_depth_cumulative[depth] = total

    def get_stats(series):
        stats = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std())
        }
        z = zscore(series.dropna())
        outlier_mask = np.abs(z) > 2.5
        outliers = series[outlier_mask]
        stats["#outliers"] = int(outlier_mask.sum())
        stats["outliers"] = [
            {"intent": df.loc[idx, "intent"], "value": float(val)}
            for idx, val in outliers.items()
        ]
        return stats

    global_stats = {
        "#children": get_stats(df["#children"]),
        "#descendants": get_stats(df["#descendants"])
    }

    per_depth_stats = {}
    for depth in sorted(df["depth"].dropna().unique()):
        subset = df[df["depth"] == depth]
        per_depth_stats[int(depth)] = {
            "#children": get_stats(subset["#children"]),
            "#descendants": get_stats(subset["#descendants"]),
            "n_topics": int(len(subset))
        }

    output = {
        "topics_per_depth": topics_per_depth,
        "topics_per_depth_cumulative": topics_per_depth_cumulative,
        "global_stats": global_stats,
        "per_depth_stats": per_depth_stats
    }

    # Load manual intents
    manual_path = os.path.join(path, "manual_utterances.csv")
    manual_intents = set(pd.read_csv(manual_path)["intent"].dropna().unique())

    # Count manual intents per depth
    manual_intents_per_depth = {}
    for depth in sorted(df["depth"].dropna().unique()):
        subset = df[df["depth"] == depth]
        intents_at_depth = set(subset["intent"].dropna().unique())
        count = len(manual_intents & intents_at_depth)
        manual_intents_per_depth[int(depth)] = count

    output["manual_intents_per_depth"] = manual_intents_per_depth

    with open(os.path.join(path, "analysis", json_out), "w") as f:
        json.dump(output, f, indent=2)

    plt.figure()
    plt.bar(topics_per_depth.keys(), topics_per_depth.values())
    plt.xlabel("Depth")
    plt.ylabel("# Topics")
    plt.title("Topics per Depth")
    plt.savefig(os.path.join(path, "analysis", "topics_per_depth.png"))

    plt.figure()
    plt.boxplot([df["#children"].dropna(), df["#descendants"].dropna()], labels=["#children", "#descendants"])
    plt.title("Children and Descendants Distribution")
    plt.savefig(os.path.join(path, "analysis", "children_descendants_boxplot.png"))

def load_and_process_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    child_counts = df['parent'].value_counts().rename('#children')
    df = df.merge(child_counts, how='left', left_on='link', right_index=True)
    df['#children'] = df['#children'].fillna(0).astype(int)

    # Add total descendant count
    df['#descendants'] = count_descendants_bottom_up(df)

    return df[["depth", '#children', '#descendants', "intent", "link", "mapping", "parent", "etag"]]

def count_descendants_bottom_up(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["#descendants"] = 0

    max_depth = df["depth"].max()
    for depth in reversed(range(max_depth + 1)):
        current = df[df["depth"] == depth]
        for parent in current["parent"].dropna().unique():
            children = current[current["parent"] == parent]
            descendants_sum = children["#descendants"].sum()
            total = len(children) + descendants_sum
            df.loc[df["link"] == parent, "#descendants"] += total

    return df["#descendants"]

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in df.iterrows():
        link = row['link']
        intent = row.get('intent', '')
        depth = row.get('depth', 0)
        children = row['#children']
        parents = row['parent']

        color = f"hsl({int(depth)*40 % 360}, 70%, 60%)"
        if pd.isna(parents):
            color = "#ff5555"

        G.add_node(
            link,
            title=f"Intent: {intent}<br>#Children: {children}",
            label=intent if pd.notna(intent) else link,
            color=color,
            group=int(depth)  # this enables clustering
        )

        if pd.notna(parents):
            for parent in str(parents).split(","):
                G.add_edge(parent.strip(), link)

    return G

def visualize_graph(G: nx.DiGraph, output_file: str = "link_structure.html"):
    net = Network(notebook=False, height="800px", width="100%", directed=True)
    net.from_nx(G)
    net.write_html(output_file)

def map_parent_intents(df: pd.DataFrame) -> pd.DataFrame:
    link_to_intent = df.set_index('link')['intent'].to_dict()

    def get_parent_intents(cell):
        if pd.isna(cell):
            return "ROOT"
        return ", ".join(link_to_intent.get(p.strip(), p.strip()) for p in str(cell).split(","))

    df['parent_intent'] = df['parent'].apply(get_parent_intents)
    return df

def show_treemap(df: pd.DataFrame, output_html: str = "treemap.html"):
    df['parent'] = df['parent'].fillna("")

    fig = go.Figure(go.Treemap(
        ids=df['link'],
        labels=df['intent'],
        parents=df['parent'],
        customdata=df[['depth', '#descendants', '#children']],
        hovertemplate="<b>%{label}</b><br>Depth: %{customdata[0]}<br>Descendants: %{customdata[1]}<br>Children: %{customdata[2]}<extra></extra>"
    ))

    fig.update_layout(margin=dict(t=30, l=10, r=10, b=10))
    fig.write_html(output_html)

def process_and_visualize_mapping(path: str = "data/az/", file: str = "mapping.csv", output_csv: str = "mapping_with_children.csv",
                          output_graph: str = "mapping_tree.html",
                          output_treemap: str = "mapping_treemap.html"):
    filepath = os.path.join(path, file)
    df = load_and_process_csv(filepath)
    df = map_parent_intents(df)
    G = build_graph(df)
    visualize_graph(G, os.path.join(path, "analysis", output_graph))
    show_treemap(df, os.path.join(path, "analysis", output_treemap))
    df.to_csv(os.path.join(path, "analysis", output_csv), index=False)
    get_mapping_stats(df, path)

def find_explosive_links_to_blacklist(path: str = "data/az/"):

    limit_children, limit_descendants = 10, 1000
    # add pages to blacklist that have #children > 10 & #descendants > 1000
    blacklist = pd.read_csv(os.path.join(path, "blacklist.csv"))#["link"].unique()
    mapping_with_children = pd.read_csv(os.path.join(path, "analysis", "mapping_with_children.csv"))
    explosive_links = mapping_with_children.loc[(mapping_with_children["#children"] > limit_children) & (mapping_with_children["#descendants"] > limit_descendants)]
    remove = []
    for l, link in explosive_links.sort_values(by="depth", ascending=False).iterrows():
        if link["depth"] < 3 or link["#descendants"] <= limit_descendants: continue
        remove.append(link["link"])
        parent_link = link["parent"]
        print(link["link"], link["parent"])
        while parent_link is not None and parent_link in mapping_with_children:
            mapping_with_children.loc[mapping_with_children["link"] == parent_link, "#descendants"] -= link["#descendants"]
            parent_link = mapping_with_children.loc[mapping_with_children["link"] == parent_link, "parent"].tolist()[0]
    if len(remove) > 0:
        logging.warning(f"Explosive links detected: {remove}")
        blacklist = pd.concat([blacklist, explosive_links.loc[explosive_links["link"].isin(remove), "link"]])
        blacklist.drop_duplicates(keep="first").to_csv(os.path.join(path, "blacklist.csv"), index=False)
        return True
    return False

def analyze_scraped_data_az(path: str = "data/az/"):
    rerun = False
    Path(os.path.join(path, "analysis")).mkdir(exist_ok=True)
    process_and_visualize_mapping(path)
    # rerun scraper if new blacklist links where found
    if find_explosive_links_to_blacklist(path): rerun = True
    return rerun

def analyze_running_az(path: str = "data/az/"):

    return

    # run main nlu training data through backup and get stats ?