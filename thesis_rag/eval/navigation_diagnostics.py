"""Offline diagnostics over frozen known-page rankings. No model calls."""

from collections import deque
from dataclasses import dataclass
from statistics import mean, median
from types import SimpleNamespace

from thesis_rag.adapters.page_benchmark import verified_page_slots
from thesis_rag.artifact_inputs import VerifiedRepositoryFiles, jsonl_records
from thesis_rag.corpus.page_scopes import load_scope_graph
from thesis_rag.interfaces import RetrievalResult


DEPTHS = (0, 1, 3, 5, 10, None)


@dataclass
class NavigationInputs:
    files: VerifiedRepositoryFiles
    suite: dict
    snapshot: dict
    questions: dict

    @classmethod
    def load(cls, root, suite_path, suite_hash):
        files = VerifiedRepositoryFiles(root)
        suite = files.object(suite_path, suite_hash)
        ref = suite['corpus']
        snapshot = files.object(ref['snapshot_manifest_path'], ref['snapshot_manifest_hash'])
        ref = suite['question_set']
        records = list(jsonl_records(files.path(ref['questions_path'], ref['questions_hash'])))
        questions = {q['query_id']: q for q in records}
        if len(questions) != len(records) or len(questions) != ref['question_count']:
            raise ValueError('Invalid navigation query inventory')
        if any(q['metadata']['corpus_hash'] != snapshot['corpus_hash'] for q in records):
            raise ValueError('Navigation questions belong to a different corpus')
        return cls(files, suite, snapshot, questions)

    def rankings(self, experiment_ids):
        """Preserve native ranks, including unmapped slots and short rankings."""
        output = {}
        definitions = {e['experiment_id']: e for e in self.suite['experiments']}
        for ref in self.suite['experiment_runs']:
            eid = ref['experiment_id']
            if eid not in experiment_ids:
                continue
            run = self.files.object(ref['run_manifest_path'], ref['run_manifest_hash'])
            path = self.files.path(run['output_path'] + '/retrieval_results.jsonl',
                                   run['output_hashes']['retrieval_results'])
            ranks = {}
            for raw in jsonl_records(path):
                result = RetrievalResult.from_dict(raw)
                if (result.query_id not in self.questions or result.query_id in ranks
                        or result.corpus_hash != self.snapshot['corpus_hash']
                        or result.config_hash != definitions[eid]['resolved_system_config_hash']):
                    raise ValueError('Navigation result identity mismatch')
                ranks[result.query_id] = verified_page_slots(result, requested_depth=20)
            if set(ranks) != set(self.questions):
                raise ValueError('Incomplete navigation rankings')
            output[eid] = ranks
        if set(output) != set(experiment_ids):
            raise ValueError('Missing navigation experiment')
        return output


def descendant_distances(graph, roots):
    """Traverse raw intents, then merge canonical pages at their shortest depth.

    The original concrete roots are retained at depth zero. Descendants admit
    both indexed language versions, as in the existing Priority scope graph.
    """
    pages = {pid: 0 for pid, _ in roots}
    for _, intent in sorted(roots):
        seen = {intent}
        pending = deque([(intent, 0)])
        while pending:
            parent, depth = pending.popleft()
            for child in sorted(graph.children.get(parent, ())):
                if child in seen:
                    continue
                seen.add(child)
                pending.append((child, depth + 1))
                for pid in graph.intent_pages.get(child, ()):
                    pages[pid] = min(pages.get(pid, depth + 1), depth + 1)
    return pages


def accepted_target_metrics(rankings, distances, depths=DEPTHS, cutoffs=(1, 5, 20)):
    """Expand only the acceptable targets, never the retrieved page ranking."""
    rows = []
    for eid, queries in rankings.items():
        if set(queries) != set(distances):
            raise ValueError('Ranking and target cohorts differ')
        for depth in depths:
            first = []
            for qid, ranking in queries.items():
                valid = {pid for pid, d in distances[qid].items() if depth is None or d <= depth}
                first.append(next((i for i, pid in enumerate(ranking, 1) if pid in valid), 21))
            for k in cutoffs:
                hits = sum(rank <= k for rank in first)
                rows.append(dict(experiment=eid, depth='unlimited' if depth is None else depth,
                                 cutoff=k, queries=len(first), hits=hits, hit_rate=hits / len(first)))
    return rows


def target_expansion(inputs, *, mapping_source, chunker_id):
    policy = SimpleNamespace(mapping_source=mapping_source, chunker_id=chunker_id)
    graph = load_scope_graph(inputs.snapshot, policy, inputs.files).graph
    cached, distances = {}, {}
    for qid, query in inputs.questions.items():
        roots = frozenset((row['page_id'], row['intent'] if row['resolution'] == 'intent'
                          else graph.mapping.pages[row['page_id']]['intent'])
                         for row in query['metadata']['source_rows'])
        if {pid for pid, _ in roots} != {t['page_id'] for t in query['acceptable_targets']}:
            raise ValueError('Historical targets differ from their source rows')
        if roots not in cached:
            cached[roots] = descendant_distances(graph, roots)
        distances[qid] = cached[roots]
    sizes = []
    for depth in DEPTHS:
        counts = [sum(depth is None or d <= depth for d in ds.values()) for ds in distances.values()]
        sizes.append(dict(depth='unlimited' if depth is None else depth, queries=len(counts),
                          mean_pages=mean(counts), median_pages=median(counts), max_pages=max(counts)))
    return distances, sizes
