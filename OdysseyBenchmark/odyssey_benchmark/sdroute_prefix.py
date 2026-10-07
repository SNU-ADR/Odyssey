"""SDF-first continuous-prefix RC shared by runtime and offline validation.

The existing SDF HMM supplies route identity. This adapter retains its selected
states and stitched edge occurrences, then freezes progress at the first scored
off-route edge. Later recovery never adds credit. Original SD-polyline metres
remain the numerator and denominator; HD polygons are not used.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from odyssey.manager.sdroute_progress import build_reference
from .sdroute_sdf import endpoint_scored_edges as _endpoint_scored_edges


def arc(xy):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def traced_match(g, points, matcher):
    """SDF Viterbi with sample/transition provenance retained.

    Identical costs, sampling and stitching to matcher.match when all samples
    have candidates. Missing-candidate samples retain their original indices;
    unlike the old matcher they cannot shift the observation coordinates.
    """
    m = matcher
    q = m.resample(np.asarray(points, float), m.STEP)
    if len(q) < 2:
        return None
    psi = m.headings(q)
    states, indices, missing = [], [], []
    for k, (p, heading) in enumerate(zip(q, psi)):
        candidates = []
        for edge in g.candidates(p, m.RADIUS):
            cost, s, d, angle = m.emission(g, edge, p, heading)
            candidates.append((cost, int(edge), s, d, angle))
        candidates.sort(key=lambda x: x[0])
        if candidates:
            states.append(candidates[:m.TOPK]); indices.append(k)
        else:
            missing.append(k)
    if not states or indices[0] != 0:
        return None
    values = [[c[0] for c in states[0]]]
    backs = [[-1] * len(states[0])]
    for k in range(1, len(states)):
        distance = float(np.linalg.norm(q[indices[k]] - q[indices[k-1]]))
        row, back = [], []
        for cost, edge, s, _, _ in states[k]:
            best, arg = np.inf, -1
            for i, (_, prev_edge, prev_s, _, _) in enumerate(states[k-1]):
                if not np.isfinite(values[-1][i]):
                    continue
                gd = m.graph_dist(g, (prev_edge, prev_s), (edge, s))
                if not np.isfinite(gd) or gd > 3. * distance + 60.:
                    continue
                value = values[-1][i] + abs(gd-distance) / m.BETA
                if value < best:
                    best, arg = value, i
            row.append(best+cost); back.append(arg)
        if not np.isfinite(min(row)):
            row = [c[0] for c in states[k]]
            back = [-1] * len(row)
        values.append(row); backs.append(back)
    chosen = []
    i = int(np.argmin(values[-1]))
    for k in range(len(states)-1, -1, -1):
        _, edge, s, d, angle = states[k][i]
        predecessor = backs[k][i]
        chosen.append(dict(sample=indices[k], edge=edge, edge_s=float(s),
                           distance=float(d), heading_error=float(angle),
                           restart=bool(k > 0 and predecessor < 0)))
        i = predecessor
        if i < 0 and k > 0:
            i = int(np.argmin(values[k-1]))
    chosen.reverse()
    edges = [chosen[0]['edge']]
    events = [dict(edge=edges[0], left=0, right=0, inferred=False)]
    chosen[0]['event'] = 0
    gaps = []
    for k, state in enumerate(chosen[1:], 1):
        edge = state['edge']
        if edge != edges[-1]:
            path = m.edge_path(g, int(g.v[edges[-1]]), int(g.u[edge]))
            if path is None:
                gaps.append(state['sample'])
                path = []
            for joined in list(path) + [edge]:
                edges.append(int(joined))
                events.append(dict(edge=int(joined), left=chosen[k-1]['sample'],
                                   right=state['sample'], inferred=int(joined) != edge))
        state['event'] = len(edges)-1
    return dict(q=q, query_arc=np.arange(len(q))*m.STEP, states=chosen,
                edges=edges, events=events, missing=missing, gaps=gaps)


@dataclass
class EdgeReference:
    edges: list
    offsets: np.ndarray
    ends: np.ndarray
    graph_s: np.ndarray
    sidecar_s: np.ndarray
    max_alignment_distance_m: float

    def sd_s(self, occurrence, edge_s):
        return float(np.interp(self.offsets[occurrence] + edge_s,
                               self.graph_s, self.sidecar_s))

    def initial_occurrence(self, edge, s_start):
        candidates = [i for i, e in enumerate(self.edges) if e == edge]
        if not candidates:
            return None
        def distance(i):
            lo, hi = self.sd_s(i, 0), self.sd_s(i, self.ends[i]-self.offsets[i])
            return max(lo-s_start, s_start-hi, 0.)
        best = min(candidates, key=distance)
        # A state before the crop is harmless: it earns nothing until entering
        # the crop. Only a first state beyond the crop start can skip its prefix.
        return best if self.sd_s(best, 0.) <= s_start + 8. else None


def build_edge_reference(g, edges, sidecar_xy):
    """Map each directed-edge OCCURRENCE to the persisted SD polyline's s.

    Reference-only ordered alignment handles cropped starts and repeated edges.
    No model trace is involved. Nonrepresentable manual IDs fail explicitly.
    """
    if any(e < 0 or e >= len(g) for e in edges):
        raise ValueError('unsupported_reference_edge')
    points, offsets, ends = [], [], []
    length = 0.
    for edge in edges:
        geom = np.asarray(g.geom[edge], float)
        if points:
            length += float(np.linalg.norm(geom[0]-points[-1]))
        offsets.append(length)
        for point in geom:
            if not points or np.linalg.norm(point-points[-1]) > 1e-8:
                points.append(point)
        length += float(arc(geom)[-1])
        ends.append(length)
    match = build_reference(np.asarray(points), sidecar_xy)
    graph_s = np.array([p.s for p in match.gt_matches])
    sidecar_s = arc(sidecar_xy)
    keep = np.r_[True, np.diff(graph_s) > 1e-7]
    max_distance = max(abs(p.d) for p in match.gt_matches)
    if max_distance > 3. or keep.sum() < 2:
        raise ValueError(f'reference_alignment_failed:{max_distance:.3f}')
    return EdgeReference(list(edges), np.asarray(offsets), np.asarray(ends),
                         graph_s[keep], sidecar_s[keep], float(max_distance))


def terminal_progress(g, edges, edge_ref, occurrence, state, point, heading,
                      tail_distance, matcher):
    """Observe the <5 m residual tail without rerunning/relabeling SDF.

    Choose one terminal state using the existing emission/transition costs.
    Progress can extend only through consecutive reference edges. An off-route
    terminal hypothesis earns at most the preceding reference junction. This
    preserves SDF's endpoint deferral while avoiding truncation at the old edge.
    """
    if matcher is None:  # A supplied diagnostic trace may omit its matcher.
        return edge_ref.sd_s(occurrence, g.project(state['edge'], point)), 'same_edge'
    candidates = []
    for e in g.candidates(point, matcher.RADIUS):
        cost,s,_,_ = matcher.emission(g,e,point,heading)
        gd = matcher.graph_dist(g,(state['edge'],state['edge_s']),(int(e),s))
        if np.isfinite(gd) and gd <= 3.*tail_distance + 8.:
            candidates.append((cost+abs(gd-tail_distance)/matcher.BETA,int(e),s))
    if not candidates:
        return edge_ref.sd_s(occurrence,state['edge_s']), 'no_terminal_candidate'
    _,edge,s = min(candidates,key=lambda x:x[0])
    if edge == state['edge']:
        return edge_ref.sd_s(occurrence,s), 'same_edge'
    path = matcher.edge_path(g,int(g.v[state['edge']]),int(g.u[edge]))
    if path is None:
        return edge_ref.sd_s(occurrence,state['edge_s']), 'terminal_graph_gap'
    frontier = edge_ref.sd_s(occurrence,float(g.length[state['edge']]))
    for e in list(path)+[edge]:
        if occurrence+1 >= len(edges) or edges[occurrence+1] != e:
            return frontier, 'terminal_branch_deferred'
        occurrence += 1
        frontier = edge_ref.sd_s(occurrence,s if e==edge else float(g.length[e]))
    return frontier, 'next_reference_edge'


def score_prefix(g, edges, edge_ref, reference, points, timestamps, matcher,
                 *, departed=False, trace=None):
    """SDF verdict plus one irreversible RC cutoff from the same trace.

    Additional prefix-quality cuts (broken match, order violation, raw pose
    jump) are reported separately and do not rewrite the binary SDF verdict.
    """
    points, timestamps = np.asarray(points, float), np.asarray(timestamps, float)
    trace = traced_match(g, points, matcher) if trace is None else trace
    if trace is None:
        return dict(status='match_failed', rc=None, sdf=0 if departed else None,
                    cutoff_reason='unavailable', first_off_edge=None), None
    seq = trace['edges']
    scored, ignored = _endpoint_scored_edges(g, edges, seq, points)
    lo, hi = len(ignored['start']), len(seq)-len(ignored['end'])
    route_set = set(edges)
    first_off = next((i for i in range(lo, hi) if seq[i] not in route_set), None)
    sdf = int(first_off is None and not departed)
    movement = np.linalg.norm(np.diff(points, axis=0), axis=1)
    raw_arc = arc(points)
    hard_u, hard_reason = float(raw_arc[-1]), 'rollout_end'
    discontinuous = np.flatnonzero((movement > 25.+1e-8) |
                                  (movement > 50.*np.diff(timestamps)+1e-8))
    if len(discontinuous):
        hard_u = float(raw_arc[discontinuous[0]])
        hard_reason = 'physical_jump'
    for sample, reason in ([(s, 'no_candidate') for s in trace['missing']] +
                           [(s, 'graph_gap') for s in trace['gaps']] +
                           [(s['sample'], 'hmm_restart') for s in trace['states'] if s['restart']]):
        u = float(trace['query_arc'][max(0, sample-1)])
        if u < hard_u:
            hard_u, hard_reason = u, reason
    if departed:
        distances = np.array([abs(reference.route.project(p).d) for p in points])
        crossings = np.flatnonzero(distances > 30.)
        if len(crossings):
            i = max(0, int(crossings[0])-1)
            if raw_arc[i] < hard_u:
                hard_u, hard_reason = float(raw_arc[i]), 'departure_30m'
        else:
            # Historical termination is definitive for SDF, but cannot locate
            # an unseen crossing in a coarse saved trajectory.
            hard_reason = 'departure_boundary_unobserved'
    frontier = float(reference.s_start)
    reason = hard_reason
    occurrence = None
    previous_event = -1
    last_state = None
    cutoff_u = hard_u
    cutoff_bracket = None
    progress = []
    off_seen = False
    terminal_status = 'not_used'
    for state in trace['states']:
        u = float(trace['query_arc'][state['sample']])
        if u > hard_u + 1e-8:
            break
        stop = False
        for event_i in range(previous_event+1, state['event']+1):
            event = trace['events'][event_i]
            if event_i < lo:
                continue
            e = event['edge']
            if event_i == first_off or event_i >= hi:
                # A connected transition locates the route exit at its junction,
                # even if the first off-route edge was inserted between samples.
                if occurrence is not None and int(g.v[edges[occurrence]]) == int(g.u[e]):
                    frontier = max(frontier, edge_ref.sd_s(occurrence, float(g.length[edges[occurrence]])))
                reason = 'first_off_route' if event_i == first_off else 'deferred_endpoint'
                off_seen = event_i == first_off
                cutoff_bracket = [float(trace['query_arc'][event['left']]),
                                  float(trace['query_arc'][event['right']])]
                cutoff_u = cutoff_bracket[0]  # conservative onset for plotting
                stop = True
                break
            if occurrence is None:
                occurrence = edge_ref.initial_occurrence(e, reference.s_start)
                if occurrence is None:
                    reason, cutoff_u, stop = 'start_occurrence_mismatch', 0., True
                    break
            elif occurrence+1 < len(edges) and edges[occurrence+1] == e:
                occurrence += 1
            else:
                reason, cutoff_u, stop = 'route_order_violation', u, True
                break
        previous_event = state['event']
        if stop:
            break
        if occurrence is not None:
            # No credit at the handoff itself, regardless of lane-offset bias.
            if u > 1e-8:
                frontier = max(frontier, edge_ref.sd_s(occurrence, state['edge_s']))
            last_state = state
        progress.append([u, float(np.clip(frontier-reference.s_start, 0., reference.length))])
    # The HMM samples every 5 m and drops the residual tail. Keep its verdict,
    # but observe the terminal pose with the same local matching costs.
    if last_state is not None and reason == hard_reason and occurrence is not None and hard_u > 1e-8:
        p = np.array([np.interp(hard_u, raw_arc, points[:, j]) for j in range(2)])
        last_u = float(trace['query_arc'][last_state['sample']])
        tail = max(0.,hard_u-last_u)
        if tail > 1e-7:
            direction = p-trace['q'][last_state['sample']]
            heading = float(np.arctan2(direction[1],direction[0]))
            final_s,terminal_status = terminal_progress(g,edges,edge_ref,occurrence,
                last_state,p,heading,tail,matcher)
            frontier = max(frontier,final_s)
    frontier = float(np.clip(frontier, reference.s_start, reference.s_end))
    rc = (frontier-reference.s_start)/reference.length
    sample = int(np.searchsorted(raw_arc, cutoff_u, side='right')-1)
    recovered = first_off is not None and any(e in route_set for e in seq[first_off+1:hi])
    record = dict(status='ok', rc=float(rc), sdf=sdf,
                  cutoff_reason=reason, cutoff_model_index=max(0, sample),
                  cutoff_model_arc_m=float(cutoff_u), cutoff_bracket_arc=cutoff_bracket,
                  cutoff_sd_s=frontier, credited_m=frontier-reference.s_start,
                  reference_m=reference.length,
                  first_off_edge=seq[first_off] if first_off is not None else None,
                  first_off_event=first_off, off_reached_before_quality_cut=off_seen,
                  recovered_after_off=bool(recovered), ignored_start=ignored['start'],
                  ignored_end=ignored['end'], matcher_gaps=len(trace['gaps']),
                  missing_candidates=len(trace['missing']),
                  restarts=sum(s['restart'] for s in trace['states']),
                  terminal_refinement=terminal_status,
                  alignment_max_m=edge_ref.max_alignment_distance_m,
                  trace_progress=progress, matched_edges=seq)
    return record, trace
