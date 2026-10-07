"""Post-rollout RC from the last SDF-one prefix, checked every travelled metre.

Final SDF-one episodes use the complete path. Final SDF-zero episodes scan
forward once; the first zero permanently freezes the last valid prefix match.
Original polyline vertices are preserved. Only the cut endpoint is interpolated.
Runtime and offline validation share this implementation; binary SDF is unchanged.
"""
from __future__ import annotations
import numpy as np

from .sdroute_sdf import endpoint_scored_edges as _endpoint_scored_edges
from .sdroute_prefix import arc, score_prefix


class MatcherCache:
    """Reuse identical HMM query geometry; endpoint scoring still runs each time."""
    def __init__(self,g,edges,matcher):
        self.g,self.edges,self.matcher=g,edges,matcher
        self.cache={};self.calls=0;self.hits=0

    def verdict(self,points):
        q=self.matcher.resample(points)
        key=(q.shape,q.tobytes())
        if key not in self.cache:
            self.cache[key]=self.matcher.match(self.g,points);self.calls+=1
        else:self.hits+=1
        matched=self.cache[key]
        if matched is None:
            return None,None,dict(start=[],end=[]),None
        seq=[int(e) for e in matched['edges']]
        scored,ignored=_endpoint_scored_edges(self.g,self.edges,seq,points)
        route=set(self.edges)
        first=next((e for e in scored if e not in route),None)
        return int(first is None),seq,ignored,first


def crop_at_distance(points,times,distance):
    """Cut the original polyline, preserving bends and interpolating one endpoint."""
    p,t=np.asarray(points,float),np.asarray(times,float)
    a=arc(p);distance=float(distance)
    if distance<0 or distance>a[-1]+1e-7:
        raise ValueError('cut distance outside executed trajectory')
    if distance>=a[-1]-1e-8:
        return p.copy(),t.copy()  # includes stationary/irregular terminal pose
    j=int(np.searchsorted(a,distance,side='left'))
    if abs(a[j]-distance)<1e-8:
        return p[:j+1].copy(),t[:j+1].copy()
    w=(distance-a[j-1])/(a[j]-a[j-1])
    return (np.vstack([p[:j],p[j-1]+w*(p[j]-p[j-1])]),
            np.r_[t[:j],t[j-1]+w*(t[j]-t[j-1])])


def boundaries(points,times,*,spacing_m=1.,grid='metres',limit=None):
    p,t=np.asarray(points,float),np.asarray(times,float)
    total=float(arc(p)[-1]);limit=total if limit is None else min(total,float(limit))
    if grid=='frames':
        a=arc(p)
        for k in range(1,len(p)):
            if a[k]>limit+1e-8:break
            yield float(a[k]),p[:k+1].copy(),t[:k+1].copy()
        if limit>0 and not np.any(np.abs(a-limit)<1e-8):
            pp,tt=crop_at_distance(p,t,limit);yield limit,pp,tt
        return
    if grid!='metres' or not np.isfinite(spacing_m) or spacing_m<=0:
        raise ValueError('positive metre spacing and a supported grid required')
    marks=list(np.arange(spacing_m,limit+1e-8,spacing_m))
    if not marks or limit-marks[-1]>1e-8:marks.append(limit)
    for u in marks:
        pp,tt=crop_at_distance(p,t,float(u));yield float(u),pp,tt


def scan_rc(g,edges,edge_ref,reference,points,timestamps,matcher,*,
            departed=False,spacing_m=1.,grid='metres',baseline=None,cache=None,
            departure_distance_m=30.):
    """Score with an immutable final verdict and a first-zero sequential scan.

    `baseline` is the existing full-path score_prefix result; pass it to share
    comparisons. Missing matcher results are not SDF-zero. Early too-short
    prefixes are skipped; a later matching failure freezes credit explicitly.
    """
    p,t=np.asarray(points,float),np.asarray(timestamps,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)!=len(t) or len(p)<2:
        raise ValueError('at least two XY poses with matching timestamps required')
    if not np.isfinite(p).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('finite coordinates and strictly increasing time required')
    if grid not in ('metres','frames') or not np.isfinite(spacing_m) or spacing_m<=0:
        raise ValueError('invalid scan grid or spacing')
    if not np.isfinite(departure_distance_m) or departure_distance_m <= 0:
        raise ValueError('positive finite departure distance required')
    cache=MatcherCache(g,edges,matcher) if cache is None else cache
    calls_before,hits_before=cache.calls,cache.hits
    final_value,final_edges,_,final_off=cache.verdict(p)
    sdf=0 if departed else final_value
    if baseline is None:
        baseline,_=score_prefix(g,edges,edge_ref,reference,p,t,matcher,departed=departed)
    if baseline['sdf']!=sdf:
        raise ValueError('full SDF differs between original and traced matcher')
    total=float(arc(p)[-1])
    out=dict(sdf=sdf,rc=None,baseline_rc=baseline['rc'],status='unavailable',
             grid=grid,spacing_m=spacing_m if grid=='metres' else None,
             reference_m=float(reference.length),total_model_m=total,
             final_time_s=float(t[-1]),cutoff_model_m=None,cutoff_time_s=None,
             first_zero_model_m=None,first_zero_time_s=None,first_zero_edge=None,
             final_first_off_edge=final_off,last_good_edges=[],checks=[],
             rc_guard_reason=None,departure_bracket_s=None)

    def finish():
        out['prefix_checks']=len(out['checks'])
        out['matcher_calls']=cache.calls-calls_before
        out['matcher_cache_hits']=cache.hits-hits_before
        if out['rc'] is not None:
            out['credited_m']=float(out['rc']*reference.length)
        return out

    if sdf==1:
        out.update(status='full_sdf_one',rc=baseline['rc'],
                   cutoff_model_m=total,
                   cutoff_time_s=float(t[-1]),rc_guard_reason=baseline.get('cutoff_reason'),
                   last_good_edges=final_edges)
        return finish()
    if sdf is None:
        out['status']='final_sdf_unavailable';return finish()

    # Inspect raw jumps BEFORE interpolation can fill them with fake 1m poses.
    raw_arc=arc(p);movement=np.diff(raw_arc)
    bad=np.flatnonzero((movement>25.+1e-8)|(movement>50.*np.diff(t)+1e-8))
    limit=total;limit_reason='rollout_end'
    if len(bad):limit=float(raw_arc[bad[0]]);limit_reason='physical_jump'
    if departed:
        distances=np.array([abs(reference.route.project(x).d) for x in p])
        cross=np.flatnonzero(distances>departure_distance_m)
        if len(cross):
            k=int(cross[0]);before=max(0,k-1)
            out['departure_bracket_s']=[float(t[before]),float(t[k])]
            if raw_arc[before]<=limit:
                limit=float(raw_arc[before]);limit_reason='departure_30m'
        elif final_off is None:
            # The matcher judged the whole drive to be on the route (final_off is None). The run
            # ended as a departure because it got 30 m from the end of the scored segment, not
            # because it left the route: the engine measures distance to reference.crop_xy() --
            # the [s_start, s_end] the GT was matched to -- so a car that completes the segment and
            # keeps driving moves away from the end point and crosses that limit. The
            # abs(route.project(d)) measured here is against the whole route, so it cannot locate
            # that moment.
            #
            # Discarding RC because the boundary could not be located would erase the whole drive.
            # Use the same value as the sdf==1 branch -- baseline['rc'], which the matcher already
            # computed for this trajectory. RC is not invented as 1: "was on the route" differs from
            # "covered the whole route", and the covered amount is what baseline measured. SDF
            # stays 0 per the departed verdict and term_reason stays route_deviation -- the only
            # change is that RC is not abandoned.
            out.update(status='departure_past_span_end',rc=baseline['rc'],
                       cutoff_model_m=total,cutoff_time_s=float(t[-1]),
                       rc_guard_reason=baseline.get('cutoff_reason'),
                       last_good_edges=final_edges)
            return finish()
    last_good=None
    status=limit_reason
    for distance,pp,tt in boundaries(p,t,spacing_m=spacing_m,grid=grid,limit=limit):
        value,seq,ignored,off=cache.verdict(pp)
        out['checks'].append(dict(model_m=distance,time_s=float(tt[-1]),sdf=value,
                                  first_off_edge=off,ignored_start=ignored['start'],ignored_end=ignored['end']))
        if value is None:
            if last_good is not None:
                status='prefix_match_failed';break
            # Insufficient motion is the only initial failure we may skip.
            if distance>=matcher.STEP+1e-7:
                status='prefix_match_failed';break
            continue
        if value==0:
            out.update(first_zero_model_m=distance,first_zero_time_s=float(tt[-1]),first_zero_edge=off)
            status='first_prefix_zero';break
        last_good=(distance,pp,tt,seq)
    if last_good is None:
        out.update(status='no_valid_prefix' if status=='first_prefix_zero' else status,
                   rc=0. if status!='prefix_match_failed' else None,
                   cutoff_model_m=0.,cutoff_time_s=float(t[0]))
        return finish()
    distance,pp,tt,seq=last_good
    # Reconstruct ONLY the saved last-good prefix; never reuse final trace labels.
    good_result,good_trace=score_prefix(g,edges,edge_ref,reference,pp,tt,matcher)
    if good_result['sdf']!=1 or good_trace['edges']!=seq:
        raise ValueError('last-good prefix trace does not match original SDF')
    out.update(status=status,rc=good_result['rc'],cutoff_model_m=distance,
               cutoff_time_s=float(tt[-1]),last_good_edges=seq,
               rc_guard_reason=good_result.get('cutoff_reason'),
               credited_sd_s=good_result.get('cutoff_sd_s'),
               trace_progress=good_result.get('trace_progress',[]))
    if status=='rollout_end':
        # A final binary zero without a matcher zero needs an observed termination.
        out.update(status='no_zero_found',rc=None)
    return finish()
