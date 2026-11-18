"""Paired ABR experiment. Run with Python and numpy. Plotting uses Pillow."""
import argparse
import bisect
import csv
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from inventory import ROOT, SEED, prepare, write_csv


RATES = (.30,.75,1.20,1.85,2.85,4.30)
BUFFER_EDGES = (4,8,16,32)
UPSTREAM_REVISIONS = {
    "merina": "dfd01b7f664e3a9fa0d123945081f7af2ebb31f7",
    "traces": "d28c4fd48d0e358db024cc0217b111c070815fa7",
}
CHECKPOINTS = (1,3,6,12,24,40)
FIELDS = ("qoe","bitrate_mbps","stall_s","switch_mbps")


def make_env(trace, environment, video_prefix):
    t, b = trace
    return environment([t], [b], ["replay"], video_prefix)


def state(speed, buffer, prev):
    return (bisect.bisect_right(RATES[1:],speed),bisect.bisect_right(BUFFER_EDGES,buffer),prev)


def buffer_action(buffer):
    target = RATES[0] + (RATES[-1]-RATES[0])*min(1,max(0,(buffer-4)/12))
    return bisect.bisect_right(RATES,round(target,12))-1


def play(env, policy, q=None, fixed=0, rng=None, epsilon=0, details=False):
    obs, prev, buffer, network_time, wall = None, 0, 0., 0., 0.
    rewards, rates, stalls, switches, log = [], [], [], [], []
    for chunk in range(env.total_chunk_num):
        if obs is None:
            a = 0
        elif policy == "fixed":
            a = fixed
        elif policy == "buffer":
            a = buffer_action(buffer)
        elif rng is not None and rng.random() < epsilon:
            a = rng.randrange(6)
        else:
            a = int(np.argmax(q[obs]))
        delay_ms, _, new_buffer, stall, size, _, done, _, _ = env.get_video_chunk(a)
        delay = float(delay_ms)/1000
        speed = 8*size/1e6/delay
        switch = abs(RATES[a]-RATES[prev]) if chunk else 0.
        reward = RATES[a]-4.3*stall-switch
        new_obs = state(speed,new_buffer,a)
        if rng is not None and obs is not None:
            target = reward if done else reward+.95*np.max(q[new_obs])
            q[obs+(a,)] += .1*(target-q[obs+(a,)])
        # Upstream advances the trace for payload transfer and buffer drain, but not RTT.
        sleep = max(0.,max(0.,buffer-delay)+4-new_buffer)
        if details:
            log.append(dict(chunk=chunk+1,decision_wall_s=wall,network_time_s=network_time,buffer_before_s=buffer,
                            action=a,bitrate_mbps=RATES[a],download_s=delay,measured_mbps=speed,
                            stall_s=float(stall),switch_mbps=switch,qoe=float(reward)))
        wall += delay+sleep
        network_time += delay-.080+sleep
        obs, prev, buffer = new_obs,a,float(new_buffer)
        rewards.append(reward); rates.append(RATES[a]); stalls.append(stall); switches.append(switch)
    metrics = (mean(rewards), mean(rates), sum(stalls), mean(switches[1:]))
    return dict(zip(FIELDS, map(float, metrics))), log


def checks(q, fixed, environment, video_prefix):
    assert state(.75,4,0)==(1,1,0) and state(4.30,32,5)==(5,4,5)
    assert buffer_action(4)==0 and buffer_action(16)==5
    for policy in ("fixed","buffer","q"):
        def history(ts,bs):
            return play(make_env((ts,bs), environment, video_prefix),policy,q,fixed,details=True)[1]
        original = history([0,10000],[2,2])
        cutoff = original[9]["network_time_s"]
        slow = history([0,cutoff,10000],[2,2,.25])
        fast = history([0,cutoff,10000],[2,2,12])
        for a,b in zip(slow[:9],fast[:9]):
            assert a["action"]==b["action"] and math.isclose(a["download_s"],b["download_s"])
        assert slow[9]["action"]==fast[9]["action"] and slow[9]["buffer_before_s"]==fast[9]["buffer_before_s"]
        assert slow[9]["download_s"]>fast[9]["download_s"]


def run(out, upstream):
    revisions = {
        name: subprocess.check_output(
            ["git", "-C", str(upstream / name), "rev-parse", "HEAD"], text=True
        ).strip()
        for name in UPSTREAM_REVISIONS
    }
    if revisions != UPSTREAM_REVISIONS:
        raise ValueError(f"Expected upstream revisions {UPSTREAM_REVISIONS}, got {revisions}")
    env_dir = upstream / "merina" / "envs"
    sys.path.insert(0, str(env_dir))
    from fixed_env_log import Environment

    video_prefix = str(env_dir / "video_size" / "Mao" / "video_size_")
    out.mkdir(parents=True, exist_ok=False)
    kept,series = prepare(out, upstream / "traces")
    subsets = {s:[r for r in kept if r["split"]==s] for s in ("train","validation","test")}
    envs = {k:make_env(v, Environment, video_prefix) for k,v in series.items()}
    def evaluate(split,policy,q=None,fixed=0,details=False):
        result,logs = [],{}
        for r in subsets[split]:
            m,log = play(envs[r["path"]],policy,q,fixed,details=details)
            result.append(dict(source=r["source"],capture=r["capture"],path=r["path"],policy=policy,**m))
            if details: logs[r["path"]] = log
        return result,logs
    validation = []
    def score(policy,setting,**kwargs):
        result,_ = evaluate("validation",policy,**kwargs)
        value = mean(r["qoe"] for r in result)
        validation.append(dict(policy=policy,setting=setting,mean_trace_qoe=value))
        return value
    fixed = max(range(6), key=lambda a:score("fixed",RATES[a],fixed=a))
    q = np.zeros((6,5,6,6))
    best_score = -float("inf")
    rng = random.Random(SEED)
    train = subsets["train"].copy()
    for epoch in range(1,max(CHECKPOINTS)+1):
        rng.shuffle(train)
        for r in train:
            play(envs[r["path"]],"q",q,rng=rng,epsilon=max(.05,.5*.92**(epoch-1)))
        if epoch in CHECKPOINTS:
            value = score("q",epoch,q=q)
            if value > best_score:
                best_score,best_q,best_epoch = value,q.copy(),epoch
    checks(best_q, fixed, Environment, video_prefix)
    np.save(out/"q_table.npy",best_q)
    write_csv(out/"validation.csv",validation)
    (out/"selection.json").write_text(json.dumps(dict(fixed_mbps=RATES[fixed],epochs=best_epoch,seed=SEED,commits=revisions,
        chunks=next(iter(envs.values())).total_chunk_num,causality_check="passed for all 3 policies"),indent=2))
    all_rows, histories = [],{}
    frozen = best_q.copy()
    for p in ("fixed","buffer","q"):
        rr,histories[p] = evaluate("test",p,best_q,fixed,True)
        all_rows.extend(rr)
    assert np.array_equal(frozen,best_q)
    write_csv(out/"per_trace.csv",all_rows)
    summary = []
    for src in ("FCC","HSDPA"):
        for p in histories:
            rr = [r for r in all_rows if r["source"]==src and r["policy"]==p]
            summary.append(dict(source=src,policy=p,traces=len(rr),**{m:mean(r[m] for r in rr) for m in FIELDS}))
    write_csv(out/"summary.csv",summary)
    spreads = defaultdict(list)
    for r in all_rows: spreads[r["path"]].append(r["qoe"])
    case = max(sorted(spreads),key=lambda k:max(spreads[k])-min(spreads[k]))
    case_rows = [dict(path=case,policy=p,**step) for p in histories for step in histories[p][case]]
    write_csv(out/"case_chunks.csv",case_rows)
    from case_plot import plot
    plot(out,case,series[case],histories)
    print((out/"counts.csv").read_text(),(out/"summary.csv").read_text(),case,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=ROOT / "upstream")
    args = parser.parse_args()
    run(args.out.resolve(), args.upstream.resolve())
