"""Create an explicit, small research evidence bundle; never sends or publishes it."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent;DEST=ROOT.parent/'kforcing-research-handoff'
ARCHIVE=ROOT.parent/'kforcing-research-handoff.zip'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    assert not DEST.exists() and not ARCHIVE.exists(), 'Refusing to overwrite handoff'
    cost=read(ROOT/'results/compute-budget/manifest.json')
    names=set(cost['input_hashes'])|{'results/compute-budget/'+n for n in cost['outputs']}
    names|={'results/compute-budget/manifest.json','results/compute-budget/integrity_audit.json',
        'verify_compute_budget.py','verify_research_evidence.py','研究交付摘要.md','checkpoint_manifest.json',
        'STAGE10_PROTOCOL.md','STAGE16_PROTOCOL.md','DECISION_CONFIRM_PLAN.md',
        'results/decision-confirm/locked_model.json'}
    names|={'results/baseline-016/'+n for n in ['analysis.json','scores.jsonl','prefixes.json']}
    names|={'results/early-010/'+n for n in ['analysis.json','scores.jsonl','samples.jsonl','timings.jsonl','locked_calibration.json','calibration.jsonl','frozen_model.json']}
    # Prior scientific inputs must agree with the last frozen research snapshot.
    prior=read(ROOT/'compute_budget_file_hashes.json')
    for name in names:
        if name in prior:assert sha(ROOT/name)==prior[name],name
    assert all(not n.startswith('results/blind-quality/') for n in names)
    DEST.mkdir()
    origins={}
    for name in sorted(names):
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        origins[name]=dict(project_relative_path=name,sha256=sha(ROOT/name),byte_identical=True)
    shutil.copyfile(ROOT/'RESEARCH_HANDOFF_README.md',DEST/'README.md')
    origins['README.md']=dict(project_relative_path='RESEARCH_HANDOFF_README.md',sha256=sha(ROOT/'RESEARCH_HANDOFF_README.md'),byte_identical=True)
    (DEST/'requirements-evidence.txt').write_text('numpy==2.5.3\n',encoding='utf-8')
    prov=read(ROOT/'provenance.json')
    dump(DEST/'sources.json',{k:prov[k] for k in ['upstream_url','upstream_commit','checkpoint_repo','checkpoint_revision','paper']})
    dump(DEST/'review_status.json',dict(as_of_utc=datetime.now(timezone.utc).isoformat(),human_ratings_received=0,
        human_quality='not_evaluated',reviewer_availability='User reports no reviewers currently available.',
        ai_ratings_counted_as_human=0,blind_review_packet_included=False,method_labeled_outputs_included=True,
        not_for_blind_reviewers=True,externally_sent=False))
    definitions=[
        ('baseline_gap','Local AR/K4 baseline scores remain unmatched to the original paper configuration.',
         'results/baseline-016/analysis.json',['/sources','/methods/ar/gen_ppl','/methods/ar/gen_ppl_interval95','/methods/fixed4/gen_ppl','/methods/fixed4/gen_ppl_interval95','/paper_reference_not_hypothesis_test']),
        ('early_online_failure','The earlier hidden82 online decoder failed the joint budget/performance gate; this is not an online test of combined19.',
         'results/early-010/analysis.json',['/source_clusters','/summary/learned/gpt2_gen_ppl','/summary/random_cal/gpt2_gen_ppl','/primary/ppl_ratio','/primary/visible_throughput_ratio','/budget_calls_per_token_ratio','/primary_budget_and_performance_gate']),
        ('later_offline_failure','The frozen combined19 selector failed the three-comparison confirmation gate.',
         'results/decision-confirm/analysis.json',['/test_states','/test_sources','/primary/vs_random','/primary/vs_confidence14','/primary/vs_heuristic','/advance_gate_passed']),
        ('cost_sensitivity','Retrospective cost normalization does not establish an incremental combined19 benefit over confidence14.',
         'results/compute-budget/analysis.json',['/cost_normalization/calls/comparisons/combined19_vs_confidence14','/cost_normalization/candidates/comparisons/combined19_vs_confidence14','/cost_normalization/calls/posthoc_cost_calibration','/cost_normalization/candidates/posthoc_cost_calibration','/wall_clock_measured']),
        ('human_quality_pending','No human quality conclusion is available; reviewers are currently unavailable.',
         'review_status.json',['/human_ratings_received','/human_quality','/reviewer_availability','/externally_sent'])]
    claims=[]
    for ident,statement,file,pointers in definitions:
        data=read(DEST/file);refs=[]
        for pointer in pointers:
            value=data
            for k in pointer.strip('/').split('/'):value=value[k]
            refs.append(dict(file=file,pointer=pointer,value=value))
        claims.append(dict(id=ident,statement=statement,evidence=refs))
    dump(DEST/'research_claims.json',dict(scope='Existing records only; cohorts and model versions must not be pooled.',claims=claims))
    dump(DEST/'SOURCE_ORIGINS.json',origins)
    payload={p.relative_to(DEST).as_posix():sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    dump(DEST/'PAYLOAD_SHA256.json',payload)
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None and set(z.namelist())==set(payload)|{'PAYLOAD_SHA256.json'}
        for name,h in payload.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    print(json.dumps(dict(archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,files=len(payload)+1,sha256=sha(ARCHIVE),
        source_files_copied=len(origins),requires=['Python','NumPy'],model_weights_included=False,
        blind_review_key_included=False,method_labeled_outputs_included=True,not_for_blind_reviewers=True,
        external_sends=0),indent=2))

if __name__=='__main__':main()
