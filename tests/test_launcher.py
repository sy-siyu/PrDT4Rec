"""Portable fixtures only: no native assets or historical runs decoded."""

from copy import deepcopy

import math

import json

from pathlib import Path

import subprocess

import sys

import numpy as np

import pytest

import torch

from prdt4rec import models, runner, training
from prdt4rec.data.bundles import normalize

def make_fixture(root):
    """Return (strict fixture recipe, portable data root); never train on creation."""
    root = Path(root); data = root / 'data'; data.mkdir(parents=True)
    c = deepcopy(runner.read(Path(__file__).parent / 'fixture_config.json'))
    c.update(profile_id='synthetic_packaging_fixture',registry_record_sha256=None,dataset='fixture',status='synthetic_test_only',blocked_reason=None)
    c['model'].update(hidden_dim=8,layers=1,heads=2,dropout=.1,conv_window=3,horizon=4)
    c['policy'].update(epochs=2,draws_per_epoch=7,batch_size=3,cosine_horizon_updates=12,checkpoint_interval_updates=3)
    c['scorer'].update(hidden_dim=8,layers=1,heads=2,dropout=.1,epochs=2,batch_size=3,checkpoint_interval_updates=2)
    c['weighting']['collector_prior']=dict(A=1/3,B=1/3,C=1/3)
    c['data'].update(split_counts=dict(train=6,validation=4,test=0),pair_counts=dict(train=4,validation=2),preference_rule='fixture_stored_reward>3',stored_feedback_threshold=3.,training_units=dict(mapping='affine',offset=0.,scale=5.,lower_clip=0.,upper_clip=1.))
    c['evaluation'].update(users=2,horizon=4,batch_size=2,reset_block_size=2)
    profile=root/'fixture.json';profile.write_text(json.dumps(c)); assets={}
    def asset(name,arrays):
        path=data/f'{name}.npz';np.savez_compressed(path,**arrays)
        assets[name]=dict(path=path.name,sha256=runner.digest(path),arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})
    feedback=np.zeros((10,4),dtype=np.float32);actions=np.zeros((10,4),dtype=np.int64);mask=np.zeros((10,4),dtype=bool)
    for i,row in enumerate([[4.,5.],[1.],[4.,1.,2.]]*2+[[5.,4.],[1.]]*2):
        feedback[i,:len(row)]=row;actions[i,:len(row)]=np.arange(len(row))+2;mask[i,:len(row)]=True
    asset('corpus',dict(actions=actions,feedback=feedback,mask=mask,users=np.asarray([1,1,1,2,2,2,1,1,2,2],dtype=np.int64),indices=np.arange(10,dtype=np.int64),splits=np.asarray([0]*6+[1]*4,dtype=np.int8),positive_count=((feedback>3)&mask).sum(1).astype(np.float32),initial_context_actions=np.full(10,7,dtype=np.int64),n_users=np.asarray(5),n_items=np.asarray(8),max_length=np.asarray(4)))
    rng=np.random.default_rng(849);users=rng.normal(size=(5,40)).astype(np.float32);users[0]=0.
    asset('features',dict(user_vectors=users,item_vectors=rng.normal(size=(6,40)).astype(np.float32)))
    asset('pairs',dict(train_preferred=np.asarray([0,0,3,3],dtype=np.int64),train_rejected=np.asarray([1,2,4,5],dtype=np.int64),validation_preferred=np.asarray([6,8],dtype=np.int64),validation_rejected=np.asarray([7,9],dtype=np.int64)))
    asset('collectors',dict(indices=np.arange(10,dtype=np.int64),collectors=np.asarray(['A','B','C']*3+['A'])))
    asset('dev_cases',dict(raw_users=np.asarray([101,102],dtype=np.int64),encoded_users=np.asarray([3,4],dtype=np.int64),initial_context_actions=np.asarray([6,7],dtype=np.int64)))
    rewards=np.ones((2,4,6),dtype=np.float32);rewards[0,0]=5.;rewards[1,2]=5.
    asset('fixture_world',dict(raw_users=np.asarray([101,102],dtype=np.int64),rewards=rewards,terminal_after=np.asarray([1,3],dtype=np.int64)))
    b=dict(schema_version=runner.BUNDLE_SCHEMA,dataset='fixture',profile_id=c['profile_id'],assets=assets,split_counts=c['data']['split_counts'],pair_counts=c['data']['pair_counts'],units=dict(training=dict(mapping='affine',offset=0.,scale=5.,lower_clip=0.,upper_clip=1.),evaluation=dict(mapping='identity')),preference=dict(rule=c['data']['preference_rule'],stored_feedback_threshold=3.),evaluation=dict(adapter='fixture_test_v1',cases_asset='dev_cases',phase='development',horizon=4),sealed_final_users=[901])
    (data/'bundle.json').write_text(json.dumps(b));return profile,data

@pytest.fixture
def fixture(tmp_path):
    old=torch.get_num_threads();torch.set_num_threads(1)
    p,d=make_fixture(tmp_path)
    yield p,d,tmp_path/'output'
    torch.set_num_threads(old)

def assert_same(a,b):
    if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a)is type(b) and len(a)==len(b)
        for x,y in zip(a,b):assert_same(x,y)
    else:assert type(a)is type(b) and a==b

def rewrite(p,fn):
    c=runner.read(p);fn(c);p.write_text(json.dumps(c))

def rewrite_asset(d,name,fn):
    b=runner.read(d/'bundle.json');path=d/b['assets'][name]['path']
    with np.load(path,allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
    fn(a);np.savez_compressed(path,**a)
    b['assets'][name]['sha256']=runner.digest(path);b['assets'][name]['arrays']={k:dict(shape=list(v.shape),dtype=str(v.dtype))for k,v in a.items()}
    (d/'bundle.json').write_text(json.dumps(b))

def train(p,d,o):
    runner.prepare(p,d,o);training.scorer(p,d,o,device='cpu');return training.policy(p,d,o,device='cpu')

def test_import_help_no_torch_or_training():
    r=subprocess.run([sys.executable,'-c',"import sys;import prdt4rec.runner;assert 'torch' not in sys.modules;prdt4rec.runner.main(['--help'])"],cwd=runner.PROJECT,text=True,capture_output=True)
    assert r.returncode==0 and '--data-root' in r.stdout

def test_metadata_dry_run_no_output_or_model(fixture,monkeypatch):
    p,d,o=fixture;monkeypatch.setattr(models,'make_model',lambda*a,**k:pytest.fail('model during check'))
    assert runner.main(['check','--dry-run','--profile',str(p),'--data-root',str(d),'--output-root',str(o)])==0
    assert not o.exists()

@pytest.mark.parametrize('mutation',['schema','unknown','loss','schedule','final','prior','beta','budget'])
def test_strict_recipe_schema_and_no_science_drift(fixture,mutation):
    p,d,o=fixture
    def change(c):
        if mutation=='schema':c['schema_version']='prdt-reproduction-profile-registry-v1'
        elif mutation=='unknown':c['Q']=True
        elif mutation=='loss':c['policy']['loss']='N*w*SUM_NLL'
        elif mutation=='schedule':c['policy']['cosine_horizon_updates']=2
        elif mutation=='final':c['evaluation']['allow_final']=True
        elif mutation=='prior':c['weighting']['collector_prior']['A']=-1.
        elif mutation=='beta':c['weighting']['beta']=float('nan')
        elif mutation=='budget':c['policy']['draws_per_epoch']=True
    rewrite(p,change)
    with pytest.raises(ValueError):runner.check(p,d)
    assert not o.exists()

def test_relocation_same_identity_and_path_escape_rejected(fixture,tmp_path):
    import shutil
    p,d,o=fixture;first=runner.check(p,d);other=tmp_path/'other';shutil.copytree(d,other)
    assert runner.check(p,other)['identity']==first['identity']
    b=runner.read(d/'bundle.json');b['assets']['corpus']['path']='../data/corpus.npz';(d/'bundle.json').write_text(json.dumps(b))
    with pytest.raises(ValueError,match='relative'):runner.check(p,d)

def test_asset_change_before_model_or_output(fixture,monkeypatch):
    p,d,o=fixture;(d/'features.npz').write_bytes(b'changed')
    monkeypatch.setattr(models,'make_model',lambda*a,**k:pytest.fail('model before hashes'))
    with pytest.raises(ValueError,match='changed portable asset'):runner.prepare(p,d,o)
    assert not o.exists()

@pytest.mark.parametrize('mutation',['split','padding','label','pair_user','pair_direction','features','sealed_final','collector'])
def test_prepare_checks_real_arrays_and_frozen_labels(fixture,mutation):
    p,d,o=fixture
    if mutation=='split':rewrite_asset(d,'corpus',lambda a:a['splits'].__setitem__(0,2))
    elif mutation=='padding':rewrite_asset(d,'corpus',lambda a:a['mask'].__setitem__((1,3),True))
    elif mutation=='label':rewrite_asset(d,'corpus',lambda a:a['positive_count'].__setitem__(0,0))
    elif mutation=='pair_user':rewrite_asset(d,'pairs',lambda a:a['train_rejected'].__setitem__(0,4))
    elif mutation=='pair_direction':rewrite_asset(d,'pairs',lambda a:a['train_preferred'].__setitem__(0,1))
    elif mutation=='features':rewrite_asset(d,'features',lambda a:a['user_vectors'][0].fill(1.))
    elif mutation=='collector':rewrite_asset(d,'collectors',lambda a:a['indices'].__setitem__(0,100))
    else:
        b=runner.read(d/'bundle.json');b['sealed_final_users']=[101];(d/'bundle.json').write_text(json.dumps(b))
    with pytest.raises(ValueError):runner.prepare(p,d,o)
    assert not(o/'run.json').exists()

def test_prepare_idempotent_collision(fixture):
    p,d,o=fixture;first=runner.prepare(p,d,o);before=(o/'run.json').read_bytes()
    assert runner.prepare(p,d,o)==first
    with pytest.raises(ValueError,match='collision'):runner.prepare(p,d,o,seed=1)
    assert(o/'run.json').read_bytes()==before

def test_actual_tiny_pipeline_mid_scorer_and_policy_resume_exact(fixture,tmp_path):
    p,d,full=fixture;resumed=tmp_path/'resumed';train(p,d,full);runner.prepare(p,d,resumed)
    assert training.scorer(p,d,resumed,device='cpu',stop_after_updates=1)['updates']==1
    training.scorer(p,d,resumed,device='cpu')
    assert training.policy(p,d,resumed,device='cpu',stop_after_updates=2)['updates']==2
    result=training.policy(p,d,resumed,device='cpu')
    assert result['optimizer_updates']==6 and result['trajectory_draws']==14
    assert [r['draws']for r in result['history']]==[7,7] and [r['updates']for r in result['history']]==[3,3]
    for filename in('scorer_latest.pt','policy_final.pt'):
        assert_same(torch.load(full/filename,map_location='cpu',weights_only=False),torch.load(resumed/filename,map_location='cpu',weights_only=False))
    with np.load(full/'scores.npz')as a,np.load(resumed/'scores.npz')as b:assert all(np.array_equal(a[k],b[k])for k in a.files)
    saved=torch.load(resumed/'policy_final.pt',map_location='cpu',weights_only=False)
    assert saved['optimizer']['state'] and saved['rng']['loader'].dtype==torch.uint8
    rates=[r for h in saved['state']['history']for r in h['applied_learning_rates']]
    assert rates==[training.learning_rate(runner.read(p)['policy'],i)for i in range(6)]
    assert rates[-1]>runner.read(p)['policy']['minimum_learning_rate']

def test_complete_fit_reuse_no_model(fixture,monkeypatch):
    p,d,o=fixture;train(p,d,o);before={n:(o/n).read_bytes()for n in('policy_final.pt','policy.json','scorer.json')}
    monkeypatch.setattr(models,'make_model',lambda*a,**k:pytest.fail('duplicate completed fit'))
    assert training.scorer(p,d,o,device='cpu')['status']=='scorer_complete'
    assert training.policy(p,d,o,device='cpu')['status']=='policy_complete'
    assert all((o/k).read_bytes()==v for k,v in before.items())

@pytest.mark.parametrize('mutation',['identity','backend','config','counter','rates','draws','schedule'])
def test_checkpoint_state_mismatch_rejected(fixture,mutation):
    p,d,o=fixture;runner.prepare(p,d,o);training.scorer(p,d,o,device='cpu');training.policy(p,d,o,device='cpu',stop_after_updates=1)
    path=o/'policy_latest.pt';s=torch.load(path,map_location='cpu',weights_only=False)
    if mutation=='identity':s['identity']='changed'
    elif mutation=='backend':s['device_type']='cuda'
    elif mutation=='config':s['config']['policy']['learning_rate']*=2
    elif mutation=='counter':s['state']['updates']+=1
    elif mutation=='rates':s['state']['epoch_rates'][0]=.1
    elif mutation=='draws':s['state']['draws']+=1
    elif mutation=='schedule':s['schedule']['total_updates']=6
    torch.save(s,path)
    with pytest.raises(ValueError,match='checkpoint'):training.policy(p,d,o,device='cpu')

def test_global_and_fixed_prior_keep_within_collector_ratios(fixture):
    p,d,o=fixture;runner.prepare(p,d,o);training.scorer(p,d,o,device='cpu');run,data=runner.open_run(p,d,o,0)
    other=deepcopy(run);other['config']['weighting'].update(rule='global_softmax',collector_prior=None)
    q0,e0=training.probabilities(other,data,d,o);q1,e1=training.probabilities(run,data,d,o)
    assert e0['rule']=='global_softmax' and e1['rule']=='fixed_collector_prior'
    for c in('A','B','C'):
        ids=[i for i in range(6)if data['collectors'][i]==c]
        assert torch.allclose(q1[ids].sum(),torch.tensor(1/3,dtype=torch.float64))
        assert torch.allclose(q1[ids]/q0[ids],torch.full((len(ids),),float(q1[ids][0]/q0[ids][0]),dtype=torch.float64))

def test_cached_scores_no_fit_exact_seed_only(fixture,monkeypatch):
    p,d,o=fixture;b=runner.read(d/'bundle.json');path=d/'scores.npz';a=dict(indices=np.arange(6,dtype=np.int64),scores=np.arange(6,dtype=np.float32));np.savez_compressed(path,**a)
    b['assets']['scores']=dict(path=path.name,sha256=runner.digest(path),arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype))for k,v in a.items()});b['score_cache_seed']=0;(d/'bundle.json').write_text(json.dumps(b))
    runner.prepare(p,d,o);monkeypatch.setattr(models,'make_model',lambda*a,**k:pytest.fail('new scorer'))
    assert training.scorer(p,d,o,device='cpu')['new_scorer_updates']==0
    with pytest.raises(ValueError,match='exact training seed'):runner.check(p,d,seed=1)

class ToyAdapter:
    calls=[]
    def fork(self):return type(self)()
    def reset(self,user,context):self.user,self.steps=user,0;return context
    def step(self,action):
        self.steps+=1;type(self).calls.append((self.user,self.steps,action));done=self.steps==(1 if self.user==101 else 3)
        return(5. if done else 1.),done,dict(terminal=done)
    def feedback(self,reward,action):return reward/10.

class ProbePolicy(torch.nn.Module):
    seen=[]
    def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(()))
    def load_state_dict(self,value,strict=True):return None
    def forward(self,users,actions,feedback,mask,*,initial_context_actions):
        type(self).seen.append((users.clone(),actions.clone(),feedback.clone(),mask.clone(),initial_context_actions.clone()))
        logits=torch.zeros((*actions.shape,8));logits[...,2]=3.;return logits

def test_eval_terminal_reward_active_mask_actual_feedback_no_teacher_forcing(fixture,monkeypatch):
    p,d,o=fixture;train(p,d,o);ProbePolicy.seen=[];ToyAdapter.calls=[]
    monkeypatch.setattr(models,'make_model',lambda*a,**k:ProbePolicy())
    r=runner.evaluate(p,d,o,device='cpu',adapter_factory=lambda*a:ToyAdapter())
    assert r['mean_length']==2. and r['mean_cumulative_reward']==6. and r['pooled_reward_per_step']==3.
    assert r['terminal_reward_included'] and ToyAdapter.calls==[(101,1,2),(102,1,2),(102,2,2),(102,3,2)]
    assert [len(row[0])for row in ProbePolicy.seen]==[2,1,1]
    assert ProbePolicy.seen[0][1].tolist()==[[0],[0]] and ProbePolicy.seen[1][1].tolist()==[[2,0]]
    assert torch.allclose(ProbePolicy.seen[2][2],torch.tensor([[.1,.1,0.]]))
    raw=runner.read(o/'development_raw.json')['episodes'];assert [e['rewards']for e in raw]==[[5.],[1.,1.,5.]]
    assert runner.evaluate(p,d,o,device='cpu',adapter_factory=lambda*a:pytest.fail('duplicate eval'))==r

def test_partial_eval_no_reroll(fixture,monkeypatch):
    p,d,o=fixture;train(p,d,o);runner.put(o/'evaluation_attempt.json',dict(partial=True))
    monkeypatch.setattr(models,'make_model',lambda*a,**k:pytest.fail('partial attempt model'))
    with pytest.raises(ValueError,match='never silently reroll'):runner.evaluate(p,d,o,device='cpu',adapter_factory=lambda*a:pytest.fail('partial attempt environment'))

def test_no_absolute_runner_or_codex_implementation_dependency():
    source=Path(runner.__file__).read_text();assert 'subprocess'not in source and '/Users/'not in source and '.codex'not in source

def test_same_output_flock_collision(fixture):
    p,d,o=fixture;runner.prepare(p,d,o);before=(o/'run.json').read_bytes()
    with runner.locked(o):
        with pytest.raises(RuntimeError,match='already owned'):training.policy(p,d,o,device='cpu')
    assert(o/'run.json').read_bytes()==before

def test_cli_fixture_evaluate_and_resume_are_complete_pipeline(fixture,capsys):
    p,d,o=fixture
    common=['--profile',str(p),'--data-root',str(d),'--output-root',str(o),'--device','cpu']
    assert runner.main(['prepare',*common])==0
    assert runner.main(['resume',*common,'--through','evaluate'])==0
    result=runner.read(o/'evaluation.json')
    assert result['users']==2 and result['terminal_reward_included'] and result['mean_cumulative_reward']==6.
    before=(o/'development_logits.npz').read_bytes()
    assert runner.main(['resume',*common,'--through','evaluate'])==0
    assert(o/'development_logits.npz').read_bytes()==before

@pytest.mark.parametrize('mutation',['unit_mapping','unit_scale','unit_clip','world_shape','payload_shape'])
def test_dry_run_deep_payload_units_without_nn_or_reset(fixture,mutation,monkeypatch):
    p,d,o=fixture
    if mutation.startswith('unit'):
        b=runner.read(d/'bundle.json')
        if mutation=='unit_mapping':b['units']['training']['mapping']='guess'
        elif mutation=='unit_scale':b['units']['training']['scale']=float('inf')
        else:b['units']['training'].update(lower_clip=2.,upper_clip=1.)
        (d/'bundle.json').write_text(json.dumps(b))
    elif mutation=='world_shape':rewrite_asset(d,'fixture_world',lambda a:a.update(rewards=np.ones((2,3,6))))
    else:rewrite_asset(d,'features',lambda a:a.update(item_vectors=np.ones((6,39),dtype=np.float32)))
    monkeypatch.setattr(models,'make_model',lambda*a,**k:pytest.fail('preflight NN'))
    monkeypatch.setattr(runner.FixtureDEV,'reset',lambda*a,**k:pytest.fail('preflight environment reset'))
    assert runner.main(['check','--dry-run','--profile',str(p),'--data-root',str(d)])==2
    assert not o.exists()

def test_sampler_chronology_exact_weighted_loader_base_seed_and_final_batch(fixture):
    from torch.utils.data import DataLoader,WeightedRandomSampler
    p,d,o=fixture;result=train(p,d,o)
    run,data=runner.open_run(p,d,o,0);q,_=training.probabilities(run,data,d,o)
    generator=torch.Generator().manual_seed(77)
    expected_hash=[]
    for _ in range(2):
        draws=torch.cat(list(DataLoader(range(6),batch_size=3,sampler=WeightedRandomSampler(q,7,replacement=True,generator=generator),generator=generator)))
        expected_hash.append(__import__('hashlib').sha256(draws.numpy().tobytes()).hexdigest())
    assert [h['global_draw_stream_sha256']for h in result['history']]==expected_hash
    saved=torch.load(o/'policy_final.pt',map_location='cpu',weights_only=False)
    assert torch.equal(saved['rng']['loader'],generator.get_state())

def test_native_numpy_scorer_and_sampler_resume_exact_fixture(fixture,tmp_path):
    p,d,full=fixture
    rewrite(p,lambda c:(c['scorer'].update(pair_order_rng='numpy_epoch_seed',full_context_batches=True),c['policy'].update(sampling_rng='numpy_choice_persistent',full_context_batches=True)))
    train(p,d,full);resumed=tmp_path/'numpy_resumed';runner.prepare(p,d,resumed)
    training.scorer(p,d,resumed,device='cpu',stop_after_updates=1);training.scorer(p,d,resumed,device='cpu')
    training.policy(p,d,resumed,device='cpu',stop_after_updates=4);training.policy(p,d,resumed,device='cpu')
    for filename in('scorer_latest.pt','policy_final.pt'):
        assert_same(torch.load(full/filename,map_location='cpu',weights_only=False),torch.load(resumed/filename,map_location='cpu',weights_only=False))
    final=torch.load(resumed/'policy_final.pt',map_location='cpu',weights_only=False)
    run,data=runner.open_run(p,d,resumed,0);q,_=training.probabilities(run,data,d,resumed)
    gen=np.random.default_rng(77)
    for _ in range(2):gen.choice(6,7,replace=True,p=q.numpy())
    assert_same(final['numpy_sampler_state'],gen.bit_generator.state)

def test_foreign_nonempty_output_never_adopted(fixture):
    p,d,o=fixture;o.mkdir();(o/'unrelated.txt').write_text('keep')
    with pytest.raises(ValueError,match='foreign nonempty'):runner.prepare(p,d,o)
    assert(o/'unrelated.txt').read_text()=='keep' and not(o/'run.json').exists()

def test_full_context_scorer_vs_dynamic_policy_padding_and_kuairec_zero_users(fixture):
    p,d,o=fixture;run=runner.check(p,d);data=runner.load_data(run,d)
    run['config']['model']['user_features']='zero';run['config']['scorer']['full_context_batches']=True
    rows=[data['corpus'].get(1)]
    scorer=training.collate(rows,run,scorer=True);policy=training.collate(rows,run)
    assert scorer['actions'].shape==(1,4) and policy['actions'].shape==(1,1)
    assert scorer['users'].tolist()==[0] and policy['users'].tolist()==[0]
    assert scorer['mask'].tolist()==[[True,False,False,False]]

def test_known_movielens_cohort_cannot_be_replaced_by_final_or_fake_roster(fixture):
    p,d,o=fixture;run=runner.check(p,d)
    run['config']['dataset']='MovieLens';run['bundle']['sealed_final_users']=[]
    with pytest.raises(ValueError,match='exact retained DEV'):runner.load_data(run,d)

def make_public_fixture(root):
    p,d=make_fixture(root)
    c=runner.read(p)
    for k in ('registry_record_sha256','status','runnable','blocked_reason'): c.pop(k)
    c['schema_version']=runner.PUBLIC_SCHEMA
    c['data'].update(preference_rule='cumulative_logged_feedback',stored_feedback_threshold=None,
        validation_preference_rule='positive_feedback_count',validation_stored_feedback_threshold=3.,
        policy_training_units=dict(mapping='affine',offset=0.,scale=5.,lower_clip=None,upper_clip=None))
    p.write_text(json.dumps(c))
    def corpus(a):
        a['raw_users']=a['users'].copy()
        a['label_feedback']=a['feedback'].copy();a['label_feedback'][3,:2]=[3.,4.]
        a['policy_feedback']=np.where(a['mask'],3.,0.).astype(np.float32)
        score=np.where(a['mask'],a['label_feedback'],0).sum(1,dtype=np.float64)
        a['preference_score']=np.where(a['splits']==0,score,a['positive_count'])
        a.pop('positive_count')
    rewrite_asset(d,'corpus',corpus)
    def pairs(a):
        a['train_preferred'][0]=1;a['train_rejected'][0]=0
        a['train_targets']=np.array([0.,1.,1.,.5],dtype=np.float32)
        a['validation_targets']=np.ones(2,dtype=np.float32)
    rewrite_asset(d,'pairs',pairs)
    b=runner.read(d/'bundle.json')
    b['units']['policy_training']=c['data']['policy_training_units']
    b['preference'].update(rule='cumulative_logged_feedback',stored_feedback_threshold=None,
        validation_rule='positive_feedback_count',validation_stored_feedback_threshold=3.)
    cp=d/'cohorts.json';cp.write_text(json.dumps(dict(train_users=[1,2],validation_users=[1,2],development_users=[101,102],final_users=[901])))
    b['assets']['cohorts']=dict(path='cohorts.json',sha256=runner.digest(cp),arrays={})
    (d/'bundle.json').write_text(json.dumps(b));return p,d

def test_public_cumulative_targets_and_separate_feedback(tmp_path):
    p,d=make_public_fixture(tmp_path);run=runner.check(p,d);data=runner.deep_check(run,d)
    assert data['pairs']['train_targets'].tolist()==[0.,1.,1.,.5]
    assert data['corpus'].get(3).positive_count==7.
    assert data['corpus'].get(6).positive_count==2. # original count validation
    assert data['corpus'].get(0).feedback.tolist()==[4.,5.]
    assert data['policy_corpus'].get(0).feedback.tolist()==[3.,3.]
    b=training.collate([data['policy_corpus'].get(0)],run)
    assert torch.allclose(normalize(b,run['bundle']['units']['policy_training'])['feedback'],torch.full((1,2),.6))

@pytest.mark.parametrize('change', ['target','score','terminal','split','final'])
def test_public_label_and_cohort_corruption_rejected(tmp_path,change):
    p,d=make_public_fixture(tmp_path)
    if change=='target':rewrite_asset(d,'pairs',lambda a:a['train_targets'].__setitem__(0,1.))
    if change=='score':rewrite_asset(d,'corpus',lambda a:a['preference_score'].__setitem__(0,1.))
    if change=='terminal':rewrite_asset(d,'corpus',lambda a:a['label_feedback'].__setitem__((0,1),0.))
    if change=='split':rewrite_asset(d,'pairs',lambda a:a['train_rejected'].__setitem__(0,6))
    if change=='final':
        b=runner.read(d/'bundle.json');cp=d/'cohorts.json';roles=runner.read(cp);roles['final_users']=[101];cp.write_text(json.dumps(roles));b['assets']['cohorts']['sha256']=runner.digest(cp);b['sealed_final_users']=[101];(d/'bundle.json').write_text(json.dumps(b))
    with pytest.raises(ValueError):runner.deep_check(runner.check(p,d),d)

def test_public_soft_label_loss_used_and_resume_exact(tmp_path,monkeypatch):
    p,d=make_public_fixture(tmp_path);old=torch.get_num_threads();torch.set_num_threads(1)
    observed=[];original=torch.nn.functional.binary_cross_entropy_with_logits
    def loss(logits,targets,*args,**kwargs):
        observed.extend(targets.detach().tolist());return original(logits,targets,*args,**kwargs)
    monkeypatch.setattr(torch.nn.functional,'binary_cross_entropy_with_logits',loss)
    try:
        a,b=tmp_path/'full',tmp_path/'resumed';runner.prepare(p,d,a);runner.prepare(p,d,b)
        training.scorer(p,d,a,device='cpu');training.scorer(p,d,b,device='cpu',stop_after_updates=1);training.scorer(p,d,b,device='cpu')
        assert set(observed)=={0.,.5,1.}
        sa=torch.load(a/'scorer_latest.pt',weights_only=False);sb=torch.load(b/'scorer_latest.pt',weights_only=False)
        assert_same(sa,sb)
        training.policy(p,d,a,device='cpu');training.policy(p,d,b,device='cpu',stop_after_updates=1);training.policy(p,d,b,device='cpu')
        assert_same(torch.load(a/'policy_final.pt',weights_only=False),torch.load(b/'policy_final.pt',weights_only=False))
        z=torch.zeros(3,requires_grad=True);original(z,torch.tensor([0.,.5,1.])).backward()
        assert z.grad[0]>0 and z.grad[1]==0 and z.grad[2]<0
    finally:torch.set_num_threads(old)

def test_public_context40_evaluation30_and_no_registry_requirement():
    for name in ('movielens','kuairec','coat'):
        c=runner.load_config(name)
        assert c['schema_version']==runner.PUBLIC_SCHEMA and 'registry_record_sha256' not in c
        assert c['model']['horizon']==40 and c['evaluation']['horizon']==30
        assert c['policy']['epochs']==25
    c=runner.load_config('coat');c['evaluation']['horizon']=41
    with pytest.raises(ValueError):runner.validate_config(c)

def test_coat_shared_user_item_constructor_matches_canonical(tmp_path):
    p,d=make_public_fixture(tmp_path);run=runner.check(p,d);data=runner.deep_check(run,d)
    c=run['config'];c['dataset']='Coat'
    from prdt4rec.training import seed_everything
    corpus, features = data['corpus'], data['features']
    seed_everything(29)
    actual = models.make_model(c, 'cpu', corpus=corpus, features=features)
    actual_rng = torch.get_rng_state()
    seed_everything(29)
    geometry = c['model']
    expected = models.RecommendationPolicy(
        corpus.n_users, corpus.n_items, corpus.max_length, backbone='convformer',
        **{k: geometry[k] for k in ('hidden_dim', 'layers', 'heads', 'dropout', 'conv_window')})
    item_table = torch.zeros((corpus.n_items, geometry['feature_dim']))
    item_table[2:] = torch.as_tensor(features['item_vectors'])
    for name, table in (('user_embedding', torch.tensor(features['user_vectors'])),
                        ('item_embedding', item_table)):
        setattr(expected, name, torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(table, freeze=True, padding_idx=0),
            torch.nn.Linear(table.shape[1], geometry['hidden_dim'], bias=False)))
    assert_same(actual.state_dict(),expected.state_dict())
    assert torch.equal(actual_rng, torch.get_rng_state())

@pytest.mark.parametrize('dataset,draw',[('MovieLens',False),('KuaiRec',False),('Coat',True)])
def test_public_validation_rng_matches_native_source(tmp_path,dataset,draw):
    p,d=make_public_fixture(tmp_path);run=runner.check(p,d);data=runner.deep_check(run,d)
    model=models.make_model(run['config'], 'cpu', scorer=True, corpus=data['corpus'], features=data['features'])
    run['config']['dataset']=dataset
    before={k:v.clone() for k,v in model.state_dict().items()}
    torch.manual_seed(936);initial=torch.get_rng_state().clone()
    if draw:torch.empty((),dtype=torch.int64).random_().item()
    expected=torch.get_rng_state().clone();torch.set_rng_state(initial)
    metrics=training._scorer_validation(run,data,model,torch.device('cpu'))
    assert torch.equal(torch.get_rng_state(),expected)
    assert metrics['validation_pairs']==2 and model.training
    assert_same(before,model.state_dict())

def test_public_coat_softplus_targets_match_native_formula():
    z=torch.tensor([-3.,0.,5.],requires_grad=True);targets=torch.tensor([0.,.5,1.])
    expected=(targets*torch.nn.functional.softplus(-z)+(1-targets)*torch.nn.functional.softplus(z)).mean()
    actual=training._public_preference_loss(z,targets,'Coat')
    assert torch.equal(actual,expected)
    actual.backward();assert z.grad[0]>0 and z.grad[1]==0 and z.grad[2]<0

def test_public_randperm_scorer_resume_and_rng(tmp_path):
    p,d=make_public_fixture(tmp_path)
    rewrite(p,lambda c:c['scorer'].update(pair_order_rng='torch_randperm'))
    a,b=tmp_path/'full',tmp_path/'resumed';runner.prepare(p,d,a);runner.prepare(p,d,b)
    training.scorer(p,d,a,device='cpu');training.scorer(p,d,b,device='cpu',stop_after_updates=1)
    training.scorer(p,d,b,device='cpu')
    ca=torch.load(a/'scorer_latest.pt',weights_only=False);cb=torch.load(b/'scorer_latest.pt',weights_only=False)
    assert_same(ca,cb)
    gen=torch.Generator().manual_seed(0)
    for _ in range(2):torch.randperm(4,generator=gen)
    assert torch.equal(ca['rng']['loader'],gen.get_state())

PUBLIC_PROFILE_IDENTITIES = {'movielens': 'ead2bf47a5721323a920fb1de07d19febe72ef10b652236eaa3c6dc20e698380', 'kuairec': '431171fd51510e9458bc8a0d54bd989398b70c23df21f1b7fbb7ae0a23afef42', 'coat': 'fcd2e8edc6e9a4f3a03dac4804cc7454948c72787fea4114e5e81d279a397ea3'}

def test_shared_config_overrides_are_recursive_and_independent(tmp_path):
    document = runner.read(runner.PUBLIC_CONFIG)
    document['defaults']['policy']['learning_rate'] = .0002
    document['datasets']['coat'].setdefault('policy', {})['learning_rate'] = .0003
    path = tmp_path / 'prdt.json'; path.write_text(json.dumps(document))
    ml = runner.load_config(str(path) + '#movielens')
    coat = runner.load_config(str(path) + '#coat')
    assert ml['policy']['learning_rate'] == .0002
    assert coat['policy']['learning_rate'] == .0003
    assert coat['policy']['epochs'] == 25
    coat['model']['layers'] = 99
    assert runner.load_config(str(path) + '#coat')['model']['layers'] == 4
    assert runner.load_config('movielens')['policy']['learning_rate'] == .0001

def test_shared_config_cli_and_unknown_override(tmp_path, capsys):
    document = runner.read(runner.PUBLIC_CONFIG)
    path = tmp_path / 'prdt.json'; path.write_text(json.dumps(document))
    assert runner.main(['list', '--config', str(path)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 6
    document['datasets']['coat']['policy'] = {'learning_rate_typo': .01}
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='unknown or missing'):
        runner.public_profiles(path)

def test_public_file_needs_no_format_labels_and_legacy_shared_copy_loads(tmp_path):
    document = runner.read(runner.PUBLIC_CONFIG)
    assert set(document) == {"defaults", "datasets"}
    assert 'schema_version' not in document['defaults']
    assert not any('schema_version' in c for c in document['datasets'].values())
    old = deepcopy(document)
    old['schema_version'] = runner.PROFILES_SCHEMA
    old['defaults']['schema_version'] = runner.PUBLIC_SCHEMA
    path = tmp_path / 'old.json'; path.write_text(json.dumps(old))
    assert runner.public_profiles(path) == runner.public_profiles()
    old['schema_version'] = 'unrecognized'
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='configuration file'):
        runner.public_profiles(path)

def test_unversioned_recipe_and_bundle_preserve_resolved_values(tmp_path):
    p, d = make_public_fixture(tmp_path)
    original = runner.check(p, d)
    rewrite(p, lambda c: c.pop('schema_version'))
    rewrite(d / 'bundle.json', lambda b: b.pop('schema_version'))
    current = runner.check(p, d)
    assert current['config'] == original['config']
    assert current['bundle'] == original['bundle']
    runner.deep_check(current, d)

def test_public_package_command_is_clean_and_propagates_errors(tmp_path):
    command = [sys.executable, '-m', 'prdt4rec']
    help_result = subprocess.run(command + ['--help'], cwd=runner.PROJECT, text=True, capture_output=True)
    assert help_result.returncode == 0
    assert 'PrDT4Rec' in help_result.stdout
    assert 'plain-prdt' not in help_result.stdout and 'v2' not in help_result.stdout
    listing = subprocess.run(command + ['list'], cwd=runner.PROJECT, text=True, capture_output=True)
    assert listing.returncode == 0
    assert {c['profile'] for c in json.loads(listing.stdout)} == {
        'movielens', 'kuairec', 'coat', 'yahoor3', 'kuairand', 'virtualtb',
    }
    missing = subprocess.run(command + ['check', '--data-root', str(tmp_path / 'missing')],
                             cwd=runner.PROJECT, text=True, capture_output=True)
    assert missing.returncode == 2

@pytest.mark.parametrize("name", ["yahoor3", "kuairand", "virtualtb"])
def test_additional_profiles_resolve_from_default_and_explicit_config(name):
    for profile in (name, str(runner.PUBLIC_CONFIG) + "#" + name):
        assert runner.load_config(profile)["profile_id"] == name


def configure_catalog_fixture(profile, root, name, adapter, window, threshold):
    config = runner.load_config(name)
    config["model"].update(hidden_dim=8, heads=2, layers=1, conv_window=3)
    config["scorer"].update(hidden_dim=8, heads=2, layers=1, epochs=2, batch_size=3,
                           checkpoint_interval_updates=1)
    config["policy"].update(epochs=2, draws_per_epoch=7, batch_size=3, cosine_horizon_updates=12,
                           checkpoint_interval_updates=1)
    config["evaluation"].update(users=2, reset_block_size=2, batch_size=2)
    config["data"].update(split_counts=dict(train=6, validation=4, test=0),
                          pair_counts=dict(train=4, validation=2))
    profile.write_text(json.dumps(config))

    width = config["model"]["feature_dim"]
    items = 3327 if name == "kuairec" else 6
    item_ids = np.arange(items, dtype=np.int64)
    users = np.array([101, 102])
    manifest = runner.read(root / "bundle.json")
    manifest["assets"].pop("fixture_world")
    manifest.update(dataset=config["dataset"], profile_id=name, units=dict(
        training=config["data"]["training_units"], policy_training=config["data"]["policy_training_units"],
        evaluation=dict(mapping="item_train_range_minus_response_min") if name == "kuairec"
        else dict(mapping="affine", offset=0., scale=5., lower_clip=None, upper_clip=None)),
        preference={k: config["data"][v] for k, v in (
            ("rule", "preference_rule"), ("validation_rule", "validation_preference_rule"),
            ("stored_feedback_threshold", "stored_feedback_threshold"),
            ("validation_stored_feedback_threshold", "validation_stored_feedback_threshold"))},
        evaluation=dict(adapter=adapter, phase="development", horizon=30, leave_window=window,
                        leave_threshold=threshold, cases_asset="dev_cases"))

    def add(name, **arrays):
        path = root / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        manifest["assets"][name] = dict(path=path.name, sha256=runner.digest(path),
                                       arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype))
                                               for k, v in arrays.items()})

    add("world", rewards=np.full((2, items), 1.0, np.float32), raw_users=users, raw_items=item_ids)
    if name == "kuairec":
        add("categories", categories=np.zeros((items, 1), dtype=np.int64))
        add("response_transform", raw_items=item_ids, watch_min=np.zeros(items),
            watch_max=np.ones(items), response_matrix_minimum=np.array(0.0))
    else:
        add("distance", distance=np.zeros((items, items)))
    cohorts = root / "cohorts.json"
    cohorts.write_text(json.dumps(dict(train_users=[1, 2], validation_users=[1, 2],
                                       development_users=users.tolist(), final_users=[901])))
    manifest["assets"]["cohorts"] = dict(path=cohorts.name, sha256=runner.digest(cohorts), arrays={})
    (root / "bundle.json").write_text(json.dumps(manifest))

    def corpus(a):
        a["raw_users"] = a["users"].copy()
        a["n_items"] = np.array(items + 2)
        a["max_length"] = np.array(40)
        for key in ("actions", "feedback", "mask"):
            a[key] = np.pad(a[key], ((0, 0), (0, 36)))
        if name == "kuairec":
            a["feedback"] *= np.float32(.01)
        a["policy_feedback"] = a["feedback"].copy()
        a["label_feedback"] = a["feedback"].copy()
        cumulative = np.where(a["mask"], a["label_feedback"], 0.).sum(axis=1, dtype=np.float64)
        score = cumulative.copy()
        for split, prefix in ((0, ""), (1, "validation_")):
            threshold = config["data"][prefix + "stored_feedback_threshold"]
            if threshold is not None:
                rows = a["splits"] == split
                score[rows] = ((a["feedback"] > threshold) & a["mask"]).sum(axis=1)[rows]
        a["preference_score"] = score
        a.pop("positive_count")

    rewrite_asset(root, "corpus", corpus)
    with np.load(root / "corpus.npz", allow_pickle=False) as arrays:
        scores = arrays["preference_score"].copy()

    def pairs(a):
        for split in ("train", "validation"):
            high, low = a[f"{split}_preferred"], a[f"{split}_rejected"]
            a[f"{split}_targets"] = np.sign(scores[high] - scores[low]).astype(np.float32) / 2 + .5

    rewrite_asset(root, "pairs", pairs)
    if name == "movielens":
        rewrite_asset(root, "collectors", lambda a: a.update(
            collectors=np.array(["A2C", "DORL", "PPOC"] * 3 + ["A2C"])))
    if name == "kuairec":
        rewrite_asset(root, "features", lambda a: (
            a.pop("user_vectors"), a.update(item_vectors=np.zeros((items, width), dtype=np.float32))))


@pytest.mark.parametrize("name,adapter,window,threshold", [
    ("movielens", "movielens_native_v1", 7, 75),
    ("coat", "coat_native_v1", 7, 6),
    ("kuairec", "kuairec_native_v1", 9, 1),
])
def test_retained_catalog_profiles_fit_resume_and_evaluate(fixture, tmp_path, name, adapter, window, threshold):
    profile, root, output = fixture
    configure_catalog_fixture(profile, root, name, adapter, window, threshold)
    complete = tmp_path / "complete"
    for path in (complete, output):
        assert runner.prepare(profile, root, path)["policy_updates"] == 6
    training.scorer(profile, root, complete, device="cpu")
    assert training.scorer(profile, root, output, device="cpu", stop_after_updates=1)["updates"] == 1
    training.scorer(profile, root, output, device="cpu")
    training.policy(profile, root, complete, device="cpu")
    assert training.policy(profile, root, output, device="cpu", stop_after_updates=1)["updates"] == 1
    policy = training.policy(profile, root, output, device="cpu")
    assert policy["optimizer_updates"] == 6 and policy["trajectory_draws"] == 14
    for checkpoint in ("scorer_latest.pt", "policy_final.pt"):
        assert_same(torch.load(complete / checkpoint, weights_only=False),
                    torch.load(output / checkpoint, weights_only=False))
    evaluation = runner.evaluate(profile, root, output, device="cpu")
    assert evaluation["adapter"] == adapter and evaluation["final_evaluation"] is False
    assert evaluation["users"] == 2 and math.isfinite(evaluation["mean_cumulative_reward"])
    assert runner.evaluate(profile, root, output, device="cpu") == evaluation


def test_initial_release_checkpoint_binding_rejects_changed_sources(fixture):
    profile, root, _ = fixture
    configure_catalog_fixture(profile, root, "kuairec", "kuairec_native_v1", 9, 1)
    current = runner.check(profile, root)
    previous = deepcopy(current)
    for key in ("input_type", "user_features", "catalog_items"):
        previous["config"]["model"].pop(key)
    previous["implementation_sha256"] = {
        **runner._INITIAL_RELEASE_SOURCES,
        "kuairec.py": "41d5a709246219bd4e537fca4b6c85bd2d072622d4575e449359ec63a727cbd6",
        "kuairec_models.py": runner._INITIAL_RELEASE_SOURCES["models.py"],
    }
    previous.pop("identity")
    previous["identity"] = runner.identity(previous)
    bound = runner._bound_run(current, previous)
    assert bound["identity"] == previous["identity"]
    assert bound["config"]["model"]["user_features"] == "zero"
    assert bound["initial_release_config"] == previous["config"]
    previous["implementation_sha256"]["plain.py"] = "changed"
    with pytest.raises(ValueError, match="identity collision"):
        runner._bound_run(current, previous)


def test_previous_environment_layout_binding_preserves_checkpoint_identity(fixture):
    profile, root, _ = fixture
    configure_catalog_fixture(profile, root, "coat", "coat_native_v1", 7, 6)
    current = runner.check(profile, root)
    previous = deepcopy(current)
    previous["implementation_sha256"] = {
        name: runner._PRE_MOVE_SOURCES[name] for name in (
            "runner.py", "environments.py", "data.py", "models.py", "training.py", "objective.py")
    }
    previous.pop("identity")
    previous["identity"] = runner.identity(previous)
    assert runner._bound_run(current, previous)["identity"] == previous["identity"]
    previous["seed"] += 1
    with pytest.raises(ValueError, match="identity collision"):
        runner._bound_run(current, previous)
