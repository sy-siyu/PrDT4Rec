import numpy as np
import pytest
from prdt4rec.environments.common import MovieLensDEV,KuaiRecDEV


def setup(n=60):
    return np.full((2,n),4.),np.array([11,29]),np.arange(n)+100,dict(raw_users=np.array([11]),initial_context_actions=np.array([2]))


def test_movielens_literal_negative_slice_and_terminal_reward():
    args=setup();distance=np.full((60,60),100.);np.fill_diagonal(distance,0)
    env=MovieLensDEV(*args,distance);assert env.reset(11,2)==2
    widths=[]
    for x in range(8):
        reward,done,info=env.step(x+2);assert reward==4 and not done;widths.append(len(info['actual_window']))
    assert widths[1:]==[1,2,3,3,2,1,7]
    reward,done,_=env.step(3);assert reward==4 and done
    with pytest.raises(RuntimeError):env.step(4)
    assert env.feedback(4,3)==.8


def test_reset_context_not_execution_and_final_users_sealed():
    args=setup();env=MovieLensDEV(*args,np.zeros((60,60)))
    with pytest.raises(ValueError):env.reset(29,2)
    with pytest.raises(ValueError):env.reset(11,3)
    env.reset(11,2);assert env.step(2)[1] is False
    assert env.step(2)[1] is True


def test_h40_keeps_terminal_reward():
    env=MovieLensDEV(*setup(),np.full((60,60),100.));env.reset(11,2)
    total=0
    for t in range(40):
        reward,done,_=env.step(t+2);total+=reward;assert done==(t==39)
    assert total==160


def test_kuairec_native_category_multiplicity_and_feedback():
    args=setup();transform=dict(raw_items=args[2],watch_min=np.zeros(60),watch_max=np.full(60,2.),response_matrix_minimum=np.array(-.2))
    env=KuaiRecDEV(*args,np.zeros((60,1),dtype=np.int64),transform);env.reset(11,2)
    assert env.step(2)[1] is False
    assert env.step(3)[1] is False
    assert env.step(4)[1] is True
    assert env.feedback(4,2)==2.2 # no clipping to1
    env.range[0]=0
    assert env.feedback(4,2)==.2


def test_coat_h30_native_threshold_and_terminal_inclusion():
    from prdt4rec.environments.common import CoatDEV
    args=setup();distance=np.full((60,60),7.);np.fill_diagonal(distance,0)
    env=CoatDEV(*args,distance);env.reset(11,2)
    for t in range(30):
        reward,done,info=env.step(t+2)
        assert reward==4. and done==(t==29)
    assert info['reason']=='H30_cap'
    env.reset(11,2);assert not env.step(2)[1]
    assert env.step(2)[1] # native threshold6 terminates repeated item
    assert env.feedback(4.,2)==.8


def test_movielens_and_kuairec_h30_keep_h40_training_separate():
    args=setup();distance=np.full((60,60),100.)
    ml=MovieLensDEV(*args,distance,horizon=30);ml.reset(11,2)
    cats=np.arange(60).reshape(-1,1)
    transform=dict(raw_items=args[2],watch_min=np.zeros(60),watch_max=np.ones(60),response_matrix_minimum=np.array(0.))
    kuai=KuaiRecDEV(*args,cats,transform,horizon=30);kuai.reset(11,2)
    for t in range(30):
        assert ml.step(t+2)[1]==(t==29)
        assert kuai.step(t+2)[1]==(t==29)


def test_coat_public_adapter_routes_h30_and_rejects_wrong_native_rule(tmp_path):
    from prdt4rec.environments.common import CoatDEV, build_adapter
    rewards, raw_users, raw_items, cases=setup()
    arrays={'world':dict(rewards=rewards,raw_users=raw_users,raw_items=raw_items),
            'dev_cases':cases,'distance':dict(distance=np.full((60,60),7.))}
    assets={}
    for name, values in arrays.items():
        np.savez(tmp_path/f'{name}.npz',**values)
        assets[name]={'path':f'{name}.npz'}
    manifest=dict(dataset='Coat',assets=assets,evaluation=dict(
        adapter='coat_native_v1',phase='development',horizon=30,
        leave_window=7,leave_threshold=6))
    env=build_adapter(manifest,tmp_path)
    assert isinstance(env,CoatDEV) and env.horizon==30
    env.reset(11,2)
    for step in range(30):
        reward,done,_=env.step(step+2)
        assert reward==4. and done==(step==29)
    manifest['evaluation']['leave_threshold']=75
    with pytest.raises(ValueError,match='threshold6'):build_adapter(manifest,tmp_path)
