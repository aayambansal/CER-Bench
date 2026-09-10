"""Offline disclosure tests; no corpus dereferencing or provider calls."""
import importlib.util
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import kendalltau

P = Path(__file__).resolve().parents[1] / "scripts/51_qrel_disclosure_experiment.py"
spec = importlib.util.spec_from_file_location("disclosure", P)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.mark.parametrize("text,expected",[(" relevant: yes",True),("not_relevant\nRELEVANT mentioned",False)])
def test_normalization(text,expected):
    assert d.h.verdict(text) is expected


@pytest.mark.parametrize("text",["MAYBE",None,"RELEVANTNESS",""])
def test_unknown_error(text):
    with pytest.raises(ValueError,match="unknown judgment"):
        d.h.verdict(text)


def test_nested_exact_counts_determinism():
    p = d.permutations_for(42,108)
    assert np.array_equal(p,d.permutations_for(42,108))
    assert not np.array_equal(p,d.permutations_for(43,108))
    assert np.all(np.sort(p,axis=-1)==np.arange(30))
    for i,f in enumerate(d.FRACTIONS):
        a = d.reveal_indices(p,f)
        assert a.shape == (108,3*i)
        assert np.array_equal(a,p[:,:3*i])
    with pytest.raises(ValueError):
        d.reveal_indices(p,1.1)


def test_ties_votes_and_tau():
    assert d.describe(np.full(1000,44/81))["mean"] == 44/81
    x = np.array([[4,4,2],[1,2,3]])
    ranks,votes = d.rank_votes(x)
    assert ranks.tolist()==[[1.5,1.5,3],[3,2,1]]
    assert votes.tolist()==[[.5,.5,0],[0,0,1]]
    assert np.all(votes.sum(axis=-1)==1)
    signs = np.array([[0,1,1],[-1,-1,-1]])
    assert d.tau_from_signs(signs[0],signs[1]) == pytest.approx(kendalltau(x[0],x[1]).statistic)
    assert ((signs[0]*signs[1])<0).sum()==2


def test_preserve_seed_negative_no_leakage():
    data = {"ids":["q"],"sizes":np.array([1]),"lcm":2,
            "hits":np.ones((1,10),dtype=int),"additions":np.zeros((1,30),dtype=int),
            "newhits":np.zeros((1,30,10),dtype=int)}
    # One positive outside rankings revealed last; other 29 rows are negatives.
    data["additions"][0,29]=1
    units,counts = d.score_permutation(data,np.arange(30)[None,:])
    assert np.all(units[:-1]==2)
    assert np.all(units[-1]==1)
    assert counts.tolist()==[1]*10+[2]
    with pytest.raises(ValueError,match="seed/judgment contradiction"):
        d.h.reconstruct({"q":["s"]},[{"task_id":"q","doc_id":"s","judgment":"NOT_RELEVANT"}],{"s"})


def test_complete_universe_fail_closed():
    with pytest.raises(ValueError,match="missing rows"):
        d.h.exact_keys(["q"],["q","missing"],"complete universe")
    with pytest.raises(ValueError,match="unknown IDs"):
        d.h.exact_keys(["q","extra"],["q"],"complete universe")


def test_real_endpoints_and_no_corpus_access(monkeypatch):
    original = Path.open
    def guarded(self,*args,**kwargs):
        assert "data/processed" not in str(self), "no corpus or repaired metadata access"
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    data = d.load_inputs()
    assert len(data["ids"])==108 and len(data["empty"])==17
    for seed in [42,43]:
        units,counts = d.score_permutation(data,d.permutations_for(seed,108))
        assert (counts[0],counts[-1])==(264,1370)
        for f,name in [(0,"seed"),(-1,"reconstructed_expanded")]:
            for j,m in enumerate(d.POOL):
                assert Fraction(int(units[f,j]),data["lcm"]*108)==Fraction(data["exact"][name][m])
        assert d.POOL[np.argmax(units[-1])]=="bge"
        assert float(Fraction(int(units[-1,2]),data["lcm"]*108))==pytest.approx(.416792975187)
        ranks,votes = d.rank_votes(units)
        assert np.all(votes.sum(axis=-1)==1)
