import numpy as np
from analyze_benefit import fit, predict, selection_lift, corr


def test_ridge_matches_one_dimensional_closed_form():
    x=np.array([[-1.],[0.],[1.]])
    y=np.array([-1.,1.,3.])
    model=fit(x,y,3.)
    # Standardized x has squared norm 3; alpha=3 shrinks slope by 1/2.
    np.testing.assert_allclose(predict(model,x),[0.,1.,2.],atol=1e-12)


def test_matched_budget_lift_and_rank_orientation():
    y=np.array([0.,1.,2.,3.])
    assert selection_lift(y,y)==1.
    assert selection_lift(-y,y)==-1.
    assert abs(corr(-y,y)+1.)<1e-12


def test_ridge_is_invariant_to_feature_units():
    x=np.array([[1.,3.],[2.,0.],[3.,2.],[4.,5.]])
    y=np.array([-.5,2.,1.,4.])
    scaled=x*np.array([1000.,.001])+np.array([34.,-12.])
    np.testing.assert_allclose(predict(fit(x,y,1.),x),predict(fit(scaled,y,1.),scaled),atol=1e-10)
