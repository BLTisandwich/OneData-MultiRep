"""Optional raw-data integration test; never includes observations in the release."""
import os
from pathlib import Path
import numpy as np
import pytest


def test_expert_replay_matches_reference(tmp_path):
    source=os.environ.get('OMNIBENCH_INTEGRATION_EPISODE')
    expected=os.environ.get('OMNIBENCH_INTEGRATION_REFERENCE')
    if not source or not expected:pytest.skip('Set both integration episode/reference paths to run the real simulation check')
    from omnibench.rollout import ExpertReplay,run_one
    source=Path(source);destination=tmp_path/'replayed'
    result=run_one(source,destination,ExpertReplay(source))
    assert result['actions']==96 and result['controller']=='ExpertReplay'
    with np.load(destination/'robot_object_state.npz') as actual,np.load(Path(expected)/'robot_object_state.npz') as reference:
        for key in ['ee_position_world_m','object_position_world_m']:
            assert np.allclose(actual[key],reference[key],atol=1e-6)
