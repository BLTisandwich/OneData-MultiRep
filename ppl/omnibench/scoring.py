"""Task goal and required-release scoring; penetration is diagnostic only."""
import numpy as np

def completion_audit(folder,result):
    with np.load(folder/'robot_object_state.npz') as z:held=bool(z['scripted_attachment_active'][-1])
    names=[e['name'] for e in result['events']]
    needs_release=result['task_id'] in ['T1','T3','T4','T7']
    return {'source_success_flag':bool(result['success']),'held_at_end':held,
            'grasp_event':'assisted_grasp_start' in names,'release_event':'assisted_release' in names,
            'goal_and_release_audit':bool(result['success'] and
               (not needs_release or ('assisted_release' in names and not held))),
            'scope':'Separate diagnostic semantics check; does not overwrite frozen source benchmark success.'}
