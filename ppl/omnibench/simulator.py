"""Adapted source simulator. Executed-state interactions; original notebook provenance is in NOTICE.md.
Scene, rendering, physical/proxy sensing and final task predicates come from the
pinned notebook. Grasp/push/press triggers depend on executed state, never an
expert phase/clock. This remains assisted kinematic manipulation.
"""
from .sim_helpers import *
from .signals import observed_tactile, observed_audio

def controlled_episode(spec,cfg,out,controller):
    import pybullet as pb, pybullet_data
    from scipy.io import wavfile
    import imageio.v2 as imageio
    import trimesh
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(spec['seed']); tid=spec['task_id']; duration=cfg.duration_s
    assert cfg.physics_hz % cfg.fps == 0 and cfg.fps >= 2
    dt=1/cfg.physics_hz; frames=int(round(duration*cfg.fps)); sub=cfg.physics_hz//cfg.fps
    cid=pb.connect(pb.DIRECT)
    try:
        pb.setAdditionalSearchPath(pybullet_data.getDataPath());pb.setTimeStep(dt)
        pb.setGravity(0,0,-9.81);pb.setPhysicsEngineParameter(numSolverIterations=80)
        floor=pb.loadURDF('plane.urdf',[0,0,-.002]); pb.changeVisualShape(floor,-1,rgbaColor=spec['style']['background'])
        robot=pb.loadURDF('franka_panda/panda.urdf',[0,0,0],useFixedBase=True)
        assert pb.getNumJoints(robot)>=12
        arm_j=list(range(7)); fingers=[9,10]; ee_link=11
        edge=spec['cube_edge_m']; bx,by,bz=spec['initial_xyz'];gx,gy,_=spec['target_xyz']
        def add_box(pos,size,color,mass=0.0):
            c=pb.createCollisionShape(pb.GEOM_BOX,halfExtents=[v/2 for v in size]);
            v=pb.createVisualShape(pb.GEOM_BOX,halfExtents=[v/2 for v in size],rgbaColor=color)
            body=pb.createMultiBody(baseMass=mass,baseCollisionShapeIndex=c,baseVisualShapeIndex=v,basePosition=pos)
            pb.changeDynamics(body,-1,lateralFriction=1.4,restitution=.03)
            return body
        cube=add_box([bx,by,bz],[edge]*3,spec['style']['target'],mass=(0 if tid in ('T5','T6') else .09))
        static={}; dynamic={'target':cube}
        # Goal marker does not collide with dynamics.
        marker=pb.createVisualShape(pb.GEOM_CYLINDER,radius=.055,length=.003,rgbaColor=spec['style']['goal'])
        goalbody=pb.createMultiBody(baseMass=0,baseVisualShapeIndex=marker,basePosition=[gx,gy,.002])
        static['goal']=goalbody
        if tid=='T3':
            # Stable support height exactly 0.054m; goal at support center.
            support=add_box([gx,gy,.027],[.075,.075,.054],spec['style']['other'],0)
            static['support']=support
        elif tid=='T4':
            # Open container: separate solid walls with no floor collision ambiguity.
            w=.14; h=.075; thick=.009
            static['bin_base']=add_box([gx,gy,.005],[w,w,.01],[.55,.58,.61,1])
            for name,dx,dy,s in [('north',0,w/2,[w,thick,h]),('south',0,-w/2,[w,thick,h]),
                                  ('east',w/2,0,[thick,w,h]),('west',-w/2,0,[thick,w,h])]:
                static['bin_'+name]=add_box([gx+dx,gy+dy,h/2+.005],s,[.52,.62,.69,1])
        elif tid=='T7':
            for i,pos in enumerate(spec['distractor_xyz']):
                static['distractor_'+str(i)]=add_box(pos,[.06]*3,spec['style']['other'],0)
        elif tid in ('T5','T6'):
            # The primary static object serves as the virtual button/tap target;
            # do not create overlapping collision boxes at the same position.
            pass
        # Camera transform records are used for RGB-D unprojection.
        cams={
          'front':([1.21,-1.05,.88],[.42,-.01,.14]),
          'overhead':([.65,0,1.24],[.48,0,.04]),
          'side':([.91,.92,.68],[.46,0,.15]),
        }
        near=.04;far=3.0; W=cfg.width;H=cfg.height
        projections={};viewmats={};cameras={}
        for cname in cfg.camera_names:
            eye,target=cams[cname]; v=pb.computeViewMatrix(eye,target,[0,0,1]);
            proj=pb.computeProjectionMatrixFOV(54,W/H,near,far)
            cameras[cname]={'eye':eye,'target':target,'view_col_major':list(v),
                            'projection_col_major':list(proj),'near_m':near,'far_m':far,
                            'width':W,'height':H}
            viewmats[cname]=v;projections[cname]=proj
        qquat=pb.getQuaternionFromEuler([math.pi,0,math.pi/4])
        wps=waypoints(spec); wt=np.asarray([x[0] for x in wps]);
        lower=[pb.getJointInfo(robot,j)[8] for j in arm_j];
        upper=[pb.getJointInfo(robot,j)[9] for j in arm_j]
        ranges=[u-l for l,u in zip(lower,upper)];rest=[0,-.35,0,-2.2,0,1.9,.79]
        qplan=[];errors=[]
        for t,xyz,_,phase in wps:
            q=pb.calculateInverseKinematics(robot,ee_link,xyz,qquat,
                lowerLimits=lower,upperLimits=upper,jointRanges=ranges,
                restPoses=rest,maxNumIterations=180,residualThreshold=1e-5)
            q=np.asarray(q[:7]);qplan.append(q)
            for j,v in zip(arm_j,q): pb.resetJointState(robot,j,float(v))
            actual=np.asarray(pb.getLinkState(robot,ee_link,computeForwardKinematics=True)[4])
            errors.append(float(np.linalg.norm(actual-np.asarray(xyz))))
        if max(errors)>.14: raise RuntimeError('IK waypoint error too large '+str(max(errors)))
        def plan_at(t):
            k=int(np.clip(np.searchsorted(wt,t,side='right')-1,0,len(wps)-2))
            alpha=float(np.clip((t-wt[k])/(wt[k+1]-wt[k]),0,1));alpha=alpha*alpha*(3-2*alpha)
            q=(1-alpha)*qplan[k]+alpha*qplan[k+1]
            pos=(1-alpha)*np.asarray(wps[k][1])+alpha*np.asarray(wps[k+1][1]);
            g=(1-alpha)*wps[k][2]+alpha*wps[k+1][2]
            return q,pos,float(g),wps[k+1][3]
        for j,q in zip(arm_j,qplan[0]):pb.resetJointState(robot,j,float(q))
        for j in fingers:pb.resetJointState(robot,j,.04)
        # Kinematic expert attachment is not a physical force-closure grasp.
        held=False; relative=None; proxy_push=False; prev_ee=None; events=[]; event_once=set();
        motion=[]; vision={n:{'rgb':[],'depth_mm':[],'seg':[]} for n in cfg.camera_names}
        fast={'time_s':[],'command_xyz_world_m':[],'actual_ee_xyz_world_m':[],
              'object_xyz_world_m':[],'gripper_command_m':[],
              'joint_position_rad':[],'robot_object_contact_N':[]}
        physical={k:[] for k in ['t','q','dq','ee_pos','ee_quat','cube_pos','cube_quat','cube_vel','contact_N','held']}
        grasp_time=2.80;release_time=6.75
        last_ee=None; last_q=None
        for step in range(int(round(duration*cfg.physics_hz))):
            t=(step+1)*dt
            if step==0:
                # Initial observation matches the source scenario, before control.
                q,pos,g,phase=plan_at(t)
            else:
                alpha=((step-1)%sub+1)/sub
                q=(1-alpha)*control_q_start+alpha*control_q_target
                pos=control_xyz_target
                g=(1-alpha)*control_g_start+alpha*control_g_target
                phase='model'
            for j,v in zip(arm_j,q): pb.resetJointState(robot,j,float(v))
            for j in fingers:pb.resetJointState(robot,j,float(np.clip(g,0,.04)))
            ee=pb.getLinkState(robot,ee_link,computeForwardKinematics=True)
            ee_xyz=np.asarray(ee[4]);ee_quat=ee[5]
            old_cube_pos=np.asarray(pb.getBasePositionAndOrientation(cube)[0])
            near_object=np.linalg.norm(ee_xyz-old_cube_pos)<.085
            new_held=(tid in ('T1','T3','T4','T7','T8')) and ((held and g<.025) or (not held and g<.015 and near_object))
            if new_held and not held:
                invp,invq=pb.invertTransform(ee[4],ee[5]);cp,cq=pb.getBasePositionAndOrientation(cube)
                relative=pb.multiplyTransforms(invp,invq,cp,cq);held=True
                if 'grasp' not in event_once:
                    events.append({'name':'assisted_grasp_start','time_s':t,'provenance':'scripted_attachment'});event_once.add('grasp')
            if held and not new_held:
                held=False; events.append({'name':'assisted_release','time_s':t,'provenance':'scripted_attachment'})
            if held:
                cp,cq=pb.multiplyTransforms(ee[4],ee[5],relative[0],relative[1]);
                pb.resetBasePositionAndOrientation(cube,cp,cq);pb.resetBaseVelocity(cube,[0,0,0],[0,0,0])
            if tid=='T2' and ee_xyz[2]<.12 and last_ee is not None:
                # Explicit assist, activated only in a spatial contact corridor.
                if abs(ee_xyz[0]-old_cube_pos[0]) < .09 and -.055 < ee_xyz[1]-old_cube_pos[1] < .025:
                    delta=float(np.clip(ee_xyz[1]-last_ee[1],0,.012))
                    cp=[float(old_cube_pos[0]),float(old_cube_pos[1]+delta),float(bz)]
                    pb.resetBasePositionAndOrientation(cube,cp,[0,0,0,1]);
                    pb.resetBaseVelocity(cube,[0,0,0],[0,0,0]);proxy_push=True
            pb.stepSimulation()
            if held:
                fresh=pb.getLinkState(robot,ee_link,computeForwardKinematics=True)
                cp,cq=pb.multiplyTransforms(fresh[4],fresh[5],relative[0],relative[1]);
                pb.resetBasePositionAndOrientation(cube,cp,cq);pb.resetBaseVelocity(cube,[0,0,0],[0,0,0])
            last_ee=ee_xyz.copy()
            fast['time_s'].append(t)
            fast['command_xyz_world_m'].append(np.asarray(pos,np.float32))
            fast['actual_ee_xyz_world_m'].append(ee_xyz.astype(np.float32))
            fast['object_xyz_world_m'].append(np.asarray(pb.getBasePositionAndOrientation(cube)[0],np.float32))
            fast['gripper_command_m'].append(g)
            fast['joint_position_rad'].append(q.astype(np.float32))
            fast['robot_object_contact_N'].append(float(sum(cp[9] for cp in pb.getContactPoints(bodyA=robot,bodyB=cube))))
            if step % sub==0:
                contact_N=float(sum(cp[9] for cp in pb.getContactPoints(bodyA=robot,bodyB=cube)))
                if tid in ('T5','T6'):
                    # Button/tap interaction is a geometry-triggered virtual event.
                    # It is not misreported as measured force or a physical button joint.
                    if (np.linalg.norm(ee_xyz[:2]-np.array([bx,by]))<.075
                         and ee_xyz[2] < .12):
                        key='button_activated' if tid=='T5' else 'tap_detected'
                        if key not in event_once:
                            events.append({'name':key,'time_s':t,'provenance':'geometric_virtual_switch'});event_once.add(key)
                if contact_N>.5 and 'contact' not in event_once:
                    events.append({'name':'physical_contact_detected','time_s':t,'provenance':'pybullet_contact'});event_once.add('contact')
                pose,rot=pb.getBasePositionAndOrientation(cube)
                vel,avel=pb.getBaseVelocity(cube)
                qnow=np.array([pb.getJointState(robot,j)[0] for j in arm_j])
                dqnow=(qnow-last_q)/(sub*dt) if last_q is not None else np.zeros(7)
                last_q=qnow.copy()
                physical['t'].append(t);physical['q'].append(qnow);physical['dq'].append(dqnow)
                physical['ee_pos'].append(ee_xyz);physical['ee_quat'].append(ee_quat)
                physical['cube_pos'].append(pose);physical['cube_quat'].append(rot)
                physical['cube_vel'].append(vel);physical['contact_N'].append(contact_N)
                physical['held'].append(held)
                motion.append(dict(phase=phase,command_pos=pos,command_quat=qquat,
                    gripper_cmd=g,ee_pos=ee_xyz,ee_quat=ee_quat,
                    robot_q=qnow,finger_positions=[pb.getJointState(robot,j)[0] for j in fingers]))
                for cname in cfg.camera_names:
                    img=pb.getCameraImage(W,H,viewmats[cname],projections[cname],
                                          renderer=pb.ER_TINY_RENDERER,
                                          flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
                    rgb=np.asarray(img[2],dtype=np.uint8).reshape(H,W,4)[:,:,:3]
                    zb=np.asarray(img[3],dtype=np.float32).reshape(H,W)
                    dm=far*near/(far-(far-near)*zb)
                    segmentation=np.asarray(img[4],dtype=np.int32).reshape(H,W)
                    vision[cname]['rgb'].append(rgb.copy());
                    vision[cname]['depth_mm'].append(np.clip(dm*1000,0,65535).astype(np.uint16));
                    vision[cname]['seg'].append(segmentation)

                # Only recomputed sensor observations cross this interface.
                raw={cname: {channel:vision[cname][channel][-1] for channel in ('rgb','depth_mm','seg')} for cname in cfg.camera_names}
                raw.update(timestamp_s=t, q=qnow.copy(), dq=dqnow.copy(), ee=ee_xyz.copy(),
                           ee_quat=np.asarray(ee_quat), fingers=np.asarray(motion[-1]['finger_positions']),
                           force=np.asarray([np.mean(fast['robot_object_contact_N'][-sub:]),np.max(fast['robot_object_contact_N'][-sub:])]),
                           tactile=observed_tactile(held,contact_N,events,t,tid),
                           impact_wave=observed_audio(events,t,spec['seed']))
                action=np.asarray(controller(raw),dtype=np.float64)
                if action.shape!=(4,) or not np.isfinite(action).all():raise ValueError('Invalid model action')
                requested=ee_xyz+action[:3]
                control_xyz_target=np.clip(ee_xyz+np.clip(action[:3],-.10,.10),[.20,-.50,.025],[.85,.50,.60])
                controller.record_execution_limit(requested,control_xyz_target)
                control_q_start=qnow.copy()
                control_q_target=np.asarray(pb.calculateInverseKinematics(robot,ee_link,control_xyz_target,qquat,
                    lowerLimits=lower,upperLimits=upper,jointRanges=ranges,restPoses=rest,
                    maxNumIterations=180,residualThreshold=1e-5)[:7])
                control_g_start=float(np.mean(raw['fingers']))
                control_g_target=float(np.clip(action[3],0,1)*.04)

        assert len(motion)==frames,(len(motion),frames)
        np.savez_compressed(out/'physics_240hz.npz',**{key:np.asarray(vals,dtype=np.float32) for key,vals in fast.items()})
        for name in physical:physical[name]=np.asarray(physical[name])
        for cname in cfg.camera_names:
            for ch in ('rgb','depth_mm','seg'):vision[cname][ch]=np.asarray(vision[cname][ch])
        M=lambda field:np.asarray([m[field] for m in motion])
        ts=physical['t'].astype(np.float64)
        # Expert commanded EE targets vs actually observed EE poses are distinct.
        cmds=M('command_pos').astype(np.float32)
        actual=M('ee_pos').astype(np.float32)
        target_g=M('gripper_cmd').astype(np.float32)
        # Critical causal label: o_t predicts expert command for the NEXT time step.
        # Using cmds[t] - observed[t] would be nearly zero under kinematic IK!
        next_cmd=np.vstack((cmds[1:],cmds[-1:])).astype(np.float32)
        next_grip=np.r_[target_g[1:],target_g[-1]].astype(np.float32)
        actions=dict(timestamp_s=ts,commanded_ee_target_world_m=cmds,
                     expert_next_ee_target_world_m=next_cmd,
                     expert_next_ee_delta_from_current_world_m=(next_cmd-actual).astype(np.float32),
                     commanded_ee_quaternion_world_xyzw=M('command_quat').astype(np.float32),
                     executed_ee_position_world_m=actual,
                     executed_ee_quaternion_world_xyzw=M('ee_quat').astype(np.float32),
                     commanded_delta_world_m=(cmds-actual).astype(np.float32),
                     executed_next_ee_delta_world_m=np.vstack((np.diff(actual,axis=0),np.zeros((1,3)))).astype(np.float32),
                     gripper_command_m=target_g,
                     gripper_next_command_m=next_grip,
                     action_valid=np.r_[np.ones(frames-1,bool),False],
                     expert_phase=np.array([m['phase'] for m in motion]))
        np.savez_compressed(out/'actions.npz',**actions)
        # All physics-relevant replay data and parameters; torque is intentionally absent
        # from kinematic joint-state reset (it would be meaningless to fabricate).
        ee_vel=np.vstack((np.zeros((1,3)),np.diff(actual,axis=0)*cfg.fps)).astype(np.float32)
        object_vel_fd=np.vstack((np.zeros((1,3)),np.diff(physical['cube_pos'],axis=0)*cfg.fps)).astype(np.float32)
        button_state=np.zeros(frames,dtype=np.uint8); tap_state=np.zeros(frames,dtype=np.uint8)
        for event in events:
            if event['name']=='button_activated':button_state[ts>=event['time_s']]=1
            if event['name']=='tap_detected':tap_state[ts>=event['time_s']]=1
        np.savez_compressed(out/'robot_object_state.npz',
            timestamp_s=ts, joint_position_rad=physical['q'].astype(np.float32),
            joint_velocity_rad_s=physical['dq'].astype(np.float32),
            finger_position_m=M('finger_positions').astype(np.float32),
            ee_position_world_m=actual,
            ee_velocity_fd_world_m_s=ee_vel,
            ee_quaternion_world_xyzw=M('ee_quat').astype(np.float32),
            object_position_world_m=physical['cube_pos'].astype(np.float32),
            object_quaternion_world_xyzw=physical['cube_quat'].astype(np.float32),
            object_velocity_reported_pybullet_world_m_s=physical['cube_vel'].astype(np.float32),
            object_velocity_finite_difference_world_m_s=object_vel_fd,
            virtual_button_active=button_state,virtual_tap_detected=tap_state,
            robot_object_physical_contact_N=physical['contact_N'].astype(np.float32),
            scripted_attachment_active=physical['held'])
        # Use all camera streams at the same timestamps; preserve raw images in NPZ.
        visdump={'timestamp_s':ts}
        for cname in cfg.camera_names:
            visdump[cname+'_rgb']=vision[cname]['rgb']
            visdump[cname+'_depth_mm']=vision[cname]['depth_mm']
            visdump[cname+'_segmentation_int32']=vision[cname]['seg']
        # Optical flow is derived from RGB, not physics ground truth.
        import cv2
        first=vision[cfg.camera_names[0]]['rgb']
        flow=np.zeros((frames,H,W,2),dtype=np.float16)
        for k in range(1,frames):
            prev=cv2.cvtColor(first[k-1],cv2.COLOR_RGB2GRAY)
            cur=cv2.cvtColor(first[k],cv2.COLOR_RGB2GRAY)
            flow[k]=cv2.calcOpticalFlowFarneback(prev,cur,None,.5,2,13,2,5,1.1,0).astype(np.float16)
        visdump['front_optical_flow_rgb_estimated_px']=flow
        np.savez_compressed(out/'multiview_rgb_depth_seg_flow.npz',**visdump)
        # Dense point clouds are reconstructible from depth + full calibration.
        # Store per-frame sampled world-frame 3D to exercise 3D encoders directly.
        pcs={};valids={}
        xx,yy=np.meshgrid(np.arange(W),np.arange(H));xclip=2*(xx+.5)/W-1
        yclip=1-2*(yy+.5)/H
        for cname in cfg.camera_names:
            V=np.asarray(viewmats[cname]).reshape(4,4,order='F')
            P=np.asarray(projections[cname]).reshape(4,4,order='F')
            inv=np.linalg.inv(P@V)
            d=vision[cname]['depth_mm'].astype(np.float64)/1000
            seg=vision[cname]['seg'];pc=[];mask=[]
            for k in range(frames):
                ids=np.where(seg[k]<0,-1,seg[k]&((1<<24)-1))
                ys,xs=np.where((ids>=0)&(d[k]<far-.05)&(d[k]>near))
                if len(xs)==0:
                    pc.append(np.zeros((cfg.sample_points,3),np.float32));mask.append(False);continue
                ix=rng.choice(len(xs),cfg.sample_points,replace=len(xs)<cfg.sample_points)
                px=xs[ix];py=ys[ix];z=d[k,py,px];zb=(far-far*near/z)/(far-near)
                clip=np.stack([xclip[py,px],yclip[py,px],2*zb-1,np.ones_like(z)],axis=-1)
                world=clip@inv.T;world=world[:,:3]/world[:,3:4]
                pc.append(world.astype(np.float32));mask.append(True)
            pcs[cname+'_world_xyz_m']=np.stack(pc)
            valids[cname+'_valid']=np.asarray(mask)
        np.savez_compressed(out/'pointcloud_world.npz',timestamp_s=ts,**pcs,**valids)
        # Frame-wise contact proxy is *not* independent measured tactile sensing.
        YY,XX=np.mgrid[0:cfg.tactile_resolution,0:cfg.tactile_resolution]
        pad=np.exp(-((XX-7.5)**2+(YY-7.5)**2)/(2*2**2))
        tactile=np.zeros((frames,2,cfg.tactile_resolution,cfg.tactile_resolution),np.float32)
        for k in range(frames):
            proxy=5. if physical['held'][k] else float(physical['contact_N'][k])
            if tid in ('T5','T6') and any(e['name'] in ('button_activated','tap_detected') and abs(e['time_s']-ts[k])<.15 for e in events):
                proxy=max(proxy,3.)
            tactile[k]=pad[None]*proxy
        np.savez_compressed(out/'tactile_proxy.npz',timestamp_s=ts,
            pressure_2pads_PROXY=tactile,source=np.asarray('kinematic_grip_or_contact_proxy'))
        # Audio is synthesized at recorded event timestamps, not measured sound.
        sr=16000;sfx=np.zeros(int(duration*sr),np.float32)
        for i,e in enumerate(events):
            start=int(e['time_s']*sr);n=min(int(.18*sr),len(sfx)-start)
            if n<=0:continue
            tt=np.arange(n)/sr;noise=np.random.default_rng(400+spec['seed']+i).standard_normal(n)
            signal=(np.sin(2*np.pi*(390+110*i)*tt)+.1*noise)*np.exp(-23*tt)
            sfx[start:start+n]+=(.17*signal).astype(np.float32)
        wavfile.write(str(out/'impact_audio_SYNTHETIC.wav'),sr,np.int16(np.clip(sfx,-1,1)*32767))
        instruction=make_instruction(spec)
        exe=shutil.which('espeak-ng') or shutil.which('espeak')
        if not exe:raise RuntimeError('Real speech synthesis unavailable. Install espeak-ng; do not substitute silence.')
        cp=subprocess.run([exe,'-s','139','-w',str(out/'spoken_instruction.wav'),instruction],
                          capture_output=True,text=True)
        if cp.returncode or (out/'spoken_instruction.wav').stat().st_size<512:
            raise RuntimeError('espeak-ng failed: '+cp.stderr[-500:])
        # Mesh target object in LOCAL frame; URDF robot mesh refs from pybullet_data.
        meshpath=out/'target_cube_local.obj'
        trimesh.creation.box(extents=[edge]*3).export(meshpath)
        # Video is a lossy preview only; the original RGB arrays are preserved.
        with imageio.get_writer(str(out/'expert_preview.mp4'),fps=cfg.fps,codec='libx264',quality=6,
                                macro_block_size=16) as writer:
            for im in first:writer.append_data(im)
        pos_end=np.asarray(pb.getBasePositionAndOrientation(cube)[0])
        rel=float(np.linalg.norm(pos_end[:2]-np.array([gx,gy])))
        special_names={e['name'] for e in events}
        if tid=='T3':
            success=rel<.045 and pos_end[2]>.075
        elif tid=='T4':
            success=rel<.047 and pos_end[2]<.085
        elif tid=='T5':success='button_activated' in special_names
        elif tid=='T6':success='tap_detected' in special_names
        else:success=rel<.06 and pos_end[2]<.085
        # Scene and camera invariants, with full source provenance.
        sim_meta={
            'engine':'PyBullet DIRECT + TinyRenderer','physics_hz':cfg.physics_hz,
            'fps':cfg.fps,'duration_s':duration,'num_frames':frames,
            'camera_calibration':cameras,'object_body_ids':{'robot':robot,'target':cube,**static},
            'robot_urdf':'pybullet_data/franka_panda/panda.urdf',
            'robot_joint_ids':arm_j,'finger_joint_ids':fingers,'ee_link_id':ee_link,
            'mesh':'target_cube_local.obj','object_size_m':edge,
            'waypoints':wps,'max_ik_position_error_m':max(errors),
            'assisted_grasp':tid in ('T1','T3','T4','T7','T8'),
            'assisted_push':tid=='T2' and proxy_push,
            'virtual_contact_state_machine':tid in ('T5','T6'),
            'contact_force_source':'pybullet getContactPoints; does not include attachment forces',
            'tactile_source':'synthetic proxy generated from grip state/contact force',
            'impact_audio_source':'synthetic tones aligned to recorded events',
            'physics_limitations':'Kinematic joint reset; not dynamic actuation/real force closure',
            'success_criteria':'task-specific simulator measured final state or virtual button/tap event'
        }
        dump_json(out/'scene_and_replay.json',sim_meta)
        dump_json(out/'events.json',events)
        dump_json(out/'task.json',dict(spec, instruction=instruction,
                                      success=bool(success),final_object_position_world_m=pos_end.tolist(),
                                      final_goal_distance_xy_m=rel))
        return {'success':bool(success),'events':events,'duration_s':duration,
                'num_frames':frames,'final_error_xy_m':rel,'final_pos':pos_end.tolist(),
                'instruction':instruction,'max_ik_error_m':max(errors),
                'actual_ee_first':actual[0].tolist(),'actual_ee_last':actual[-1].tolist(),
                'phase_list':[m['phase'] for m in motion],
                'contact_peak_N':float(np.max(physical['contact_N'])),
                'grasp_assisted':bool(tid in ('T1','T3','T4','T7','T8'))}
    finally:
        pb.disconnect(cid)
