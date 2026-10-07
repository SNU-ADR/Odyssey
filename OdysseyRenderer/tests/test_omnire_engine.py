"""Native renderer contracts using CPU tensors and explicit rasterizer stubs."""
import copy
from types import SimpleNamespace

import json
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from odyssey_renderer.omnire import engine as module
from odyssey_renderer.omnire.engine import OmniReRenderEngine as Engine
from odyssey_renderer.omnire.camera_contract import validate_exposure
from odyssey_renderer.mtgs import mtgs

NAMES = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]


def metadata():
    times = np.array([1_000_000, 1_100_237, 1_200_043], dtype=np.int64)
    residual = np.tile(np.eye(4), (3, 8, 1, 1))
    residual[:, :, 0, 3] = np.arange(8)
    ego = np.tile(np.eye(4), (3, 1, 1)); ego[:, :3, 3] = [100., 200., 3.]
    exposure = dict(schema_version=1, mode="camera_frame", camera_structure="affine", frame_structure="affine",
        camera_names=NAMES, camera_ids=list(range(8)), training_timestamps_us=times,
        exposure=torch.cat([torch.eye(3)*2, torch.zeros(3,1)],1).repeat(8,1,1),
        residual=torch.zeros(8,3,3,4))
    return dict(schema_version=1, projection_mode="training_undistorted", near_plane=.1,
        rasterize_mode="classic", rigid_camera_time=True, camera_pose_policy="logged_residual_at_tick",
        camera_names=NAMES, camera_ids=list(range(8)), training_timestamps_us=times,
        image_timestamps_us=times[:,None].repeat(8,1), camera_to_ego=residual,
        recon2world_translation=np.array([100.,200.,0.]), exposure=exposure,
        ego_to_global=ego, camera_to_global=ego[:,None]@residual)



def manifest():
    return dict(schema_version=1, population_authority="checkpoint_instance_id_map",
                activation_policy="finite_positive_gaussian_dynamic_actors", actors={})

def set_asset(engine, asset):
    asset["background"]["config"]["omnire_actor_manifest"] = manifest()
    return engine.set_asset(asset)


def engine():
    e = Engine.__new__(Engine)
    e._actor_manifest = manifest()
    e.device = "cpu"; e.rasterize_mode = "classic"; e.radius_clip = 0.; e.render_depth = False
    e._calibration = metadata(); e.recon2global_translation = np.array([100.,200.,0.])
    e._simulation_base = 1_000_000; e._simulation_dt_us = 100_000
    e._camera_cache = {}; e.sensor_mapping = {}
    e._exposure = validate_exposure(e._calibration["exposure"], e._calibration["training_timestamps_us"], NAMES)
    e._clear_asset_caches()
    e._packed_rigid=None; e._rigid_tables={}; e._rigid_indices={}; e._actor_visibility={}
    e._rigid_valid_rows={}; e._vehicle_tokens=set(); e._render_agent_states={}
    e._render_agent_rows={}; e._suppressed_actors=frozenset()
    e._vehicle_pose_refs={}
    e.actor_pose_source="checkpoint"; e.lift="tick"
    e._horizon_factor=1.0; e._beyond_horizon=False
    e.calibrate_agent_state()
    return e


def test_native_tick_and_exact_saved_clock_select_same_row():
    e=engine()
    assert e._resolve_frame(1_100_000)==1
    assert e._resolve_frame(1_100_237)==1
    for time in (999_999,1_050_000,1_300_000,float("nan")):
        with pytest.raises(ValueError): e._resolve_frame(time)


def test_camera_calibration_reproduces_logged_and_virtual_ego_without_subtick():
    e=engine(); cams={name:dict(height=2,width=3,intrinsic=np.eye(3)) for name in NAMES[:3]}
    logged=e._ego_at_tick([0.,0.,0.],1)
    got,_,_,_,_=e._camera_inputs(cams,logged,1)
    expected=e._calibration["camera_to_global"][1,:3].copy(); expected[:,:3,3]-=e.recon2global_translation
    np.testing.assert_allclose(got.numpy(),expected,atol=1e-6)
    shifted=e._ego_at_tick([5.,-2.,0.],1)
    out,_,_,_,_=e._camera_inputs(cams,shifted,1)
    np.testing.assert_allclose((out-got)[:,:3,3],np.tile([5.,-2.,0.],(3,1)),atol=1e-6)
    assert len(e._camera_cache)==3


def test_render_copies_states_batches_postprocess_and_exposes_after_sky(monkeypatch):
    monkeypatch.delenv("ODYSSEY_RENDER_CAMS",raising=False)
    _scene_cameras(monkeypatch)
    e=engine(); calls=[]; outputs=[]
    def collect(frame,camera_index,camera,common,assembler=None):
        calls.append((frame,camera_index))
        return dict(means=torch.zeros(1,3),scales=torch.ones(1,3),quats=torch.tensor([[1.,0.,0.,0.]]),
                    opacities=torch.ones(1),rgbs=torch.zeros(1,3)), {"actor"+str(camera_index)}
    e._collect_camera=collect
    def raster(**kwargs):
        assert kwargs["near_plane"]==.1 and kwargs["colors"].shape[0]==1
        assert kwargs["Ks"].device.type=="cpu"
        return torch.full((1,2,3,3),.2),torch.ones(1,2,3,1),{}
    monkeypatch.setattr(mtgs,"rasterization",raster)
    e.composite_background=lambda rgb,*args:rgb+.1
    def post(rgb,render,alpha,maps,shape,mode):
        outputs.append(rgb.clone()); assert rgb.shape==(3,2,3,3)
        return dict(cameras={},lidars={})
    e._postprocess_render=post
    states={"ego":np.array([100.,200.,0.]),"actor":np.array([101.,202.,.1])}
    before=copy.deepcopy(states)
    result=e.render(dict(timestamp=1_100_000,agent_state=states,
        cameras={name:dict(height=2,width=3,intrinsic=np.eye(3)) for name in NAMES[:3]}))
    assert calls==[(1,0),(1,1),(1,2)] and len(outputs)==1
    torch.testing.assert_close(outputs[0],torch.full((3,2,3,3),.6))
    for name in states: np.testing.assert_array_equal(states[name],before[name])
    assert e.rendered_tokens()=={"actor0","actor1","actor2"}
    assert result["omnire"]["source_timestamp_us"]==1_100_237
    np.testing.assert_allclose(result["ego2global"],e._calibration["ego_to_global"][1])
    calibration=result["camera_calibrations"]
    assert list(calibration)==NAMES[:3]
    for name in NAMES[:3]:
        row=calibration[name]
        np.testing.assert_array_equal(row["camera_to_ego"],e._calibration["camera_to_ego"][1,NAMES.index(name)])
        np.testing.assert_array_equal(row["cam_intrinsic"],np.eye(3,dtype=np.float32))
        np.testing.assert_array_equal(row["distortion"],np.zeros(5))
        assert row["projection_semantics"]=="training_undistorted"
        assert row["camera_to_ego"].shape==(4,4) and row["cam_intrinsic"].shape==(3,3)
    calibration["CAM_F0"]["camera_to_ego"][0,3]=999.
    assert e._calibration["camera_to_ego"][1,0,0,3]==0.


def test_an_unlifted_render_reports_the_rig_not_the_logged_residual(monkeypatch):
    """The reported calibration is the rig relative to the reconstructed ego. The saved camera_to_ego is
    relative to the logged ego, so per row it is off by the logged ego height error against the reconstructed
    road; even with lift off (tick) the image comes from the reconstructed camera.
    (With lift on, the render residual itself is the rig: see calibrate_agent_state.)"""
    monkeypatch.delenv("ODYSSEY_RENDER_CAMS",raising=False)
    _scene_cameras(monkeypatch)
    for lift in ("tick",):
        e=engine()
        rig=e._calibration["camera_to_ego"].copy()                  # in the metadata the rig equals the saved residual
        e._calibration["camera_to_ego"][...,2,3]+=np.array([-2.,1.,3.])[:,None]   # logged ego height error
        e._calibration["ego_to_global"][:,2,3]-=np.array([-2.,1.,3.])             # keeps the camera unchanged
        e.lift=lift; e._road_surface=None; e.calibrate_agent_state()
        e._collect_camera=lambda frame,camera_index,camera,common,assembler=None:(dict(means=torch.zeros(1,3),
            scales=torch.ones(1,3),quats=torch.tensor([[1.,0.,0.,0.]]),opacities=torch.ones(1),
            rgbs=torch.zeros(1,3)),set())
        monkeypatch.setattr(mtgs,"rasterization",lambda **kw:(torch.full((1,2,3,3),.2),torch.ones(1,2,3,1),{}))
        e.composite_background=lambda rgb,*args:rgb
        e._postprocess_render=lambda *args:dict(cameras={},lidars={})
        result=e.render(dict(timestamp=1_100_000,agent_state={"ego":np.array([100.,200.,0.])},
            cameras={name:dict(height=2,width=3,intrinsic=np.eye(3)) for name in NAMES[:3]}))
        for name in NAMES[:3]:
            np.testing.assert_allclose(result["camera_calibrations"][name]["camera_to_ego"],
                                       rig[1,NAMES.index(name)],atol=1e-9)

class Model:
    def __init__(self):self.geometry_calls=0;self.color_calls=0
    def get_global_gaussians(self,**kwargs):
        self.geometry_calls+=1
        return dict(means=torch.zeros(1,3),scales=torch.ones(1,3),quats=torch.tensor([[1.,0.,0.,0.]]),opacities=torch.ones(1))
    def get_gaussian_rgbs(self,**kwargs):
        self.color_calls+=1
        return torch.full((1,3),.5)


def test_tl_bypasses_actor_geometry_and_sh_caches_and_scorer(monkeypatch):
    class TL(Model):pass
    monkeypatch.setattr(module,"OmniReTrafficLightSubModel",TL)
    e=engine();bg=Model();tl=TL();hidden=Model()
    e.submodel_names={"background":"bg","lights":"tl","deform":"hidden"}
    e.gaussian_models={"bg":bg,"tl":tl,"hidden":hidden}
    e._actor_visibility={"deform":np.array([True,False,True])}
    e._global_gaussians_cached=lambda name,model,q,t,stamp:model.get_global_gaussians(timestamp=stamp)
    def colors(name,model):
        assert model is not tl
        return None
    e._sh_colors_cached=colors
    common={}
    for camera in (0,1,2):
        gs,tokens=e._collect_camera(1,camera,torch.eye(4),common)
        assert len(gs["means"])==2 and tokens==set()
    assert bg.geometry_calls==tl.geometry_calls==1
    assert tl.color_calls==3 and hidden.geometry_calls==0


def test_rigid_sampling_uses_per_camera_time_and_current_visibility(monkeypatch):
    e=engine(); e._calibration["image_timestamps_us"][1,0]+=40_000
    e._calibration["image_timestamps_us"][1,1]-=40_000
    class Rigid(Model):
        def get_means(self,global_quat,global_trans):return global_trans[None]
        def get_scales(self):return torch.ones(1,3)
        def get_quats(self,**kwargs):return torch.tensor([[1.,0.,0.,0.]])
        def get_opacity(self):return torch.ones(1)
    rigid=Rigid(); e.submodel_names={"actor":"r"};e.gaussian_models={"r":rigid}
    positions=torch.zeros(3,1,3);positions[:,0,0]=torch.arange(3,dtype=torch.float32)
    quats=torch.tensor([1.,0.,0.,0.]).repeat(3,1,1);visible=torch.ones(3,1,dtype=torch.bool)
    e._rigid_tables={"actor":(positions,quats,visible)};e._packed_rigid=(positions,quats,visible)
    e._rigid_indices={"actor":0};e._actor_visibility={"actor":np.ones(3,dtype=bool)}
    e._sh_colors_cached=lambda *args:None
    front,_=e._collect_camera(1,0,torch.eye(4),{})
    left,_=e._collect_camera(1,1,torch.eye(4),{})
    assert front["means"][0,0]>1 and left["means"][0,0]<1
    visible[1]=False
    with pytest.raises(ValueError,match="no Gaussian"):e._collect_camera(1,0,torch.eye(4),{})


def test_reactive_vehicle_uses_simulated_pose_even_after_logged_visibility_ends():
    e=engine()
    class Rigid(Model):
        def get_means(self,global_quat,global_trans):return global_trans[None]
        def get_scales(self):return torch.ones(1,3)
        def get_quats(self,global_quat,global_trans):return global_quat[None]
        def get_opacity(self):return torch.ones(1)
    rigid=Rigid(); background=Model()
    e.submodel_names={"background":"bg","vehicle":"r"}
    e.gaussian_models={"bg":background,"r":rigid}
    positions=torch.tensor([[[1.,2.,3.]],[[2.,2.,4.]],[[0.,0.,10000.]]])
    quats=torch.tensor([1.,0.,0.,0.]).repeat(3,1,1)
    visible=torch.tensor([[True],[True],[False]])
    e._rigid_tables={"vehicle":(positions,quats,visible)}
    e._packed_rigid=(positions,quats,visible)
    e._rigid_indices={"vehicle":0}
    e._rigid_valid_rows={"vehicle":torch.tensor([0,1])}
    e._actor_visibility={"vehicle":np.array([True,True,False])}
    e._vehicle_tokens={"vehicle"}
    e.actor_pose_source="scenario"
    e._ground_tree=cKDTree([[1.,2.],[10.,5.]])
    e._ground_z=np.array([2.,8.])
    e._vehicle_heights={"vehicle":2.}
    e._render_agent_states={"vehicle":np.array([10.,5.,0.,0.,0.,np.pi/2])}
    e._global_gaussians_cached=lambda name,model,q,t,stamp:model.get_global_gaussians()
    e._sh_colors_cached=lambda *args:None

    gs,tokens=e._collect_camera(2,0,torch.eye(4),{})
    assert tokens=={"vehicle"}
    torch.testing.assert_close(gs["means"][-1],torch.tensor([10.,5.,9.]))
    torch.testing.assert_close(gs["quats"][-1],
                               torch.tensor([2**-.5,0.,0.,2**-.5]),atol=1e-6,rtol=0)
    e._vehicle_pose_refs={}
    e._render_agent_states={"vehicle":np.array([1.,2.,0.,0.,0.,0.])}
    gs,_=e._collect_camera(2,0,torch.eye(4),{})
    torch.testing.assert_close(gs["means"][-1],torch.tensor([1.,2.,3.]))
    e._beyond_horizon=True
    gs,tokens=e._collect_camera(2,0,torch.eye(4),{})
    assert tokens=={"vehicle"} and len(gs["means"])==2
    e._render_agent_states={}
    gs,tokens=e._collect_camera(2,0,torch.eye(4),{})
    assert tokens==set() and len(gs["means"])==1


def _reactive_on_road(monkeypatch, static):
    """One reactive vehicle (1.6 m tall) drives on the sloped road z = 2 + 0.1 x."""
    e = engine()
    surface = module.RoadSurface.__new__(module.RoadSurface)
    surface.offset = 0.4
    surface.ground = lambda xy, yaw: (2.4 + 0.1 * np.atleast_2d(xy)[:, 0],
                                      np.tile([0., 0., 1.], (len(np.atleast_2d(xy)), 1)))
    e._path_lift = surface
    trans = torch.tensor([[[5., 0., 7.0]], [[5., 0., 7.0]], [[5., 0., 7.0]]])
    quats = torch.tensor([1., 0., 0., 0.]).repeat(3, 1, 1)
    e._packed_rigid = (trans, quats, torch.ones(3, 1, dtype=torch.bool))
    e._rigid_tables = {"car": (trans, quats, torch.ones(3, 1, dtype=torch.bool))}
    e._rigid_indices = {"car": 0}
    e._rigid_valid_rows = {"car": torch.tensor([0, 1, 2])}
    e._vehicle_tokens = {"car"}
    e.actor_pose_source = "scenario"
    e._vehicle_dims = {}
    tracks = {"car": {"state": {"height": np.full(3, 1.6), "length": np.full(3, 4.6), "width": np.full(3, 1.9)}}}
    stub = SimpleNamespace(managers={"scenario_manager": SimpleNamespace(current_scene={"object_track": tracks})})
    monkeypatch.setattr(Engine, "engine", property(lambda self: stub))
    e._actor_height = e._measure_actor_height()
    e._static_actors = np.array([static])
    return e


@pytest.mark.parametrize("x", [0.0, 10.0, 30.0])
def test_a_reactive_vehicle_stands_on_the_road_under_it_now(monkeypatch, x):
    """A simulator-driven vehicle has no trained row, so it stands on the road under its current (x, y), box bottom on the road."""
    e = _reactive_on_road(monkeypatch, static=False)
    pos, quat = e._stand_simulated_vehicle_on_road(0, "car", np.array([x, 0., 0., 0., 0., 0.]))
    assert pos[2] == pytest.approx(2.0 + 0.1 * x + 0.8, abs=1e-6)   # road + box height / 2
    # The nose pitches up along the 0.1 grade (5.7 deg); yaw stays 0.
    ang = module.quat_to_angle(torch.as_tensor(quat, dtype=torch.float64), focus=["yaw", "pitch"])
    assert float(ang["yaw"]) == pytest.approx(0.0, abs=1e-6)
    assert abs(float(ang["pitch"])) == pytest.approx(np.arctan(0.1), abs=1e-4)


def test_a_parked_vehicle_in_a_reactive_run_keeps_the_learned_height(monkeypatch):
    """A parked vehicle excluded from the road bake keeps its trained checkpoint height in reactive runs too."""
    e = _reactive_on_road(monkeypatch, static=True)
    e._ground_tree = None                                    # nearby ground points are not read
    pos, quat = e._simulated_vehicle_pose("car", np.array([5., 0., 0., 0., 0., 0.]), keep_learned_z=True)
    assert float(pos[2]) == pytest.approx(7.0)


ON_ROAD = 2.0 + 0.1 * 20.0 + 0.8


@pytest.mark.parametrize("static, held, want_z", [
    (False, False, ON_ROAD),     # moving vehicle
    (True, True, 7.0),           # parked in the log and held at its log pose by the simulator
    (True, False, ON_ROAD),      # parked in the log but driven off by IDM (otherwise it floats)
    (False, True, ON_ROAD),      # replays its log pose but was moving in the log -- on the road, as in log replay
])
def test_collect_camera_keeps_the_learned_height_only_for_a_parked_car_the_simulator_holds(
        monkeypatch, static, held, want_z):
    """In _collect_camera the trained height is kept only for a road-excluded parked car the simulator holds at its log pose."""
    class Rigid(Model):
        def get_means(self, global_quat, global_trans): return global_trans[None]
        def get_scales(self): return torch.ones(1, 3)
        def get_quats(self, global_quat, global_trans): return global_quat[None]
        def get_opacity(self): return torch.ones(1)
    e = _reactive_on_road(monkeypatch, static=static)
    e._held_at_log_pose = frozenset({"car"} if held else ())
    e.submodel_names = {"car": "r"}
    e.gaussian_models = {"r": Rigid()}
    e._actor_visibility = {"car": np.array([True, True, True])}
    e._ground_tree = cKDTree([[5., 0.]]); e._ground_z = np.array([3.0]); e._vehicle_heights = {"car": 1.6}  # the old path would give 3.8
    e._actor_road = e._smooth_actor_road()
    e._render_agent_states = {"car": np.array([20., 0., 0., 0., 0., 0.])}
    e._sh_colors_cached = lambda *args: None
    gs, tokens = e._collect_camera(1, 0, torch.eye(4), {})
    assert tokens == {"car"}
    assert float(gs["means"][-1][2]) == pytest.approx(want_z, abs=1e-5)
    assert float(gs["means"][-1][0]) == pytest.approx(20.0)       # xy comes from the simulator


def test_shared_native_postprocess_keeps_single_restorer_batch(monkeypatch):
    e=engine();e.sensor_mapping={name:i for i,name in enumerate(NAMES[:3])}
    calls=[]
    class Restorer:
        def restore_batch(self,images):calls.append(len(images));return images
    monkeypatch.setattr(mtgs,"get_restorer",lambda:Restorer())
    monkeypatch.setattr(mtgs,"_gpu_undistort_enabled",lambda:False)
    raw=torch.tensor([.1,.2,.3]).repeat(3,2,3,1)
    y,x=np.mgrid[:2,:3].astype(np.float32)
    result=e._postprocess_render(raw,raw,torch.ones(3,2,3,1),[(x,y)]*3,dict(height=2,width=3),"RGB")
    assert calls==[3]
    np.testing.assert_array_equal(result["cameras"]["CAM_F0"]["image"][0,0],[76,51,25])


@pytest.mark.parametrize("mutation",["missing", "baked", "wrong_raster", "missing_ego"])
def test_noncanonical_banks_fail_before_model_construction(mutation):
    e=engine(); asset=dict(background=dict(config=dict(omnire_calibration=metadata(),recon2world_translation=[100.,200.,0.])))
    cfg=asset["background"]["config"]
    if mutation=="missing":del cfg["omnire_calibration"]
    if mutation=="baked":cfg["c018_render"]={"rigid_camera_time_baked":True}
    if mutation=="wrong_raster":e.rasterize_mode="antialiased"
    if mutation=="missing_ego":
        del cfg["omnire_calibration"]["ego_to_global"];del cfg["omnire_calibration"]["camera_to_global"]
    with pytest.raises(ValueError):set_asset(e, asset)


def test_set_asset_keeps_cpu_batched_rigid_and_all_actor_visibility(monkeypatch):
    e=engine(); moves=[]
    class TL:
        def to(self,device):moves.append(device);return self
    monkeypatch.setattr(module,"OmniReTrafficLightSubModel",TL)
    def construct(self,asset):
        self.gaussian_models={"traffic_lights":TL()}
        self.recon2global_translation=asset["background"]["config"]["recon2world_translation"]
    monkeypatch.setattr(module.MTGSRenderEngine,"set_asset",construct)
    cfg=dict(type="VanillaGaussianSplattingModel",omnire_calibration=metadata(),
        recon2world_translation=[100.,200.,0.],omnire_environment=torch.zeros(6,2,2,3))
    valid=torch.tensor([True,False,True])
    rigid=dict(config=dict(type="RigidSubModel",log_timestamps=metadata()["training_timestamps_us"]),
        state_dict=dict(instance_trans=torch.zeros(3,3),instance_quats=torch.tensor([1.,0.,0.,0.]).repeat(3,1),instance_valid=valid))
    deform=dict(config=dict(type="OmniReDeformableSubModel"),state_dict=dict(instance_valid=valid))
    asset=dict(background=dict(config=cfg),na_rigid_abc=rigid,na_deform_def=deform)
    set_asset(e, asset)
    assert moves==["cpu"]
    assert e._packed_rigid[0].shape==(3,1,3) and e._packed_rigid[0].device.type=="cpu"
    assert e._rigid_indices=={"abc":0}
    assert not e._actor_visibility["def"][1]
    assert e.original_sky.shape==(6,2,2,3)
    del deform["state_dict"]["instance_valid"]
    with pytest.raises(ValueError,match="instance_valid"):set_asset(e, asset)


def test_extended_horizon_holds_last_saved_row_and_still_bounds_ticks():
    e=engine(); e._horizon_factor=2.0
    assert e._tick_index(1_100_237)==(1,1)
    assert e._tick_index(1_300_000)==(3,2)          # first tick past the three saved rows
    assert e._tick_index(1_500_000)==(5,2)          # last tick inside 3 rows x 2
    assert e._resolve_frame(1_500_000)==2
    for time in (1_600_000,1_350_000,999_999):      # past 2x, off-grid, before base
        with pytest.raises(ValueError): e._tick_index(time)


def test_horizon_extension_factor_must_be_at_least_one():
    with pytest.raises(ValueError,match="horizon_extension_factor"):
        Engine(horizon_extension_factor=0.5)


def test_beyond_saved_horizon_hides_actors_but_keeps_background_and_lights(monkeypatch):
    class TL(Model):pass
    monkeypatch.setattr(module,"OmniReTrafficLightSubModel",TL)
    e=engine();bg=Model();tl=TL();actor=Model()
    e.submodel_names={"background":"bg","lights":"tl","actor":"a"}
    e.gaussian_models={"bg":bg,"tl":tl,"a":actor}
    e._actor_visibility={"actor":np.ones(3,dtype=bool)}
    e._global_gaussians_cached=lambda name,model,q,t,stamp:model.get_global_gaussians(timestamp=stamp)
    e._sh_colors_cached=lambda *args:None
    gs,tokens=e._collect_camera(2,0,torch.eye(4),{})
    assert len(gs["means"])==3 and tokens=={"actor"}
    e._beyond_horizon=True
    gs,tokens=e._collect_camera(2,0,torch.eye(4),{})
    assert len(gs["means"])==2 and tokens==set()


def _scene_cameras(monkeypatch, extra=None):
    """Scenario the renderer reads for lift: CAM_F0 sensor2ego is identity (same as the CAM_F0 residual in the test metadata)."""
    scene = {"cameras": {"CAM_F0": {"sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0],
                                    "sensor2ego_translation": [0.0, 0.0, 0.0]}}}
    scene.update(extra or {})
    stub = SimpleNamespace(managers={"scenario_manager": SimpleNamespace(current_scene=scene)})
    monkeypatch.setattr(Engine, "engine", property(lambda self: stub))


def test_road_surface_moves_the_actors_but_the_ego_stays_on_the_logged_path(monkeypatch):
    """The camera stays in the frame of the reconstructed logged pose. If the closed-loop ego drifts
    onto a curb or driveway, the field height there is not applied to the camera."""
    class Field:                            # a field reporting a road 5 m above the logged trajectory
        cover, offset, spread = 1.0, 0.0, 0.0
        def lift(self, x, y, heading):
            pose = np.eye(4); pose[:3, 3] = [x, y, 8.0]; return pose
    monkeypatch.setattr(module.GroundField, "load", staticmethod(lambda path: None))
    monkeypatch.setattr(module, "RoadSurface", lambda field, path, anchor: Field())
    e = engine()
    e._calibration["ego_to_global"][:, 0, 3] = [100., 110., 120.]    # a logged trajectory long enough to project onto
    e._calibration["camera_to_global"][:, :, 0, 3] += np.array([0., 10., 20.])[:, None]
    _scene_cameras(monkeypatch)
    e.lift, e._road_surface = "all", "baked.npz"
    e.calibrate_agent_state()
    assert isinstance(e._path_lift, Field)                 # actors read the field
    pose = e._ego_at_tick([0., 0., 0.], 1)
    assert float(pose[2, 3]) == pytest.approx(3.0)         # the ego keeps the logged trajectory height


def test_road_surface_with_a_camera_only_lift_is_refused(monkeypatch):
    monkeypatch.delenv("ODYSSEY_OMNIRE_ROAD_SURFACE", raising=False)
    with pytest.raises(ValueError, match="lift=all"):
        Engine(lift="ego", road_surface="baked.npz")


def _lidar_engine(monkeypatch, heights):
    """Engine with a RoadSurface: the road is the plane z = 2.0 + offset (0.4); two actors."""
    e = engine()
    surface = module.RoadSurface.__new__(module.RoadSurface)
    surface.offset = 0.4
    surface.ground = lambda xy, yaw: (np.full(len(np.atleast_2d(xy)), 2.4),
                                      np.tile([0.0, 0.0, 1.0], (len(np.atleast_2d(xy)), 1)))
    e._path_lift = surface
    trans = torch.tensor([[[0., 0., 9.], [5., 0., -3.]]] * 3, dtype=torch.float64)  # the trained z is deliberately wrong
    quats = torch.tensor([[[1., 0., 0., 0.]] * 2] * 3, dtype=torch.float64)
    e._packed_rigid = (trans, quats, torch.ones(3, 2, dtype=torch.bool))
    e._rigid_indices = {"car_a": 0, "car_b": 1}
    tracks = {t: {"state": {"height": np.full(3, h), "length": np.full(3, 4.6), "width": np.full(3, 1.9)}}
              for t, h in heights.items()}
    stub = SimpleNamespace(managers={"scenario_manager": SimpleNamespace(
        current_scene={"object_track": tracks})})
    monkeypatch.setattr(Engine, "engine", property(lambda self: stub))
    return e, trans, quats


def test_lidar_road_puts_the_box_bottom_on_it_and_ignores_the_learned_z(monkeypatch):
    e, trans, quats = _lidar_engine(monkeypatch, {"car_a": 1.6, "car_b": 3.0})
    e._actor_height = e._measure_actor_height()
    e._actor_road = e._smooth_actor_road()
    pos, _ = e._stand_actors_on_road(trans[0], quats[0], 0)
    # origin = road (2.0) + box height / 2. The trained z (9, -3) is not used.
    np.testing.assert_allclose(pos[:, 2].numpy(), [2.0 + 0.8, 2.0 + 1.5])


def test_a_static_actor_stays_where_the_checkpoint_put_it(monkeypatch, tmp_path):
    """A stationary actor excluded by the road bake is not moved by the renderer either.

    The rigid node is not the whole car: most of a parked vehicle is baked into the background, so
    lifting only the node onto the road misaligns it with the rest and slices the car horizontally.
    The verdict reads drop_static, which the bake writes next to the road surface.
    """
    e, trans, quats = _lidar_engine(monkeypatch, {"car_a": 1.6, "car_b": 3.0})
    trans = trans.clone()
    trans[1, 1, 0], trans[2, 1, 0] = 8.0, 11.0                  # only car_b moves (6 m)
    e._packed_rigid = (trans, quats, torch.ones(3, 2, dtype=torch.bool))
    (tmp_path / "baked.json").write_text(json.dumps({"drop_static": 2.0}))
    e._road_surface = str(tmp_path / "baked.npz")
    e._actor_height = e._measure_actor_height()
    e._actor_road = e._smooth_actor_road()
    e._static_actors = e._left_where_the_checkpoint_put_them()
    assert list(e._static_actors) == [True, False]
    pos, rot = e._stand_actors_on_road(trans[0], quats[0], 0)
    assert float(pos[0, 2]) == pytest.approx(9.0)               # parked car: trained z kept
    assert float(pos[1, 2]) == pytest.approx(2.0 + 1.5)         # moving car: on the road
    np.testing.assert_allclose(rot[0].numpy(), quats[0, 0].numpy())


def test_without_a_bake_verdict_every_actor_is_still_lifted(monkeypatch, tmp_path):
    """A road surface without drop_static (baked earlier) keeps the old behaviour."""
    e, trans, quats = _lidar_engine(monkeypatch, {"car_a": 1.6, "car_b": 3.0})
    e._road_surface = str(tmp_path / "baked.npz")               # no json next to it
    assert e._left_where_the_checkpoint_put_them() is None


def test_lidar_road_refuses_an_actor_without_a_box_height(monkeypatch):
    e, _, _ = _lidar_engine(monkeypatch, {"car_a": 1.6})
    with pytest.raises(ValueError, match="car_b"):
        e._measure_actor_height()


def _moving_car(monkeypatch, road):
    """One car moves 0.37 m per row along x (out of phase with the bump period). road(xy) -> road height."""
    e = engine()
    surface = module.RoadSurface.__new__(module.RoadSurface)
    surface.offset = 0.0
    surface.ground = lambda xy, yaw: (road(np.atleast_2d(xy)), np.tile([0., 0., 1.], (len(np.atleast_2d(xy)), 1)))
    e._path_lift = surface
    n = 60
    trans = torch.zeros(n, 1, 3, dtype=torch.float64); trans[:, 0, 0] = 0.37 * torch.arange(n, dtype=torch.float64)
    quats = torch.zeros(n, 1, 4, dtype=torch.float64); quats[..., 0] = 1.0
    e._packed_rigid = (trans, quats, torch.ones(n, 1, dtype=torch.bool))
    e._rigid_indices = {"car": 0}
    stub = SimpleNamespace(managers={"scenario_manager": SimpleNamespace(current_scene={"object_track": {
        "car": {"state": {"height": np.full(3, 1.6), "length": np.full(3, 4.6), "width": np.full(3, 1.9)}}}})})
    monkeypatch.setattr(Engine, "engine", property(lambda self: stub))
    e._actor_height = e._measure_actor_height()
    e._actor_road = e._smooth_actor_road()
    return e, trans, quats


def test_a_bumpy_road_does_not_shake_the_car(monkeypatch):
    """+-10 cm bumps every 0.25 m: sampling a single point would make the car jump 20 cm per row."""
    bump = lambda xy: 0.1 * np.sign(np.sin(xy[:, 0] * np.pi / 0.25 + 0.3))
    e, trans, quats = _moving_car(monkeypatch, bump)
    assert np.abs(np.diff(bump(trans[10:50, 0, :2].numpy()))).max() > 0.19   # the jump a single-point sample would give
    z = np.array([float(e._stand_actors_on_road(trans[r], quats[r], r)[0][0, 2]) for r in range(10, 50)])
    assert np.abs(np.diff(z)).max() < 0.02


def test_the_wheel_plane_reads_the_grade_and_caps_it(monkeypatch):
    e, trans, quats = _moving_car(monkeypatch, lambda xy: 0.05 * xy[:, 0] + 0.02 * xy[:, 1])
    np.testing.assert_allclose(e._actor_road[1][10:50, 0], [[0.05, 0.02]] * 40, atol=1e-9)
    # a sub-row position (0.4 m ahead) also stands on the same plane
    moved = trans[20].clone(); moved[0, 0] += 0.4
    pos, _ = e._stand_actors_on_road(moved, quats[20], 20)
    assert float(pos[0, 2]) == pytest.approx(0.05 * (20 * 0.37 + 0.4) + 0.8, abs=1e-9)
    steep, _, _ = _moving_car(monkeypatch, lambda xy: 0.5 * xy[:, 0])
    assert np.degrees(np.arctan(np.linalg.norm(steep._actor_road[1][30, 0]))) == pytest.approx(8.0)


def test_a_lifted_ego_behind_the_log_gets_the_camera_height_of_where_it_is(monkeypatch):
    """GT ego z is flat (3 m) while the reconstructed camera climbs (4, 5, 6 m), so the saved per-row
    calibration absorbs the difference (1, 2, 3 m). An ego lagging the log at row 0's position, rendered
    at row 2, must get the camera height there, 4 m, not GT z (3) + row-2 calibration (3) = 6 m."""
    e = engine()
    e._calibration["ego_to_global"][:, 0, 3] = [100., 110., 120.]
    e._calibration["camera_to_global"][:, :, 0, 3] += np.array([0., 10., 20.])[:, None]
    e._calibration["camera_to_global"][:, :, 2, 3] = np.array([4., 5., 6.])[:, None]
    e._calibration["camera_to_ego"][:, :, 2, 3] = np.array([1., 2., 3.])[:, None]
    _scene_cameras(monkeypatch)
    e.lift = "ego"; e._road_surface = None
    e.calibrate_agent_state()
    ego = e._ego_at_tick([0., 0., 0.], 2)                     # row 0 position (reconstruction coordinates)
    cams = {name: dict(height=2, width=3, intrinsic=np.eye(3)) for name in NAMES[:1]}
    camera, _, _, _, _ = e._camera_inputs(cams, ego, 2)
    assert float(camera[0, 2, 3]) == pytest.approx(4.0)
    np.testing.assert_allclose(e._camera_residual[:, 0, 2, 3], 0.0, atol=1e-9)   # the calibration does not depend on the row


def test_a_time_shifted_actor_is_drawn_at_its_own_row_not_the_sim_frame():
    """An actor shifted by the replay clock must be drawn at the row the simulator placed it on.

    Drawing it at the sim frame splits the scored position from the visible one. Only VEHICLE is
    drawn at the simulated pose, so pedestrians and cyclists are not covered by that bypass.
    The row is the source of truth.
    """
    e = engine()

    class Rigid(Model):
        def get_means(self, global_quat, global_trans): return global_trans[None]
        def get_scales(self): return torch.ones(1, 3)
        def get_quats(self, global_quat, global_trans): return global_quat[None]
        def get_opacity(self): return torch.ones(1)

    e.submodel_names = {"background": "bg", "walker": "w"}
    e.gaussian_models = {"bg": Model(), "w": Rigid()}
    positions = torch.tensor([[[1., 1., 0.]], [[2., 2., 0.]], [[3., 3., 0.]]])
    quats = torch.tensor([1., 0., 0., 0.]).repeat(3, 1, 1)
    visible = torch.tensor([[True], [True], [True]])
    e._rigid_tables = {"walker": (positions, quats, visible)}
    e._packed_rigid = (positions, quats, visible)
    e._rigid_indices = {"walker": 0}
    e._actor_visibility = {"walker": np.array([True, False, True])}
    e._global_gaussians_cached = lambda name, model, q, t, stamp: model.get_global_gaussians()
    e._sh_colors_cached = lambda *args: None

    # without row information, unchanged: the pose at sim frame 0
    gs, tokens = e._collect_camera(0, 0, torch.eye(4), {})
    torch.testing.assert_close(gs["means"][-1], torch.tensor([1., 1., 0.]))

    # if the simulator is replaying this actor at row 2, draw that row's pose
    e._render_agent_rows = {"walker": 2}
    gs, tokens = e._collect_camera(0, 0, torch.eye(4), {})
    assert tokens == {"walker"}
    torch.testing.assert_close(gs["means"][-1], torch.tensor([3., 3., 0.]))

    # visibility is also read at that row; row 1 is a frame where this actor is not reconstructed
    e._render_agent_rows = {"walker": 1}
    gs, tokens = e._collect_camera(0, 0, torch.eye(4), {})
    assert tokens == set()


def test_a_time_shifted_actor_gets_its_own_row_timestamp_for_deformation():
    """Pedestrians evaluate deformation and colour at a timestamp, which must also come from the actor's row."""
    e = engine()
    seen = {}

    class Deformable(Model):
        def get_global_gaussians(self, quat=None, trans=None, timestamp=None):
            seen["gauss"] = timestamp
            return {"means": torch.zeros(1, 3), "scales": torch.ones(1, 3),
                    "quats": torch.tensor([[1., 0., 0., 0.]]), "opacities": torch.ones(1)}

        def get_gaussian_rgbs(self, camera_to_worlds, timestamp, device=None):
            seen["rgb"] = timestamp
            return torch.zeros(1, 3)

    e.submodel_names = {"background": "bg", "walker": "w"}
    e.gaussian_models = {"bg": Model(), "w": Deformable()}
    e._global_gaussians_cached = lambda name, model, q, t, stamp: \
        model.get_global_gaussians(quat=q, trans=t, timestamp=stamp)
    e._sh_colors_cached = lambda *args: None
    e._render_agent_rows = {"walker": 2}

    e._collect_camera(0, 0, torch.eye(4), {})
    stamps = e._calibration["training_timestamps_us"]
    assert seen["gauss"] == int(stamps[2])      # its own row 2, not sim frame 0
    assert seen["rgb"] == int(stamps[2])        # pose and colour must use the same time


def _two_actor_engine():
    """An engine drawing two checkpoint-posed actors plus the background."""
    e = engine()
    e.submodel_names = {"background": "bg", "a": "ma", "b": "mb"}
    e.gaussian_models = {"bg": Model(), "ma": Model(), "mb": Model()}
    e._actor_visibility = {"a": np.array([True] * 3), "b": np.array([True] * 3)}
    e._global_gaussians_cached = lambda name, model, q, t, stamp: model.get_global_gaussians()
    e._sh_colors_cached = lambda *args: None
    return e


@pytest.mark.parametrize("pose_source", ["checkpoint", "scenario"])
def test_a_suppressed_actor_is_neither_drawn_nor_reported_as_rendered(pose_source):
    """An actor the manager held out of the world must not reach the image.

    The renderer poses actors from the checkpoint, so absence from AGENT_STATE does not
    remove them -- under `checkpoint` nothing is gated by it at all. The suppression set
    is what makes the drawn world equal the simulated one, and it has to hold for every
    pose source and every actor type, not just the reactive vehicles.
    """
    e = _two_actor_engine()
    e.actor_pose_source = pose_source
    e._suppressed_actors = frozenset({"a"})

    gs, tokens = e._collect_camera(0, 0, torch.eye(4), {})

    assert tokens == {"b"}
    assert e.gaussian_models["ma"].geometry_calls == 0    # never even posed
    assert e.gaussian_models["mb"].geometry_calls == 1
    assert len(gs["means"]) == 2                          # background + b


def test_an_empty_suppression_set_draws_every_actor():
    """The default must be "draw everything" -- a renderer used without a render manager
    (every existing unit test) never sets the field."""
    e = _two_actor_engine()

    _, tokens = e._collect_camera(0, 0, torch.eye(4), {})

    assert tokens == {"a", "b"}


def test_engine_owns_actor_collection_without_historical_inheritance():
    assert "_collect_camera" in Engine.__dict__
    assert Engine.__module__.endswith("omnire.engine")
