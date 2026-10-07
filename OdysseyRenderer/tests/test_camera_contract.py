import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

PATH = Path(__file__).parents[1] / "odyssey_renderer/omnire/camera_contract.py"
spec = importlib.util.spec_from_file_location("camera_contract_module", PATH)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def exposure():
    return dict(schema_version=1, mode="camera_frame", camera_names=list(cc.CAMERA_NAMES),
        camera_ids=list(range(8)), training_timestamps_us=np.array([100000, 200000], dtype=np.int64),
        exposure=torch.zeros(8, 3, 4), residual=torch.zeros(8, 2, 3, 4),
        camera_structure="affine", frame_structure="affine")


def metadata():
    return dict(schema_version=1, camera_names=list(cc.CAMERA_NAMES),
        training_timestamps_us=np.array([100000, 200000], dtype=np.int64),
        image_timestamps_us=np.broadcast_to(np.array([[100000], [200000]],dtype=np.int64),(2,8)).copy(),
        camera_to_ego=np.broadcast_to(np.eye(4),(2,8,4,4)).copy(), anchor=np.array([5.,6.,7.]),
        exposure=exposure(), projection_mode="training_undistorted",near_plane=.1,
        rasterize_mode="classic",rigid_camera_time=True,camera_pose_policy="logged_residual_at_tick")


def test_exposure_actual_base_and_row_matmul_not_transpose():
    data=exposure()
    matrix=torch.tensor([[.2,.4,.6,.1],[.1,.3,.5,.2],[.4,.2,.1,.3]])
    data["exposure"][0]=matrix
    data["residual"][0,1,:,3]=.05
    result=cc.validate_exposure(data,data["training_timestamps_us"],["CAM_F0"])
    rgb=torch.tensor([[[.2,.3,.4]]])
    expected=(rgb@matrix[:,:3]+matrix[:,3]+.05).clamp(0,1)
    torch.testing.assert_close(cc.apply_exposure(rgb,result,1,"CAM_F0"),expected)
    assert not torch.allclose(expected,(rgb@matrix[:,:3].T+matrix[:,3]+.05).clamp(0,1))


def test_exposure_strict_frame_camera_and_dtype():
    data=exposure()
    result=cc.validate_exposure(data,data["training_timestamps_us"],["CAM_F0"])
    for frame in (-1,2,.5):
        with pytest.raises(ValueError):
            cc.apply_exposure(torch.ones(1,1,3),result,frame,"CAM_F0")
    with pytest.raises(ValueError):
        cc.apply_exposure(torch.ones(1,1,3),result,0,"CAM_L0")
    data["exposure"]=data["exposure"].double()
    with pytest.raises(ValueError):
        cc.validate_exposure(data,data["training_timestamps_us"],["CAM_F0"])


def rigid():
    times=np.array([0,100000,200000],dtype=np.int64)
    image=np.broadcast_to(times[:,None],(3,2)).copy()
    image[1]=[125000,75000]
    trans=torch.zeros(3,2,3);trans[:,:,0]=torch.tensor([0.,10.,30.])[:,None]
    quats=torch.zeros(3,2,4);quats[:,:,0]=torch.tensor([2.,-3.,4.])[:,None]
    fv=torch.ones(3,2,dtype=torch.bool)
    return times,image,trans,quats,fv


def test_camera_specific_offsets_and_shortest_arc_nlerp():
    times,image,trans,quats,fv=rigid()
    first=cc.sample_rigid_camera_pose(times,image,1,0,trans,quats,fv)
    second=cc.sample_rigid_camera_pose(times,image,1,1,trans,quats,fv)
    torch.testing.assert_close(first[0][:,0],torch.tensor([15.,15.]))
    torch.testing.assert_close(second[0][:,0],torch.tensor([7.5,7.5]))
    torch.testing.assert_close(first[1].norm(dim=-1),torch.ones(2))
    assert first[2].tolist()==[True,True]
    assert first[3]["interpolated_actor_count"]==2
    assert trans[1,0,0]==10


def test_current_absence_never_interpolates_and_neighbor_absence_falls_back():
    times,image,trans,quats,fv=rigid()
    fv[1,0]=False;fv[2,1]=False
    p,q,present,diag=cc.sample_rigid_camera_pose(times,image,1,0,trans,quats,fv)
    assert present.tolist()==[False,True]
    assert p[:,0].tolist()==[10.,10.]
    assert diag["interpolated_actor_count"]==0


def test_active_masks_gate_both_rows():
    times,image,trans,quats,fv=rigid()
    active=fv.clone();active[1,0]=False;active[2,1]=False
    p,q,present,diag=cc.sample_rigid_camera_pose(times,image,1,0,trans,quats,fv,active)
    assert present.tolist()==[False,True]
    assert p[:,0].tolist()==[10.,10.]


def test_boundary_keeps_current_and_offset_limit_rejects():
    times,image,trans,quats,fv=rigid();image[0,0]=-10000
    p,q,present,diag=cc.sample_rigid_camera_pose(times,image,0,0,trans,quats,fv)
    assert p[:,0].tolist()==[0.,0.]
    assert diag["fallback"]=="neighbor_out_of_range"
    image[1,0]=150001
    with pytest.raises(ValueError,match="0.05"):
        cc.sample_rigid_camera_pose(times,image,1,0,trans,quats,fv)


def test_metadata_aligns_anchor_timestamps_camera_and_se3():
    value=metadata()
    result=cc.validate_time_calibration(value,expected_timestamps=value["training_timestamps_us"],
        expected_anchor=[5,6,7],expected_camera_names=["CAM_F0","CAM_R0"])
    assert result["camera_to_ego"].shape==(2,8,4,4)
    with pytest.raises(ValueError,match="anchor"):
        cc.validate_time_calibration(value,expected_anchor=[5,6,8])
    value["camera_to_ego"][0,0,0,0]=2
    with pytest.raises(ValueError,match="rotation"):
        cc.validate_time_calibration(value)


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(mode="frame"),
    lambda x: x.update(camera_ids=list(reversed(range(8)))),
    lambda x: x.update(camera_names=["CAM_F0"]*8),
    lambda x: x.update(training_timestamps_us=np.array([100000,200001],dtype=np.int64)),
    lambda x: x.update(residual=torch.zeros(8,3,3,4)),
    lambda x: x["exposure"].fill_(float("nan")),
])
def test_exposure_malformed_saved_contract_rejected(mutate):
    data=exposure();mutate(data)
    with pytest.raises(ValueError):
        cc.validate_exposure(data,np.array([100000,200000],dtype=np.int64),["CAM_F0"])


def test_camera_subset_order_maps_saved_exposure_rows():
    data=exposure()
    data["exposure"][2,:,3]=.7;data["exposure"][0,:,3]=.2
    prepared=cc.validate_exposure(data,data["training_timestamps_us"],["CAM_R0","CAM_F0"])
    rgb=torch.ones(2,3,3)
    torch.testing.assert_close(cc.apply_exposure(rgb,prepared,0,"CAM_R0"),torch.full_like(rgb,.7))
    torch.testing.assert_close(cc.apply_exposure(rgb,prepared,0,"CAM_F0"),torch.full_like(rgb,.2))


def test_global_calibration_recomposition_and_absolute_anchor_tolerance():
    data=metadata();data.pop("anchor")
    data["recon2world_translation"]=[588068.,4475491.,196.]
    data["ego_to_global"]=np.broadcast_to(np.eye(4),(2,4,4)).copy()
    data["ego_to_global"][:,:3,3]=[588069.,4475492.,196.]
    data["camera_to_global"]=data["ego_to_global"][:,None]@data["camera_to_ego"]
    assert cc.validate_time_calibration(data)["camera_ids"]==list(range(8))
    with pytest.raises(ValueError,match="anchor"):
        cc.validate_time_calibration(data,expected_anchor=[588069.,4475491.,196.])
    data["camera_to_global"][0,0,0,3]+=.01
    with pytest.raises(ValueError,match="recomposition"):
        cc.validate_time_calibration(data)


@pytest.mark.parametrize("key,value",[("projection_mode","legacy_distorted"),
    ("near_plane",.01),("rigid_camera_time",False),("camera_pose_policy","guess")])
def test_unknown_render_policy_rejected(key,value):
    data=metadata();data[key]=value
    with pytest.raises(ValueError):cc.validate_time_calibration(data)


def test_metadata_clock_map_and_camera_identity_rejected():
    data=metadata();data["map_name"]="singapore"
    with pytest.raises(ValueError,match="map"):
        cc.validate_time_calibration(data,expected_map_name="boston")
    with pytest.raises(ValueError,match="timestamps"):
        cc.validate_time_calibration(data,expected_timestamps=np.array([100000,200001],dtype=np.int64))
    with pytest.raises(ValueError,match="cameras"):
        cc.validate_time_calibration(data,expected_camera_names=["CAM_UNKNOWN"])
    data["image_timestamps_us"][0,0]+=50001
    with pytest.raises(ValueError,match="0.05"):
        cc.validate_time_calibration(data)


def test_nontrivial_quaternion_short_arc_and_half_offset():
    times,image,trans,quats,fv=rigid();image[1,0]=150000
    # Current is identity via negative quaternion; neighbor is +90deg about Z.
    quats[2,:,0]=np.sqrt(.5);quats[2,:,3]=np.sqrt(.5)
    p,q,present,diag=cc.sample_rigid_camera_pose(times,image,1,0,trans,quats,fv)
    expected=torch.tensor([-np.cos(np.pi/8),0,0,-np.sin(np.pi/8)],dtype=torch.float32)
    torch.testing.assert_close(q,expected[None].expand(2,4))
    torch.testing.assert_close(p[:,0],torch.tensor([20.,20.]))
    assert diag["alpha"]==.5


def test_zero_offset_and_static_active_mask():
    times,image,trans,quats,fv=rigid();image[1,0]=100000
    p,q,present,diag=cc.sample_rigid_camera_pose(times,image,1,0,trans,quats,fv,torch.tensor([True,False]))
    assert diag["fallback"]=="zero_offset"
    assert present.tolist()==[True,False]
    torch.testing.assert_close(p,trans[1])


def test_neighbor_tick_shorter_than_offset_rejected():
    times,image,trans,quats,fv=rigid();times[2]=110000
    with pytest.raises(ValueError,match="neighbor tick"):
        cc.sample_rigid_camera_pose(times,image,1,0,trans,quats,fv)


def test_invalid_neighbor_is_ignored_when_not_visible():
    times,image,trans,quats,fv=rigid();fv[2]=False;quats[2]=0;trans[2]=float("nan")
    p,q,present,diag=cc.sample_rigid_camera_pose(times,image,1,0,trans,quats,fv)
    torch.testing.assert_close(p,trans[1])
    assert present.all()


def test_calibration_and_pose_inputs_are_unchanged():
    data=metadata();result=cc.validate_time_calibration(data)
    result["camera_to_ego"][0,0,0,3]=10
    assert data["camera_to_ego"][0,0,0,3]==0
    times,image,trans,quats,fv=rigid();before=quats.clone()
    cc.sample_rigid_camera_pose(times,image,1,0,trans,quats,fv)
    torch.testing.assert_close(quats,before)


def test_invalid_absent_current_pose_does_not_abort_other_actors():
    times,image,trans,quats,fv=rigid();fv[1,0]=False
    trans[1,0]=float("nan");quats[1,0]=float("nan")
    p,q,present,diag=cc.sample_rigid_camera_pose(times,image,1,0,trans,quats,fv)
    assert present.tolist()==[False,True]
    assert p[1,0]==15
    assert torch.isfinite(q[1]).all()
