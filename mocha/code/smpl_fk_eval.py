"""Eval-time SMPL->H36M FK, fully self-contained (no smplx / chumpy / timm).
Vendored LBS from smplx.lbs (torch+numpy only) so it runs inside the eval Docker
image. Validated to match scripts/smpl_fk.py (training path) bit-for-bit.
"""
import os, numpy as np, torch
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))


# ---- vendored SMPL LBS (from smplx.lbs, torch-only) ----
def _rodrigues(rot_vecs, eps=1e-8):
    bs = rot_vecs.shape[0]; dev, dt = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rd = rot_vecs / angle
    cos = torch.cos(angle).unsqueeze(1); sin = torch.sin(angle).unsqueeze(1)
    rx, ry, rz = torch.split(rd, 1, dim=1)
    zeros = torch.zeros((bs, 1), dtype=dt, device=dev)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(bs, 3, 3)
    ident = torch.eye(3, dtype=dt, device=dev).unsqueeze(0)
    return ident + sin * K + (1 - cos) * torch.bmm(K, K)


def _transform_mat(R, t):
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def _rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    joints = joints.unsqueeze(-1)
    rel = joints.clone(); rel[:, 1:] -= joints[:, parents[1:]]
    tm = _transform_mat(rot_mats.reshape(-1, 3, 3), rel.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    chain = [tm[:, 0]]
    for i in range(1, parents.shape[0]):
        chain.append(torch.matmul(chain[parents[i]], tm[:, i]))
    transforms = torch.stack(chain, dim=1)
    return transforms[:, :, :3, 3]


def _lbs_verts(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights):
    bs = max(betas.shape[0], pose.shape[0]); dev, dt = betas.device, betas.dtype
    v_shaped = v_template + torch.einsum('bl,mkl->bmk', [betas, shapedirs])
    J = torch.einsum('bik,ji->bjk', [v_shaped, J_regressor])
    ident = torch.eye(3, dtype=dt, device=dev)
    rot_mats = _rodrigues(pose.view(-1, 3)).view(bs, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:] - ident).view(bs, -1)
    pose_offsets = torch.matmul(pose_feature, posedirs).view(bs, -1, 3)
    v_posed = pose_offsets + v_shaped
    J_tr = _rigid_transform(rot_mats, J, parents, dtype=dt)
    # NOTE: for joints-only via h36m regressor we only need v_posed*A skinning -> verts.
    # Recompute A (rel transforms) for skinning:
    rel = _rel_transforms(rot_mats, J, parents, dt)
    W = lbs_weights.unsqueeze(0).expand(bs, -1, -1)
    nj = J_regressor.shape[0]
    T = torch.matmul(W, rel.view(bs, nj, 16)).view(bs, -1, 4, 4)
    homo = torch.ones([bs, v_posed.shape[1], 1], dtype=dt, device=dev)
    vph = torch.cat([v_posed, homo], dim=2)
    v_homo = torch.matmul(T, vph.unsqueeze(-1))
    return v_homo[:, :, :3, 0]


def _rel_transforms(rot_mats, joints, parents, dtype=torch.float32):
    joints = joints.unsqueeze(-1)
    rel = joints.clone(); rel[:, 1:] -= joints[:, parents[1:]]
    tm = _transform_mat(rot_mats.reshape(-1, 3, 3), rel.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    chain = [tm[:, 0]]
    for i in range(1, parents.shape[0]):
        chain.append(torch.matmul(chain[parents[i]], tm[:, i]))
    transforms = torch.stack(chain, dim=1)
    joints_h = F.pad(joints, [0, 0, 0, 1])
    return transforms - F.pad(torch.matmul(transforms, joints_h), [3, 0, 0, 0, 0, 0, 0, 0])


class SMPLH36M:
    def __init__(self, npz_path=None, h36m_reg_path=None, device="cpu"):
        npz_path = npz_path or os.path.join(_DIR, "smpl_neutral_clean.npz")
        h36m_reg_path = h36m_reg_path or os.path.join(_DIR, "J_regressor_h36m_correct.npy")
        d = np.load(npz_path); self.device = torch.device(device)
        self.v_template = torch.as_tensor(d["v_template"], dtype=torch.float32, device=self.device)
        self.shapedirs = torch.as_tensor(d["shapedirs"], dtype=torch.float32, device=self.device)
        pd = d["posedirs"]
        self.posedirs = torch.as_tensor(pd.reshape(pd.shape[0] * 3, -1).T.copy(), dtype=torch.float32, device=self.device)
        self.J_regressor = torch.as_tensor(d["J_regressor"], dtype=torch.float32, device=self.device)
        parents = d["kintree_table"][0].astype(np.int64).copy(); parents[0] = -1
        self.parents = torch.as_tensor(parents, device=self.device)
        self.weights = torch.as_tensor(d["weights"], dtype=torch.float32, device=self.device)
        self.h36m_reg = torch.as_tensor(np.load(h36m_reg_path), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def joints(self, pose, trans, betas=None, batch=2048):
        # ascontiguousarray (not asarray): run.py passes pose[::stride] for fps>30 walks,
        # a non-contiguous view; _rodrigues(pose.view(-1,3)) crashes on non-contiguous ->
        # predict() would silently fall back to class 1 for every high-fps hidden walk.
        pose = np.ascontiguousarray(pose, np.float32); trans = np.ascontiguousarray(trans, np.float32)
        T = pose.shape[0]
        betas = np.zeros(10, np.float32) if betas is None else np.asarray(betas, np.float32).reshape(-1)[:10]
        outs = []
        for s in range(0, T, batch):
            e = min(T, s + batch); n = e - s
            p = torch.as_tensor(pose[s:e], dtype=torch.float32, device=self.device)
            b = torch.as_tensor(np.broadcast_to(betas, (n, 10)).copy(), dtype=torch.float32, device=self.device)
            tr = torch.as_tensor(trans[s:e], dtype=torch.float32, device=self.device)
            verts = _lbs_verts(b, p, self.v_template[None].expand(n, -1, -1), self.shapedirs,
                               self.posedirs, self.J_regressor, self.parents, self.weights)
            verts = verts + tr[:, None, :]
            j17 = torch.einsum('bik,ji->bjk', [verts, self.h36m_reg])
            outs.append(j17.cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float32)


def resample_30fps(arr, fps):
    return arr[::max(1, int(round(float(fps) / 30.0)))]


if __name__ == "__main__":  # validate equivalence vs the training-time FK (smpl_fk.py / smplx.lbs)
    import argparse, pickle, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="a CARE-PD *_canonical.pkl to compare on")
    args = ap.parse_args()
    sys.path.insert(0, _DIR)
    from smpl_fk import SMPLH36M as TrainFK
    train = TrainFK(device="cpu")
    ev = SMPLH36M(npz_path=os.path.join(_DIR, "smpl_neutral_clean.npz"),
                  h36m_reg_path=os.path.join(_DIR, "J_regressor_h36m_correct.npy"),
                  device="cpu")
    data = pickle.load(open(args.pkl, "rb"))
    maxd = 0.0
    for si, (subj, walks) in enumerate(data.items()):
        if si >= 3: break
        for wid, e in list(walks.items())[:2]:
            a = train.joints(e["pose"], e["trans"]); b = ev.joints(e["pose"], e["trans"])
            maxd = max(maxd, float(np.abs(a - b).max()))
    print(f"max |train_FK - eval_FK| = {maxd:.3e}  (should be ~0 -> eval FK is faithful)")
