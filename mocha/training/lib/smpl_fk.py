"""SMPL (canonical pickle) -> H36M 17-joint 3D, chumpy-free.
Uses smplx.lbs (torch-only; vendor lbs.py for eval). Arrays from smpl_neutral_clean.npz.
Canonical convention: x=lateral, y=up, z=forward. betas=0 (privacy) by default.
"""
import os, numpy as np, torch
from smplx.lbs import lbs, vertices2joints

_DIR = os.path.dirname(os.path.abspath(__file__))
# Both live in assets/ — see assets/README.md for how to obtain them.
_ASSETS = os.path.join(_DIR, "..", "..", "assets")
DEF_NPZ = os.environ.get("SMPL_NPZ", os.path.join(_ASSETS, "smpl_neutral_clean.npz"))
DEF_REG = os.environ.get("H36M_REGRESSOR", os.path.join(_ASSETS, "J_regressor_h36m_correct.npy"))

# H36M-17 joint indices (regressor order): 0 pelvis,1 RHip,2 RKnee,3 RAnkle,4 LHip,5 LKnee,
# 6 LAnkle,7 Spine,8 Thorax,9 Neck/Nose,10 Head,11 LSho,12 LElb,13 LWri,14 RSho,15 RElb,16 RWri

class SMPLH36M:
    def __init__(self, npz_path=DEF_NPZ, h36m_reg_path=DEF_REG, device="cpu"):
        d = np.load(npz_path)
        self.device = torch.device(device)
        self.v_template = torch.as_tensor(d["v_template"], dtype=torch.float32, device=self.device)
        self.shapedirs = torch.as_tensor(d["shapedirs"], dtype=torch.float32, device=self.device)
        pd = d["posedirs"]  # (6890,3,207)
        self.posedirs = torch.as_tensor(pd.reshape(pd.shape[0] * 3, -1).T.copy(), dtype=torch.float32, device=self.device)  # (207, 6890*3)
        self.J_regressor = torch.as_tensor(d["J_regressor"], dtype=torch.float32, device=self.device)
        parents = d["kintree_table"][0].astype(np.int64).copy(); parents[0] = -1
        self.parents = torch.as_tensor(parents, device=self.device)
        self.weights = torch.as_tensor(d["weights"], dtype=torch.float32, device=self.device)
        self.h36m_reg = torch.as_tensor(np.load(h36m_reg_path), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def joints(self, pose, trans, betas=None, batch=2048):
        """pose (T,72) axis-angle, trans (T,3) -> h36m joints (T,17,3) in canonical world frame."""
        # ascontiguousarray (not asarray): strided views from pose[::stride] (fps>30
        # cohorts e.g. BMCLab@150) are non-contiguous; lbs does pose.view(-1,3) which
        # fails on non-contiguous tensors. Force a contiguous copy.
        pose = np.ascontiguousarray(pose, np.float32); trans = np.ascontiguousarray(trans, np.float32)
        T = pose.shape[0]
        if betas is None:
            betas = np.zeros(10, np.float32)
        betas = np.asarray(betas, np.float32).reshape(-1)[:10]
        outs = []
        for s in range(0, T, batch):
            e = min(T, s + batch); n = e - s
            p = torch.as_tensor(pose[s:e], dtype=torch.float32, device=self.device)
            b = torch.as_tensor(np.broadcast_to(betas, (n, 10)).copy(), dtype=torch.float32, device=self.device)
            tr = torch.as_tensor(trans[s:e], dtype=torch.float32, device=self.device)
            verts, _ = lbs(b, p, self.v_template[None].expand(n, -1, -1), self.shapedirs,
                           self.posedirs, self.J_regressor, self.parents, self.weights, pose2rot=True)
            verts = verts + tr[:, None, :]                       # add global translation (matches smpl2h36m)
            j17 = vertices2joints(self.h36m_reg, verts)          # (n,17,3)
            outs.append(j17.cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float32)


def resample_30fps(arr, fps):
    """Subsample frames to ~30 fps (matches CARE-PD exfps=30, stride=int(fps/30))."""
    stride = max(1, int(round(float(fps) / 30.0)))
    return arr[::stride]


if __name__ == "__main__":
    import pickle
    from dataio import DATA_DIR
    m = SMPLH36M(device="cuda" if torch.cuda.is_available() else "cpu")
    for ds in ["3DGait_canonical.pkl", "BMCLab_canonical.pkl", "PD-GaM_canonical.pkl"]:
        path = os.path.join(DATA_DIR, ds)
        if not os.path.exists(path):
            print(ds, "MISSING"); continue
        data = pickle.load(open(path, "rb"))
        subj = list(data.keys())[0]; walk = list(data[subj].keys())[0]
        e = data[subj][walk]
        j = m.joints(e["pose"], e["trans"], betas=None)
        up = j[:, :, 1]  # y axis (canonical up)
        pelvis_y = j[:, 0, 1].mean(); head_y = j[:, 10, 1].mean()
        # vertical extent ~ stature
        stature = up.max() - up.min()
        # left/right hip separation along lateral (x)
        hip_sep = np.abs(j[:, 1, 0] - j[:, 4, 0]).mean()
        print(f"{ds}: T={j.shape[0]} fps={e.get('fps')} | head_y-pelvis_y={head_y-pelvis_y:.3f} "
              f"stature={stature:.3f} hip_sep={hip_sep:.3f} "
              f"y[min,max]=[{up.min():.2f},{up.max():.2f}] z(fwd)range={j[:,:,2].max()-j[:,:,2].min():.2f}")
