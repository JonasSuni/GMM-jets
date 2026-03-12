import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import *


r_e = 6.371e6
m_p = 1.672621898e-27
q_p = 1.602176634e-19
mu0 = 1.25663706212e-06
kb = 1.380649e-23

# wrkdir_DNR = os.environ["WRK"] + "/"
homedir = os.environ["HOME"] + "/"

# wrkdir_DNR = wrkdir_DNR + "jets_3D/"
wrkdir_NEW = "/turso/home/jesuni/wrk/jets_3D/"
wrkdir_DNR = wrkdir_NEW
wrkdir_other = os.environ["WRK"] + "/"

bulkpath_FIF = "/wrk-vakka/group/spacephysics/vlasiator/3D/FIF/bulk1/"


def read_file(cellid, fnr):

    txtdir = wrkdir_DNR + "vdf_txts/"

    vdfdata = np.loadtxt(txtdir + "c{}/f{}.txt".format(int(cellid), int(fnr)))
    vc_coord_arr = vdfdata[:, [0, 1, 2]]
    vc_val_arr = vdfdata[:, 3]

    return (vc_coord_arr, vc_val_arr)


def fit_gmm(cellid, fnr, nMaxwellians, inertia=0.0):

    outdir = wrkdir_DNR + "vdf_gmm/"

    vc_coord_arr, vc_val_arr = read_file(cellid, fnr)

    model = GeneralMixtureModel(
        [Normal()] * nMaxwellians, verbose=True, inertia=inertia
    ).fit(vc_coord_arr, sample_weight=vc_val_arr)

    predicted_cluster = model.predict(vc_coord_arr)

    out_arr = []

    for idx in range(nMaxwellians):
        means = model.distributions[idx].means.numpy().flatten()
        covs = model.distributions[idx].covs.numpy().flatten()
        print(np.append(means, covs).flatten())
        out_arr.append(np.append(means, covs).flatten())

    out_arr = np.array(out_arr)

    if not os.path.exists(outdir + "n{}".format(nMaxwellians)):
        try:
            os.makedirs(outdir + "n{}".format(nMaxwellians))
        except OSError:
            pass

    if not os.path.exists(outdir + "n{}/c{}".format(nMaxwellians, cellid)):
        try:
            os.makedirs(outdir + "n{}/c{}".format(nMaxwellians, cellid))
        except OSError:
            pass

    np.savetxt(outdir + "n{}/c{}/f{}.fit".format(nMaxwellians, cellid, fnr), out_arr)
    np.savetxt(
        outdir + "n{}/c{}/f{}.pred".format(nMaxwellians, cellid, fnr), predicted_cluster
    )


def process_all_gmm(nMaxwellians=1, inertia=0.0):

    dirlist = os.listdir(wrkdir_DNR + "vdf_txts")
    cellids = np.array([d[1:] for d in dirlist]).astype(int)
    for ci in cellids:
        fnrlist = os.listdir(wrkdir_DNR + "vdf_txts/c{}".format(ci))
        fnrs = np.array([f.split(".")[0][1:] for f in fnrlist])
        for fnr in fnrs:
            fit_gmm(ci, fnr, nMaxwellians, inertia=inertia)
