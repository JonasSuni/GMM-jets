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


def fit_gmm(cellid, fnr, nMaxwellians):

    outdir = wrkdir_DNR + "vdf_gmm/"

    vc_coord_arr, vc_val_arr = read_file(cellid, fnr)

    model = GeneralMixtureModel([Normal()] * nMaxwellians, verbose=True).fit(
        vc_coord_arr, model_weight=vc_val_arr
    )

    predicted_cluster = model.predict(vc_coord_arr)

    covs_list = []
    means_list = []

    for idx in range(nMaxwellians):
        covs_list.append(model.distributions[idx].covs.numpy())
        means_list.append(model.distributions[idx].covs.numpy())

    covs_arr = np.array(covs_list)
    means_arr = np.array(means_list)

    out_arr = np.array(
        [covs_arr, means_arr, predicted_cluster, vc_coord_arr, vc_val_arr]
    )

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

    np.save(outdir + "n{}/c{}/f{}".format(cellid, fnr), out_arr)


def process_all_gmm(nMaxwellians=1, prepost_time=30):

    archer_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archer_intervals.txt", dtype=int
    )
    koller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/koller_intervals.txt", dtype=int
    )
    archerkoller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archerkoller_intervals.txt", dtype=int
    )

    for p in archer_data:
        ci, t0, t1, tjet = p
        for fnr in np.arange(t0 - prepost_time, t1 + prepost_time + 0.1, 1, dtype=int):
            fit_gmm(ci, fnr, nMaxwellians)

    for p in koller_data:
        ci, t0, t1, tjet = p
        for fnr in np.arange(t0 - prepost_time, t1 + prepost_time + 0.1, 1, dtype=int):
            fit_gmm(ci, fnr, nMaxwellians)

    for p in archerkoller_data:
        ci, t0, t1, tjet = p
        for fnr in np.arange(t0 - prepost_time, t1 + prepost_time + 0.1, 1, dtype=int):
            fit_gmm(ci, fnr, nMaxwellians)
