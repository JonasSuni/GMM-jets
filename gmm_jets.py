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
    if not vdfdata.size:
        raise Exception
    vc_coord_arr = vdfdata[:, [0, 1, 2]]
    vc_val_arr = vdfdata[:, 3]

    return (vc_coord_arr, vc_val_arr)


def evaluate_maxwellian(X, mean, cov):

    invcov = np.linalg.inv(cov)
    res = (
        (2 * np.pi) ** (-3.0 / 2)
        * np.linalg.det(cov) ** (-1.0 / 2)
        * np.exp(-0.5 * np.vecdot((X - mean), np.matmul(invcov, (X - mean).T).T))
    )
    return res


def fit_gmm(
    cellid,
    fnr,
    nMaxwellians,
    inertia=0.0,
    debug=False,
    mincov=0.0,
    skip=True,
    maxiter=1000,
):

    outdir = wrkdir_DNR + "vdf_gmm/"
    if (
        os.path.isfile(outdir + "n{}/c{}/f{}.fit".format(nMaxwellians, cellid, fnr))
        and skip
        and not debug
    ):
        print("File already exists and skip is True, exiting.")
        return None

    try:
        vc_coord_arr, vc_val_arr = read_file(cellid, fnr)
    except:
        return None

    vmean = np.nanmean(vc_coord_arr, axis=0)
    vmeanmag = np.linalg.norm(vmean)

    onemodel = Normal().fit(vc_coord_arr, sample_weight=vc_val_arr)
    onecovs = onemodel.covs

    det_means = [[-750e3, 0, 0], [-187.5e3, 0, 0], [650e3, -375e3, 0], [0.0, 0.0, 0.0]]

    distribs = []
    for idx in range(nMaxwellians):
        distribs.append(
            Normal(
                means=det_means[idx],
                min_cov=mincov,
                covs=onecovs,
                covariance_type="full",
            )
        )

    if nMaxwellians == 1:
        model = distribs[0].fit(vc_coord_arr, sample_weight=vc_val_arr)
    else:
        model = GeneralMixtureModel(
            distribs, verbose=True, inertia=inertia, max_iter=maxiter
        ).fit(vc_coord_arr, sample_weight=vc_val_arr)

    if nMaxwellians > 1:
        predicted_cluster = model.predict(vc_coord_arr)

    out_arr = []

    likelihoods = np.zeros_like(vc_val_arr)

    for idx in range(nMaxwellians):
        if nMaxwellians > 1:
            weight = model.priors.numpy()[idx]
            means = model.distributions[idx].means.numpy()
            covs = model.distributions[idx].covs.numpy()
        else:
            weight = 1
            means = model.means.numpy()
            covs = model.covs.numpy()
        print(
            "Weight: {}\nMean: {}\nCovariance:\n{}\n".format(
                weight, means / 1e3, covs * m_p / kb / 1e6
            )
        )
        likelihoods = likelihoods + weight * evaluate_maxwellian(
            vc_coord_arr, means, covs
        )
        out_arr.append([weight] + means.tolist() + covs.flatten().tolist())

    loglikelihood = np.sum(np.log(likelihoods))

    print("Log-likelihood is {}".format(loglikelihood))

    for idx in range(len(out_arr)):
        out_arr[idx] = out_arr[idx] + [loglikelihood]

    out_arr = np.array(out_arr)

    if debug:
        if nMaxwellians > 1:
            return (
                model,
                vc_coord_arr,
            )
        else:
            return model

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
    if nMaxwellians > 1:
        np.savetxt(
            outdir + "n{}/c{}/f{}.pred".format(nMaxwellians, cellid, fnr),
            predicted_cluster,
        )


def plot_loglikelihoods():

    outdir = outdir = wrkdir_DNR + "vdf_gmm/"

    loglikes = np.empty((4, 100000), dtype=float)
    loglikes.fill(np.nan)

    for nMaxwellians in [1, 2, 3, 4]:

        counter = 0
        dirlist = os.listdir(outdir + "n{}".format(nMaxwellians))
        for dir in dirlist:
            fnrfiles = os.listdir(outdir + "n{}/{}".format(nMaxwellians, dir))
            for fnr in fnrfiles:
                data = np.loadtxt(
                    outdir + "n{}/{}/{}".format(nMaxwellians, dir, fnr), ndmin=2
                )
                loglike = data[0][-1]
                loglikes[nMaxwellians - 1, counter] = loglike
                counter += 1

    loglikes = loglikes[~np.isnan(loglikes)].T

    loglikes = loglikes.T
    narr = np.array([1, 2, 3, 4])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="compressed")

    ax.boxplot(loglikes)

    fig.savefig(wrkdir_DNR + "Figs/loglikelihoods.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


def process_all_gmm(nMaxwellians=1, inertia=0.0, mincov=0.0, skip=True, maxiter=1000):

    dirlist = os.listdir(wrkdir_DNR + "vdf_txts")
    cellids = np.array([d[1:] for d in dirlist]).astype(int)
    for ci in cellids:
        fnrlist = os.listdir(wrkdir_DNR + "vdf_txts/c{}".format(ci))
        fnrs = np.array([f.split(".")[0][1:] for f in fnrlist])
        for fnr in fnrs:
            fit_gmm(
                ci,
                fnr,
                nMaxwellians,
                inertia=inertia,
                mincov=mincov,
                skip=skip,
                maxiter=maxiter,
            )
