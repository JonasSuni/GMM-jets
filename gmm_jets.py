import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import *
from copy import deepcopy


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

bulkpath_FIF = "/turso/group/spacephysics/vlasiator/data/L0/3D/FIF/bulk1/"


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
    old_means=None,
    old_covs=None,
    old_priors=None,
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

    distribs = []
    if old_means or old_covs or old_priors:
        if not (old_means and old_covs and old_priors):
            raise ValueError(
                "Error: Old priors, means, and covariances must all be defined! Exiting."
            )
        if (
            len(old_means) != nMaxwellians
            or len(old_covs) != nMaxwellians
            or len(old_priors) != nMaxwellians
        ):
            raise ValueError(
                "Error: Old priors, means, and covariances must have lengths equal to number of Maxwellians! Exiting."
            )

        for idx in range(nMaxwellians):
            distribs.append(
                Normal(
                    means=deepcopy(old_means[idx]),
                    covs=deepcopy(old_covs[idx]),
                    covariance_type="full",
                )
            )

    else:
        onemodel = Normal().fit(vc_coord_arr, sample_weight=vc_val_arr)
        onecovs = onemodel.covs
        onemeans = onemodel.means
        cov_sphere = np.trace(onecovs.numpy()) / 3.0
        std_sphere = np.sqrt(cov_sphere)

        thetas = np.linspace(0, 2 * np.pi, nMaxwellians + 1)[:-1]

        det_means = []
        for idx in range(thetas.size):
            det_means.append(
                onemeans.numpy()
                + std_sphere * np.array([np.cos(thetas[idx]), np.sin(thetas[idx]), 0.0])
            )

        for idx in range(nMaxwellians):
            distribs.append(
                Normal(
                    means=deepcopy(det_means[idx].astype(float)),
                    min_cov=mincov,
                    covs=deepcopy(onecovs.numpy().astype(float)),
                    covariance_type="full",
                )
            )

    if nMaxwellians == 1:
        model = distribs[0].fit(vc_coord_arr, sample_weight=vc_val_arr)
    else:
        model = GeneralMixtureModel(
            distribs, verbose=True, inertia=inertia, max_iter=maxiter, priors=old_priors
        ).fit(vc_coord_arr, sample_weight=vc_val_arr)

    if nMaxwellians > 1:
        predicted_cluster = model.predict(vc_coord_arr).numpy()
        predict_proba = model.predict(vc_coord_arr).numpy()
        predicted_cluster = np.hstack((predicted_cluster, predict_proba))

    out_arr = []

    likelihoods = np.zeros_like(vc_val_arr)

    means_list = []
    covs_list = []
    weights_list = []

    for idx in range(nMaxwellians):
        if nMaxwellians > 1:
            weight = model.priors.numpy()[idx]
            means = model.distributions[idx].means.numpy()
            covs = model.distributions[idx].covs.numpy()
            means_list.append(model.distributions[idx].means)
            covs_list.append(model.distributions[idx].covs)
            weights_list.append(model.priors[idx])
        else:
            weight = 1
            means = model.means.numpy()
            covs = model.covs.numpy()
            means_list.append(model.means)
            covs_list.append(model.covs)
            weights_list.append(1)
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
        out_arr[idx] = out_arr[idx] + [loglikelihood] + [vc_coord_arr[:, 0].size]

    out_arr = np.array(out_arr)

    if debug:
        if nMaxwellians > 1:
            return (
                model,
                vc_coord_arr,
            )
        else:
            return model

    create_dir_if_not_exist(outdir + "n{}".format(nMaxwellians))

    create_dir_if_not_exist(outdir + "n{}/c{}".format(nMaxwellians, cellid))

    np.savetxt(outdir + "n{}/c{}/f{}.fit".format(nMaxwellians, cellid, fnr), out_arr)
    if nMaxwellians > 1:
        np.savetxt(
            outdir + "n{}/c{}/f{}.pred".format(nMaxwellians, cellid, fnr),
            predicted_cluster,
        )

    return (means_list, covs_list, weights_list)


def create_dir_if_not_exist(outdir):

    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
            print("Created directory {}".format(outdir))
        except OSError:
            print("Directory {} already exists".format(outdir))
            pass


def plot_jet_loglikes(prepost_time=30, tjet_only=False, skip_mono=False):

    outdir = wrkdir_DNR + "Figs/loglikes/"
    create_dir_if_not_exist(outdir)
    create_dir_if_not_exist(outdir + "archer")
    create_dir_if_not_exist(outdir + "koller")
    create_dir_if_not_exist(outdir + "archerkoller")

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
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), layout="compressed")
        if tjet_only:
            plot_loglike_tjet(
                ax, 4, ci, t0, t1, tjet, prepost_time, skip_mono=skip_mono
            )
        else:
            plot_loglike_onejet(ax, 4, ci, t0, t1, tjet, prepost_time)
        fig.savefig(
            outdir + "archer/c{}_t{}_{}.png".format(ci, t0, t1),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    for p in koller_data:
        ci, t0, t1, tjet = p
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), layout="compressed")
        if tjet_only:
            plot_loglike_tjet(
                ax, 4, ci, t0, t1, tjet, prepost_time, skip_mono=skip_mono
            )
        else:
            plot_loglike_onejet(ax, 4, ci, t0, t1, tjet, prepost_time)
        fig.savefig(
            outdir + "koller/c{}_t{}_{}.png".format(ci, t0, t1),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    for p in archerkoller_data:
        ci, t0, t1, tjet = p
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), layout="compressed")
        if tjet_only:
            plot_loglike_tjet(
                ax, 4, ci, t0, t1, tjet, prepost_time, skip_mono=skip_mono
            )
        else:
            plot_loglike_onejet(ax, 4, ci, t0, t1, tjet, prepost_time)
        fig.savefig(
            outdir + "archerkoller/c{}_t{}_{}.png".format(ci, t0, t1),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_loglike_tjet(ax, nMaxwellians, ci, tjet, penalty=True, skip_mono=False):

    maxwell_arr = np.arange(1, nMaxwellians + 1)
    loglikes_arr = np.zeros(nMaxwellians, dtype=float)
    for idx in range(nMaxwellians):
        try:
            data = np.loadtxt(
                wrkdir_DNR + "vdf_gmm/n{}/c{}/f{}.fit".format(idx + 1, ci, int(tjet)),
                ndmin=2,
            )
        except:
            print("Something went wrong when reading loglike")
            loglikes_arr[idx] = np.nan
            continue

        pen = 0.0
        if penalty:
            knz = idx + 1
            pred_len = data[0][-1]
            pen = (
                0.5 * knz * (9 + 1)
                + knz * 0.5 * np.log(pred_len / 12)
                + 0.5 * 9 * (knz * np.log(pred_len / 12) + np.sum(np.log(data[:, 0])))
            )
        loglikes_arr[idx] = data[0][-2] - pen

    ax.plot(maxwell_arr, loglikes_arr, "o-")

    if skip_mono:
        ax.set_xlim(2 - 0.1, nMaxwellians + 0.1)
    else:
        ax.set_xlim(1 - 0.1, nMaxwellians + 0.1)
    ax.grid()
    ax.set(xlabel="# Maxwellians", ylabel="Log-likelihood")


def plot_loglike_onejet(
    ax, nMaxwellians, ci, t0, t1, tjet, prepost_time=30, penalty=True
):

    fnr_arr = np.arange(t0 - prepost_time, t1 + prepost_time + 0.1, 1, dtype=int)
    loglikes_arr = np.zeros((nMaxwellians, fnr_arr.size), dtype=float)
    for idx in range(nMaxwellians):
        for idx2, fnr in enumerate(fnr_arr):
            try:
                data = np.loadtxt(
                    wrkdir_DNR
                    + "vdf_gmm/n{}/c{}/f{}.fit".format(idx + 1, ci, int(fnr)),
                    ndmin=2,
                )
            except:
                print("Something went wrong when reading loglike")
                loglikes_arr[idx, idx2] = np.nan
                continue

            pen = 0.0
            if penalty:
                knz = idx + 1
                pred_len = data[0][-1]
                pen = (
                    0.5 * knz * (9 + 1)
                    + knz * 0.5 * np.log(pred_len / 12)
                    + 0.5
                    * 9
                    * (knz * np.log(pred_len / 12) + np.sum(np.log(data[:, 0])))
                )
            loglikes_arr[idx, idx2] = data[0][-2] - pen

    for idx in range(nMaxwellians):
        ax.plot(fnr_arr, loglikes_arr[idx, :], label="n = {}".format(idx + 1))

    ax.set_xlim(fnr_arr[0], fnr_arr[-1])
    ax.grid()
    ax.legend()
    ax.set(xlabel="t [s]", ylabel="Log-likelihood")
    ax.axvline(tjet, linestyle="dashed", color="black")
    ax.fill_between(
        fnr_arr,
        0,
        1,
        where=np.logical_and(fnr_arr >= t0, fnr_arr <= t1),
        color="green",
        alpha=0.2,
        transform=ax.get_xaxis_transform(),
        linewidth=0,
    )


def plot_loglikelihoods():

    outdir = outdir = wrkdir_DNR + "vdf_gmm/"

    loglikes = np.empty((4, 100000), dtype=float)
    loglikes.fill(np.nan)
    print(loglikes.shape)

    for nMaxwellians in [1, 2, 3, 4]:

        counter = 0
        dirlist = os.listdir(outdir + "n{}".format(nMaxwellians))
        for dir in dirlist:
            fnrfiles = os.listdir(outdir + "n{}/{}".format(nMaxwellians, dir))
            fitfiles = [fnr for fnr in fnrfiles if "fit" in fnr]
            fnr = fitfiles[30]
            data = np.loadtxt(outdir + "n{}/{}/{}".format(nMaxwellians, dir, fnr))
            # print(data.shape)
            if nMaxwellians > 1:
                loglike = data[0][-2]
            else:
                loglike = data[-2]
            loglikes[nMaxwellians - 1, counter] = loglike
            counter += 1

    loglikes = loglikes[~np.isnan(loglikes)].reshape(
        (int(loglikes[~np.isnan(loglikes)].size / 4), 4)
    )
    print(loglikes.shape)
    narr = np.array([[1, 2, 3, 4]] * int(loglikes.shape[0]), dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="compressed")

    ax.boxplot(loglikes, showmeans=True)
    ax.scatter(narr, loglikes, alpha=0.3, edgecolors="none")

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


def process_all_jet_gmm(nMaxwellians=4, skip=True, prepost_time=30, tjet_only=False):

    archer_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archer_intervals.txt", dtype=int
    )
    koller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/koller_intervals.txt", dtype=int
    )
    archerkoller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archerkoller_intervals.txt", dtype=int
    )

    all_data = np.vstack((archer_data, koller_data, archerkoller_data))

    for p in all_data:
        ci, t0, t1, tjet = p
        fnr_arr_pre = np.arange(t0 - prepost_time, tjet, 1, dtype=int)[::-1]
        fnr_arr_post = np.arange(tjet + 1, t1 + prepost_time + 0.1, 1, dtype=int)
        tjet_means, tjet_covs, tjet_priors = fit_gmm(
            ci,
            int(tjet),
            nMaxwellians,
            skip=skip,
            old_covs=None,
            old_means=None,
            old_priors=None,
        )
        if not tjet_only:
            old_means, old_covs, old_priors = (tjet_means, tjet_covs, tjet_priors)
            for fnr in fnr_arr_pre:
                try:
                    out = fit_gmm(
                        ci,
                        fnr,
                        nMaxwellians,
                        skip=skip,
                        old_covs=old_covs,
                        old_means=old_means,
                        old_priors=old_priors,
                    )
                    old_means, old_covs, old_priors = out
                except:
                    print("File not found, continuing.")
                    continue
            old_means, old_covs, old_priors = (tjet_means, tjet_covs, tjet_priors)
            for fnr in fnr_arr_post:
                try:
                    out = fit_gmm(
                        ci,
                        fnr,
                        nMaxwellians,
                        skip=skip,
                        old_covs=old_covs,
                        old_means=old_means,
                        old_priors=old_priors,
                    )
                    old_means, old_covs, old_priors = out
                except:
                    print("File not found, continuing.")
                    continue
