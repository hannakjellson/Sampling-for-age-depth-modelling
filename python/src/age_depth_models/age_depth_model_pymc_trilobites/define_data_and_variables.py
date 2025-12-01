import numpy as np
import pandas as pd
import os

def read_data():
    base_path = "../../../../data/"
    dc13_df = pd.read_csv(os.path.join(base_path, "StratoBayes/cambrian/cambrian_signal.csv"))
    radiometric_df = pd.read_csv(os.path.join(base_path, "StratoBayes/cambrian/cambrian_ties.csv"))

    dc13_depths = dc13_df["height"].to_numpy()
    dc13 = dc13_df["d13C"].to_numpy()
    dc13_site = dc13_df["Site.detail"].to_numpy()

    radiometric_depths = radiometric_df["height"].to_numpy()
    radiometric = radiometric_df["mean"].to_numpy()
    radiometric_site = radiometric_df["site"].to_numpy()
    radiometric_sd = radiometric_df["sd"].to_numpy()

    return (
        dc13_depths,
        dc13,
        dc13_site,
        radiometric_depths,
        radiometric,
        radiometric_site,
        radiometric_sd,
    )


def get_data():
    (
        dc13_depths,
        dc13,
        dc13_site, 
        radiometric_depths,
        radiometric,
        radiometric_site,
        radiometric_sd,
    ) = read_data()
    Tiout_mask = np.where(dc13_site == "Tiout")
    Talat_Nyssi_mask = np.where(dc13_site == "MS3-Talat n' Yissi")
    Oued_Sdas_mask = np.where(dc13_site == "MS7 Oued Sdas")
    Sukharikha_mask = np.where(dc13_site == "Sukharikha")

    Tiout_dc13_depths = dc13_depths[Tiout_mask]
    Tiout_dc13_depths -=  np.max(Tiout_dc13_depths)
    Tiout_dc13_depths *= -1
    Tiout_dc13 = dc13[Tiout_mask]

    Talat_Nyssi_dc13_depths = dc13_depths[Talat_Nyssi_mask]
    Talat_Nyssi_dc13_depths -=  np.max(Talat_Nyssi_dc13_depths)
    Talat_Nyssi_dc13_depths *= -1
    Talat_Nyssi_dc13 = dc13[Talat_Nyssi_mask]

    Oued_Sdas_dc13_depths = dc13_depths[Oued_Sdas_mask]
    Oued_Sdas_dc13_depths -=  np.max(Oued_Sdas_dc13_depths)
    Oued_Sdas_dc13_depths *= -1
    Oued_Sdas_dc13 = dc13[Oued_Sdas_mask]

    Sukharikha_dc13_depths = dc13_depths[Sukharikha_mask]
    Sukharikha_dc13_depths -=  np.max(Sukharikha_dc13_depths)
    Sukharikha_dc13_depths *= -1
    Sukharikha_dc13 = dc13[Sukharikha_mask]

    Tiout_mask = np.where(radiometric_site == "MS6")
    Talat_Nyssi_mask = np.where(radiometric_site == "MS3")
    Oued_Sdas_mask = np.where(radiometric_site == "MS7")

    Tiout_radiometric_depths = radiometric_depths[Tiout_mask]
    Tiout_radiometric_depths -=  np.max(Tiout_radiometric_depths)
    Tiout_radiometric_depths *= -1
    Tiout_radiometric = radiometric[Tiout_mask]
    Tiout_radiometric_sd = radiometric_sd[Tiout_mask]

    Talat_Nyssi_radiometric_depths = radiometric_depths[Talat_Nyssi_mask]
    Talat_Nyssi_radiometric_depths -=  np.max(Talat_Nyssi_radiometric_depths)
    Talat_Nyssi_radiometric_depths *= -1
    Talat_Nyssi_radiometric = radiometric[Talat_Nyssi_mask]
    Talat_Nyssi_radiometric_sd = radiometric_sd[Talat_Nyssi_mask]

    Oued_Sdas_radiometric_depths = radiometric_depths[Oued_Sdas_mask]
    Oued_Sdas_radiometric_depths -=  np.max(Oued_Sdas_radiometric_depths)
    Oued_Sdas_radiometric_depths *= -1
    Oued_Sdas_radiometric = radiometric[Oued_Sdas_mask]
    Oued_Sdas_radiometric_sd = radiometric_sd[Oued_Sdas_mask]

    data = {
        "Tiout_dc13_depths": Tiout_dc13_depths,
        "Tiout_dc13": Tiout_dc13,
        "Talat_Nyssi_dc13_depths": Talat_Nyssi_dc13_depths,
        "Talat_Nyssi_dc13": Talat_Nyssi_dc13,
        "Oued_Sdas_dc13_depths": Oued_Sdas_dc13_depths,
        "Oued_Sdas_dc13": Oued_Sdas_dc13,
        "Sukharikha_dc13_depths": Sukharikha_dc13_depths,
        "Sukharikha_dc13": Sukharikha_dc13,

        "Tiout_radiometric_depths": Tiout_radiometric_depths,
        "Tiout_radiometric": Tiout_radiometric,
        "Tiout_radiometric_sd": Tiout_radiometric_sd,
        "Talat_Nyssi_radiometric_depths": Talat_Nyssi_radiometric_depths,
        "Talat_Nyssi_radiometric": Talat_Nyssi_radiometric,
        "Talat_Nyssi_radiometric_sd": Talat_Nyssi_radiometric_sd,
        "Oued_Sdas_radiometric_depths": Oued_Sdas_radiometric_depths,
        "Oued_Sdas_radiometric": Oued_Sdas_radiometric,
        "Oued_Sdas_radiometric_sd": Oued_Sdas_radiometric_sd,
    }

    return data


def get_hmc_config():
    N = 10
    H = 20
    delta_c = H / N
    cs = np.linspace(0, H, N + 1)
    num_samples = 100000
    num_chains = 4
    a = 1.5
    b = 0.27

    config = {
        "N": N,
        "H": H,
        "delta_c": delta_c,
        "cs": cs,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "a": a,
        "b": b
    }

    config_str = "_".join(
    f"{k}{v:.2f}" if isinstance(v, float) else f"{k}{v}"
    for k, v in config.items()
    if isinstance(v, (int, float))
    )   

    return config, config_str