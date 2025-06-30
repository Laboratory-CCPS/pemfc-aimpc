import os
from contextlib import redirect_stderr
from datetime import datetime

import numpy as np
import pandas as pd
import tqdm

from src.model_suh.conversions import rpm2rad


# %%
class Mpcdatagenerator:
    def __init__(
        self,
        mpc_solver,
        n_samples: 10000,
        reset_chance=0.25,
        x_high: np.array = np.array([0.22e5, 1.73e5, rpm2rad(100e3), 2.65e5]),
        x_low: np.array = np.array([0.09e5, 0.7e5, rpm2rad(45e3), 1.35e5]),
        load_high: float = 330.0,
        load_low: float = 90.0,
        fnames: list | str = ["p_O2", "p_N2", "w_cp", "p_sm", "I_load"],
        lnames: list | str = ["I_st", "v_cm"],
        filename: str = None,
        future_I_load_known: bool = False,
    ):

        self.fnames = fnames
        self.lnames = lnames
        self.mpcsolver = mpc_solver
        self.nsamples = n_samples
        self.resetchance = reset_chance
        self.xhigh = x_high
        self.xlow = x_low
        self.lhigh = load_high
        self.llow = load_low
        self.filename = filename
        self.future_I = future_I_load_known

        self.data = None

    def get_new_init(self):
        I_load = np.random.uniform(high=self.lhigh, low=self.llow)
        x_0 = np.random.uniform(high=self.xhigh, low=self.xlow)
        return I_load, x_0

    def save_data(self, changed_filename: str = None, save_dir: str = ".//data"):
        if len(save_dir) > 0:
            os.makedirs(save_dir, exist_ok=True)

        now = datetime.now().strftime(r"%Y_%m_%d_%H")
        if changed_filename is None:
            if self.filename is None:
                filename = os.path.join(
                    save_dir,
                    f"MPC_{self.mpcsolver.N}_samples_{self.nsamples}_datapoints_"
                    + now
                    + ".csv",
                )
            else:
                filename = os.path.join(
                    save_dir,
                    f"{self.filename}_MPC_{self.mpcsolver.N}_samples_{self.nsamples}_datapoints_"
                    + now
                    + ".csv",
                )
        else:
            filename = os.path.join(
                save_dir,
                f"{changed_filename}_MPC_{self.mpcsolver.N}_samples_{self.nsamples}_datapoints_"
                + now
                + ".csv",
            )

        self.data.to_csv(filename, index=False)
        return filename

    def generate_data(self):
        log_dict = {f"{x}": np.empty(self.nsamples) for x in self.fnames + self.lnames}

        # Initialization
        I_load_step, x_init = self.get_new_init()
        lam_g = None

        u_first_init = self.mpcsolver.get_u_init()
        u_init = u_first_init

        v_cm_init = self.mpcsolver.get_signal_value("v_cm", u=u_init)
        v_cm_prev = v_cm_init

        for i in tqdm.tqdm(range(self.nsamples)):
            if np.random.random() < self.resetchance:
                I_load_step, x_init = self.get_new_init()
                u_init = u_first_init
                v_cm_prev = v_cm_init

            # Try to solve with original x0; if it fails try with different x0 until it works
            trysolve = True
            while trysolve:
                try:
                    with redirect_stderr(None):
                        # resample load always
                        step_result = self.mpcsolver.solve_step(
                            x0=x_init,
                            u_init=u_init,
                            lam_g=lam_g,
                            I_load=I_load_step,
                            v_cm_prev=v_cm_prev,
                        )
                        trysolve = False
                except RuntimeError as _:
                    I_load_step, x_init = self.get_new_init()
                    u_init = u_first_init
                    v_cm_prev = v_cm_init

            lam_g = step_result["lam_g"]

            u_i = self.mpcsolver.get_complete_input(
                step_result["u_opt"][:, [0]], step_result["u_fixed"][:, 0], 1
            )

            # logging
            log_dict["I_st"][i] = self.mpcsolver.get_signal_value("I_st", u=u_i)
            log_dict["v_cm"][i] = self.mpcsolver.get_signal_value("v_cm", u=u_i)

            for statename in self.fnames:
                if statename != "I_load":
                    log_dict[statename][i] = self.mpcsolver.get_signal_value(
                        statename, x=step_result["x0"]
                    )
                else:
                    log_dict[statename][i] = I_load_step

            u_init[:, :-1] = step_result["u_opt"][:, 1:]
            u_init[:, -1] = step_result["u_opt"][:, -1]

            v_cm_prev = self.mpcsolver.get_signal_value("v_cm", u=u_init)

            if self.mpcsolver.multiple_shooting:
                x_init = np.hstack(
                    (step_result["x_opt"], step_result["x_opt"][:, [-1]])
                )
            else:
                x_init = step_result["x_opt"][:, 0]

        self.data = pd.DataFrame(log_dict)
        filename = self.save_data()

        print(
            f"Data generation complete.\nGenerated {self.data.shape[0]} data points for an MPC with {self.mpcsolver.N} steps.\nResult stored under {filename}"
        )
        return filename
