import random

import numpy as np
import tqdm

from src.model_suh import load_generator, model_suh
from src.model_suh.conversions import bar2pa, rpm2rad
from src.mpc import mpc_problem, mpc_solver
from src.mpc.mpc_datastore import MpcDataStore
from src.ode_modelling.ode_model import get_model


def main():
    p = model_suh.params()
    model = get_model(model_suh.model, p)

    limits = {
        "p_O2": (bar2pa(0.05), bar2pa(0.3)),
        "p_N2": (bar2pa(0.25), bar2pa(2.5)),
        "w_cp": (rpm2rad(15e3), rpm2rad(105e3)),
        "p_sm": (bar2pa(1), bar2pa(3.5)),
    }

    rel_std = {
        "p_O2": 0.025,
        "p_N2": 0.025,
        "w_cp": 0.025,
        "p_sm": 0.025,
    }

    abs_std = {k: v * (limits[k][1] - limits[k][0]) for k, v in rel_std.items()}

    n_explorations_per_step = 5

    mpcproblem = mpc_problem.MpcProblem(model)

    mpcproblem.add_free_input("I_st", 0, np.inf, 100)
    mpcproblem.add_free_input("v_cm", 50, 250, 110)

    mpcproblem.add_constraint("p_O2", 0, np.inf)
    mpcproblem.add_constraint("p_N2", 0, np.inf)
    mpcproblem.add_constraint("p_sm", 0, np.inf)
    mpcproblem.add_constraint("w_cp", rpm2rad(20e3), rpm2rad(100e3))
    mpcproblem.add_constraint("lambda_O2", 1.5, 5, feedthrough=True)
    mpcproblem.add_constraint("lambda_O2", 1.5, 5, feedthrough=False)

    mpcproblem.add_quadratic_cost("I_st", ref="#I_load", weight_stage=5e-2)
    mpcproblem.add_quadratic_cost("lambda_O2", ref=2, weight_stage=1, weight_end=1)

    mpcproblem.add_quadratic_diffquot_cost(
        "v_cm", diff_quot=(-1, 1), ts_order=0, weight=1e-5
    )

    Ts_mpc = 0.025
    Npredict = 10

    discretization = "rk1"
    substeps = 2
    scale_model = True
    use_lambda = True
    multiple_shooting = True

    plugin_options = {}
    solver_options = {}
    solver_options_expl = {"max_iter": 100}

    load_gen = load_generator.LoadGenerator(
        I_init=130.0,
        I_range=(20.0, 320.0),
        init_duration=1.0,
        load_change_step_prop=0.75,
        load_step_hold_duration_range=(0.1, 2.5),
        load_ramp_duration_range=(0.1, 5.0),
        load_ramp_hold_prob=0.25,
        load_ramp_hold_duration_range=(0.1, 2.0),
    )

    [_, x0] = model_suh.get_validation_input(Ts_mpc)

    T_end = 10

    mpcsolver = mpc_solver.MpcSolver(
        mpcproblem,
        Ts_mpc,
        Npredict,
        discretization,
        substeps=substeps,
        scale_model=scale_model,
        use_lambda=use_lambda,
        multiple_shooting=multiple_shooting,
        solver_options=solver_options,
        plugin_options=plugin_options,
        verbose=False,
    )

    mpcsolver_expl = mpc_solver.MpcSolver(
        mpcproblem,
        Ts_mpc,
        Npredict,
        discretization,
        substeps=substeps,
        scale_model=scale_model,
        use_lambda=use_lambda,
        multiple_shooting=multiple_shooting,
        solver_options=solver_options_expl,
        plugin_options=plugin_options,
        verbose=False,
    )

    u_init = mpcsolver.get_u_init()
    x_init = x0
    lam_g = None

    n_mpc_steps = int(np.floor(T_end / Ts_mpc))

    u_mpc = np.nan * np.zeros((len(mpcsolver.free_inputs), n_mpc_steps))
    u_fixed = np.nan * np.zeros((len(mpcsolver.fixed_inputs), n_mpc_steps))
    I_load = np.nan * np.zeros((1, n_mpc_steps + 1))

    data_store = MpcDataStore(
        mpcsolver,
        initial_capacity=n_mpc_steps * (n_explorations_per_step + 1),
        add_columns=("predecessor",),
    )

    cur_predecessor = np.nan

    n_fails_exception = 0
    n_fails_no_success = 0
    n_success_expl = 0
    n_invalid_state = 0

    for i in tqdm.tqdm(range(n_mpc_steps)):

        t_step = i * mpcsolver.Ts
        I_load_step = load_gen(t_step)

        step_result = mpcsolver.solve_step(
            x0=x_init, u_init=u_init, lam_g=lam_g, I_load=I_load_step
        )

        if step_result["opti_sol"].stats()["success"]:
            (x_init, u_init, lam_g) = mpcsolver.get_next_initial_values(step_result)

        u_mpc[:, i] = step_result["u_opt"][:, 0]
        u_fixed[:, i] = step_result["u_fixed"][:, 0]
        I_load[:, i] = I_load_step

        if not step_result["opti_sol"].stats()["success"]:
            print(step_result["opti_sol"].stats()["return_status"])
            continue

        data_store.append(step_result, predecessor=cur_predecessor)
        cur_predecessor = data_store.get_len() - 1

        i_success = 0

        for _ in range(n_explorations_per_step):
            x0_expl = x_init[:, 0].copy()
            u_init_expl = u_init.copy()

            while True:
                for i, xname in enumerate(mpcsolver.x_names):
                    val = np.inf

                    while val < limits[xname][0] or val > limits[xname][1]:
                        val = random.normalvariate(x0_expl[i], abs_std[xname])

                    x0_expl[i] = val

                (valid, v_cm) = model_suh.check_state(x0_expl, p)

                if valid:
                    v_cm_init = mpcsolver.extract_mpc_input_values(
                        "I_st", u_free=u_init_expl
                    )
                    v_cm_init[v_cm_init < v_cm] = v_cm

                    u_init_expl = mpcsolver.get_u_init(
                        u_init=u_init_expl, v_cm=v_cm_init
                    )

                    break

                n_invalid_state += 1

            try:
                step_result = mpcsolver_expl.solve_step(
                    x0=x0_expl, u_init=u_init, lam_g=lam_g, I_load=I_load_step
                )
            except RuntimeError:
                n_fails_exception += 1
                continue

            if step_result["opti_sol"].stats()["success"]:
                data_store.append(step_result, predecessor=-cur_predecessor)
                i_success += 1
                n_success_expl += 1

                if i_success == n_explorations_per_step:
                    break
            else:
                n_fails_no_success += 1

    print(
        f"exploration trials: {n_success_expl} succeeded, "
        f"{n_fails_exception + n_fails_no_success} failed ({n_fails_exception} w/ exception)"
    )

    print(f"{n_invalid_state} invalid states rejected")

    data_store.get_dataframe().to_feather("mpc_data.feather")


if __name__ == "__main__":
    main()
