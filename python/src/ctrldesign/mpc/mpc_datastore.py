from typing import Any, Sequence

import numpy as np
import pandas as pd

from .mpc_solver import MpcSolver


class MpcDataStore:
    def __init__(
        self,
        mpcsolver: MpcSolver,
        initial_capacity: int,
        add_columns: Sequence[str] = (),
    ):
        self._x_names = mpcsolver.x_names
        self._u_free_names = tuple(f.signal for f in mpcsolver.free_inputs)
        self._params = tuple(
            (p.signal, mpcsolver.N + 1 if p.is_vector else 1)
            for p in mpcsolver.parameters
        )
        self._add_columns = tuple(add_columns)

        cols = (
            len(self._x_names)
            + len(self._u_free_names)
            + sum(p[1] for p in self._params)
            + len(self._add_columns)
            + 1
        )

        self._data = np.zeros((initial_capacity, cols))
        self._cur_row = 0

    def append(self, step_result: dict[str, Any], **add_data):
        if self._cur_row >= self._data.shape[0]:
            self._resize()

        idx1 = 0

        for aname in self._add_columns:
            idx0 = idx1
            idx1 = idx0 + 1
            self._data[self._cur_row, idx0] = add_data[aname]

        unknown_data = set(add_data.keys()) - set(self._add_columns)
        if len(unknown_data) > 0:
            raise ValueError(
                f"unknown additional data columns: {', '.join(unknown_data)}"
            )

        idx0 = idx1
        idx1 = idx0 + len(self._x_names)

        self._data[self._cur_row, idx0:idx1] = step_result["x0"].reshape((-1,))

        for pname, plen in self._params:
            idx0 = idx1
            idx1 = idx0 + plen

            if isinstance(step_result["params"][pname], np.ndarray):
                self._data[self._cur_row, idx0:idx1] = step_result["params"][
                    pname
                ].reshape((-1,))
            else:
                self._data[self._cur_row, idx0:idx1] = step_result["params"][pname]

        idx0 = idx1
        idx1 = idx0 + len(self._u_free_names)
        self._data[self._cur_row, idx0:idx1] = step_result["u_opt"][:, 0].reshape((-1,))

        if "opti_sol" in step_result:
            self._data[self._cur_row, idx1] = step_result["opti_sol"].stats()[
                "iterations"
            ]["obj"][-1]
        else:
            self._data[self._cur_row, idx1] = np.nan

        self._cur_row += 1

    def get_len(self) -> int:
        return self._cur_row

    def get_dataframe(self, mark_x0_and_u_opt: bool = True) -> pd.DataFrame:

        cols = []

        for aname in self._add_columns:
            cols.append(aname)

        if mark_x0_and_u_opt:
            cols += [f"x0[{i}]: {x}" for (i, x) in enumerate(self._x_names)]
        else:
            cols += list(self._x_names)

        for pname, plen in self._params:
            if plen == 1:
                cols.append(pname)
            else:
                cols += [f"{pname}[{i}]" for i in range(plen)]

        if mark_x0_and_u_opt:
            cols += [f"u_opt[{i}]: {u}" for (i, u) in enumerate(self._u_free_names)]
        else:
            cols += list(self._u_free_names)

        cols.append("obj")

        return pd.DataFrame(self._data[: self._cur_row, :], columns=cols)

    def _resize(self):
        new_size = max(1, int(np.ceil(self._data.shape[0] * 1.6)))

        new_data = np.zeros((new_size, self._data.shape[1]))
        new_data[: self._data.shape[0], :] = self._data

        self._data = new_data

    @staticmethod
    def remove_marks_from_df_columns(df: pd.DataFrame):
        def remove_marks(s: str) -> str:
            if ":" not in s:
                return s

            s = s.split(": ")
            return s[1]

        df.columns = [remove_marks(c) for c in df.columns]
