from dataclasses import dataclass

import numpy as np

from biosim.exceptions import InvalidParameterError
from biosim.validation import require_nonnegative, require_positive


@dataclass
class InitialConditions:
    """Initial state of the reactor.

    X0: initial biomass concentration (g/L)
    S0: initial substrate concentration (g/L)
    P0: initial product concentration (g/L)
    V0: initial working volume (L)
    C_O2_0: initial dissolved oxygen concentration (mg/L), required only when
        the chosen oxygen model tracks supply dynamics (e.g. OxygenWithKLa)
    I0: initial inducer concentration (uM); used only when the product model is
        inducible (it then enters the state vector), otherwise ignored
    """

    X0: float
    S0: float
    P0: float = 0.0
    V0: float = 1.0
    C_O2_0: float | None = None
    I0: float = 0.0

    def __post_init__(self) -> None:
        require_nonnegative("X0", self.X0)
        require_nonnegative("S0", self.S0)
        require_nonnegative("P0", self.P0)
        require_positive("V0", self.V0)
        require_nonnegative("I0", self.I0)
        if self.C_O2_0 is not None:
            require_nonnegative("C_O2_0", self.C_O2_0)


class StateLayout:
    """Maps named state variables to indices in the ODE state vector y.

    Base layout is [X, S, P, V]. Optional states are inserted before V, in this order:
    C_O2 (when oxygen supply dynamics are enabled) then I (inducer concentration, when the
    product model is inducible): [X, S, P, (C_O2), (I), V]. Built dynamically so no code
    elsewhere hardcodes vector positions.
    """

    def __init__(self, has_oxygen_state: bool, has_inducer_state: bool = False):
        self.has_oxygen_state = has_oxygen_state
        self.has_inducer_state = has_inducer_state
        self.idx_X = 0
        self.idx_S = 1
        self.idx_P = 2
        nxt = 3
        self.idx_C_O2: int | None = None
        self.idx_I: int | None = None
        if has_oxygen_state:
            self.idx_C_O2 = nxt
            nxt += 1
        if has_inducer_state:
            self.idx_I = nxt
            nxt += 1
        self.idx_V = nxt
        self.size = nxt + 1

    def build_initial_vector(self, ic: InitialConditions) -> np.ndarray:
        if self.has_oxygen_state and ic.C_O2_0 is None:
            raise InvalidParameterError(
                "C_O2_0 must be provided in InitialConditions when the oxygen "
                "model tracks dissolved-O2 supply dynamics"
            )
        y0 = np.zeros(self.size)
        y0[self.idx_X] = ic.X0
        y0[self.idx_S] = ic.S0
        y0[self.idx_P] = ic.P0
        if self.has_oxygen_state:
            y0[self.idx_C_O2] = ic.C_O2_0
        if self.has_inducer_state:
            y0[self.idx_I] = ic.I0
        y0[self.idx_V] = ic.V0
        return y0

    def unpack(self, y: np.ndarray) -> tuple[float, float, float, float | None, float]:
        X = y[self.idx_X]
        S = y[self.idx_S]
        P = y[self.idx_P]
        C_O2 = y[self.idx_C_O2] if self.has_oxygen_state else None
        V = y[self.idx_V]
        return X, S, P, C_O2, V

    def inducer(self, y: np.ndarray) -> float:
        """Inducer concentration (uM) from y; 0.0 when the layout has no inducer state."""
        return float(y[self.idx_I]) if self.has_inducer_state else 0.0

    def column_names(self) -> list[str]:
        names = ["X", "S", "P"]
        if self.has_oxygen_state:
            names.append("C_O2")
        if self.has_inducer_state:
            names.append("I")
        names.append("V")
        return names
