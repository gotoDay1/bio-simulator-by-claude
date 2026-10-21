"""Fed-batch with the standard induction recipe (300 uM at 32/56/80 h) and dose variants.

Shows the inducer state: it is raised at each dose and diluted by the feed in between, and
the product responds through a saturating beta(C).
"""

from biosim import (
    BioreactorSimulation,
    FedBatch,
    GrowthDrivenSubstrate,
    InducibleLuedekingPiretProduct,
    InitialConditions,
    OxygenDemandOnly,
    PirtMonodGrowth,
    constant_feed,
    standard_induction,
)


def run(conc: float, mode: str = "set"):
    sim = BioreactorSimulation(
        growth_model=PirtMonodGrowth(
            Yxs_ref=0.5, ms_ref=0.02, Ea_ms=50.0, qSmax_ref=50.0, Ea_qS=0.0, Ks=1e-3
        ),
        product_model=InducibleLuedekingPiretProduct(
            alpha=0.0, beta_basal=0.002, beta_max=0.05, EC50=100.0
        ),
        substrate_model=GrowthDrivenSubstrate(),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
        operation_mode=FedBatch(feed_rate_fn=constant_feed(0.05), S_feed=100.0),
        initial_conditions=InitialConditions(X0=1.0, S0=0.0, P0=0.0, V0=1.0),
        t_span=(0.0, 100.0),
        n_points=201,
        inducer_events=standard_induction(conc=conc, mode=mode) if conc > 0 else [],
    )
    return sim.run().data


if __name__ == "__main__":
    print("dose (uM) x3   final inducer (uM)   final product p = P*V (g)")
    for conc in (0.0, 150.0, 300.0, 600.0):
        last = run(conc).iloc[-1]
        print(f"{conc:10.0f}   {last.I:16.1f}   {last.P * last.V:22.1f}")
    one = run(300.0).query("t in [31.5, 32.0, 56.0, 80.0, 100.0]")[["t", "I"]]
    print("\n300 uM recipe, inducer concentration over time:")
    print(one.to_string(index=False))
