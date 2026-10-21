"""Fed-batch with a temperature shift at 48 h: feed-limited biomass at several shift targets.

Shows the model's central point: with the same feed, temperature changes the final biomass
through the maintenance coefficient m(T) (and yield Y(T)), not through a mu_max(T).
"""

from biosim import (
    BioreactorSimulation,
    ConditionProfile,
    FedBatch,
    GrowthDrivenSubstrate,
    InitialConditions,
    OxygenDemandOnly,
    PirtMonodGrowth,
    TemperatureLuedekingPiretProduct,
    constant_feed,
    shift_value,
)


def run(shift_to_C: float):
    sim = BioreactorSimulation(
        growth_model=PirtMonodGrowth(
            Yxs_ref=0.5, ms_ref=0.02, Ea_ms=50.0, qSmax_ref=50.0, Ea_qS=0.0, Ks=1e-3
        ),
        product_model=TemperatureLuedekingPiretProduct(alpha=0.0, beta_ref=0.05, Ea_beta=30.0),
        substrate_model=GrowthDrivenSubstrate(),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
        operation_mode=FedBatch(feed_rate_fn=constant_feed(0.05), S_feed=100.0),
        initial_conditions=InitialConditions(X0=1.0, S0=0.0, P0=0.0, V0=1.0),
        t_span=(0.0, 100.0),
        n_points=201,
        condition_profile=ConditionProfile(shift_value(30.0, shift_to_C, 48.0)),
    )
    return sim.run().data


if __name__ == "__main__":
    print("shift to  final X (g/L)  final biomass x=X*V (g)  final product p=P*V (g)")
    for T in (22.0, 24.0, 30.0, 36.0):
        d = run(T)
        last = d.iloc[-1]
        print(
            f"{T:5.0f} C   {last.X:10.2f}  {last.X * last.V:20.1f}  {last.P * last.V:20.1f}"
        )
