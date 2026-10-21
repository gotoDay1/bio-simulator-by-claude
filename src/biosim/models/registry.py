from biosim.models.base import GrowthModel, OxygenModel, ProductModel, SubstrateModel
from biosim.models.growth import (
    GompertzGrowth,
    LogisticGrowth,
    MonodGrowth,
    MonodGrowthO2,
    PirtMonodGrowth,
)
from biosim.models.oxygen import OxygenDemandOnly, OxygenWithKLa
from biosim.models.product import (
    LuedekingPiretProduct,
    NoProduct,
    TemperatureLuedekingPiretProduct,
)
from biosim.models.substrate import GrowthDrivenSubstrate, YieldMaintenanceSubstrate

GROWTH_MODELS: dict[str, type[GrowthModel]] = {
    "monod": MonodGrowth,
    "logistic": LogisticGrowth,
    "gompertz": GompertzGrowth,
    "monod_o2": MonodGrowthO2,
    "pirt_monod": PirtMonodGrowth,
}

PRODUCT_MODELS: dict[str, type[ProductModel]] = {
    "luedeking_piret": LuedekingPiretProduct,
    "luedeking_piret_temp": TemperatureLuedekingPiretProduct,
    "none": NoProduct,
}

SUBSTRATE_MODELS: dict[str, type[SubstrateModel]] = {
    "yield_maintenance": YieldMaintenanceSubstrate,
    "growth_driven": GrowthDrivenSubstrate,
}

OXYGEN_MODELS: dict[str, type[OxygenModel]] = {
    "demand_only": OxygenDemandOnly,
    "kla_supply": OxygenWithKLa,
}
