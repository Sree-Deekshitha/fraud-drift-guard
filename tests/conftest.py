import pytest

from fraudguard.model import fit_model
from fraudguard.simulate import Scenario, generate_weeks

SMALL = Scenario(weekly_volume=6_000, base_fraud_rate=0.03)


@pytest.fixture(scope="session")
def small_data():
    return generate_weeks(range(0, 4), SMALL)


@pytest.fixture(scope="session")
def trained(small_data, tmp_path_factory):
    train = small_data[small_data.week <= 2]
    val = small_data[small_data.week == 3]
    model = fit_model(train, val, version="test")
    path = model.save(tmp_path_factory.mktemp("model"))
    return model, path
