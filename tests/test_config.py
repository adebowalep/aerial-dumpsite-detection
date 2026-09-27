import pytest

from src.config import TrainConfig, load_config, save_config


def test_load_config_with_no_path_returns_defaults():
    config = load_config()
    assert config == TrainConfig()


def test_load_config_reads_yaml_values(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("num_epochs: 3\nlearning_rate: 0.01\n")

    config = load_config(str(yaml_path))

    assert config.num_epochs == 3
    assert config.learning_rate == 0.01
    assert config.batch_size == TrainConfig().batch_size  # untouched default


def test_load_config_rejects_unknown_yaml_key(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("not_a_real_field: 1\n")

    with pytest.raises(ValueError):
        load_config(str(yaml_path))


def test_cli_overrides_take_precedence_over_yaml(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("num_epochs: 3\n")

    config = load_config(str(yaml_path), overrides={"num_epochs": 99, "learning_rate": None})

    assert config.num_epochs == 99  # override wins
    assert config.learning_rate == TrainConfig().learning_rate  # None override is ignored


def test_overrides_reject_unknown_key():
    with pytest.raises(ValueError):
        load_config(overrides={"not_a_real_field": 1})


def test_common_size_property():
    config = TrainConfig(common_size_width=512, common_size_height=256)
    assert config.common_size == (512, 256)


def test_save_config_round_trips(tmp_path):
    config = TrainConfig(num_epochs=7)
    out_path = tmp_path / "saved.yaml"

    save_config(config, str(out_path))
    reloaded = load_config(str(out_path))

    assert reloaded == config
