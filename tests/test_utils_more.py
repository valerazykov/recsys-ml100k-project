from src import utils


def test_dict_deep_update_simple():
    base = {"a": {"b": 1}, "x": 5}
    upd = {"a": {"c": 2}, "x": 6}
    res = utils.dict_deep_update(base, upd)
    assert res["a"]["b"] == 1 and res["a"]["c"] == 2 and res["x"] == 6


def test_safe_cast_and_yaml_json(tmp_path):
    assert utils.safe_cast("10", int) == 10
    assert utils.safe_cast("notnum", int, default=0) == 0

    # write/read json
    p = tmp_path / "t.json"
    utils.save_json({"k": 1}, str(p))
    data = utils.load_json(str(p))
    assert data["k"] == 1

    # yaml
    p2 = tmp_path / "t.yaml"
    utils.save_yaml({"a": 2}, str(p2))
    loaded = utils.load_yaml(str(p2))
    assert loaded["a"] == 2
