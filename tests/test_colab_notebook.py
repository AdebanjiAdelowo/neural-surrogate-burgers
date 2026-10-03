"""Checks on colab/run_cuda.ipynb that run without Colab or a GPU: structure, the run-identity
lifecycle, and that the historical-input upload never lands in the repository checkout."""
import ast
import json
import os
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "colab" / "run_cuda.ipynb"


@pytest.fixture(scope="module")
def code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def cell(cells, marker):
    (found,) = [c for c in cells if marker in c]
    return found


def test_notebook_is_clean_and_parses(code_cells):
    nb = json.loads(NOTEBOOK.read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
    for src in code_cells:
        ast.parse(src)
    config = cell(code_cells, "REPO_URL =")
    assert 'RUN_FULL = False' in config and 'DATA_SOURCE = "upload"' in config


def test_run_id_survives_rerunning_the_configuration_cell(code_cells):
    config, helpers = cell(code_cells, "REPO_URL ="), cell(code_cells, "def run(cmd")
    # the configuration cell must not assign RUN_ID at all
    assigned = {t.id for node in ast.walk(ast.parse(config)) if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    assert "RUN_ID" not in assigned
    (id_line,) = [l for l in cell(code_cells, "init_run_id(globals()").splitlines() if "init_run_id(" in l]

    g = {}
    exec(config, g)
    exec(helpers, g)
    exec(id_line, g)                      # section 3: identity created once
    first = g["RUN_ID"]
    assert first and first.startswith("cuda_")
    exec(config, g)                       # the user re-runs section 0 ...
    g["RUN_FULL"] = True                  # ... to switch the gate on
    exec(id_line, g)                      # and re-runs section 3
    assert g["RUN_ID"] == first

    # provenance and the archive both take that same identity
    assert '"run_id": RUN_ID' in cell(code_cells, 'PROV = {"run_id"')
    package = cell(code_cells, "make_archive")
    assert 'make_archive(f"/content/{RUN_ID}", "zip", "runs", RUN_ID)' in package
    assert "assert RUN_ID and" in package


@pytest.fixture
def helpers(code_cells):
    g = {}
    exec(cell(code_cells, "def run(cmd"), g)
    return g


def test_upload_prompt_never_writes_into_the_checkout(helpers, tmp_path, monkeypatch):
    repo, content = tmp_path / "repo", tmp_path / "content"
    repo.mkdir()
    content.mkdir()
    (repo / "README.md").write_text("x")
    monkeypatch.chdir(repo)               # the notebook's working directory is the checkout

    def colab_like_upload():              # google.colab.files.upload saves into the CWD
        Path("historical_inputs.zip").write_bytes(b"zip")
        return {"historical_inputs.zip": b"zip"}

    zip_path = str(content / "historical_inputs.zip")
    got = helpers["stage_historical_zip"](zip_path, str(content / "inputs"), str(repo), colab_like_upload)
    assert got == zip_path and Path(zip_path).read_bytes() == b"zip"
    assert sorted(os.listdir(repo)) == ["README.md"]
    assert os.getcwd() == str(repo)


def test_archive_dragged_into_the_checkout_is_moved_out(helpers, tmp_path):
    repo, content = tmp_path / "repo", tmp_path / "content"
    repo.mkdir()
    content.mkdir()
    (repo / "historical_inputs.zip").write_bytes(b"zip")
    zip_path = str(content / "historical_inputs.zip")
    assert helpers["stage_historical_zip"](zip_path, str(content / "inputs"), str(repo)) == zip_path
    assert os.listdir(repo) == []


def test_archive_location_inside_the_checkout_is_refused(helpers, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="outside the repository"):
        helpers["stage_historical_zip"](str(repo / "h.zip"), str(tmp_path / "inputs"), str(repo))
    assert helpers["stage_historical_zip"](str(tmp_path / "h.zip"), str(tmp_path / "in"), str(repo)) is None
