from ocellar.molecule import Molecule
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def molecule():
    file = Path(__file__).absolute().parent / "data" / "PhEt.xyz"
    mol = Molecule(str(file))
    mol.build_geometry()
    mol.build_graph()
    mol.build_structure(cut_molecule=True)
    return mol


def test_graph_reconstruction(molecule):
    initial_graph = molecule.graph.copy()

    new_m, idxs = molecule.select([2, 3, 4, 5, 6, 7])
    mapping = {old_id: new_id for new_id, old_id in enumerate(idxs)}

    # new_m.build_graph()
    graph = new_m.graph.copy()
    assert graph is not None

    for new_id, old_id in enumerate(idxs):
        assert graph.degree[new_id] == initial_graph.degree[old_id]

        old_neighbors = list(initial_graph.neighbors(old_id))
        new_neighbors = list(graph.neighbors(new_id))

        extra = [i for i in new_neighbors if i >= len(idxs)]

        for old_neighbor in old_neighbors:
            if old_neighbor not in mapping:
                assert len(extra) > 0
                continue

            assert mapping[old_neighbor] in list(new_neighbors)
            assert (
                graph.get_edge_data(new_id, mapping[old_neighbor])["order"]
                == initial_graph.get_edge_data(old_id, old_neighbor)["order"]
            )
