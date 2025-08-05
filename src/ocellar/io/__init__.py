"""Initialize backends mapping for all available drivers."""

from ocellar.io import d_cclib, d_internal, d_mdanalysis, d_openbabel, d_ovito

BACKENDS = {
    "internal": d_internal.DInternal,
    "cclib": d_cclib.DCclib,
    "openbabel": d_openbabel.DOpenbabel,
    "MDAnalysis": d_mdanalysis.DMDAnalysis,
    "ovito": d_ovito.DOvito,
}


def Driver(backend: str):
    return BACKENDS[backend]
