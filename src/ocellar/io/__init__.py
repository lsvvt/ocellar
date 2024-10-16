from ocellar.io import d_cclib, d_openbabel, d_mdanalysis, d_internal

BACKENDS = {"internal" : d_internal.Dinternal, "cclib" : d_cclib.DCclib, "openbabel" : d_openbabel.DOpenbabel, "MDAnalysis" : d_mdanalysis.DMDAnalysis}

def Driver(backend):
    return BACKENDS[backend]