from ocellar.io import d_cclib, d_openbabel

BACKENDS = {"cclib" : d_cclib.DCclib, "openbabel" : d_openbabel.DOpenbabel}

def Driver(backend):
    return BACKENDS[backend]