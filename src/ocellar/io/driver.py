import inspect


class Driver:
    backend = None

    @classmethod
    def _build_geometry(cls, *args, **kwargs):
        raise NotImplementedError(inspect.stack()[0][3], cls.backend)

    @classmethod
    def _build_bonds(cls, *args, **kwargs):
        raise NotImplementedError(inspect.stack()[0][3], cls.backend)