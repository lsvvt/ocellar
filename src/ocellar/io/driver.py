import inspect


class Driver:
    """Base class for drivers implementing molecule operations.

    This class defines the interface for various backend drivers.
    All methods raise NotImplementedError and should be overridden by subclasses.

    Attributes
    ----------
    backend : None or str
        Identifier for the backend used by the driver. Should be set by subclasses.

    """

    backend = None

    @classmethod
    def _build_geometry(cls, *args, **kwargs):
        raise NotImplementedError(inspect.stack()[0][3], cls.backend)

    @classmethod
    def _build_bonds(cls, *args, **kwargs):
        raise NotImplementedError(inspect.stack()[0][3], cls.backend)

    @classmethod
    def _save_xyz(cls, *args, **kwargs):
        raise NotImplementedError(inspect.stack()[0][3], cls.backend)

    @classmethod
    def _save_pdb(cls, *args, **kwargs):
        raise NotImplementedError(inspect.stack()[0][3], cls.backend)

    @classmethod
    def _save_dump(cls, *args, **kwargs):
        raise NotImplementedError(inspect.stack()[0][3], cls.backend)
