from protocolquant import __version__


def test_import_and_version() -> None:
    assert isinstance(__version__, str)
    assert __version__
