import pyDive
import pytest
import os

@pytest.fixture(scope="session")
def init_pyDive(request):
    pyDive.init(os.environ["IPP_PROFILE_NAME"])