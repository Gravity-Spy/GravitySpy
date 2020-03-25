"""Unit test for GravitySpy
"""

from gravityspy.api.project import GravitySpyProject
import os

PROJECT_PICKLE = os.path.join(os.path.split(__file__)[0], '..', '..',
                              'examples', '1104.pkl')

PROJECT = GravitySpyProject.load_project_from_cache(PROJECT_PICKLE)

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'

class TestGravitySpyAPI(object):
    """`TestCase` for the GravitySpy
    """
