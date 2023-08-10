from setuptools import setup, find_packages
from typing import List

HYPHEN = '-e .'
def get_requirements(filename:str) -> List[str]:
    """Return the list of requirements from requirements.txt file"""
    requirements = []
    with open(filename) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPHEN in requirements:
            requirements.remove(HYPHEN)
            
    return requirements
            
setup(
    name= 'bspline_project',
    version= '0.0.1',
    author = 'abhishek',
    author_email = 's.apratap19@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)