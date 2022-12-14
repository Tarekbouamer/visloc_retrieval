from setuptools import setup, find_packages
from pathlib import Path

description = ['Long Term Visual Localization']

root = Path(__file__).parent
with open(str(root / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

with open(str(root / '_version.py'), 'r') as f:
    version = eval(f.read().split('__version__ = ')[1].split()[0])

with open(str(root / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')
    
setup(
    name='retrieval',
    version=version,
    packages=find_packages(),
    python_requires='>=3.7',
    author='Tarek BOUAMER',
    author_email="tarekbouamer1788@gmail.com",
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/Tarekbouamer',
    
    install_requires=['opencv-contrib-python==4.6.0.66',
                      'opencv-python==4.6.0.66',
                      'h5py',
                      'scipy',
                      'matplotlib',
                      'tqdm',
                      'tensorboardX==2.5.1',
                      'gdown==4.5.4',
                      'timm==0.6.12',
                      ],
)
