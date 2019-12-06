from distutils.core import setup
setup(
  name='woe-monotonic-binning',
  packages=['woe_monotonic_binning'],
  version='0.1.3',
  license='MIT',
  description='Optimal binning algorithm and function to apply on a pandas DataFrame',
  author='PEDRO HENRIQUE BAUMGRATZ MEIRELLES',
  author_email='pedrohbm@poli.ufrj.br',
  url='https://github.com/PedroHBM/woe-monotonic-binning',
  download_url='https://github.com/PedroHBM/woe-monotonic-binning/archive/v0.1.2-alpha.tar.gz',
  keywords=['WOE', 'BINNING', 'IV', 'LOGISTIC', 'REGRESSION'],
  install_requires=[            
          'pandas',
          'numpy',
          'scipy',
          'tqdm',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)