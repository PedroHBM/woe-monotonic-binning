from distutils.core import setup
setup(
  name='woe-monotonic-binning',
  packages=['woe-monotonic-binning'],
  version='0.1',
  license='MIT',
  description='Optimal binning algorithm and function to apply on a pandas DataFrame',
  author='PEDRO HENRIQUE BAUMGRATZ MEIRELLES',
  author_email='pedrohbm@poli.ufrj.br',
  url='https://github.com/PedroHBM/woe-monotonic-binning',
  download_url='https://github.com/user/reponame/archive/v_01.tar.gz',
  keywords=['WOE', 'BINNING', 'IV', 'LOGISTIC', 'REGRESSION'],
  install_requires=[            # I get to this in a second
          'pandas',
          'numpy',
          'scipy',
          'tqdm',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Data Scientists',      
    'Topic :: Data Science :: Analytics :: Statistics',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)