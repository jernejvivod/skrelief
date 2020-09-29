from setuptools import setup

setup(name='skrelief',
      version='0.1',
      description='scikit-learn compatible interface to a number of Relief-based feature selection algorithms',
      url='https://github.com/jernejvivod/skrelief',
      author='Jernej Vivod',
      author_email='vivod.jernej@gmail.com',
      license='MIT',
      packages=['skrelief'],
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'julia',
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Utilities'
      ],
      keywords=['data mining', 'feature selection', 'feature evaluation', 'machine learning', 'data analysis', 'artificial intelligence', 'data science'],
      include_package_data=True,
      zip_safe=False)
