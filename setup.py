"""Package setup script."""

from setuptools import find_packages, setup

setup(name='rai',
      version='0.0.1',
      author='Ali Payani',
      author_email='apayani@cisco.com',
      include_package_data=True,
      packages=find_packages(),
      data_files=[],
      scripts=[],
      url='',
      license='',
      description="Responsible AI framework.",
      long_description=open('README.md').read(),
      install_requires=[
          'redis~=4.0.1',
          'sklearn~=0.0',
          'numpy~=1.20.3',
          'pandas~=1.3.4',
          'apscheduler',
          'adversarial-robustness-toolbox',
      ])
