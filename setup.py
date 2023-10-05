# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


"""Package setup script."""

from setuptools import find_packages, setup

setup(name='openrai',
      version='0.0.6',
      author='Ali Payani',
      author_email='apayani@cisco.com',
      include_package_data=True,
      packages=find_packages(),
      package_data={
          '': ['*.json'],  # Include all JSON files in the package
          'logos': ['*.png'],
          'js': ['*.js'],
      },
      scripts=[],
      url='',
      license='',
      description="Responsible AI framework.",
      long_description=open('README.rst').read(),
      extras_require={
          'dashboard': [
              'dash~=2.5.0',
              'dash_bootstrap_components~=1.1.0',
              'dash-daq~=0.5.0',
              'flask==2.2.4'
          ],
      },
      install_requires=[
          'aif360~=0.4.0',
          'fairlearn~=0.7.0',
          'sqlitedict~=2.1.0',
          'scikit-learn~=1.0.2',
          'numpy~=1.23.5',
          'nltk~=3.7',
          'pandas~=1.3.5',
          'apscheduler',
          'adversarial-robustness-toolbox',
          'python-dotenv==1.0.0',
          'tensorflow~=2.9.1',
          'torch~=1.11.0',
          'torchvision~=0.12.0',
          'torchmetrics~=0.9.3'
      ])
