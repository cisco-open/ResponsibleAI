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
          'sqlitedict~=2.1.0',
          'scikit-learn~=1.0.2',
          'numpy~=1.23.5',
          'pandas~=1.3.5',
          'apscheduler',
          'adversarial-robustness-toolbox',
      ])
