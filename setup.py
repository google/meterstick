#!/usr/bin/python
#
# Copyright 2016 Google Inc.
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
"""This is the setup file for the project. The standard setup rules apply:
   python setup.py build
   sudo python setup.py install
"""

from setuptools import setup, find_packages

setup(
    name=u'meterstick',
    version=u"0.1.0",
    description=u'A concise grammar for specifying routine data analyses.',
    license=u'Apache License, Version 2.0',
    url=u'https://github.com/google/meterstick',
    maintainer=u'Meterstick development team',
    maintainer_email=u'meterstick@googlegroups.com',
    classifiers=[
        u'Development Status :: 3 - Alpha',
        u'Intended Audience :: Developers',
        u'Intended Audience :: Education',
        u'Intended Audience :: Science/Research',
        u'Topic :: Scientific/Engineering :: Information Analysis',
        u'Topic :: Scientific/Engineering :: Mathematics',
        u'License :: OSI Approved :: Apache Software License',
        u'Programming Language :: Python',
    ],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas'
    ]
)
