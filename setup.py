# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup file for the meterstick package."""

import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE/"README.md").read_text()

setup(
    name="meterstick",
    version="1.1.0",
    description="A grammar of data analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    URL="https://github.com/google/meterstick",
    author="Xunmo Yang, Dennis Sun, Taylor Pospisil",
    authoremail="meterstick-external@google.com",
    license="Apache License 2.0",
    packages=["meterstick"],
    installrequires=["six", "numpy", "scipy", "pandas"],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 5 - Production/Stable",
        "Framework :: IPython",
        "Framework :: Jupyter",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Programming Language :: SQL",
    ])
