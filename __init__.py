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

"""Module for meterstick."""

# Present unified interface in a single module.
# pylint: disable=wildcard-import
from meterstick import utils
from meterstick.metrics import *
from meterstick.operations import *
from meterstick.sql import *
try:
  from meterstick.models import *  # pylint: disable=g-import-not-at-top
except (ImportError, ModuleNotFoundError) as e:
  print(
      'WARNING: metersick.model is not imported because: "%s". '
      "It's OK if you don't fit model in Meterstick."
      % repr(e))
try:
  # Send a request to Google Analytics to track usage.
  import requests  # pylint: disable=g-import-not-at-top
  import uuid  # pylint: disable=g-import-not-at-top

  data = {
      'v': '1',  # API Version.
      'tid': 'UA-189724257-3',  # Tracking ID / Property ID.
      # Anonymous Client Identifier. Ideally, this should be a UUID that
      # is associated with particular user, device, or browser instance.
      'cid': str(uuid.uuid4()),
      't': 'event',  # Event hit type.
      'ec': 'import',  # Event category.
      'ea': 'import',  # Event action.
      'el': 'external import',  # Event label.
      'ev': 1,  # Event value, must be an integer
      'ua': '',  # User agent
  }
  response = requests.post(
      'https://www.google-analytics.com/collect', data=data)
except:  # pylint: disable=bare-except
  pass
