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


import logging
import dash
import dash_bootstrap_components as dbc
import RAIDashboard.config  # noqa: F401
from .db_utils import DBUtils

logger = logging.getLogger(__name__)
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]

dbUtils = DBUtils()

app = dash.Dash(external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
