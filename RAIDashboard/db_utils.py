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


import json
import logging
import os
from collections import defaultdict
import numpy as np
import traceback
import dash_bootstrap_components as dbc
from dash import html
from sqlitedict import SqliteDict

from .timer import DashboardTimer

logger = logging.getLogger(__name__)


PROJECTS = 'PROJECTS'


class DBUtils(object):
    def __init__(self):
        self._threads = []
        self.values = {}
        self.info = {}
        self._precision = 3
        self._maxlen = 100
        self._initialized = False
        self._subscribers = defaultdict(bool)
        self._current_project = {}  # Contains certificates, metrics, info, project info
        self._current_project_name = None  # Used with sqlite to get current project
        self._last_current_project_name = None
        self._load_analysis_storage = {}
        self._projects = []
        self._metrics_config = {}
        self.db = None
        self.config_db = None
        self.projects_db = self._get_db()
        self._init_monitoring()
        self._update_projects()

    def has_update(self, channel, reset=True):
        val = self._subscribers[channel]
        if reset:
            self.reset_channel(channel)
        return val

    def has_analysis_update(self, analysis, reset=True):
        current = self.db.get('analysis_update', {})
        if current and current.get(analysis) and current.get(analysis, {}).get('update') is True:
            current[analysis]['update'] = False
            self.db['analysis_update'] = current
            return True
        return False

    def reset_channel(self, channel):
        self._subscribers[channel] = False

    def _get_db(self, name=None):
        folder = os.getenv('DATABASE_FOLDER')
        if name is None:
            name = 'rai_internal'
        db = f'{folder}/{name}.sqlite'
        return SqliteDict(db, encode=json.dumps, decode=json.loads, autocommit=True)

    def _init_monitoring(self):
        def sub_handler():
            self._update_projects()
            if self._current_project_name:
                self._update_info()
                self._update_values()
            relevant = ["metric_detail", "metric_graph", "certificate"]
            for item in relevant:
                self._subscribers[item] = True
        DashboardTimer(5, sub_handler)

    def _progress_to_html(self, progress):
        return html.Div(dbc.Progress(value=progress, label=str(progress) + "%"))

    def request_start_analysis(self, analysis):
        def handler_analysis():
            try:
                keep_alive = True
                request = self.db.get('analysis_response')
                if request and request.get(analysis):
                    val = request[analysis]['val']
                    self.db['analysis_response'][analysis] = {
                        'val': val,
                        'seen': True,
                    }
                    res = self.db.get('analysis_update')
                    if analysis in res:
                        res[analysis]['update'] = True
                    else:
                        res[analysis] = {'update': True, 'progress': 100}
                    self.db['analysis_update'] = res
                    self.set_analysis_progress(self._current_project_name, analysis, html.Div(json.loads(val)))
                    self._subscribers["analysis_update|" + self._current_project_name + "|" + analysis] = True
                    keep_alive = False
            except Exception as e:
                print(f'handler_analysis timer failed: {e}', traceback.print_exc())
            finally:
                return keep_alive

        def handler_progress():
            try:
                keep_alive = True
                request = self.db.get('analysis_update')
                if request and request.get(analysis):
                    progress = request[analysis]['progress']
                    if progress != 100:
                        self.set_analysis_progress(self._current_project_name, analysis, self._progress_to_html(progress))
                    self._subscribers["analysis_update|" + self._current_project_name + "|" + analysis] = True
                    keep_alive = False
            except Exception as e:
                print(f'handler_progress timer failed: {e}')
            finally:
                return keep_alive
        self.set_analysis_progress(self._current_project_name, analysis, self._progress_to_html(0))
        self.db['analysis_response'] = {}
        self.db['analysis_update'] = {}
        self.db['start_analysis'] = {
            'data': self.get_current_dataset(),
            'analysis': analysis,
            'seen': False
        }
        DashboardTimer(1, handler_analysis)
        DashboardTimer(1, handler_progress)

    def request_available_analysis(self):
        def sub_handler():
            try:
                keep_alive = True
                request = self.db.get('available_analysis_response')
                if request and request['seen'] is False:
                    self.db['available_analysis_response'] = {
                        'data': request['data'],
                        'seen': True
                    }
                    self.set_available_analysis(json.loads(request['data']))
                    keep_alive = False
            except Exception as e:
                print(f'Analysis timer failed: {e}')
            finally:
                return keep_alive
        print("requesting available analysis")
        self.db['available_analysis'] = {
            'data': self.get_current_dataset(),
            'seen': False
        }
        DashboardTimer(0.5, sub_handler)

    def reformat(self, precision):
        self._precision = precision
        self._current_project = self._reformat_data(self._current_project)

    def _reload(self):
        self._update_projects()
        self._update_info()
        self._update_values()

    def close(self):
        self.db.close()
        self.projects_db.close()

    def get_project_info(self):
        return self._current_project.get("project_info", {})

    def get_metric_info(self):
        return self._current_project.get("metric_info", {})

    def get_certificate_info(self):
        return self._current_project.get("certificate_info", {})

    def get_certificate_values(self):
        return self._current_project.get("certificate_values", {})

    def get_metric_values(self):
        return self._current_project.get("metric_values", '[]')

    def get_current_dataset(self):
        return self._current_project.get("current_dataset", None)

    def get_data_summary(self):
        return self._current_project.get("data_summary", {})

    def get_model_interpretation(self):
        return self._current_project.get("model_interpretation", {})

    def get_available_analysis(self):
        return self._current_project.get("available_analysis", [])

    def get_analysis(self, analysis_name):
        analysis = None
        if analysis_name is not None:
            analysis = self._load_analysis_storage.get(self._current_project_name, {}).get(analysis_name, None)
            if analysis is None:
                analysis = self.db.get("analysis_response", {}).get(analysis, {}).get('val')
                if analysis is not None:
                    analysis = json.loads(analysis)
        return analysis

    def set_current_project(self, project_name):
        project_name = project_name
        logger.info(f"changing current project from {self._current_project_name} to {project_name}")
        if self._current_project_name == project_name:
            return
        self._current_project_name = project_name
        if self.config_db:
            self.config_db.close()
        self.config_db = self._get_db(f'{project_name}_config')
        self._metrics_config = self.config_db.get('options', {})
        if self.db:
            self.db.close()
        self.db = self._get_db(project_name)
        self._current_project = {}
        self._update_info()
        self._update_values()
        self.set_data_summary()
        self.set_model_interpretation()
        self._current_project["analysis"] = {}
        self._current_project = self._reformat_data(self._current_project)

    def set_current_dataset(self, dataset):
        self._current_project["current_dataset"] = dataset

    def set_data_summary(self):
        print("Current proj name: ", self._current_project_name)
        summary = self.db.get("data_summary", '{}')
        self._current_project["data_summary"] = json.loads(summary) if summary is not None else {}

    def set_model_interpretation(self):
        interpretation = self.db.get("model_interpretation", '{}')
        self._current_project["model_interpretation"] = json.loads(interpretation) if interpretation is not None else {}

    def set_available_analysis(self, available):
        self._current_project["available_analysis"] = available

    def set_analysis_progress(self, project_name, analysis_name, report):
        if project_name not in self._load_analysis_storage:
            self._load_analysis_storage[project_name] = {}
        self._load_analysis_storage[project_name][analysis_name] = report

    def _update_projects(self):
        self._projects = self.projects_db.get(PROJECTS, [])

    def get_projects_list(self):
        return self._projects

    def get_sorted_projects_list(self):
        return sorted(self._projects) if self._projects else []

    def get_dataset_list(self):
        return self._current_project.get("dataset_values", [])

    def _update_info(self):
        self.info = {}
        if self._last_current_project_name != self._current_project_name:
            print("current project name: ", self._current_project_name)
            self._last_current_project_name = self._current_project_name
        self._current_project["project_info"] = \
            json.loads(self.db.get('project_info', '{}'))
        self._current_project["certificate_info"] = \
            json.loads(self.db.get('certificate_info', '{}'))
        self._current_project["metric_info"] = \
            json.loads(self.db.get('metric_info', '{}'))

    def _update_values(self):
        self.values = {}
        self._current_project["metric_values"] = json.loads(
            self.db.get('metric_values')
        )
        self._current_project["certificate_values"] = json.loads(
            self.db.get('certificate_values')
        )

        self._current_project["dataset_values"] = []
        for item in self._current_project["metric_values"]:
            for val in item:
                if val not in self._current_project["dataset_values"]:
                    self._current_project["dataset_values"].append(val)

    def _reformat_data(self, x):
        if type(x) is float:
            return np.round(x, self._precision)
        elif type(x) is list:
            return [self._reformat_data(i) for i in x]
        elif isinstance(x, dict):
            return {k: self._reformat_data(v) for k, v in x.items()}
        else:
            return x

    def __del__(self):
        pass
    # self.close()
