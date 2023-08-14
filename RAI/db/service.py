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
import json
import subprocess
import threading
import RAI
import os
import numpy as np

from sqlitedict import SqliteDict

from RAI.Analysis import AnalysisManager
from RAI.utils.utils import MonitoringTimer

logger = logging.getLogger(__name__)

PROJECTS = 'PROJECTS'


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


__all__ = ['RaiDB']


class RaiDB:
    """
    Service used to provide sqlite functionalities. Allows for adding measurements, deleting measurements,
    exporting metadata.
    """

    def __init__(self, ai_system: RAI.AISystem = None) -> None:
        self.ai_system = ai_system
        self._threads = []
        self.analysis_manager = AnalysisManager()
        self.projects_db = self._get_db()
        self.db = self._get_db(ai_system.name)
        self._init_available_analysis_timer()
        self._init_analysis_timer()

    def _get_db(self, name=None):
        folder = os.getenv('DATABASE_FOLDER')
        if name is None:
            name = 'rai_internal'
        db = f'{folder}/{name}.sqlite'
        return SqliteDict(db, encode=json.dumps, decode=json.loads, autocommit=True)

    def _init_available_analysis_timer(self):
        def sub_handler():
            try:
                keep_alive = True
                request = self.db.get('available_analysis')
                if request and request['seen'] is False:
                    available = self.analysis_manager.get_available_analysis(self.ai_system, request['data'])
                    self.db['available_analysis_response'] = {
                        'data': json.dumps(available),
                        'seen': False
                    }
                    self.db['available_analysis'] = {
                        'data': request['data'],
                        'seen': True
                    }
                    keep_alive = False
            except Exception as e:
                print(f'Analysis timer failed: {e}')
            finally:
                return keep_alive
        MonitoringTimer(1, sub_handler)

    def _init_analysis_timer(self):
        def sub_handler():
            try:
                keep_alive = True
                request = self.db.get('start_analysis')
                if request and request['seen'] is False:
                    dataset = request['data']
                    analysis = request['analysis']
                    self.db['start_analysis'] = {
                        'data': dataset,
                        'analysis': analysis,
                        'seen': True
                    }
                    if request['analysis'] in self.analysis_manager.get_available_analysis(self.ai_system, dataset):
                        connection = self.get_progress_update_lambda(analysis)
                        x = threading.Thread(target=self._run_analysis_thread, args=(dataset, analysis, connection))
                        x.start()
            except Exception as e:
                print(f'Analysis timer failed: {e}')
            finally:
                return keep_alive
        MonitoringTimer(1, sub_handler)

    def get_progress_update_lambda(self, analysis):
        return lambda progress: self.analysis_progress_update(analysis, progress)

    def analysis_progress_update(self, analysis: str, progress):
        current = self.db.get('analysis_update', {})
        current[analysis] = {'progress': progress, 'update': True}
        self.db['analysis_update'] = current

    def Disconnect(self):
        self.projects_db.close()
        try:
            self.db.close()
        except Exception:
            pass

    def _run_analysis_thread(self, dataset, analysis, connection):
        result = self.analysis_manager.run_analysis(self.ai_system, dataset, analysis, connection=connection)
        # encoded_res = pickle.dumps(result[analysis].to_html())
        encoded_res = json.dumps(self._jsonify_analysis(result[analysis].to_html()))
        current = self.db.get('analysis_response', {})
        current[analysis] = {
            'val': encoded_res,
            'seen': False,
        }
        self.db['analysis_response'] = current

    def _jsonify_analysis(self, analysis):
        if "dash" in str(type(analysis)) or "plotly" in str(type(analysis)):
            analysis = analysis.to_plotly_json()
        if isinstance(analysis, dict):
            for key in analysis:
                analysis[key] = self._jsonify_analysis(analysis[key])
        elif isinstance(analysis, list):
            for i in range(len(analysis)):
                analysis[i] = self._jsonify_analysis(analysis[i])
        elif isinstance(analysis, np.ndarray):
            analysis = analysis.tolist()
        return analysis

    def reset_data(self, export_metadata: bool = True) -> None:
        to_delete = ["metric_values", "model_info", "metric_info", "metric", "certificate_metadata",
                     "certificate_values", "certificate"]
        for key in to_delete:
            self.db.pop(key, None)
        if export_metadata:
            self.export_metadata()

    def export_metadata(self) -> None:
        metric_info = self.ai_system.get_metric_info()
        certificate_info = self.ai_system.get_certificate_info()

        if certificate_info:
            metric_info['Certificates'] = {'meta': {'display_name': 'Certificates'}}
            for certificate in certificate_info:
                display_name = certificate_info[certificate]['display_name'].title()
                metric_info['Certificates'][display_name] = {'display_name': display_name}

        if self.ai_system.custom_metrics or self.ai_system.custom_functions:
            metric_info['Custom'] = {'meta': {'display_name': 'Custom'}}
            for metric in self.ai_system.custom_metrics:
                metric_info['Custom'][metric] = {'display_name': metric}
            if self.ai_system.custom_functions:
                for func in self.ai_system.custom_functions:
                    metric_info['Custom'][func.__name__] = {'display_name': func.__name__}

        project_info = self.ai_system.get_project_info()

        projects = self.projects_db.get(PROJECTS, [])
        projects.append(self.ai_system.name)
        self.projects_db[PROJECTS] = list(set(projects))
        self.db['metric_info'] = json.dumps(metric_info)
        self.db['certificate_info'] = json.dumps(certificate_info)
        self.db['project_info'] = json.dumps(project_info)

    def export_visualizations(self, model_interpretation_dataset: str, data_visualization_dataset: str):
        data_visualizations = ["DataVisualization"]
        interpretations = ["GradCamAnalysis"]
        for analysis in data_visualizations:
            if data_visualization_dataset:
                connection = self.get_progress_update_lambda(analysis)
                result = self.analysis_manager.run_analysis(self.ai_system, data_visualization_dataset, analysis,
                                                            connection)
                encoded_res = json.dumps(self._jsonify_analysis(result[analysis].to_html()))
                current = self.db.get('analysis_response', {})
                current[analysis] = {
                    'val': encoded_res,
                    'seen': False,
                }
                self.db['analysis_response'] = current
        for analysis in interpretations:
            if model_interpretation_dataset:
                connection = self.get_progress_update_lambda(analysis)
                result = self.analysis_manager.run_analysis(self.ai_system, model_interpretation_dataset, analysis,
                                                            connection)
                if analysis in result:
                    encoded_res = json.dumps(self._jsonify_analysis(result[analysis].to_html()))
                    current = self.db.get('analysis_response', {})
                    current[analysis] = {
                        'val': encoded_res,
                        'seen': False,
                    }
                    self.db['analysis_response'] = current

    def add_measurement(self) -> None:
        certificates = self.ai_system.get_certificate_values()
        metrics = self.ai_system.get_metric_values()
        print("Sharing: ", self.ai_system.name)
        self.db['certificate_values'] = json.dumps(certificates)
        # Leaving this for now.
        # TODO: Set up standardized to json for all metrics.
        '''
        # print("METRICS: ", metrics)
        for dataset in metrics:
            for group in metrics[dataset]:
                for m in metrics[dataset][group]:
                    print(m, "\n")

        print("testing json dumps: \n")
        for dataset in metrics:
            for group in metrics[dataset]:
                for m in metrics[dataset][group]:
                    if "moment" in m:
                        continue
                    print(m, "\n")
                    print(metrics[dataset][group][m])
                    print(json.dumps(metrics[dataset][group][m]))
        '''
        metric_values = json.loads(self.db.get('metric_values', '[]'))
        metric_values.append(metrics)  # True
        self.db['metric_values'] = json.dumps(metric_values)

    def viewGUI(self):
        gui_launcher = threading.Thread(target=self._view_gui_thread, args=[])
        gui_launcher.start()

    def _view_gui_thread(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("../../Dashboard")
        file = os.path.abspath("main.py")
        subprocess.call("start /wait python " + file, shell=True)
        print("GUI can be viewed in new terminal")
