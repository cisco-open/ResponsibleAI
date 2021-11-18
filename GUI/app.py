#!venv/bin/python
import os
import json
import pandas
import numpy as np
from flask import Flask, url_for, redirect, render_template, request, abort, jsonify, send_from_directory
import flask_admin
from flask_admin.contrib import sqla
from flask_admin import helpers as admin_helpers
from flask_admin import BaseView, expose
from wtforms import PasswordField
import redis
import math

# Create Flask application
app = Flask(__name__)
app.config.from_pyfile('config.py')

admin = flask_admin.Admin(
    app,
    'RAI',
    template_mode='bootstrap4',
)

cert_measures = pandas.read_csv(os.path.dirname(os.path.realpath(__file__)) + "\\output\\certificate_measures.csv")
cert_meta = json.load(open(os.path.dirname(os.path.realpath(__file__)) + "\\output\\certificate_metadata.json", "r"))
r = redis.Redis(host='localhost', port=6379, db=0)


# Flask views
@app.route('/')
def index():
    return redirect(url_for('admin.index'))


@app.route('/users')
def users():
    return render_template('/admin/users.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers)

@app.route('/info')
def info():
    return render_template('/admin/info.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers)


@app.route('/event')
def event():
    return render_template('/admin/event_list.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers)


@app.route('/getData/<date1>/<date2>', methods=['GET'])
def getData(date1, date2):
    date1 += "-00:00:00"
    date2 += "-00:00:00"
    data_test = r.lrange('metric_values', 0, -1)
    res = []
    for i in range(len(data_test)):
        item = json.loads(data_test[i])
        if date1 <= item['date'] <= date2:
            res.append(item)
    return json.dumps(res)


@app.route('/getMetricList', methods=['GET'])
def getMetricList():
    data_test = r.get('metric_info')
    data = json.loads(data_test)
    # print("DATA: ", data)
    result = {}

    for metric in data:
        for tag in data[metric]["tags"]:
            if tag.lower() in result:
                result[tag.lower()].append(metric)
            else:
                result[tag.lower()] = []
                result[tag.lower()].append(metric)
    return result


@app.route('/getMetricInfo', methods=['GET'])
def getMetricInfo():
    return json.loads(r.get('metric_info'))


@app.route('/getCertification/<date1>/<date2>', methods=['GET'])
def getCertification(date1, date2): # NOT REAL DATA YET.
    date1 += "-00:00:00"
    date2 += "-00:00:00"
    mask = (cert_measures['date'] >= date1) & (cert_measures['date'] <= date2)
    return cert_measures.loc[mask].to_json(orient='records')  # serialize and use JSON headers


@app.route('/getCertificationMeta', methods=['GET'])
def getCertificationMeta():
    return cert_meta


@app.route('/viewClass/<category>')
def renderClassTemplate(category):
    functional = category.replace(' ', '_').lower()
    return render_template('/admin/view_class.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           Category=category,
                           Functional=functional)


@app.route('/viewAll')
def renderAllMetrics():
    return render_template('/admin/view_all.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers)


@app.route('/learnMore/<metric>')
def learnMore(metric):
    data_test = r.get('metric_info')
    metric_info = json.loads(data_test)
    return render_template('/admin/metric_info.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           metric_display_name=metric_info[metric]['display_name'],
                           metric_range=metric_info[metric]['range'],
                           metric_has_range=metric_info[metric]['has_range'],
                           metric_explanation=metric_info[metric]['explanation'],
                           metric_type=metric_info[metric]['type'],
                           metric_tags=metric_info[metric]['tags'],
                           metric_hidden_name=metric
                           )


if __name__ == '__main__':
    # Build a sample db on the fly, if one does not exist yet.
    app_dir = os.path.realpath(os.path.dirname(__file__))
    # Start app
    app.run(debug=True)
