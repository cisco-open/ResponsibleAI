#!venv/bin/python
import os
import json
import pandas
from flask import Flask, url_for, redirect, render_template, request, abort, jsonify, send_from_directory
import flask_admin
from flask_admin import helpers as admin_helpers
import redis
import datetime

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

cache = {'metric_info': json.loads(r.get('metric_info')), 'metric_values': r.lrange('metric_values', 0, -1)}


def get_dates():
    data_test = r.lrange('metric_values', 0, -1)
    date_start = "2020-10-01"
    if len(data_test) >= 1:
        date_start = json.loads(data_test[0])['metadata > date'][:10]
    now = datetime.datetime.now()
    date_end = "{:02d}".format(now.year) + "-" + "{:02d}".format(now.month) + "-" + "{:02d}".format(now.day)
    print("START: " + date_start + " END: " + date_end)
    return date_start, date_end


# Flask views
@app.route('/')
def index():
    return redirect(url_for('admin.index'))


'''
@app.route('/users')
def users():
    return render_template('/admin/users.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers)
'''


@app.route('/info')
def info():
    model_info = r.get('model_info')
    data = json.loads(model_info)
    print("DATA: " + str(data))
    name = data['id']
    description = data['description']
    task_type = data['task_type']
    prot_attr = ["None"]
    model_type = data['model']
    features = data['features']

    if 'fairness' in data['configuration'] and 'protected_attributes' in data['configuration']['fairness']:
        prot_attr = data['configuration']['fairness']['protected_attributes']


    return render_template('/admin/info.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           name=name,
                           description=description,
                           task_type=task_type,
                           protected_attributes=prot_attr,
                           model_type=model_type,
                           features=features)


@app.route('/event')
def event():
    return render_template('/admin/event_list.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers)


@app.route('/getData/<date1>/<date2>', methods=['GET'])
def getData(date1, date2):
    date1 += " 00:00:00"
    date2 += " 99:99:99"
    data_test = r.lrange('metric_values', 0, -1)
    # data_test = cache['metric_values']
    res = []
    for i in range(len(data_test)):
        item = json.loads(data_test[i])
        print("DATE1: ", date1, ", DATE2:, ", date2, " DATE: ", item['metadata > date'])

        if date1 <= item['metadata > date'] <= date2:
            res.append(item)
    return json.dumps(res)


@app.route('/getMetricList', methods=['GET'])
def getMetricList():
    data_test = r.get('metric_info')
    data = json.loads(data_test)
    # data = cache['metric_info']

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
    # return cache['metric_info']


@app.route('/getModelInfo', methods=['GET'])
def getModelInfo():
    return json.loads(r.get('model_info'))
    # return cache['metric_info']


@app.route('/getCertification/<date1>/<date2>', methods=['GET'])
def getCertification(date1, date2):  # NOT REAL DATA YET.
    date1 += " 00:00:00"
    date2 += " 99:99:99"
    mask = (cert_measures['date'] >= date1) & (cert_measures['date'] <= date2)
    return cert_measures.loc[mask].to_json(orient='records')  # serialize and use JSON headers


@app.route('/getCertificationMeta', methods=['GET'])
def getCertificationMeta():
    return cert_meta


@app.route('/viewClass/<category>')
def renderClassTemplate(category):
    functional = category.replace(' ', '_').lower()
    start_date, end_date = get_dates()

    return render_template('/admin/view_class.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           Category=category,
                           Functional=functional,
                           start_date=start_date,
                           end_date=end_date)


@app.route('/viewAll')
def renderAllMetrics():
    start_date, end_date = get_dates()
    return render_template('/admin/view_all.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           start_date=start_date,
                           end_date=end_date)


@app.route('/learnMore/<metric>')
def learnMore(metric):
    data_test = r.get('metric_info')
    metric_info = json.loads(data_test)
    start_date, end_date = get_dates()
    # metric_info = cache['metric_info']
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
                           metric_hidden_name=metric,
                           start_date=start_date,
                           end_date=end_date
                           )


if __name__ == '__main__':
    # Build a sample db on the fly, if one does not exist yet.
    app_dir = os.path.realpath(os.path.dirname(__file__))
    # Start app
    app.run(debug=True)
