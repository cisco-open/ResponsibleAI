#!venv/bin/python
import os
import json
import pandas
from flask import Flask, url_for, redirect, render_template, request, abort, jsonify, send_from_directory
import flask_admin
from flask_admin import helpers as admin_helpers
import redis
import sys
import datetime
import threading


# if len(sys.argv) <= 1:
#    raise Exception("Please Enter Model Name")

# if len(sys.argv) > 2:
#     raise Exception("Please Enter Model Name with no spaces")

model_name = "cisco_german_fairness"  # sys.argv[1]
if len(sys.argv) == 2:
    model_name = sys.argv[1]


metric_access_stats = threading.Lock()
cert_access_stats = threading.Lock()
metric_update_required = False
cert_update_required = False


# Create Flask application
app = Flask(__name__)
app.config.from_pyfile('config.py')

admin = flask_admin.Admin(
    app,
    'RAI',
    template_mode='bootstrap4',
)

r = redis.Redis(host='localhost', port=6379, db=0)
metric_sub = r.pubsub()
metric_sub.psubscribe(model_name + '|certificate')
cert_sub = r.pubsub()
cert_sub.psubscribe(model_name + '|certificate')



def get_dates():
    data_test = r.lrange(model_name + '|metric_values', 0, -1)
    clear_streams()
    date_start = "2020-10-01"
    if len(data_test) >= 1:
        date_start = json.loads(data_test[0])['metadata > date'][:10]
    now = datetime.datetime.now()
    date_end = "{:02d}".format(now.year) + "-" + "{:02d}".format(now.month) + "-" + "{:02d}".format(now.day)
    return date_start, date_end


def get_certificate_dates():
    data_test = r.lrange(model_name + '|certificate_values', 0, -1)
    clear_streams()
    date_start = "2020-10-01"
    if len(data_test) >= 1:
        date_start = json.loads(data_test[0])['metadata > date']["value"][:10]
    now = datetime.datetime.now()
    date_end = "{:02d}".format(now.year) + "-" + "{:02d}".format(now.month) + "-" + "{:02d}".format(now.day)
    return date_start, date_end


@app.route('/')
def index():
    model_info = json.loads(r.get(model_name + '|model_info'))
    start_date, end_date = get_certificate_dates()
    # GET FAILED PER CATEGORY.
    failed = []

    data_test = r.lrange(model_name + '|certificate_values', 0, -1)
    metadata = json.loads(r.get(model_name + '|certificate_metadata'))

    # data_test = cache['metric_values']
    clear_streams()
    res = []
    item = json.loads(data_test[-1])
    scores = {"fairness": [0, 0], "explainability": [0, 0], "performance": [0, 0], "robustness": [0, 0]}
    temp_dict = {}
    for value in item:
        if metadata[value]["tags"][0] != "metadata":
            if metadata[value]["tags"][0] in scores:
                scores[metadata[value]["tags"][0]][1] += 1
                if item[value]["value"]:
                    scores[metadata[value]["tags"][0]][0] += 1
                else:
                    failed.append({"name":metadata[value]['display_name'], "category": metadata[value]["tags"][0]})
        temp_dict['metadata'] = {"date": item['metadata > date'], "description": item['metadata > description'], "scores": scores}
        res.append(temp_dict)

    return render_template('/admin/index.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           failed=failed,
                           h=admin_helpers,
                           end_date=end_date,
                           start_date=start_date,
                           model_name=model_info["display_name"])
    # return redirect(url_for('admin.index'))


@app.route('/viewCertificates/<name>')
def viewCertificates(name):
    name = name.lower()
    cert_info = r.lrange(model_name + '|certificate_values', 0, -1)
    model_info = json.loads(r.get(model_name + '|model_info'))
    metadata = json.loads(r.get(model_name + '|certificate_metadata'))
    clear_streams()
    data = json.loads(cert_info[-1])
    date = data['metadata > date']
    print("METADATA: " + str(metadata))
    result1 = []
    result2 = []
    print("DATA: ")
    print(str(data))
    for item in data:
        if metadata[item]["tags"][0] == name:
            print("ADDING: " + str(item))
            dict_item = {}
            if data[item]["value"]:
                dict_item['value'] = "Passed"
                dict_item['score_class'] = 'fa-check green'
            else:
                dict_item['value'] = "Failed"
                dict_item['score_class'] = 'fa-times red'
            dict_item['explanation'] = data[item]["explanation"]
            dict_item['name'] = metadata[item]['display_name']
            dict_item['backend_name'] = item
            dict_item["measurement_description"] = data["metadata > description"]
            print(metadata[item]["level"])
            if '1' in metadata[item]['level'] or metadata[item]['level'] == 1:
                result1.append(dict_item)
            elif '2' in metadata[item]['level'] or metadata[item]['level'] == 2:
                result2.append(dict_item)
    return render_template('/admin/view_certificates.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           certificate_name=name,
                           model_name=model_info["display_name"],
                           features1=result1,
                           features2=result2,
                           date=date["value"])


def getConditions(name, metadata, cert_values):
    result = []
    metric_info = json.loads(r.get(model_name + '|metric_info'))
    print("Function starting")
    cert_values = json.loads(cert_values[-1])
    for i, item in enumerate(metadata[name]['condition']['terms']):
        if item[0][0] == '@':
            print("Certificate")
            new_val = str(cert_values[name]["term_values"][i])
            display_name = str(metadata[(item[0])[1:]]["name"])
            result.append({"name": display_name + " " + str(item[1]) + " " + str(item[2]), "result": new_val})
        elif item[0][0] == '&':
            print("Metric")
            new_val = str(cert_values[name]["term_values"][i])
            display_name = str(metric_info[(item[0])[1:]]["display_name"])
            result.append({"name": display_name + " " + str(item[1]) + " " + str(item[2]), "result": new_val})
    return result



@app.route('/viewCertificate/<name>')
def viewCertificate(name):
    cert_info = r.lrange(model_name + '|certificate_values', 0, -1)
    metadata = json.loads(r.get(model_name + '|certificate_metadata'))
    model_info = json.loads(r.get(model_name + '|model_info'))

    conditions = getConditions(name, metadata, cert_info)
    print(str(conditions))

    clear_streams()
    result = []
    for i in range(len(cert_info)):
        dict_item = {}
        data = json.loads(cert_info[i])
        dict_item['date'] = data['metadata > date']["value"]
        if data[name]["value"]:
            dict_item['value'] = "Passed"
            dict_item['score_class'] = 'fa-check green'
        else:
            dict_item['value'] = "Failed"
            dict_item['score_class'] = 'fa-times red'
        dict_item['explanation'] = data[name]['explanation']
        dict_item['description'] = metadata[name]['description']
        result.append(dict_item)
    return render_template('/admin/view_certificate.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           conditions=conditions,
                           union=metadata[name]['condition']['op'],
                           model_name=model_info["display_name"],
                           certificate_name=metadata[name]["display_name"],
                           features=result)



@app.route('/viewAllCertificates')
def viewAllCertificates():
    cert_info = r.lrange(model_name + '|certificate_values', 0, -1)
    model_info = json.loads(r.get(model_name + '|model_info'))
    metadata = json.loads(r.get(model_name + '|certificate_metadata'))
    clear_streams()
    data = json.loads(cert_info[-1])
    date = data['metadata > date']
    result1 = []
    result2 = []
    for item in data:
        if 'metadata' not in metadata[item]["tags"]:
            dict_item = {}
            if data[item]["value"]:
                dict_item['value'] = "Passed"
                dict_item['score_class'] = 'fa-check green'
            else:
                dict_item['value'] = "Failed"
                dict_item['score_class'] = 'fa-times red'
            dict_item['explanation'] = data[item]["explanation"]
            dict_item['name'] = metadata[item]['display_name']
            dict_item['category'] = metadata[item]['tags'][0]
            dict_item['backend_name'] = item
            dict_item["measurement_description"] = data["metadata > description"]
            print(metadata[item]["level"])
            if '1' in metadata[item]['level'] or metadata[item]['level'] == 1:
                result1.append(dict_item)
            elif '2' in metadata[item]['level'] or metadata[item]['level'] == 2:
                result2.append(dict_item)
    return render_template('/admin/view_certificates.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           model_name=model_info["display_name"],
                           features1=result1,
                           features2=result2,
                           date=date["value"])




@app.route('/info')
def info():
    model_info = r.get(model_name + '|model_info')
    clear_streams()
    data = json.loads(model_info)
    name = data['id']
    description = data['description']
    task_type = data['task_type']
    prot_attr = ["None"]
    model_type = data['model']
    features = data['features']
    model_display_name = data["display_name"]

    if 'fairness' in data['configuration'] and 'protected_attributes' in data['configuration']['fairness']:
        prot_attr = data['configuration']['fairness']['protected_attributes']


    return render_template('/admin/info.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           name=name,
                           model_name=model_display_name,
                           description=description,
                           task_type=task_type,
                           protected_attributes=prot_attr,
                           model_type=model_type,
                           features=features)


@app.route('/event')
def event():
    result = []
    data_test = r.lrange(model_name + '|metric_values', 0, -1)
    clear_streams()
    for i in range(len(data_test)):
        item = json.loads(data_test[i])
        new_dict = {"date": item['metadata > date'], "event": "Measurement Added", "description": item['metadata > description']}
        result.append(new_dict)
    model_info = json.loads(r.get(model_name + '|model_info'))
    return render_template('/admin/event_list.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           model_name=model_info["display_name"],
                           h=admin_helpers,
                           events=result)


@app.route('/getData/<date1>/<date2>', methods=['GET'])
def getData(date1, date2):
    date1 += " 00:00:00"
    date2 += " 99:99:99"
    data_test = r.lrange(model_name + '|metric_values', 0, -1)
    clear_streams()
    # data_test = cache['metric_values']
    res = []
    for i in range(len(data_test)):
        item = json.loads(data_test[i])
        if date1 <= item['metadata > date'] <= date2:
            res.append(item)
    return json.dumps(res)


@app.route('/getMetricList', methods=['GET'])
def getMetricList():
    data_test = r.get(model_name + '|metric_info')
    clear_streams()
    data = json.loads(data_test)
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
    clear_streams()
    return json.loads(r.get(model_name + '|metric_info'))
    # return cache['metric_info']


@app.route('/getModelInfo', methods=['GET'])
def getModelInfo():
    clear_streams()
    return json.loads(r.get(model_name + '|model_info'))
    # return cache['metric_info']


@app.route('/getCertification/<date1>/<date2>', methods=['GET'])
def getCertification(date1, date2):  # NOT REAL DATA YET.
    date1 += " 00:00:00"
    date2 += " 99:99:99"
    data_test = r.lrange(model_name + '|certificate_values', 0, -1)
    metadata = json.loads(r.get(model_name + '|certificate_metadata'))

    # data_test = cache['metric_values']
    clear_streams()
    res = []
    for i in range(len(data_test)):
        item = json.loads(data_test[i])
        scores = {"fairness": [0, 0], "explainability": [0, 0], "performance": [0, 0], "robustness": [0, 0]}
        if date1 <= item['metadata > date']["value"] <= date2:
            temp_dict = {}
            for value in item:
                if metadata[value]["tags"][0] != "metadata":
                    if metadata[value]["tags"][0] not in temp_dict:
                        temp_dict[metadata[value]["tags"][0]] = []
                    metric_obj = item[value]
                    for key in metadata[value]:
                        metric_obj[key] = metadata[value][key]
                    temp_dict[metadata[value]["tags"][0]].append(metric_obj)

                    if metadata[value]["tags"][0] in scores:
                        scores[metadata[value]["tags"][0]][1] += 1
                        if item[value]["value"]:
                            scores[metadata[value]["tags"][0]][0] += 1
            temp_dict['metadata'] = {"date": item['metadata > date'], "description": item['metadata > description'], "scores": scores}
            res.append(temp_dict)
    return json.dumps(res)


@app.route('/getCertificationMeta', methods=['GET'])
def getCertificationMeta():
    model_info = r.get(model_name + '|certificate_metadata')
    data = json.loads(model_info)
    clear_streams()
    return data


@app.route('/viewClass/<category>')
def renderClassTemplate(category):
    functional = category.replace(' ', '_').lower()
    start_date, end_date = get_dates()
    model_info = json.loads(r.get(model_name + '|model_info'))
    page_js = "/static/js/add_metrics.js"
    main_function = "loadMetrics"

    return render_template('/admin/view_class.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           page_js=page_js,
                           main_function=main_function,
                           model_name=model_info["display_name"],
                           Category=category,
                           Functional=functional,
                           start_date=start_date,
                           end_date=end_date)


@app.route('/viewAll')
def renderAllMetrics():
    start_date, end_date = get_dates()
    model_info = json.loads(r.get(model_name + '|model_info'))
    page_js = "/static/js/add_all.js"
    main_function = "loadAll"
    functional = ""

    return render_template('/admin/view_class.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           Category="All",
                           main_function=main_function,
                           functional=functional,
                           page_js=page_js,
                           h=admin_helpers,
                           model_name=model_info["display_name"],
                           start_date=start_date,
                           end_date=end_date)


@app.route('/learnMore/<metric>')
def learnMore(metric):
    data_test = r.get(model_name + '|metric_info')
    model_info = json.loads(r.get(model_name + '|model_info'))
    clear_streams()
    metric_info = json.loads(data_test)
    start_date, end_date = get_dates()
    # metric_info = cache['metric_info']

    return render_template('/admin/metric_info.html',
                           admin_base_template=admin.base_template,
                           admin_view=admin.index_view,
                           get_url=url_for,
                           h=admin_helpers,
                           model_name=model_info["display_name"],
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

@app.route('/updateMetrics', methods=['GET'])
def updateMetrics():
    return json.dumps(metric_event_stream())


@app.route('/updateCertificates', methods=['GET'])
def updateCertificates():
    return json.dumps(cert_event_stream())

def clear_streams():
    metric_event_stream()
    cert_event_stream()


def metric_event_stream():
    message = metric_sub.get_message()
    result = False
    if message:
        result = message['data'] != 1
        while message:
            message = metric_sub.get_message()
    return result


def cert_event_stream():
    message = cert_sub.get_message()
    result = False
    if message:
        result = message['data'] != 1
        while message:
            message = cert_sub.get_message()
    return result


if __name__ == '__main__':
    # Build a sample db on the fly, if one does not exist yet.
    app_dir = os.path.realpath(os.path.dirname(__file__))
    # Start app
    app.run(debug=True)

