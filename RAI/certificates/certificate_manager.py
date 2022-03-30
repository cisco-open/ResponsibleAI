from .certificate import Certificate

__all__ = ['CertificateManager']



import json
import os.path

cert_list_file_name = "RAI\\certificates\\standard\\cert_list.json"
# class Certificate(object):
 
class CertificateManager(object):
    def __init__(self) -> None:
        super().__init__()
        self.certificates = {}
        self.metadata = {}
        self.results = {}
        # self.pre


    def load_stock_certificates(self):
        f = open(cert_list_file_name, "r")
        data = json.load(f)
        for item in data["certs"]:
            c = Certificate()
            c.load_from_json("RAI\\certificates\\standard\\" + item["filename"])
            self.metadata[item["name"]] = c.cert_spec["meta"]
            self.metadata[item["name"]]["condition"] = c.cert_spec["condition"]
            self.certificates[item["name"]] = c

    def load_custom_certificates(self, filename):
        f = open(filename, "r")
        data = json.load(f)
        for item in data["certs"]:
            c = Certificate()
            c.load_from_json("RAI\\certificates\\standard\\" + item["filename"])
            self.metadata[item["name"]] = c.cert_spec["meta"]
            self.metadata[item["name"]]["condition"] = c.cert_spec["condition"]
            self.certificates[item["name"]] = c


    def get_metadata(self) -> dict:
        return self.metadata
        
    def compute(self, metric_values):
        self.results = {}
        self.results = {}
        for cert_name in self.certificates:
            c = self.certificates[cert_name]
            self.results[cert_name] = {"explanation": ""}
            self.results[cert_name]["value"] = c.evaluate(metric_values, self.results)
            # self.results[cert_name]["term_values"] = c.term_values
        return self.results


