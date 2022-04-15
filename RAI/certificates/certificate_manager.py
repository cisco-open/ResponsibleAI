from .certificate import Certificate

__all__ = ['CertificateManager']

import json
import os.path
import site

# choose the first site packages folder
site_pkgs_path = site.getsitepackages()[0]
rai_pkg_path = os.path.join(site_pkgs_path, "RAI")
if not os.path.isdir(rai_pkg_path):
    rai_pkg_path = "RAI"

cert_file_folder = os.path.join(rai_pkg_path, "certificates", "standard")
cert_list_file_name = os.path.join(cert_file_folder, "cert_list.json")

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
            name = os.path.basename(item["filename"])
            c.load_from_json(os.path.join(cert_file_folder, item["filename"]))
            self.metadata[name] = c.cert_spec["meta"]
            self.metadata[name]["condition"] = c.cert_spec["condition"]
            self.certificates[name] = c

    def load_custom_certificates(self, filename):
        f = open(filename, "r")
        data = json.load(f)
        for item in data["certs"]:
            c = Certificate()
            name = os.path.basename(item["filename"])
            c.load_from_json(os.path.join(cert_file_folder, item["filename"]))
            self.metadata[name] = c.cert_spec["meta"]
            self.metadata[name]["condition"] = c.cert_spec["condition"]
            self.certificates[name] = c


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
