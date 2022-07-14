import json
import os.path
import site

from .certificate import Certificate

__all__ = ['CertificateManager']

site_pkgs_path = site.getsitepackages()[0]
rai_pkg_path = os.path.join(site_pkgs_path, "RAI")
if not os.path.isdir(rai_pkg_path):
    rai_pkg_path = "RAI"
cert_file_folder = os.path.join(os.path.dirname(__file__), "standard")
cert_list_file_name = os.path.join(os.path.dirname(__file__), 'standard/empty_cert_list.json')


class CertificateManager(object):
    """
    CertificateManager is a class automatically created by AISystems.
    This class loads a file containing information on which certificates to use,
    before creating associated Certificate Objects, as well as prompting their associated evaluations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.certificates = {}
        self.metadata = {}
        self.results = {}

    # Loads all certificates found in the stock certificate file
    # TODO: Refactor to be the same? as load_custom_certificates with default value
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

    # Loads all certificates found in a custom filepath
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
        return self.results
