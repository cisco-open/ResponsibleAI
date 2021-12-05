from RAI import certificates
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
        # self.pre
    def load_stock_certificates(self):
        f = open( cert_list_file_name , "r")
        data = json.load(f)
        
        for item in data["certs"]:
            c = Certificate()
            c.load_from_json( "RAI\\certificates\\standard\\" + item["filename"])
            self.metadata[item["name"]] =  c.cert_spec["meta"]
            self.certificates[ item["name"] ] = c

    def compute( self, metric_values):
        cert_values = {}
        for cert_name in self.certificates:
            c = self.certificates[cert_name]
            cert_values[cert_name] = { "explanation":""} 
            cert_values[cert_name]["value"] = c.evaluate( metric_values, cert_values )

        return cert_values


