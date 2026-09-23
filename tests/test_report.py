import json
from pathlib import Path

from nemesys_gnn4id.utils.report import generate_report


def test_json_report_serializes_protocol_model_path():
    result = {
        'protocol_inference': {
            'summary': {'model_path': Path('models/field_proto_model_v2.pth')},
        },
    }

    report = json.loads(generate_report(result, fmt='json'))
    assert report['protocol_inference']['summary']['model_path'] == str(Path('models/field_proto_model_v2.pth'))
