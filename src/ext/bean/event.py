class RouterRequest:
    def __init__(self, client_id, session_id, trace_id):
        self.trace_id = trace_id;
        self.client_id = client_id;
        self.session_id = session_id;

class ExtractRequest:
    def __init__(self, client_id, session_id, trace_id, root_path, dataset_root_path, target_image_path):
        self.trace_id = trace_id;
        self.session_id = session_id;
        self.client_id = client_id;
        self.root_path = root_path;
        self.dataset_root_path = dataset_root_path;
        self.target_image_path = target_image_path;

## face_image_path with old root
class CompareRequest:
    def __init__(self, client_id, session_id, trace_id, total_image_numbers, request_order, root_path, extract_root_path, face_image_path, face_extract_path, target_extract_path):
        self.trace_id = trace_id;
        self.session_id = session_id;
        self.client_id = client_id;
        self.total_image_numbers = total_image_numbers;
        self.request_order = request_order;
        self.root_path = root_path;
        self.extract_root_path = extract_root_path;
        self.face_image_path = face_image_path;
        self.face_extract_path = face_extract_path;
        self.target_extract_path = target_extract_path;

## face_image_path with old root
class AggregatorRequest:
    def __init__(self, client_id, session_id, trace_id, total_image_numbers, request_order, root_path, extract_root_path, face_image_path, result, error_message):
        self.trace_id = trace_id;
        self.session_id = session_id;
        self.client_id = client_id;
        self.total_image_numbers = total_image_numbers;
        self.request_order = request_order;
        self.root_path = root_path;
        self.extract_root_path = extract_root_path;
        self.face_image_path = face_image_path;
        self.result = result;
        self.error_message = error_message;

class ResponseRequest:
    def __init__(self, client_id, session_id, trace_id, results):
        self.trace_id = trace_id;
        self.session_id = session_id;
        self.client_id = client_id;
        self.results = results;

def convert_to_dict(obj):
    '''convert Object to Dict'''
    dict = {}
    dict.update(obj.__dict__)
    return dict