##################
## storage the data in redis in future version
##################

# store the positive result
storage = {}

# store all results
storage2 = {}

###
# key : clientId:sessionId:traceId
def put(key, value):
    if key in storage:
        array = storage[key]
        array.append(value)
    else:
        storage[key] = [value]

###
# key : clientId:sessionId:traceId
def put2(key, value):
    if key in storage2:
        array = storage2[key]
        array.append(value)
    else:
        storage2[key] = [value]

def get_numbers(key):
    if key in storage2:
        return len(storage2[key])
    return 0

def get_result(key):
    if key in storage:
        return storage[key]
    return []

def extract_result(key):
    if key in storage:
        return storage.pop(key)
    return []

def print_result(key):
    if key in storage:
        for v in storage[key]:
            print(v)
    else:
        print('nothing')

def format_key(client_id, session_id, trace_id):
    return "{0}:{1}:{2}".format(client_id, session_id, trace_id)
