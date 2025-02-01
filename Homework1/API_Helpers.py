import numpy

def fetch(url):
    import requests, gzip, os, hashlib, numpy
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(script_dir, 'tmp')
    
    fp = os.path.join(tmp_dir, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, 'rb') as f:
            dat = f.read()
            
    else:
        with open(fp, 'wb') as f:
            dat = requests.get(url).content
            f.write(dat)
    
    return numpy.frombuffer(gzip.decompress(dat), dtype=numpy.uint8)

def unpickle(file):
    import pickle
        
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return decode_dict(dict)

def decode_dict(d):
    if isinstance(d, dict):
        return {k.decode('utf-8') if isinstance(k, bytes) else k:
                decode_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [decode_dict(i) for i in d]
    elif isinstance(d, bytes):
        return d.decode('utf-8')
    else:
        return d