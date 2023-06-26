import json
import sys

def parse(str, key):
    #print(str)
    #print(key)
    str_dict = json.loads(str)
    val = str_dict[key]
    if type(val)==list:
        return ",".join(val)
    else:
        return val
    #return str_dict[key]

if __name__ == '__main__':
    parse(sys.argv[1], sys.argv[2])
