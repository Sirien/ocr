import pickle
from pprint import pprint

infer_result = pickle.load(open("infer_result3.pkl", "rb"))

total_count = sum([v[1] for v in infer_result.values()])
total_T_count = sum([v[0] for v in infer_result.values()])


ret = [[k,int(v[0]),v[1], v[0]/v[1]] for k,v in infer_result.items()]

pprint(sorted(ret, key=lambda x:x[2], reverse=True), stream=open("infer_result3.txt", "w"))



# print(total_T_count / total_count)
