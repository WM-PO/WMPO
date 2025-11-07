from math_util import evaluate_math
from code_util import evaluate_code
import json
# from math_verify import parse, verify
# Parse the gold and answer
# If you know that gold will only contain latex or expr (no latex env), use
# parse(gold, extraction_config=[LatexExtractionConfig()]) or parse(gold, extraction_config=[ExprExtractionConfig()])

# gold = parse("$(1,4.5)$")
# answer = parse("$(1,\\frac{9}{2})$")
# import os
# print(os.name)
# Order here is important!
# verify(gold, answer)

# print(verify(gold, answer))
# print(verify(parse("$\\boxed{\\frac{x}{7}+\\frac{2}{7}}$"), parse("$\\frac{x+2}{7}$")))
# print(verify(parse("$\\boxed{2022!}$"),parse("$\\frac{2023!}{2023}$")))

# print(verify(parse("$\\begin{bmatrix}\n -7 & 6 & -8 \\\\\n 11 & -9 & 12 \\\\\n 15 & -16 & 19 \n \\end{bmatrix}$"), parse("$\\begin{pmatrix}\n -7 & 6 & -8 \\\\\n 11 & -9 & 12 \\\\\n 15 & -16 & 19\n \\end{pmatrix}$")))
# print(verify(parse("$\\frac{\\sqrt{505}}{7}$"), parse("$\\frac{\\sqrt{505}}{7}$")))
# print(verify(parse("${x^2 + y^2 + 4x - 6y + 13}$"), "${(x + 2)^2 + (y - 3)^2}$"))
# print(verify(parse("$\\boxed{1 -frac{1950!}{(1950-m)! \\cdot 1950^m}}$"),parse("$1 - \\frac{\\binom{1950}{m}m!}{1950^m}$")))
"""
print(evaluate_math("(1,4.5)","(1,\\frac{9}{2})"))
print(evaluate_math("\\boxed{\\frac{x}{7}+\\frac{2}{7}}","\\frac{x+2}{7}"))
print(evaluate_math("\\boxed{2022!}","\\frac{2023!}{2023}"))
print(evaluate_math("\\begin{bmatrix}\n -7 & 6 & -8 \\\\\n 11 & -9 & 12 \\\\\n 15 & -16 & 19 \n \\end{bmatrix}",
                    "\\begin{pmatrix}\n -7 & 6 & -8 \\\\\n 11 & -9 & 12 \\\\\n 15 & -16 & 19\n \\end{pmatrix}"))
print(evaluate_math("\\frac{\\sqrt{505}}{7}", "\\frac{\\sqrt{505}}{7}"))
print(evaluate_math("x^2 + y^2 + 4x - 6y + 13", "(x + 2)^2 + (y - 3)^2"))
print(evaluate_math("\\boxed{1 -frac{1950!}{(1950-m)! \\cdot 1950^m}}","1 - \\frac{\\binom{1950}{m}m!}{1950^m}"))
"""



test_cases = """{"fn_name": "standardize_medical_data", "inputs": [[[{"age": 40, "bmi": 30.0, "children": 2}, {"age": 50, "bmi": 25.0, "children": 3}, {"age": 60, "bmi": 20.0, "children": 1}]], [[{"age": 28, "bmi": 24.0, "child
ren": 2}, {"age": 32, "bmi": 26.0, "children": 1}, {"age": 36, "bmi": 28.0, "children": 3}]], [[{"age": 29, "bmi": 23.5, "children": 1}, {"age": 31, "bmi": 27.5, "children": 2}, {"age": 33, "bmi": 31.5, "children": 3}]], [[{"age": 45
, "bmi": 30.0, "children": 3}, {"age": 55, "bmi": 35.0, "children": 4}, {"age": 65, "bmi": 40.0, "children": 5}]], [[{"age": 27, "bmi": 22.5, "children": 0}, {"age": 29, "bmi": 26.5, "children": 1}, {"age": 31, "bmi": 30.5, "children
": 2}]], [[{"age": 39, "bmi": 22.0, "children": 2}, {"age": 41, "bmi": 24.0, "children": 3}, {"age": 43, "bmi": 26.0, "children": 4}]]], "outputs": [[[{"age": -1.0, "bmi": 1.0, "children": 0.0}, {"age": 0.0, "bmi": 0.0, "children": 1
.0}, {"age": 1.0, "bmi": -1.0, "children": -1.0}]], [[{"age": -1.0, "bmi": -1.0, "children": 0.0}, {"age": 0.0, "bmi": 0.0, "children": -1.0}, {"age": 1.0, "bmi": 1.0, "children": 1.0}]], [[{"age": -1.0, "bmi": -1.0, "children": -1.0
}, {"age": 0.0, "bmi": 0.0, "children": 0.0}, {"age": 1.0, "bmi": 1.0, "children": 1.0}]], [[{"age": -1.0, "bmi": -1.0, "children": -1.0}, {"age": 0.0, "bmi": 0.0, "children": 0.0}, {"age": 1.0, "bmi": 1.0, "children": 1.0}]], [[{"ag
e": -1.0, "bmi": -1.0, "children": -1.0}, {"age": 0.0, "bmi": 0.0, "children": 0.0}, {"age": 1.0, "bmi": 1.0, "children": 1.0}]], [[{"age": -1.0, "bmi": -1.0, "children": -1.0}, {"age": 0.0, "bmi": 0.0, "children": 0.0}, {"age": 1.0,
 "bmi": 1.0, "children": 1.0}]]]}"""        

test_cases = json.dumps({"fn_name": "standardize_medical_data", "inputs": [[[{"age": 40, "bmi": 30.0, "children": 2}, {"age": 50, "bmi": 25.0, "children": 3}, {"age": 60, "bmi": 20.0, "children": 1}]],], "outputs": [[[{"age": -1.0, "bmi": 1.0, "children": 0.0}, {"age": 0.0, "bmi": 0.0, "children": 1.0}, {"age": 1.0, "bmi": -1.0, "children": -1.0}]],]} , ensure_ascii=False)                                                                                                                                                                                           



completion = """      
```python                                                                                                                                                                                                            
from typing import List, Dict                                                                                                                                                                                      
from statistics import mean, stdev                                                                                                                                                                                 
                                                                                                                                                                                                                    
def standardize_medical_data(data: List[Dict[str, float]]) -> List[Dict[str, float]]:                                                                                                                              
    standardized_data = []                                                                                                                                                                                         
                                                                                                                                                                                                                
    for patient in data:                                                                                                                                                                                           
        standardized_patient = {}                                                                                                                                                                                  
        patient_mean = mean(patient[key] for key in patient if key.isdigit())                                                                                                                                      
        patient_std = stdev(patient[key] for key in patient if key.isdigit())                                                                                                                                      
                                                                                                                                                                                                                
        for key, value in patient.items():                                                                                                                                                                         
            if key.isdigit():                                                                                                                                                                                      
                standardized_patient[key] = (value - patient_mean) / patient_std                                                                                                                                   
            else:                                                                                                                                                                                                  
                standardized_patient[key] = value                                                                                                                                                                  
                                                                                                                                                                                                                
        standardized_data.append(standardized_patient)                                                                                                                                                             
                                                                                                                                                                                                                
    return standardized_data      
```
"""

test_cases = json.dumps({"fn_name": "standardize_medical_data", "inputs": [[[{"age": 40, "bmi": 30.0, "children": 2}, {"age": 50, "bmi": 25.0, "children": 3}, {"age": 60, "bmi": 20.0, "children": 1}]]], "outputs": [[[{"age": 40, "bmi": 30.0, "children": 2}, {"age": 50, "bmi": 25.0, "children": 3}, {"age": 60, "bmi": 20.0, "children": 1}]]]} , ensure_ascii=False)                                                                                                                                                                                           

test_cases = json.dumps({"fn_name": "standardize_medical_data", "inputs": [["abc"]], "outputs": [["abc"]]} , ensure_ascii=False)     
                                                                                                                                                                                      
test_cases = "{\"fn_name\": \"standardize_medical_data\", \"inputs\": [[\"abc\",[3,4],[{3:5},4]]], \"outputs\": [[\"abc\",[3,4],[{3:5},4]]]}"    

test_cases = json.dumps({"fn_name": "standardize_medical_data", "inputs": [["abc",[3,4],[{True:5},4]]], "outputs": [["abc",[3,4],[{True:5},4]]]})  
print("mttestcase",test_cases)
print(json.loads(test_cases))
#test_cases = "{ \"inputs\": [\"3\"], \"outputs\": [\"3\"]}" 

completion = """      
```python           
import sys            
from statistics import mean, stdev           
from typing import List                                                                                                                                                                                                                                                                                                                                                                                           
def standardize_medical_data(a,y,z):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    return a,y  ,z    
```
"""


# completion = """import sys
# from collections import deque

# def main():
#     data = sys.stdin.read().split()
#     it = iter(data)
    
#     # Read start and target positions
#     x0, y0, x1, y1 = int(next(it)), int(next(it)), int(next(it)), int(next(it))
    
#     n = int(next(it))
#     allowed = set()
#     # The total number of allowed cells is at most 10^5.
#     for _ in range(n):
#         r = int(next(it))
#         a = int(next(it))
#         b = int(next(it))
#         for c in range(a, b + 1):
#             allowed.add((r, c))
    
#     # Directions for the king (8 neighboring cells)
#     directions = [(-1, -1), (-1, 0), (-1, 1),
#                   (0, -1),           (0, 1),
#                   (1, -1),  (1, 0),  (1, 1)]
    
#     start = (x0, y0)
#     target = (x1, y1)
    
#     # BFS initialization
#     queue = deque()
#     queue.append((x0, y0, 0))
#     # Mark the starting cell as visited by removing it from allowed set.
#     allowed.discard(start)
    
#     while queue:
#         x, y, moves = queue.popleft()
#         if (x, y) == target:
#             print(moves)
#             return
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             if (nx, ny) in allowed:
#                 allowed.remove((nx, ny))
#                 queue.append((nx, ny, moves + 1))
    
#     print(-1)

# if __name__ == '__main__':
#     main()
# """
# test_cases =  """{\n \"inputs\": [\n \"5 7 6 11\\n3\\n5 3 8\\n6 7 11\\n5 2 5\\n\",\n \"3 4 3 10\\n3\\n3 1 4\\n4 5 9\\n3 10 10\\n\",\n \"1 1 2 10\\n2\\n1 1 3\\n2 6 10\\n\",\n \"9 8 7 8\\n9\\n10 6 6\\n10 6 6\\n7 7 8\\n9 5 6\\n8 9 9\\n9 5 5\\n9 8 8\\n8 5 6\\n9 10 10\\n\",\n \"6 15 7 15\\n9\\n6 15 15\\n7 14 14\\n6 15 15\\n9 14 14\\n7 14 16\\n6 15 15\\n6 15 15\\n7 14 14\\n8 15 15\\n\",\n \"13 16 20 10\\n18\\n13 16 16\\n20 10 10\\n19 10 10\\n12 15 15\\n20 10 10\\n18 11 11\\n19 10 10\\n19 10 10\\n20 10 10\\n19 10 10\\n20 10 10\\n20 10 10\\n19 10 10\\n18 11 11\\n13 16 16\\n12 15 15\\n19 10 10\\n19 10 10\\n\",\n \"89 29 88 30\\n16\\n87 31 31\\n14 95 95\\n98 88 89\\n96 88 88\\n14 97 97\\n13 97 98\\n100 88 88\\n88 32 32\\n99 88 89\\n90 29 29\\n87 31 31\\n15 94 96\\n89 29 29\\n88 32 32\\n97 89 89\\n88 29 30\\n\",\n \"30 14 39 19\\n31\\n35 7 11\\n37 11 12\\n32 13 13\\n37 5 6\\n46 13 13\\n37 14 14\\n31 13 13\\n43 13 19\\n45 15 19\\n46 13 13\\n32 17 17\\n41 14 19\\n30 14 14\\n43 13 17\\n34 16 18\\n44 11 19\\n38 13 13\\n40 12 20\\n37 16 18\\n46 16 18\\n34 10 14\\n36 9 10\\n36 15 19\\n38 15 19\\n42 13 19\\n33 14 15\\n35 15 19\\n33 17 18\\n39 12 20\\n36 5 7\\n45 12 12\\n\",\n \"2 1 1 1\\n2\\n1 1 2\\n2 1 2\\n\",\n \"1 1 1 2\\n5\\n1000000000 1 10000\\n19920401 1188 5566\\n1000000000 1 10000\\n1 1 10000\\n5 100 200\\n\",\n \"1 1 1000000000 2\\n5\\n1000000000 1 10000\\n19920401 1188 5566\\n1000000000 1 10000\\n1 1 10000\\n5 100 200\\n\"\n ],\n \"outputs\": [\n \"4\\n\",\n \"6\\n\",\n \"-1\\n\",\n \"2\\n\",\n \"1\\n\",\n \"-1\\n\",\n \"1\\n\",\n \"9\\n\",\n \"1\\n\",\n \"1\\n\",\n \"-1\\n\"\n ]\n}"""
 


success, metadata_list, format_correctness = evaluate_code(completion, test_cases)
print(success)
print(metadata_list)
print(format_correctness)
