import sys

file_path = sys.argv[1]

results = []
with open(file_path,'r') as file:
    while True:
        line_content = file.readline()
        if not line_content:
            break
        result = line_content[:-1].split("\\n\\n")[-1]
        results.append(result)
num_correct = 0
for index,i in enumerate(results):
    if "Incorrect" in i:
        pass
    else:
        num_correct +=1
print(results)

print("-"*100)
print(num_correct)
print(num_correct / 800 * 100)
