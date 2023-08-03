import re
import verif.metric
import verif.output

metrics = [metric for metric in verif.metric.get_all() if metric[1].is_valid()]
outputs = [output for output in verif.output.get_all() if output[1].is_valid()]

print("Metrics: %d" % len(metrics))
print("Outputs: %d" % len(outputs))
print("Total: %d" % (len(metrics) + len(outputs)))

description = verif.driver.show_description()
lines0 = description.split('\n')
reg = re.compile("^  -")
lines0 = [line for line in lines0 if reg.match(line)]
lines = list()
for i, line in enumerate(lines0):
    line = line.split(' ')
    line = [q for q in line if q != '']
    line = line[0]
    if line[0:2] == '--':
        continue
    lines += [line]
print("(real) options: %d" % len(lines))
