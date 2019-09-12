import re
import verif.metric
import verif.output

metrics = [metric for metric in verif.metric.get_all() if metric[1].is_valid()]
outputs = [output for output in verif.output.get_all() if output[1].is_valid()]

print "Metrics: %d" % len(metrics)
print "Outputs: %d" % len(outputs)
print "Total: %d" % (len(metrics) + len(outputs))

description = verif.driver.show_description()
lines = description.split('\n')
reg = re.compile("^  -")
lines = [line for line in lines if reg.match(line)]
for i in range(0, len(lines)):
    line = lines[i]
    line = line.split(' ')
    line = [q for q in line if q != '']
    lines[i] = line[0]
print "Options: %d" % len(lines)
