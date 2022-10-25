import os
import re
import sys
import matplotlib.pylab as mpl
import verif.driver
import verif.metric
import verif.output

"""
This script creates a bash completion script by parsing command-line options from verif's
description as well as reading all classed form verif.metric and verif.output
"""

description = verif.driver.show_description()
lines = description.split('\n')
reg = re.compile("^  -")
lines = [line for line in lines if reg.match(line)]
for i in range(0, len(lines)):
    line = lines[i]
    line = line.split(' ')
    line = [q for q in line if q != '']
    lines[i] = line[0]

aggregators = [agg.name() for agg in verif.aggregator.get_all()]
aggregators.sort()

axes = [axis[0].lower() for axis in verif.axis.get_all()]
axes.sort()

metrics = verif.metric.get_all()
outputs = verif.output.get_all()
metricOutputs = metrics + outputs
metricOutputs.sort(key=lambda x: x[0].lower(), reverse=False)

types = ["csv", "plot", "text", "map", "rank", "maprank", "impact", "mapimpact"]
types.sort()

print("# START verif completion")
print("_verif()")
print("{")
print("local cur prev opts")

print('COMPREPLY=()')
print('cur="${COMP_WORDS[COMP_CWORD]}"')
print('prev="${COMP_WORDS[COMP_CWORD-1]}"')

# Files
print('if [ "$cur" = "" ] || [[ "$cur" =~ -* ]]; then')
print("   COMPREPLY=( $( compgen -f -W '",  end = '')
string = ' '.join(lines)
print(string, end = '')
print("' -- $cur ) )")
print('fi')

# Metrics
print('if [ "$prev" = "-m" ]; then')
print("   COMPREPLY=( $( compgen -W '", end = '')
for m in metricOutputs:
    name = m[0].lower()
    if(m[1].is_valid()):
        desc = m[1].get_class_name()
        print(name + " ", end = '')
print("' -- $cur ) )")

# Cmap
print('elif [ "$prev" = "-cmap" ]; then')
print("   COMPREPLY=( $( compgen -W '", end = '')
cmaps = list(mpl.cm.cmap_d.keys())
cmaps.sort()
print(' '.join(cmaps), end = '')
print("' -- $cur ) )")

# Agg
print('elif [ "$prev" = "-agg" ]; then')
print("   COMPREPLY=( $( compgen -W '", end = '')
print(' '.join(aggregators), end = '')
print("' -- $cur ) )")

# Type
print('elif [ "$prev" = "-type" ]; then')
print("   COMPREPLY=( $( compgen -W '", end = '')
print(' '.join(types), end = '')
print("' -- $cur ) )")

# Axis
print('elif [ "$prev" = "-x" ]; then')
print("   COMPREPLY=( $( compgen -W '", end = '')
print(' '.join(axes), end = '')
print("' -- $cur ) )")
print('fi')
print('return 0')
print('}')
print('complete -ofilenames -F _verif verif')
print('# END verif completion')
