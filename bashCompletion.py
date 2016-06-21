import sys
import verif.Metric as Metric
import verif.Output as Output

if len(sys.argv) < 2:
   print "Error: missing input file"
   sys.exit()
filename = sys.argv[1]

metrics = Metric.getAllMetrics()
outputs = Output.getAllOutputs()
metricOutputs = metrics + outputs
metricOutputs.sort(key=lambda x: x[0].lower(), reverse=False)

print "# START verif completion"
print "_verif()"
print "{"
print "local cur prev opts"

print 'COMPREPLY=()'
print 'cur="${COMP_WORDS[COMP_CWORD]}"'
print 'prev="${COMP_WORDS[COMP_CWORD-1]}"'
 

print 'if [ "$cur" = "" ] || [[ "$cur" =~ -* ]]; then'
print "   COMPREPLY=( $( compgen -f -W '",
file = open(filename, "r")
for line in file.read().split('\n'):
   print line,
file.close()
print "' -- $cur ) )"
print 'fi'

# Metrics
print 'if [ "$prev" = "-m" ]; then'
print "   COMPREPLY=( $( compgen -W '",
for m in metricOutputs:
   name = m[0].lower()
   if(m[1].isValid()):
      desc = m[1].summary()
      print name + " ",
print "' -- $cur ) )"
print 'fi'
print 'return 0'
print '}'
print 'complete -F _verif verif'
print '# END verif completion'
