# START verif completion
_verif()
{
local cur prev opts
COMPREPLY=()
cur="${COMP_WORDS[COMP_CWORD]}"
prev="${COMP_WORDS[COMP_CWORD-1]}"
if [ "$cur" = "" ] || [[ "$cur" =~ -* ]]; then
   COMPREPLY=( $( compgen -f -W ' -m --list-times --list-locations --list-quantiles --list-thresholds --version -d -elevrange -l -lx -latrange -lonrange -o -r -q -t -x -acc -agg -b -c -C -fcst -hist -obs -sort -aspect -bottom -clim -cmap -dpi -f -fs -labfs -lc -left -leg -legfs -legloc -ls -lw -maptype -ms -nogrid -nomargin -right -simple -sp -tickfs -title -titlefs -top -type -xlabel -xlim -xlog -xrot -xticks -xticklabels -ylabel -ylim -ylog -yrot -yticks -xticklabels ' -- $cur ) )
fi
if [ "$prev" = "-m" ]; then
   COMPREPLY=( $( compgen -W ' a  against  alphaindex  b  baserate  bias  biasfreq  bs  bsrel  bsres  bss  bsunc  c  change  cmae  cond  corr  d  derror  diff  discrimination  dmb  droc  droc0  drocnorm  dscore  economicvalue  edi  eds  ef  error  ets  fa  far  fcst  fcstrate  freq  hit  hss  ign0  igncontrib  impact  invreliability  kendallcorr  kss  leps  lor  mae  marginal  marginalratio  mbias  meteo  miss  n  nsec  obs  obsfcst  or  pc  performance  pitdev  pithist  qq  quantilescore  rankcorr  reliability  rmse  rmsf  roc  scatter  sedi  seds  spherical  spreadskill  stderror  taylor  threat  timeseries  within  yulesq  ' -- $cur ) )
fi
return 0
}
complete -F _verif verif
# END verif completion
