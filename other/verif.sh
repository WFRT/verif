# START verif completion
_verif()
{
local cur prev opts
COMPREPLY=()
cur="${COMP_WORDS[COMP_CWORD]}"
prev="${COMP_WORDS[COMP_CWORD-1]}"
if [ "$cur" = "" ] || [[ "$cur" =~ -* ]]; then
   COMPREPLY=( $( compgen -f -W ' -m --config --list-times --list-dates --list-locations --list-quantiles --list-thresholds --version -d -elevrange -l -lx -latrange -lonrange -o -obsrange -r -q -t -tod -x -acc -agg -b -c -C -fcst -hist -obs -sort -a -aspect -bottom -clabel -clim -cmap -dpi -f -fs -labfs -lc -left -leg -legfs -legloc -ls -lw -maptype -ma -ms -nogrid -nomargin -obsleg -proj -right -simple -sp -tickfs -title -titlefs -top -type -xlabel -xlim -xlog -xrot -xticks -xticklabels -ylabel -ylim -ylog -yrot -yticks -yticklabels ' -- $cur ) )
fi
if [ "$prev" = "-m" ]; then
   COMPREPLY=( $( compgen -W ' a  against  alphaindex  autocorr  autocov  b  baserate  bias  biasfreq  bs  bsrel  bsres  bss  bssrel  bssres  bsunc  c  change  cmae  cond  corr  d  derror  diff  discrimination  dmb  droc  droc0  dscore  economicvalue  edi  eds  ef  error  ets  fa  far  fcst  fcstrate  freq  fss  hit  hss  ign0  igncontrib  invreliability  kendallcorr  kss  leps  lor  mae  marginal  marginalratio  mbias  meteo  miss  n  nsec  obs  obsfcst  or  pc  performance  pit  pithist  pithistdev  pithistshape  pithistslope  qq  quantilescore  rankcorr  ratio  reliability  rmse  rmsf  roc  scatter  sedi  seds  spherical  spreadskill  stderror  taylor  threat  timeseries  within  yulesq  ' -- $cur ) )
elif [ "$prev" = "-cmap" ]; then
   COMPREPLY=( $( compgen -W ' Spectral summer coolwarm viridis Wistia_r pink_r Set1 Set2 Set3 brg_r Dark2 hot PuOr_r afmhot_r plasma terrain_r PuBuGn_r RdPu gist_ncar_r tab10_r gist_yarg_r Dark2_r YlGnBu RdYlBu hot_r gist_rainbow_r copper_r gnuplot_r cool_r cool gray gist_stern Greens_r GnBu gist_ncar spring_r gist_rainbow gist_heat_r Wistia OrRd_r CMRmap tab20_r bone gist_stern_r RdYlGn Pastel2_r spring afmhot terrain YlOrRd_r Set2_r brg tab20 PuBu RdGy_r rainbow flag_r jet_r RdPu_r gist_yarg cividis_r BuGn Paired_r hsv_r bwr cubehelix YlOrRd Greens PRGn gist_heat Paired hsv Oranges_r prism_r tab10 Pastel2 Pastel1_r Pastel1 gray_r jet plasma_r Spectral_r gnuplot2_r BuGn_r YlGnBu_r copper gist_earth_r Set3_r OrRd PuBu_r ocean_r winter_r gnuplot2 PuRd_r bone_r gist_earth Oranges RdYlGn_r PiYG inferno CMRmap_r YlGn binary_r gist_gray_r Accent viridis_r BuPu_r gist_gray flag seismic_r RdBu_r BrBG Reds summer_r GnBu_r BrBG_r Reds_r RdGy PuRd Accent_r Blues inferno_r tab20b_r cividis Greys autumn cubehelix_r magma nipy_spectral_r PRGn_r Greys_r pink binary winter gnuplot RdYlBu_r prism YlOrBr coolwarm_r rainbow_r magma_r Purples_r BuPu PiYG_r YlGn_r Blues_r YlOrBr_r seismic Purples tab20c tab20b bwr_r RdBu autumn_r ocean Set1_r PuOr PuBuGn nipy_spectral tab20c_r ' -- $cur ) )
elif [ "$prev" = "-agg" ]; then
   COMPREPLY=( $( compgen -W ' absmean count iqr max mean meanabs median min quantile range std sum variance ' -- $cur ) )
elif [ "$prev" = "-type" ]; then
   COMPREPLY=( $( compgen -W ' plot text csv map rank maprank impact mapimpact ' -- $cur ) )
elif [ "$prev" = "-x" ]; then
   COMPREPLY=( $( compgen -W ' all axis day dayofmonth dayofyear elev fcst lat leadtime leadtimeday location lon month monthofyear no obs threshold time timeofday week year ' -- $cur ) )
fi
return 0
}
complete -F _verif verif
# END verif completion
