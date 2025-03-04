# START verif completion
_verif()
{
local cur prev opts
COMPREPLY=()
cur="${COMP_WORDS[COMP_CWORD]}"
prev="${COMP_WORDS[COMP_CWORD-1]}"
if [ "$cur" = "" ] || [[ "$cur" =~ -* ]]; then
   COMPREPLY=( $( compgen -f -W '-m --config --list-times --list-dates --list-locations --list-quantiles --list-thresholds --help -d -elevrange -l -lx -latrange -lonrange -o -obsrange -r -q -t -tod -x -acc -agg -b -c -C -fcst -hist -obs -sort -T -Tagg -Tx -a -af -afs -aspect -bottom -clabel -clim -cmap -dpi -gc -gs -gw -f -fs -labfs -lc -left -leg -legfs -legloc -ls -lw -maptype -ma -ms -nogrid -nomargin -obsleg -right -simple -sp -tickfs -title -titlefs -top -type -xlabel -xlim -xlog -xrot -xticks -xticklabels -ylabel -ylim -ylog -yrot -yticks -yticklabels' -- $cur ) )
fi
if [ "$prev" = "-m" ]; then
   COMPREPLY=( $( compgen -W 'a against alphaindex autocorr autocov b baserate bias biasfreq bs bsdecomp bsrel bsres bss bssrel bssres bsunc c change cmae cond corr d derror diff discrimination dmb droc droc0 dscore economicvalue edi eds ef error ets fa far fcst fcstrate freq fss hit hss ign0 igncontrib invreliability kendallcorr kge kss leps lor mae marginal marginalratio mbias meteo miss murphy n nnsec nsec obs obsfcst or pc performance pit pithist pithistdev pithistshape pithistslope qq quantile quantilecoverage quantilescore rankcorr ratio reliability rmse rmsf roc scatter sedi seds spherical spread spreadskill spreadskillratio stderror obsstddev fcststddev taylor threat threshold timeseries within yulesq ' -- $cur ) )
elif [ "$prev" = "-cmap" ]; then
   COMPREPLY=( $( compgen -W 'Accent Accent_r Blues Blues_r BrBG BrBG_r BuGn BuGn_r BuPu BuPu_r CMRmap CMRmap_r Dark2 Dark2_r GnBu GnBu_r Grays Grays_r Greens Greens_r Greys Greys_r OrRd OrRd_r Oranges Oranges_r PRGn PRGn_r Paired Paired_r Pastel1 Pastel1_r Pastel2 Pastel2_r PiYG PiYG_r PuBu PuBuGn PuBuGn_r PuBu_r PuOr PuOr_r PuRd PuRd_r Purples Purples_r RdBu RdBu_r RdGy RdGy_r RdPu RdPu_r RdYlBu RdYlBu_r RdYlGn RdYlGn_r Reds Reds_r Set1 Set1_r Set2 Set2_r Set3 Set3_r Spectral Spectral_r Wistia Wistia_r YlGn YlGnBu YlGnBu_r YlGn_r YlOrBr YlOrBr_r YlOrRd YlOrRd_r afmhot afmhot_r autumn autumn_r berlin berlin_r binary binary_r bone bone_r brg brg_r bwr bwr_r cividis cividis_r cool cool_r coolwarm coolwarm_r copper copper_r cubehelix cubehelix_r flag flag_r gist_earth gist_earth_r gist_gray gist_gray_r gist_grey gist_grey_r gist_heat gist_heat_r gist_ncar gist_ncar_r gist_rainbow gist_rainbow_r gist_stern gist_stern_r gist_yarg gist_yarg_r gist_yerg gist_yerg_r gnuplot gnuplot2 gnuplot2_r gnuplot_r gray gray_r grey grey_r hot hot_r hsv hsv_r inferno inferno_r jet jet_r magma magma_r managua managua_r nipy_spectral nipy_spectral_r ocean ocean_r pink pink_r plasma plasma_r prism prism_r rainbow rainbow_r seismic seismic_r spring spring_r summer summer_r tab10 tab10_r tab20 tab20_r tab20b tab20b_r tab20c tab20c_r terrain terrain_r turbo turbo_r twilight twilight_r twilight_shifted twilight_shifted_r vanimo vanimo_r viridis viridis_r winter winter_r' -- $cur ) )
elif [ "$prev" = "-agg" ] || [ "$prev" = "-Tagg" ]; then
   COMPREPLY=( $( compgen -W 'abschange absmean change count iqr max mean meanabs median min quantile range std sum variance' -- $cur ) )
elif [ "$prev" = "-type" ]; then
   COMPREPLY=( $( compgen -W 'csv impact map mapimpact maprank plot rank text' -- $cur ) )
elif [ "$prev" = "-x" ]; then
   COMPREPLY=( $( compgen -W 'all axis day dayofmonth dayofyear elev fcst lat leadtime leadtimeday location lon month monthofyear no obs threshold time timeofday week year' -- $cur ) )
elif [ "$prev" = "-Tx" ]; then
   COMPREPLY=( $( compgen -W ' leadtime time ' -- $cur ) )
fi
return 0
}
complete -ofilenames -F _verif verif
# END verif completion
