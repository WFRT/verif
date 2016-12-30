let b:did_ftplugin = 1
set tabstop=3
set softtabstop=3
set shiftwidth=3
set expandtab
set cindent shiftwidth=3
autocmd BufNewFile,BufRead *.py setlocal textwidth=79
if (version >= 7.30 && version < 100) || (version >= 703) 
   autocmd BufNewFile,BufRead *.py setlocal colorcolumn=+1
endif
