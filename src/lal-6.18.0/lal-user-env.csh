# source this file to access LAL
if ( ! ${?OCTAVE_PATH} ) setenv OCTAVE_PATH
setenv OCTAVE_PATH `echo "$OCTAVE_PATH" | /opt/local/bin/gsed -e 's|/Users/ram/git/lalsuite/_inst/lib/octave/4.2.0/site/oct/x86_64-apple-darwin16.4.0:||g;'`
setenv OCTAVE_PATH "/Users/ram/git/lalsuite/_inst/lib/octave/4.2.0/site/oct/x86_64-apple-darwin16.4.0:$OCTAVE_PATH"
if ( ! ${?LAL_DATADIR} ) setenv LAL_DATADIR
setenv LAL_DATADIR "/Users/ram/git/lalsuite/_inst/share/lal"
if ( ! ${?PKG_CONFIG_PATH} ) setenv PKG_CONFIG_PATH
setenv PKG_CONFIG_PATH `echo "$PKG_CONFIG_PATH" | /opt/local/bin/gsed -e 's|/Users/ram/git/lalsuite/_inst/lib/pkgconfig:||g;'`
setenv PKG_CONFIG_PATH "/Users/ram/git/lalsuite/_inst/lib/pkgconfig:$PKG_CONFIG_PATH"
if ( ! ${?PYTHONPATH} ) setenv PYTHONPATH
setenv PYTHONPATH `echo "$PYTHONPATH" | /opt/local/bin/gsed -e 's|/Users/ram/git/lalsuite/_inst/lib/python2.7/site-packages:||g;'`
setenv PYTHONPATH "/Users/ram/git/lalsuite/_inst/lib/python2.7/site-packages:$PYTHONPATH"
if ( ! ${?PATH} ) setenv PATH
setenv PATH `echo "$PATH" | /opt/local/bin/gsed -e 's|/Users/ram/git/lalsuite/_inst/bin:||g;'`
setenv PATH "/Users/ram/git/lalsuite/_inst/bin:$PATH"
if ( ! ${?MANPATH} ) setenv MANPATH
setenv MANPATH `echo "$MANPATH" | /opt/local/bin/gsed -e 's|/Users/ram/git/lalsuite/_inst/share/man:||g;'`
setenv MANPATH "/Users/ram/git/lalsuite/_inst/share/man:$MANPATH"
if ( ! ${?LAL_PREFIX} ) setenv LAL_PREFIX
setenv LAL_PREFIX "/Users/ram/git/lalsuite/_inst"
