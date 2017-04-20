# source this file to access LAL
export OCTAVE_PATH
OCTAVE_PATH=`echo "$OCTAVE_PATH" | /opt/local/bin/gsed -e 's|/Users/ram/git/lalsuite/_inst/lib/octave/4.2.0/site/oct/x86_64-apple-darwin16.4.0:||g;'`
OCTAVE_PATH="/Users/ram/git/lalsuite/_inst/lib/octave/4.2.0/site/oct/x86_64-apple-darwin16.4.0:$OCTAVE_PATH"
export LAL_DATADIR
LAL_DATADIR="/Users/ram/git/lalsuite/_inst/share/lal"
export PKG_CONFIG_PATH
PKG_CONFIG_PATH=`echo "$PKG_CONFIG_PATH" | /opt/local/bin/gsed -e 's|/Users/ram/git/lalsuite/_inst/lib/pkgconfig:||g;'`
PKG_CONFIG_PATH="/Users/ram/git/lalsuite/_inst/lib/pkgconfig:$PKG_CONFIG_PATH"
export PYTHONPATH
PYTHONPATH=`echo "$PYTHONPATH" | /opt/local/bin/gsed -e 's|/Users/ram/git/lalsuite/_inst/lib/python2.7/site-packages:||g;'`
PYTHONPATH="/Users/ram/git/lalsuite/_inst/lib/python2.7/site-packages:$PYTHONPATH"
export PATH
PATH=`echo "$PATH" | /opt/local/bin/gsed -e 's|/Users/ram/git/lalsuite/_inst/bin:||g;'`
PATH="/Users/ram/git/lalsuite/_inst/bin:$PATH"
export MANPATH
MANPATH=`echo "$MANPATH" | /opt/local/bin/gsed -e 's|/Users/ram/git/lalsuite/_inst/share/man:||g;'`
MANPATH="/Users/ram/git/lalsuite/_inst/share/man:$MANPATH"
export LAL_PREFIX
LAL_PREFIX="/Users/ram/git/lalsuite/_inst"
