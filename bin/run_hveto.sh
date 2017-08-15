ifo=$1
label=$2
gpsStart=$3
gpsEnd=$4

path=`pwd`

dur=$(expr ${gpsEnd} - ${gpsStart})

filter_psql -d $ifo -l $label -o $path/$ifo-triggers-${gpsStart}-${dur}.csv -f segs_${gpsStart}_${gpsEnd}.dat -p 1 -s $gpsStart -e $gpsEnd --database O1GlitchClassificationUpdate

ls $path/$ifo-triggers-${gpsStart}-${dur}.csv | lalapps_path2cache > cache.lcf

hveto $gpsStart $gpsEnd --ifo $ifo --config-file h1l1-hveto-daily-o2.ini -p cache.lcf -o ~/public_html/HVeto/$ifo/$label/$gpsStart-$gpsEnd --nproc 32 --omega-scans 5

