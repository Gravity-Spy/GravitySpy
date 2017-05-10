ifo=$1
label=$2
gpsStart=$3
gpsEnd=$4

path=`pwd`

python filter_psql.py -d $ifo -l $label -o triggers_${gpsStart}_${gpsEnd}.csv -f segs_${gpsStart}_${gpsEnd}.dat -p 1 -s $gpsStart -e $gpsEnd

ls $path/triggers_${gpsStart}_${gpsEnd}.csv > cache.lcf

hveto $gpsStart $gpsEnd --ifo $ifo --config-file h1l1-hveto-daily-o2.ini -p cache.lcf -o ~/public_html/HVeto/$ifo/$label/$gpsStart-$gpsEnd
