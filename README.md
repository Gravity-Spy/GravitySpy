# GravitySpy
This repo contains code for the GravitySpy Citizen Science project.

RUN source /home/detchar/opt/gwpysoft/etc/gwpy-user-env.sh
RUN mkdir -p /home/$username$/detchar/dmt_wplot/
RUN cp /home/scoughlin/detchar/GlitchZoo/dmt_wplot/*.py /home/$username$/detchar/dmt_wplot/
RUN cp /home/scoughlin/detchar/GlitchZoo/dmt_wplot/*.jar /home/$username$/detchar/dmt_wplot/
RUN source /home/scoughlin/detchar/GlitchZoo/dmt_wplot/setup_paths.sh
1.) RUN python read_omicron_triggers.py --gpsStart 1126100000 --gpsEnd 1127800000
2.) RUN python read_omicron_triggers.py --gpsStart 1127800000 --gpsEnd 1129500000
3.) RUN python read_omicron_triggers.py --gpsStart 1129500000 --gpsEnd 1131400000 
