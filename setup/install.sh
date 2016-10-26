git clone https://github.com/gwpy/gwpysoft /home/scoughlin/src/O2/gwpysoft
cd /home/scoughlin/src/O2/gwpysoft
./gwpysoft-init /home/scoughlin/opt/O2/GravitySpy ./packages.txt
cd /home/scoughlin/opt/O2/GravitySpy/
bin/pip install --upgrade git+https://github.com/gwpy/gwpy
bin/pip install scikit-image
bin/pip install Theano
bin/pip install keras
bin/pip install Jinja2
bin/pip install pandas
bin/pip install git+git://github.com/zooniverse/panoptes-python-client.git
