mkdir ~/python      
cd ~/python
wget https://www.python.org/ftp/python/2.7.13/Python-2.7.13.tgz
tar zxfv Python-2.7.13.tgz
cd Python-2.7.13

./configure --prefix=$HOME/python
make && make altinstall
