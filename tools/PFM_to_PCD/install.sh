##linux
#build code 
#!/bin/bash

sudo apt-get update
sudo apt-get upgrade
mkdir build
cd build
rm -rf *
cmake .. 
make
sudo make install

