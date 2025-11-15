#! /bin/bash

cd ~/Documents/CudaTags

sudo apt-get update
sudo apt-get upgrade

bash install/pre.bash
bash install/wpi.bash
bash install/self.bash