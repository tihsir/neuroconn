#! /bin/bash

username=$USER

echo Copying working directory into your local directory
cp -r neuroconn ../../home/$USER/

chown -r $USER ../../home/$USER/neuroconn/

cd ../../home/$USER
