#!/bin/bash
cp -R game/src/game/* submission/game 
cp mcts/src/mcts/*.py submission/mcts
cp model/src/model/*.py submission/model 
cp agent/src/agent/*.py submission/agent
cp logger/src/logger/*.py submission/logger
cp resources/models/best_model_p* submission/resources/models 
cd submission

file_name="submission_${1}.tar.gz"
# read file_name
tar cvfz $file_name *
mv $file_name ..
cd ..
