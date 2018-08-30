import os
from gqcnn import GQCNN, SGDOptimizer, GQCNNAnalyzer
from autolab_core import YamlConfig


""" Training config templates
       cfg/tools/training.yaml
       cfg/tools/analyze_gqcnn_performance.yaml 
       cfg/tools/gqcnn_prediction_visualizer.yaml
"""

print('==============================')

# Configuration for the training
configfile = '/home/simon/sandbox/graspitmod/catkin_ws/src/gqcnn/custom/trainingConfig.yaml' 
train_config = YamlConfig('/home/simon/sandbox/graspitmod/catkin_ws/src/gqcnn/custom/trainingConfig.yaml')
gqcnn_config = train_config['gqcnn_config']

# refine existing...
existing_set = os.environ.get('gqcnn_train_main_modeldir')
if existing_set is not None and os.path.isdir(existing_set) and os.path.exists(existing_set):
    print('loading existing set: ' + existing_set)
    gqcnn = GQCNN.load(existing_set)
else:
    print('making new set')
    gqcnn = GQCNN(gqcnn_config)


SGDOptimizer = SGDOptimizer(gqcnn, train_config)

with gqcnn.get_tf_graph().as_default():
     SGDOptimizer.optimize()
