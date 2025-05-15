import os
import sys
from icedata3d_util import DATA_PATH, collect_point_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

ice_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/ice_paths.txt'))]
print(ice_paths)
print('\n')
# ice_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/ice_paths.txt'))]
ice_paths = [os.path.join(DATA_PATH, p) for p in ice_paths]
print(ice_paths)
output_folder = os.path.join(ROOT_DIR, 'data/icedata_3d')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for ice_path in ice_paths:
    print('icepath=.  ',ice_path)
    try:
        elements = ice_path.split('/')
        print('elements=.  ',elements)
        out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
        print('out_filename=.  ',out_filename)
        print(os.path.join(output_folder, out_filename), '    done\n')
        collect_point_label(ice_path, os.path.join(output_folder, out_filename), 'numpy')
        
    except:
        print(ice_path, 'ERROR!!')
