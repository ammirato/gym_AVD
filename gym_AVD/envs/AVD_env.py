import gym
#from gym import error, utils
from gym.spaces import Discrete,Box
#from gym.utils import seeding
import os
import random
import sys
import json
import cv2

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms

class AVDEnv(gym.Env):
    '''
    Gym environment for Active Vision Dataset.

    Task: From an initial position, navigate as close as possible
          to a target object instance. 


    '''

    action_id_to_action_str = {0:'forward', 
                               1:'backward',
                               2:'rotate_cw',
                               3:'rotate_ccw'
                              }
    action_space = Discrete(4) 
    observation_space = Box(low=0,high=255,shape=(960,540,3)) 
    reward_range = None


    def __init__(self):
        self.scene_names = ''

    def _render(self, mode='human', close=False):
        print 'Hello, world, AVD render'  

    def _step(self, action):
        '''
            take an action in the environment.

            Valid actions: 'forward', 'backward', 'rotate_cw', 'rotate_ccw'
        '''


        action = self.action_id_to_action_str[action]
        next_img_name = self.current_scene_info[1][2][action]
        if next_img_name != '':
            img_index = int(next_img_name[5:11]) 
            self.current_scene_info = self.dataset[img_index-1]
            self.current_obs = [self.current_scene_info[0], self.target_imgs]
        reward = 0
        done = False
        info = {}
        if self.current_scene_info[1][1] in self.goal_img_names:
            reward = 1
            done = True

        return self.current_obs, reward, done, info
    def _reset(self):
        ''' 
        Reset environment.  Returns an initial observation
        
        chooses a new instance, scene, starting position, and goal

        '''
        if len(self.scene_names) == 0:
            print 'Call setup before using environment!'
            sys.exit(0)

        #choose a target and scene for this run
        #first choose scene, then an instance that is in that scene
        scene = random.choice(self.scene_names)
        scene_present_ids = self.get_scenes_instance_ids(scene)
        possible_inst_ids = set(scene_present_ids) & set(self.instance_ids)
        self.chosen_inst_id = random.choice(list(possible_inst_ids))

        #get scene images loader
        #only consider boxes from the chosen class
        pick_trans = AVD_transforms.PickInstances([self.chosen_inst_id],
                                                  max_difficulty=4)
        self.dataset = AVD.AVD(root=self.AVD_path,
                             scene_list=[scene],
                             target_transform=pick_trans,
                             classification=False,
                             class_id_to_name=self.id_to_name,
                             fraction_of_no_box=1)

        #set goal points...top 5 biggest boxes of target in sene
        annotations = json.load(open(os.path.join(self.AVD_path,scene,'annotations.json')))
        target_boxes = []
        for img_name in annotations.keys():
            boxes = annotations[img_name]['bounding_boxes']
            for box in boxes:
                if box[4] == self.chosen_inst_id:
                    area = (box[3]-box[1])*(box[2]-box[0])
                    target_boxes.append((area,img_name))
        target_boxes.sort(key=lambda tup: tup[0])
        num_goals = min(len(target_boxes), 5)
        goal_boxes = target_boxes[-1*num_goals:] 
        self.goal_img_names = [name for _,name in goal_boxes]

        #get target images
        chosen_target_paths = self.target_img_paths[self.chosen_inst_id]
        target_imgs = []
        for target_type in range(len(chosen_target_paths)):
            img = cv2.imread(random.choice(chosen_target_paths[target_type]))
            target_imgs.append(img)
        self.target_imgs = target_imgs 
            
        #pick random initial frame 
        self.current_scene_info = self.dataset[random.choice(range(len(self.dataset)))]
        scene_img = self.current_scene_info[0]
        
        self.current_obs = [scene_img,self.target_imgs]   
        return  self.current_obs





    def setup(self,scene_names='Home_001_1', 
                   instance_ids=[], 
                   AVD_path='', 
                   target_path=''):
        '''
        KWARGS for init:

            scene_names: '': a random single scene
                        'scene_name': the specified single scene
                        [x]: x random scenes (x is of type int) 
                        ['scene1','scene2',...]: specificed list of scenes

            instance_ids: -1: a random single instance
                         x: the specified single instance, x is of type int
                         [x]: x random instances (x is of type int)
                         [id1, id2, ...]: specified list of instances (int list)
                         []: all possible instances (based on scene choices)
            
            AVD_path: root directory of AVD dataset.

            target_path: root directory of target images


        '''
        #TODO: ensure every scene has at least one chosen instance

        self.AVD_path = AVD_path
        self.target_path = target_path
        #choose scene_names
        possible_scene_names = [name for name in os.listdir(AVD_path)
                                if os.path.isdir(os.path.join(AVD_path,name))]
        if type(scene_names) is list:
            if type(scene_names[0]) is int:
                random.shuffle(possible_scene_names)
                scene_names = possible_scene_names[0:scene_names[0]]
            else:
                assert type(scene_names[0]) is str
        else:
            assert type(scene_names) is str
            if len(scene_names) == 0:
                scene_names = [random.choice(possible_scene_names)]    
            else:
                scene_names = [scene_names]
        self.scene_names = scene_names

        #choose instance ids 
        self.id_to_name = self.get_class_id_to_name_dict(AVD_path)
        self.name_to_id = {v: k for k, v in self.id_to_name.iteritems()}
        all_instance_ids = self.id_to_name.keys()
        #get of objects in scenes chosen
        possible_instance_ids = set()
        for scene in self.scene_names:
            present_ids = self.get_scenes_instance_ids(scene)
            possible_instance_ids.update(present_ids) 
 
        if type(instance_ids) is list:
            if len(instance_ids) == 0:
                instance_ids = possible_instance_ids 
            elif len(instance_ids) == 1:
                random.shuffle(possible_instance_ids)
                instance_ids = possible_instance_ids[0:instance_ids[0]]
            else:
                assert type(instance_ids[1]) is int
        else:
            assert type(instance_ids) is int 
            if instance_ids < 0:
                instance_ids = [random.choice(possible_instance_ids)]    
            else:
                instance_ids = [instance_ids]
        self.instance_ids = instance_ids



        #get target image paths for all target images
        #type of target image can mean different things, 
        #probably different type is different view
        target_paths = os.listdir(target_path)
        target_paths.sort()
        target_imgs = {}
        #each target gets a list of lists, one for each type dir
        for inst_id in instance_ids:
            target_imgs[inst_id] = []
        for type_ind, t_dir in enumerate(target_paths):
            for name in os.listdir(os.path.join(target_path,t_dir)):
                if name.find('N') == -1: 
                    obj_id = self.name_to_id[name[:name.rfind('_')]]
                else:
                    obj_id = self.name_to_id[name[:name.find('N')-1]]
                #make sure object is valid, and store path
                if obj_id in instance_ids:
                    if len(target_imgs[obj_id]) <= type_ind:
                        target_imgs[obj_id].append([])
                    target_imgs[obj_id][type_ind].append(
                                            os.path.join(target_path,t_dir,name))
        self.target_img_paths = target_imgs







    def get_class_id_to_name_dict(self,root,file_name='all_instance_id_map.txt'):
        """
        Returns a dict from integer class id to string name
        """
        map_file = open(os.path.join(root,file_name),'r')
        id_to_name_dict = {}
        for line in map_file:
            line = str.split(line)
            id_to_name_dict[int(line[1])] = line[0]
        return id_to_name_dict


    def get_scenes_instance_ids(self,scene_name):
        in_file = open(os.path.join(self.AVD_path,scene_name,'present_instance_names.txt'))
        ids = []
        for line in in_file:
            ids.append(self.name_to_id[line.split()[0]])
        return ids 

