import gym
#from gym import error, utils
from gym.spaces import Discrete,Box, Dict
#from gym.utils import seeding
import os
import random
import sys
import json
import cv2
import numpy as np

import active_vision_dataset_processing.data_loading.active_vision_dataset as AVD
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

    scene_img_shape = (405,720,3)
    observation_space = Dict({'scene_image': Box(low=0,high=255,
                                             #shape=(540,960,3),dtype=np.uint8),
                                             shape=scene_img_shape,dtype=np.uint8),
                              'target_image': Box(low=0,high=255,
                                             shape=(100,100,3), dtype=np.uint8)})
        
    
    target_data = None                
    reward_range = None
    _seed = None

    def __init__(self):
        self.scene_names = ''

    def render(self, mode='human'):
        print('Hello, world, AVD render')
    def close(self):
        print('Hello, world, AVD close')

    def step(self, action):
        '''
            take an action in the environment.

            Valid actions: 'forward', 'backward', 'rotate_cw', 'rotate_ccw'
        '''
        r_mult = 50

        self.num_steps +=1 
        action = self.action_id_to_action_str[action]
        next_img_name = self.current_scene_info[1][2][action]
        if next_img_name != '':
            img_index = int(next_img_name[5:11]) 
            self.current_scene_info = self.dataset[img_index-1]
            self.current_obs = {'scene_image':self.current_scene_info[0],
                                'target_image': self.target_imgs}

        box = self.current_scene_info[1][0]
        area = 0
        if len(box) > 0:
            box = box[0]
            area = (box[3]-box[1]) * (box[2]-box[0])
        #reward = (area/self.max_area)  * r_mult
        reward = (1-(area/self.max_area)) * -.1
        done = False
        info = {}
        if self.current_scene_info[1][1] in self.goal_img_names:
            reward = 1*r_mult
            done = True

        if self.num_steps >= self.max_steps:
            done =True

        if done:
            self.current_obs = self.reset()


        return self.current_obs, reward, done, info
    def reset(self):
        ''' 
        Reset environment.  Returns an initial observation
        
        chooses a new instance, scene, starting position, and goal

        '''
        if len(self.scene_names) == 0:
            print('Call setup before using environment!')
            sys.exit(0)

        #choose a target and scene for this run
        #first choose scene, then an instance that is in that scene
        scene = random.choice(self.scene_names)
        scene_present_ids = self.get_scenes_instance_ids(scene)
        ids_with_imgs = self.get_instance_ids_with_target_images()
        possible_inst_ids = set(scene_present_ids) & set(self.instance_ids) & set(ids_with_imgs)
        self.chosen_inst_id = random.choice(list(possible_inst_ids))

        #get scene images loader
        #only consider boxes from the chosen class
        pick_trans = AVD_transforms.PickInstances([self.chosen_inst_id],
                                                  max_difficulty=4)
        resize_trans = AVD_transforms.ResizeImage((self.scene_img_shape[0],self.scene_img_shape[1]))
        self.dataset = AVD.AVD(root=self.AVD_path,
                             scene_list=[scene],
                             transform=resize_trans,
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
        self.max_area = target_boxes[-1][0]

        #get target images
        chosen_target_paths = self.target_img_paths[self.chosen_inst_id]
        #if not(len(chosen_target_paths) > 0):
        #    print(self.chosen_inst_id)
        #assert(len(chosen_target_paths)>0)
        target_imgs = []
        for target_type in range(len(chosen_target_paths)):
            img = cv2.imread(random.choice(chosen_target_paths[target_type]))
            target_imgs.append(img)
        #assert len(target_imgs)>0, '{}'.format(self.chosen_inst_id)
        self.target_imgs = self.match_and_concat_images_list(target_imgs)
               
        self.observation_space = Dict({'scene_image': Box(low=0,high=255,
                                                         shape=(960,540,3),
                                                         dtype=np.uint8),
                                        'target_image': Box(low=0,high=255,
                                                 shape=self.target_imgs.shape,
                                                 dtype=np.uint8)})
  
        #pick random initial frame 
        self.current_scene_info = self.dataset[random.choice(range(len(self.dataset)))]
        scene_img = self.current_scene_info[0]
        
        self.current_obs = {'scene_image': scene_img,
                            'target_image': self.target_imgs}

        self.num_steps = 0
        return  self.current_obs





    def setup(self,scene_names='Home_001_1', 
                   instance_ids=[], 
                   AVD_path='', 
                   target_path='',
                   max_steps=300):
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

            max_steps: max number of steps before done
        '''
        #TODO: ensure every scene has at least one chosen instance

        self.AVD_path = AVD_path
        self.target_path = target_path
        self.max_steps = max_steps
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
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}
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

    def get_instance_ids_with_target_images(self):
        ids = []
        for k in self.target_img_paths.keys():
            if len(self.target_img_paths[k]) > 0:
                ids.append(k)
        return ids

    def match_and_concat_images_list(self,img_list, min_size=None):
        """ 
        Stacks image in a list into a single ndarray 

        Input parameters:
            img_list: (list) list of ndarrays, images to be stacked. If images
                      are not the same shape, zero padding will be used to make
                      them the same size. 

            min_size (optional): (int) If not None, ensures images are at least
                                 min_size x min_size. Default: None 

        Returns:
            (ndarray) a single ndarray with first dimension equal to the 
            number of elements in the inputted img_list    
        """
        #find size all images will be
        max_rows = 0 
        max_cols = 0 
        for img in img_list:
            max_rows = max(img.shape[0], max_rows)
            max_cols = max(img.shape[1], max_cols)
        if min_size is not None:
            max_rows = max(max_rows,min_size)
            max_cols = max(max_cols,min_size)

        #resize and stack the images
        for il,img in enumerate(img_list):
            resized_img = np.zeros((max_rows,max_cols,img.shape[2]))
            resized_img[0:img.shape[0],0:img.shape[1],:] = img 
            img_list[il] = resized_img
        return np.stack(img_list,axis=0)


