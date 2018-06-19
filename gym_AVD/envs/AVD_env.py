import gym
from gym.spaces import Discrete,Box, Dict
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

    scene_img_shape = (1080,1920,3)
    target_img_shape = (100,100,3)
    observation_space = Dict({'scene_image': Box(low=0,high=255,
                                             shape=scene_img_shape,dtype=np.uint8),
                              'target_image': Box(low=0,high=255,
                                             shape=target_img_shape, dtype=np.uint8)})
        
    
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

        self.num_steps +=1 
        action = self.action_id_to_action_str[action]
        next_img_name = self.current_scene_info[1][2][action]
        if next_img_name != '':
            img_index = int(next_img_name[5:11]) 
            self.current_scene_info = self.dataset[img_index-1]
            self.current_obs = {'scene_image':self.current_scene_info[0],
                                'target_image': self.target_imgs}

        reward= -.1
        done = False
        info = {}

        if self.current_scene_info[1][1] in self.goal_img_names:
            reward = 1
            done = True
        if self.num_steps >= self.max_steps:
            done =True

        if done and self.reset_on_done:
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
        if self.choose_sequentially:
            try:
                scene = self.scene_names[self.current_scene_ind]
            except:
                #we are done with all scenes
                return -1
            #update init pos
            self.initial_positions = json.load(open(os.path.join(self.AVD_path,
                                                                scene,
                                                                'AOS_initial_positions.json')))
            scene_present_ids = list(set(self.get_scenes_instance_ids(scene)) & set(self.instance_ids))
            self.chosen_inst_id = scene_present_ids[self.current_instance_ind]
            starting_poses = self.initial_positions[str(self.chosen_inst_id)] 
            try:
                self.current_init_pos_ind +=1
                starting_name = starting_poses[self.current_init_pos_ind]
            except:
                #we are done with this instance
                try:
                    self.current_instance_ind += 1
                    self.chosen_inst_id = scene_present_ids[self.current_instance_ind]
                    
                except:
                    #we have done all instances for this scene
                    self.current_scene_ind += 1                     
                    try:
                        scene = self.scene_names[self.current_scene_ind]
                    except:
                        #we are done with all scenes
                        return -1
                    self.current_instance_ind = 0 
                    self.current_init_pos_ind = 0 
                    self.reset()
 
                starting_poses = self.initial_positions[str(self.chosen_inst_id)] 
                self.current_init_pos_ind = 0
                starting_name = starting_poses[self.current_init_pos_ind]

        else: #choose randomly
            scene = random.choice(self.scene_names)
            scene_present_ids = self.get_scenes_instance_ids(scene)
            ids_with_imgs = self.get_instance_ids_with_target_images()
            possible_inst_ids = set(scene_present_ids) & set(self.instance_ids) & set(ids_with_imgs)
            self.chosen_inst_id = random.choice(list(possible_inst_ids))
            #pick initial frame 
            self.initial_positions = json.load(open(os.path.join(self.AVD_path,
                                                                scene,
                                                                'AOS_initial_positions.json')))
            starting_poses = self.initial_positions[str(self.chosen_inst_id)] 
            starting_name = random.choice(starting_poses)

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

        self.destination_imgs = json.load(open(os.path.join(self.AVD_path,
                                                            scene,
                                                            'destination_images.json')))
        self.goal_img_names = self.destination_imgs[str(self.chosen_inst_id)]
        for ind,img_name in enumerate(self.goal_img_names):
            self.goal_img_names[ind] = str(img_name)

        #get target images
        try:
            chosen_target_paths = self.target_img_paths[self.chosen_inst_id]
            target_imgs = []
            for target_type in range(len(chosen_target_paths)):
                img = cv2.imread(random.choice(chosen_target_paths[target_type]))
                target_imgs.append(img)
            self.target_imgs = self.resize_target_images(target_imgs,size=self.target_img_shape)
        except:
            self.target_imgs = np.zeros((2,) + self.target_img_shape)
               
        self.observation_space = Dict({'scene_image': Box(low=0,high=255,
                                                         #shape=(960,540,3),
                                                         shape=self.scene_img_shape,
                                                         dtype=np.uint8),
                                        'target_image': Box(low=0,high=255,
                                                 shape=self.target_imgs.shape,
                                                 dtype=np.uint8)})
  
        starting_index = self.dataset.get_name_index(starting_name)
        self.current_scene_info = self.dataset[starting_index]

        scene_img = self.current_scene_info[0]
        self.current_obs = {'scene_image': scene_img,
                            'target_image': self.target_imgs}

        self.num_steps = 0
        return  self.current_obs


    def setup(self,scene_names='Home_001_1', 
                   instance_ids=[], 
                   AVD_path='', 
                   target_path=None,
                   choose_sequentially=False,
                   reset_on_done=False,
                   max_steps=3000):
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
            choose_sequentially (optionial): (bool) if this env should go
                                             sequentially through all combinations 
                                             of scene/instance/starting-position.
                                             if False, on env.reset(), and random
                                             combo is chosen. 
            max_steps: max number of steps before done
        '''
        #TODO: ensure every scene has at least one chosen instance

        self.AVD_path = AVD_path
        self.target_path = target_path
        self.max_steps = max_steps
        self.choose_sequentially = choose_sequentially
        self.reset_on_done = reset_on_done
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
        if target_path is not None:
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
        else:
            self.target_img_paths = {} 
        self.num_steps = 0

        if self.choose_sequentially:
            self.finished_scenes = []
            self.finished_instances = []
            self.finished_init_pos_inds = []
            self.current_scene_ind = 0
            self.current_instance_ind = 0 
            self.current_init_pos_ind  = -1 
        return None

    def get_current_env_info(self):
        '''
        Returns [scene_name, instance_id, initial_position_img_name]
        '''
        try:
            return [self.scene_names[self.current_scene_ind], self.chosen_inst_id,
                    self.initial_positions[str(self.chosen_inst_id)][self.current_init_pos_ind]]
        except:
            return -1



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
        in_file = open(os.path.join(self.AVD_path,scene_name,'instances_for_AOS.txt'))
        ids = []
        for line in in_file:
            ids.append(int(line.split()[0]))
        return ids 

    def get_instance_ids_with_target_images(self):
        ids = []
        for k in self.target_img_paths.keys():
            if len(self.target_img_paths[k]) > 0:
                ids.append(k)
        return ids


    def resize_target_images(self,img_list, size=[100,100], random_bg=False):
        """
        Stacks image in a list into a single ndarray 
        Input parameters:
            img_list: (list) list of ndarrays, images to be resized and stacked. 
            size (optional): ([int,int]) size of resized target images. Default [100,100]
        Returns:
            (ndarray) a single ndarray with first dimension equal to the 
            number of elements in the inputted img_list    
        """

        #resize and stack the images
        for il,img in enumerate(img_list):
            max_dim = max(img.shape)
            scale = 100.0/max_dim
            img = cv2.resize(img,(0,0),fx=scale,fy=scale)

            if random_bg:
                resized_img = np.random.randint(0,high=255,size=(size[0],size[1],img.shape[2]))
            else:
                if img.mean() < 127:
                    resized_img = 255*np.ones((size[0],size[1],img.shape[2]))
                else:
                    resized_img = np.zeros((size[0],size[1],img.shape[2]))
            resized_img[0:img.shape[0],0:img.shape[1],:] = img
            img_list[il] = resized_img
        return np.stack(img_list,axis=0) 


