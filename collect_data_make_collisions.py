import os,sys,random,time
import carla
import numpy as np
import cv2

#settings
IM_W,IM_H = (210,140)
time_step = 0.1
image_save_path = r'/home/riset/Documents/script-alison/data_make_collision' #change this
seq_len = 8
# how much sequences we want to get?
num_of_seq = 270 					#yang ini 
max_sample = 8
number_env_vehicles = 100
#if not os.path.exists(os.path.join(image_save_path)):
#    os.makedirs(os.path.join(image_save_path))
#create main carla objects
client = carla.Client('10.6.46.136',2000) #change this
client.set_timeout(20)

world = client.load_world('Town10HD')

blueprint_library = world.get_blueprint_library()

class Carla_session:
    def __init__(self):
        self.actors = []
        self.vehicle = []
        self.counter = 0
        self.n_seq = len(os.listdir(image_save_path))
        self.n_seq = 2954 				#yang ini juga
        self.n_eps = 0
        self.collision_flag = False
        self.episode_images = []
        self.track_cleanup =[]
        self.env_actors = []

    def add_vehicles(self):
        env_vehicles_bp = blueprint_library.filter('vehicle.*')
        env_vehicles_bp = [x for x in env_vehicles_bp if int(x.get_attribute('number_of_wheels')) == 4]
        env_vehicles_bp = [x for x in env_vehicles_bp if not x.id.endswith('isetta')]
        env_vehicles_bp = [x for x in env_vehicles_bp if not x.id.endswith('carlacola')] 
        spawn_points = world.get_map().get_spawn_points()      
        self.env_actors = []
        num_vehicle = 0
        for n, transform in enumerate(spawn_points):
            if num_vehicle >= number_env_vehicles:
                break
            env_vehicle_bp = random.choice(env_vehicles_bp)
            if env_vehicle_bp.has_attribute('color'):
                env_vehicle_bp.set_attribute('color', random.choice(env_vehicle_bp.get_attribute('color').recommended_values))
            env_vehicle_bp.set_attribute('role_name', 'autopilot')
            try:
                env_vehicle = world.spawn_actor(env_vehicle_bp,transform)
            except:
                continue
            env_vehicle.set_autopilot(True)
            self.env_actors.append(env_vehicle)
            num_vehicle+=1

    def add_actors(self):
        #set vehicle
        vehicle_bp = blueprint_library.find('vehicle.nissan.micra')
        not_created = True
        while not_created:
            try:
                start_point = random.choice(world.get_map().get_spawn_points())
                self.vehicle = world.spawn_actor(vehicle_bp,start_point)
                not_created = False
            except Exception as e:
                print(f"{e}. recreate the actor")

        #get and set sensors
        collision_sensor_bp = blueprint_library.find('sensor.other.collision')
        camera_sensor_bp = blueprint_library.find('sensor.camera.rgb')
        camera_sensor_bp.set_attribute('image_size_x',str(IM_W))
        camera_sensor_bp.set_attribute('image_size_y',str(IM_H))
        camera_sensor_bp.set_attribute('sensor_tick',str(time_step))
        camera_sensor_bp.set_attribute('fov',str(100))

        sensor_location = carla.Transform(carla.Location(x=0,y=0,z=2))
        self.camera = world.spawn_actor(camera_sensor_bp, sensor_location, attach_to = self.vehicle)
        self.collision_sensor = world.spawn_actor(collision_sensor_bp, sensor_location, attach_to = self.vehicle)
        self.actors.extend([self.vehicle,self.camera,self.collision_sensor])
        self.camera.listen(lambda image: self.add_image(image))
        self.collision_sensor.listen(lambda collision: self.end_seq(collision,'collision'))  

    def start_new_seq(self):
        self.add_actors()
        self.collision_flag = False
        print('starting new seq')
        self.counter = 0
        #self.n_seq += 1
        self.track_cleanup.append(self.n_seq)
        path = os.path.join(image_save_path, str(self.n_seq))
        if not os.path.exists(path):
            os.makedirs(path)

    def add_image(self,image):
        self.counter += 1
        img = np.reshape(image.raw_data,(IM_H,IM_W,4))
        img = img[:,:,:3][:]
        path = os.path.join(image_save_path, str(self.n_seq))
        cv2.imwrite(os.path.join(path,'{}.png'.format(self.counter)),img)
        

    def delete_images(self):
        imagestodelete = self.counter-seq_len
        path = os.path.join(image_save_path,str(self.n_seq))
        for i in range(imagestodelete):
            os.remove(os.path.join(path,f"{i+1}.png"))

    def save_images(self):
        for ind,img in enumerate(self.episode_images[-seq_len:]):
            cv2.imwrite(os.path.join(image_save_path,str(self.n_seq),'{}.png'.format(ind)),img)
    
    def end_seq(self,cause_obj,cause):
        print("collision happened")
        print(f"caused by: {cause_obj.other_actor}")
        self.collision_flag =True
        print(f"num of imgs: {self.counter}. to be deleted: {self.counter-seq_len}")
        self.delete_images()
        self.destroy_actors()
        print(f"seq {self.n_seq} end")

    def destroy_actors(self):
        for actor in self.actors:
            actor.destroy()
        self.actors = []
    
    def get_directions(self):
        thr = random.choice([0.8,0.7,0.6])
        steer = random.choice([-0.3,0.0,0.0,0.0,0.3,0.1,-0.1])
        return carla.VehicleControl(thr,steer)  
       
    def drive_around(self,num_of_seq):
        self.add_vehicles()
        for j in range(num_of_seq):
            self.n_seq+=1
            print(f"seq {self.n_seq}")
            try:
                self.start_new_seq()
                while not self.collision_flag:
                    self.vehicle.apply_control(self.get_directions())
                    if self.collision_flag:
                        break
                        
                    if self.vehicle.is_at_traffic_light():
                        traffic_light = self.vehicle.get_traffic_light()
                        if traffic_light.get_state() == carla.TrafficLightState.Red:
                            traffic_light.set_state(carla.TrafficLightState.Green)
                    time.sleep(1)
            except Exception as e:
                print(e)
        self.env_actors =[]


c = Carla_session()
c.drive_around(num_of_seq)
