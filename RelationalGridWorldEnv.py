# Import Modules
import sys
import time
import os
import numpy as np
from gym.envs.toy_text import discrete
import pygame

# Define colors and import icons
dir_path = os.path.dirname(os.path.realpath(__file__))

white = (255,255,255)
black = (0,0,0)

pit_img = pygame.image.load(dir_path + '/icon/pit.png')
mountain_img = pygame.image.load(dir_path + '/icon/mountain.png')
teleport_img = pygame.image.load(dir_path + '/icon/teleport.png')
wall_img = pygame.image.load(dir_path + '/icon/wall.png')
sword_img = pygame.image.load(dir_path + '/icon/sword.png')
terminal_img = pygame.image.load(dir_path + '/icon/terminal.png')
enemy_img = pygame.image.load(dir_path + '/icon/enemy.png')
agent_img = pygame.image.load(dir_path + '/icon/agent.png')

# Define actions
up    = 0
right = 1
down  = 2
left  = 3

# Define environment class
class RGW(discrete.DiscreteEnv): 

    # Define boundaries of observations
    def env_bound(self, pos):
        x = min(pos[0], self.shape[0] - 1)
        x = max(x, 0)
        y = min(pos[1], self.shape[1] - 1)
        y = max(y, 0)
        return np.array([x,y])

    # Define state connections, object states and rewards
    def env_dynamics(self, current, delta, state, nS_pos):
        
        ## Rewards for states 
        r_trans    =    0    # transition reward   
        r_pit      =   -1    # pit reward
        r_mountain = -0.1    # mountain transtion reward
        r_enemy    =   -1    # enemy reward
        r_kill     =    1    # kill reward
        r_term     =    10   # term reward

        new_position = np.array(current) + np.array(delta)
        new_position = self.env_bound(new_position).astype(int)

        obj_id = 0           # empty grid id

        ## Don't change position if new position is on wall
        if self._wall[tuple(new_position)]:
            obj_id = 6
            new_position = np.array(current)    

        ## Teleportation
        if np.sum(new_position == self.teleport_in) == 2:
            obj_id = 3
            new_position = self.teleport_out
        elif np.sum(new_position == self.teleport_out) == 2:
            obj_id = 3
            new_position = self.teleport_in

       ## Sword
        if state >= nS_pos:
            new_state = np.ravel_multi_index(tuple(new_position), self.shape) + nS_pos

        else:
            new_state = np.ravel_multi_index(tuple(new_position), self.shape)
            if self._sword[tuple(new_position)]:
                new_state = new_state + nS_pos
                obj_id = 1

        ## State rewarding
        if self._pit[tuple(new_position)]:
            reward = r_pit
            obj_id = 7

        elif  self._mountain[tuple(new_position)]:
            obj_id = 2
            reward = r_mountain

        elif self._enemy[tuple(new_position)]:
            obj_id = 4
            if state > nS_pos:
                reward = r_kill
            else:
                reward = r_enemy

        else:
            reward = r_trans

        ## Env restarts when the state is terminal
        if tuple(new_position) == self.pos_term:
            reward = r_term
            obj_id = 5
        is_done = self._pit[tuple(new_position)] or (tuple(new_position) == self.pos_term)
        return [(obj_id, new_state, reward, is_done)]  

    def __init__(self): 
        ## Grid size (MxM)  
        self.M = 10       
        self.shape = (self.M, self.M)
        
        ## Define state space and action space 
        nS_p = np.prod(self.shape)     # positional state space
        nS_s = 2                       # memory states (e.g sword availability[False, True])
        nS =  nS_p * nS_s              # state space      
        nA = 4                         # action space

        ## Starting point
        self.pos_0 = (0,0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index(self.pos_0, self.shape)] = 1.0

        ## Terminal position  
        self.pos_term = (9,0)

        ## Pit positions
        self._pit = np.zeros(self.shape, dtype=np.bool)
        self._pit[7, 8] = True
        self._pit[8, 9] = True

        ## Wall positions
        self._wall = np.zeros(self.shape, dtype=np.bool)
        self._wall[6, 0:8] = True
        self._wall[4:6, 3] = True
        self._wall[8, 0:2] = True

        ## Teleport positions
        self._teleport = np.zeros(self.shape, dtype=np.bool)
        self.teleport_in = np.array([3,9])   # teleport in
        self.teleport_out = np.array([8,8])  # teleport out
        self._teleport[self.teleport_in[0], self.teleport_in[1]] = True
        self._teleport[self.teleport_out[0], self.teleport_out[1]] = True  

        ## Mountain positions
        self._mountain = np.zeros(self.shape, dtype=np.bool)
        self._mountain[0, 5] = True
        self._mountain[0, 7:10] = True
        self._mountain[1, 6:10] = True
        self._mountain[2, 7:10] = True
        self._mountain[3, 8:10] = True
        self._mountain[4, 9:10] = True

        ## Sword positions
        self._sword = np.zeros(self.shape, dtype=np.bool)
        self._sword[0, 6] = True

        ## Enemy positions
        self._enemy = np.zeros(self.shape, dtype=np.bool)
        self._enemy[9, 1] = True

        ## Create screenshots folder and initialize screenshot counter
        if not os.path.exists('screenshot'):
            os.makedirs('screenshot')
        self.folder_name = time.strftime("%Y%m%d-%H%M%S")

        self.screenshot_n = 0

        self.rect_len = 1000/self.M
        self.screen = pygame.Surface((1000,1000))
        self.screen.fill(white)

        ## Calculate states and corresponding rewards
        E = {}
        for s in range(nS):
            position = np.unravel_index(s%nS_p, self.shape)
            E[s] = { a : [] for a in range(nA) } 
            E[s][up] = self.env_dynamics(position, [-1, 0], s, nS_p)
            E[s][right] = self.env_dynamics(position, [0, 1], s, nS_p)
            E[s][down] = self.env_dynamics(position, [1, 0], s, nS_p)
            E[s][left] = self.env_dynamics(position, [0, -1], s, nS_p)

        super(RGW, self).__init__(nS, nA, E, isd)
    
    # Assign specific features to objects
    def mygrey(self):

            state_matrix = np.zeros([100, 1])
            
            for s in range(int(self.nS/2)):
                position = np.unravel_index(s, self.shape)
                if self.s%(self.nS/2) == s:  
                    state_matrix[s,0] = 128              

                elif position == self.pos_term:
                    state_matrix[s,0] =  255        

                elif self._teleport[position]:
                    state_matrix[s,0] = 200        

                elif self._sword[position]:
                    if self.s >= (self.nS/2): 
                        state_matrix[s,0] = 220     
                    else:
                        state_matrix[s,0] = 75      

                elif self._enemy[position]:
                    state_matrix[s,0] = 175         

                elif self._mountain[position]:
                    state_matrix[s,0] = 150         

                elif self._pit[position]: 
                    state_matrix[s,0] = 30          

                elif self._wall[position]:   
                    state_matrix[s,0] = 0           

                else:
                    state_matrix[s,0] = 220         
           
            return state_matrix

    def myrender_rgb(self):
        ## Pygame visualization

        for i in range(self.M):
            for j in range(self.M):
                pygame.draw.rect(self.screen, black, (i*self.rect_len,j*self.rect_len,self.rect_len,self.rect_len), 2)
   
        for s in range(int(self.nS/2)):
            position = np.unravel_index(s, self.shape)
            if self.s%(self.nS/2) == s:  
                self.screen.blit(agent_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+10))

            elif position == self.pos_term:
                self.screen.blit(terminal_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+10))
               
            elif self._teleport[position]:
                self.screen.blit(teleport_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+5))
            
            elif self._sword[position]:
                if self.s >= (self.nS/2): 
                    pygame.draw.rect(self.screen, white, (position[1]*self.rect_len+10,position[0]*self.rect_len+10,self.rect_len-20,self.rect_len-20), 0)
                  
                else:
                    self.screen.blit(sword_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+10))
                    
            elif self._enemy[position]:
                self.screen.blit(enemy_img,(position[1]*self.rect_len+20,position[0]*self.rect_len+10))

            elif self._mountain[position]:
                self.screen.blit(mountain_img,(position[1]*self.rect_len,position[0]*self.rect_len))
                
            elif self._pit[position]: 
                self.screen.blit(pit_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+10))
               
            elif self._wall[position]:   
                self.screen.blit(wall_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+10))

        imgdata = pygame.surfarray.array3d(self.screen)
        imgdata = imgdata.swapaxes(0,1)

        return imgdata

    #   Rendering
    def myrender(self):
        ## Initialize pygame visualization
        screen = pygame.display.set_mode((1000,1000))
        screen.fill(white)

        ## Update visual according to state of agent
        ## and paste icons on it
        for i in range(self.M):
            for j in range(self.M):
                pygame.draw.rect(screen, black, (i*self.rect_len,j*self.rect_len,self.rect_len,self.rect_len), 2)
   
        for s in range(int(self.nS/2)):
            position = np.unravel_index(s, self.shape)
            if self.s%(self.nS/2) == s:  
                screen.blit(agent_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+10))

            elif position == self.pos_term:
                screen.blit(terminal_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+10))
               
            elif self._teleport[position]:
                screen.blit(teleport_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+5))
            
            elif self._sword[position]:
                if self.s >= (self.nS/2): 
                    pygame.draw.rect(screen, white, (position[1]*self.rect_len+10,position[0]*self.rect_len+10,self.rect_len-20,self.rect_len-20), 0)
                  
                else:
                    screen.blit(sword_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+10))
                    
            elif self._enemy[position]:
                screen.blit(enemy_img,(position[1]*self.rect_len+20,position[0]*self.rect_len+10))

            elif self._mountain[position]:
                screen.blit(mountain_img,(position[1]*self.rect_len,position[0]*self.rect_len))
                
            elif self._pit[position]: 
                screen.blit(pit_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+10))
               
            elif self._wall[position]:   
                screen.blit(wall_img,(position[1]*self.rect_len+10,position[0]*self.rect_len+10))

        pygame.display.update() 
        self.screenshot_n += 1
            
        rect = pygame.Rect(0, 0, 1000, 1000)
        sub = screen.subsurface(rect)

        # Save screenshots
        if not os.path.exists('screenshot/'+self.folder_name):
            os.makedirs('screenshot/'+self.folder_name)
        pygame.image.save(sub, "screenshot/"+self.folder_name+"/screenshot_"+str(self.screenshot_n)+".jpg")
        
        return

    def stopmyrender(self):
        pygame.quit()