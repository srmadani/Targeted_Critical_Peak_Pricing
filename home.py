import numpy as np
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import pandas as pd
import pickle
from itertools import product
pd.set_option('display.max_columns', None)
import random
random.seed(0)


class Homes(Env):
    def __init__(self, tar= False, option= 'WCO', rate = 99):
        self.option = option.upper()
        self.tar = tar
        file = open('datasets/db.pkl', 'rb')
        db = pickle.load(file)
        self.d = db['d']
        self.df = db['data']
        # self.d['PC'] = self.d['PC'] #* 2
        self.train_set = [[2017, 12], [2013, 3], [2018, 2], [2018, 12], [2015, 12], [2017, 3], [2016, 1], [2017, 1], [2018, 3], [2014, 1], 
                          [2015, 3], [2016, 3], [2019, 2], [2014, 2], [2015, 2], [2015, 1], [2013, 12], [2019, 3], [2013, 1], [2016, 2], [2014, 12]]
        self.validation_set = [[2016, 12], [2013, 2], [2018, 1]]
        self.test_set = [[2017, 2], [2014, 3], [2019, 1]]


        # [l1/1000,l2/1000, l3/1000, t1, t2, self.rh/25, self.peak/((self.rate+1)*1000), self.day/self.H, [profile techs if targeted]]
        # action set is a 1 binary vector. if cell is 0 -> no CPE; if cell is 1 -> 6-9 is chosen; if cell is 2 -> 16-20 is chosen; if cell is 3 -> both 6-9 and 16-20 are chosen
        self.action_space = Discrete(4)
        if self.tar:
            self.observation_space = Box(low= 0, high= 1, shape= (3+2+1+1+1+4,))
            self.ids = []
            for PV,EV,BAT,DR in product([False, True], [False, True], [False, True], [False, True]):
                self.ids.append((PV,EV,BAT,DR))
        else:
            self.observation_space = Box(low= 0, high= 1, shape= (3+2+1+1+1,))
        # set inactive consumer rate
        self.rate = rate
        self.scales = db['scales']
        
    def step(self, action):
        if self.tar:
            if action == 3 and self.rh[self.n-1] >= 7:
                self.rh[self.n-1] -= 7
            elif action == 1 and self.rh[self.n-1] >= 3:
                self.rh[self.n-1] -= 3
            elif action == 2 and self.rh[self.n-1] >= 4:
                self.rh[self.n-1] -= 4
            else:
                action = 0
            q = f"Day == {self.day} and PV == {self.ids[self.n-1][0]} and EV == {self.ids[self.n-1][1]} \
                and BAT == {self.ids[self.n-1][2]} and DR == {self.ids[self.n-1][3]}"
            self.LL = self.data.query(q)["Load"].values.astype(float)
            self.returned_load = self.data.query(q)[f"{self.option}{action}"].values.astype(float)
            b40 = max(np.sum(self.LL) - 40, 0)
            self.beyound40 = max(np.sum(self.returned_load) - 40, 0) #+ self.rate * tmp1
            price = self.data.query(q)["P"].values.astype(float)
            self.daily_load[self.day-1] += self.returned_load + self.rate * self.LL 
            if self.option == 'WCO':
                reward =  np.sum((self.returned_load + self.rate * self.LL) * (self.d['P']['WCO'] - price) ) \
                        + np.sum((self.d["PH"]['WCO'] - self.d["PR"]['WCO']) * (self.beyound40 + self.rate * b40))\
                        # - np.abs(self.rh[self.n-1] - np.mean(self.rh))
                if action != 0:
                    self.saving = self.data.query(q)[f"Saving{action}"].sum()
                    reward -= np.sum(self.d["PP"] * self.saving)
                else:
                    self.saving = 0
            else:
                reward =  np.sum(self.returned_load * (self.d['P']['FXD'][action] - price) + self.rate * self.LL * (self.d['P']['WCO'] - price)) \
                        + np.sum((self.d["PH"]['FXD'] - self.d["PR"]['FXD']) * self.beyound40 + (self.d["PH"]['WCO'] - self.d["PR"]['WCO']) * self.rate * b40) \
                        # - np.abs(self.rh[self.n-1] - np.mean(self.rh))
                
            #updates
            if self.n == 16:
                if self.daily_load[self.day-1].max() > self.peak:
                    self.peak = self.daily_load[self.day-1].max()
                self.peaks.append(self.peak)
                if self.day == self.H:
                    reward -= self.d["PC"] * self.peak
                    done = True
                    self.n = 1
                else:
                    done = False
                    self.n = 1
                    self.day += 1
                    # self.daily_load = np.zeros(self.d['T'])
                    L = self.data.query(f"Day == {self.day} and PV == {self.ids[self.n-1][0]} \
                                        and EV == {self.ids[self.n-1][1]} and BAT == {self.ids[self.n-1][2]} \
                                        and DR == {self.ids[self.n-1][3]}")[f"{self.option}0"].values.astype(float)
                    l1, l2, l3 = L[6:9].max(), L[16:20].max(), np.concatenate((L[:6], L[9:16], L[20:])).max()
                    t1, t2 = (self.data.query(f'Day == {self.day}')['T'][6:9].min() - self.t_min)/(self.t_max - self.t_min), (self.data.query(f'Day == {self.day}')['T'][16:20].min() - self.t_min)/(self.t_max - self.t_min)
                    self.state = np.array([l1/1000,l2/1000, l3/1000, t1, t2] + list([self.rh[self.n-1]/25]) + [self.peak/((self.rate+1)*1000), self.day/self.H] + list(self.ids[self.n-1]), dtype=np.float32)
            else:
                done = False
                self.n += 1
                L = self.data.query(f"Day == {self.day} and PV == {self.ids[self.n-1][0]} \
                                    and EV == {self.ids[self.n-1][1]} and BAT == {self.ids[self.n-1][2]} \
                                    and DR == {self.ids[self.n-1][3]}")[f"{self.option}0"].values.astype(float)
                l1, l2, l3 = L[6:9].max(), L[16:20].max(), np.concatenate((L[:6], L[9:16], L[20:])).max()
                t1, t2 = (self.data.query(f'Day == {self.day}')['T'][6:9].min() - self.t_min)/(self.t_max - self.t_min), (self.data.query(f'Day == {self.day}')['T'][16:20].min() - self.t_min)/(self.t_max - self.t_min)
                self.state = np.array([l1/1000,l2/1000, l3/1000, t1, t2] + list([self.rh[self.n-1]/25]) + [self.peak/((self.rate+1)*1000), self.day/self.H] + list(self.ids[self.n-1]), dtype=np.float32)
        else:
            if action == 3 and self.rh >= 7:
                self.rh -= 7
            elif action == 1 and self.rh >= 3:
                self.rh -= 3
            elif action == 2 and self.rh >= 4:
                self.rh -= 4
            else:
                action = 0
            self.LL = self.data.query(f"Day == {self.day}").groupby(['Hour'])["Load"].sum().values.astype(float) # day's hourly basic load
            tmp1 = self.data.query(f"Day == {self.day}").groupby(['PV','EV','BAT','DR'])["Load"].sum() - 40 # load beyond 40
            tmp1[tmp1 < 0] = 0
            b40 = np.sum(tmp1)
            self.returned_load = self.data.query(f"Day == {self.day}").groupby(['Hour'])[f"{self.option}{action}"].sum().values.astype(float) # returned load based on the CPE announcement
            tmp = self.data.query(f"Day == {self.day}").groupby(['PV','EV','BAT','DR'])[f"{self.option}{action}"].sum() - 40 # beyond 40 for the actual returned load
            tmp[tmp < 0] = 0
            self.beyound40 = np.sum(tmp)
            price = self.data.query(f"Day == {self.day}").groupby(['Hour'])["P"].sum().values.astype(float)/16
            if self.option == "WCO":
                reward =  np.sum((self.returned_load + self.rate * self.LL) * (self.d['P']['WCO'] - price) ) \
                        + np.sum((self.d["PH"]['WCO'] - self.d["PR"]['WCO']) * (self.beyound40 + self.rate * b40))
                if action != 0:
                    self.saving = self.data.query(f"Day == {self.day}").groupby(['Hour'])[f"Saving{action}"].sum().sum()
                    reward -= np.sum(self.d["PP"] * self.saving)
                else:
                    self.saving = 0
            else:
                reward =  np.sum(self.returned_load * (self.d['P']['FXD'][action] - price) + self.rate * self.LL * (self.d['P']['WCO'] - price)) \
                        + np.sum((self.d["PH"]['FXD'] - self.d["PR"]['FXD']) * self.beyound40
                                  + (self.d["PH"]['WCO'] - self.d["PR"]['WCO']) * self.rate * b40)
            
            #updates
            tl = self.returned_load + self.rate * self.LL
            if tl.max() > self.peak:
                self.peak = tl.max()
            self.peaks.append(self.peak)
        
            # check if it's end of the month, add capacity cost to the total cost
            if self.day == self.H:
                reward -= self.d["PC"] * self.peak
                done = True
            else:
                done = False
                self.day += 1
                L = self.data.query(f"Day == {self.day}").groupby(['Hour'])[f"{self.option}0"].sum().values.astype(float)
                l1, l2, l3 = L[6:9].max(), L[16:20].max(), np.concatenate((L[:6], L[9:16], L[20:])).max()
                t1, t2 = (self.data.query(f'Day == {self.day}')['T'][6:9].min() - self.t_min)/(self.t_max - self.t_min), (self.data.query(f'Day == {self.day}')['T'][16:20].min() - self.t_min)/(self.t_max - self.t_min)
                self.state = np.array([l1/1000,l2/1000, l3/1000, t1, t2] + list([self.rh/25]) + [self.peak/((self.rate+1)*1000), self.day/self.H], dtype=np.float32)
               
        if self.is_neg:
            reward = (reward-self.scale)/np.abs((self.H-1)*self.scale)
        else:
            reward /= self.scale
        info = {}
        self.acts.append(action)
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self, mode= 'train',scale= True, set_num = 0, dataset_mode = False, year = 2013, month = 1):
        if dataset_mode:
            self.year, self.month = year, month
            self.is_neg, self.scale = False, 1
        else:
            if mode == 'train':
                self.year, self.month = random.choice(self.train_set)
            elif mode == 'validation':
                self.year, self.month = self.validation_set[set_num]
            else:
                self.year, self.month = self.test_set[set_num]
            if scale:
                self.scale = self.scales[self.rate, self.year, self.month] 
            else:
                self.scale = 1
            self.is_neg = True if self.scale<0 else False
        idx = self.df.query(f"Year == {self.year} and Month == {self.month}").index
        self.data = self.df.loc[idx,:]
        self.returned_load = np.zeros((self.d['T'],))
        self.beyound40 = 0
        self.saving = 0
        # H: day numbers in the month
        self.H = self.data.loc[:,'Day'].unique().max()
        self.t_min, self.t_max = self.data['T'].min(), self.data['T'].max()
        # Reset day
        self.day = 1
        self.peak = 0
        self.acts = []
        #number of current profile 1 -> 16
        self.n = 1
        self.peaks = []
        self.LL = np.zeros(self.d['T'])
        self.rh = 25 * np.ones(16) if self.tar else 25
        # Reset state
        L = self.data.query(f"Day == {self.day}").groupby(['Hour'])[f"{self.option}0"].sum().values.astype(float)
        l1, l2, l3 = L[6:9].max(), L[16:20].max(), np.concatenate((L[:6], L[9:16], L[20:])).max()
        t1, t2 = (self.data.query(f'Day == {self.day}')['T'][6:9].min() - self.t_min)/(self.t_max - self.t_min), (self.data.query(f'Day == {self.day}')['T'][16:20].min() - self.t_min)/(self.t_max - self.t_min)
        if self.tar:
            self.daily_load = np.zeros((self.H,self.d['T']))
            self.state = np.array([l1/1000,l2/1000, l3/1000, t1, t2] + list([self.rh[self.n-1]/25]) + [self.peak/((self.rate+1)*1000), self.day/self.H] + list(self.ids[self.n-1]), dtype=np.float32)
        else:
            self.state = np.array([l1/1000,l2/1000, l3/1000, t1, t2] + list([self.rh/25]) + [self.peak/((self.rate+1)*1000), self.day/self.H], dtype=np.float32)
        return self.state


# if __name__ == "__main__":