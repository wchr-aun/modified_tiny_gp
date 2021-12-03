### THIS CODE IS MODIFIED FROM https://github.com/moshesipper/tiny_gp BY moshesipper

import math
import numpy as np
from random import random, randint
from copy import deepcopy
import time
from util import *

POP_SIZE        = 120
MIN_DEPTH       = 2
MAX_DEPTH       = 5
GENERATIONS     = 500
TOURNAMENT_SIZE = 10
XO_RATE         = 0.8
PROB_MUTATION   = 0.5
LOG             = True
LOG_FILE_NAME   = './log_q3_most_generalize.txt'
ITERATIONS      = 10 # Amount of iterations to run. For 1 time run, set this value to 1

def add(x, y): return x + y
def mul(x, y): return x * y
# def cos(x): return math.cos(2*math.pi*x)
def cos(x, y): return math.cos(x*y)

# OPERATIONS = [add, mul]
# TERMINALS = ['x', 'y']
OPERATIONS = [add, mul, cos]
TERMINALS = ['x', 'y', 20, -10, 2, math.pi]
FUNCTIONS = []

def target_func(x, y): # evolution's target
    # return x ** 2 + y ** 2
    return 20 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))

def generate_dataset():
  x = np.random.uniform(-5.12, 5.12, 100)
  y = np.random.uniform(-5.12, 5.12, 100)
  return np.array([x, y, target_func(x, y)]).T

class GPTree:
  def __init__(self, data = None, left = None, right = None):
    self.data  = data
    self.left  = left
    self.right = right

  def node_label(self): # string label
    if (self.data in OPERATIONS):
      return self.data.__name__
    else: 
      return str(self.data)

  def formula(self):
    if self.left and self.right: return symbols(self.node_label(), self.left.formula(), self.right.formula())
    elif self.left: return symbols(self.node_label(), self.left.formula())
    else: return symbols(self.node_label())

  def compute_tree(self, x, y): 
    if (self.data in OPERATIONS):
      if self.data in FUNCTIONS:
        return self.data(self.left.compute_tree(x, y))
      else:
        return self.data(self.left.compute_tree(x, y), self.right.compute_tree(x, y))
    elif self.data == 'x': return x
    elif self.data == 'y': return y
    else: return self.data
          
  def random_tree(self, grow, max_depth, depth = 0): # create random tree using either grow or full method
    if depth < MIN_DEPTH or (depth < max_depth and not grow): 
      self.data = OPERATIONS[randint(0, len(OPERATIONS)-1)]
    elif depth >= max_depth:   
      self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
    else: # intermediate depth, grow
      if random () > 0.5: 
        self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
      else:
        self.data = OPERATIONS[randint(0, len(OPERATIONS)-1)]
    if self.data in OPERATIONS:
      if self.data in FUNCTIONS:
        self.left = GPTree()          
        self.left.random_tree(grow, max_depth, depth = depth + 1)
      else:
        self.left = GPTree()          
        self.left.random_tree(grow, max_depth, depth = depth + 1)            
        self.right = GPTree()
        self.right.random_tree(grow, max_depth, depth = depth + 1)

  def mutation(self):
    if random() < PROB_MUTATION: # mutate at this node
        self.random_tree(grow = True, max_depth = 2)
    elif self.left: self.left.mutation()
    elif self.right: self.right.mutation() 

  def size(self): # tree size in nodes
    if self.data in TERMINALS: return 1
    l = self.left.size()  if self.left  else 0
    r = self.right.size() if self.right else 0
    return 1 + l + r

  def build_subtree(self): # count is list in order to pass 'by reference'
    t = GPTree()
    t.data = self.data
    if self.left:  t.left  = self.left.build_subtree()
    if self.right: t.right = self.right.build_subtree()
    return t
                      
  def scan_tree(self, count, second): # note: count is list, so it's passed 'by reference'
    count[0] -= 1            
    if count[0] <= 1: 
      if not second: # return subtree rooted here
        return self.build_subtree()
      else: # glue subtree here
        self.data  = second.data
        self.left  = second.left
        self.right = second.right
    else:  
      ret = None              
      if self.left  and count[0] > 1: ret = self.left.scan_tree(count, second)  
      if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)  
      return ret

  def crossover(self, other): # xo 2 trees at random nodes
    if random() < XO_RATE:
      second = other.scan_tree([randint(1, other.size())], None) # 2nd random subtree
      self.scan_tree([randint(1, self.size())], second) # 2nd subtree 'glued' inside 1st tree

  def depth(self):     
    if self.data in TERMINALS: return 0
    l = self.left.depth()  if self.left  else 0
    r = self.right.depth() if self.right else 0
    return 1 + max(l, r)

def selection(population, fitnesses): # select one individual using tournament selection
  tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
  tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
  return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])

def init_population(): # ramped half-and-half
  pop = []
  for md in range(3, MAX_DEPTH + 1):
    for i in range(int(POP_SIZE/6)):
      t = GPTree()
      t.random_tree(grow = True, max_depth = md) # grow
      pop.append(t) 
    for i in range(int(POP_SIZE/6)):
      t = GPTree()
      t.random_tree(grow = False, max_depth = md) # full
      pop.append(t) 
  return pop

def main():
  dataset = generate_dataset()
  population = init_population() 
  best_of_run = None
  best_of_run_f = 0
  best_of_run_gen = 0
  pb_particle = deepcopy(population)
  pb_particle_f = [fitness(population[i], dataset) for i in range(POP_SIZE)]

  # go evolution!
  for gen in range(GENERATIONS):        
    nextgen_population=[]
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
    if max(fitnesses) > best_of_run_f:
      best_of_run_f = max(fitnesses)
      best_of_run_gen = gen
      best_of_run = deepcopy(population[fitnesses.index(best_of_run_f)])
      raw_fitness = 1 / (1 + mean([abs(best_of_run.compute_tree(ds[0], ds[1]) - ds[2]) for ds in dataset]))
      print('________________________')
      print(f'gen: {gen} best_of_run_f: {round(best_of_run_f,3)} raw_fitness: {round(raw_fitness, 3)}')
      print(f'Formula: {best_of_run.formula()}')
    if math.isclose(raw_fitness, 1): break # Let GP tries to simplify the formula for another 50 generations
    for i in range(POP_SIZE):
      parent1 = selection(population, fitnesses)
      parent2 = selection(population, fitnesses)
      parent1.crossover(pb_particle[i])
      parent2.crossover(best_of_run)
      parent1.mutation()
      parent2.mutation()

      fitness_parent1 = fitness(parent1, dataset)
      fitness_parent2 = fitness(parent2, dataset)

      fitness_parent = 0
      parent = None
      if fitness_parent1 > fitness_parent2:
        fitness_parent = fitness_parent1
        parent = parent1
      else:
        fitness_parent = fitness_parent2
        parent = parent2

      if fitness_parent > pb_particle_f[i]:
        pb_particle[i] = deepcopy(parent)
        pb_particle_f[i] = fitness_parent

      nextgen_population.append(parent)
    population=nextgen_population

  raw_fitness = 1 / (1 + mean([abs(best_of_run.compute_tree(ds[0], ds[1]) - ds[2]) for ds in dataset]))
  # print('-------------------------------------------------')
  print(f'-END OF RUN-\n')
  lu.print_and_log(f'best_of_run attained at generation {best_of_run_gen}th.')
  lu.print_and_log(f'Have the fitness value of {round(best_of_run_f, 3)} and raw fitness value of {round(raw_fitness, 3)}')
  lu.print_and_log(f'Final Formula: {best_of_run.formula()}')
  return raw_fitness, gen

if __name__== '__main__':
  lu = logUtil(LOG, LOG_FILE_NAME)
  lu.clearLog()
  times = []
  _fitnesses = []
  _gen = []
  for i in range(ITERATIONS):
    lu.print_and_log('-------------------------------------------------')
    lu.print_and_log(f'-START OF {i + 1}th RUN-\n')
    start = time.time()
    fs, gen = main()
    _fitnesses.append(fs)
    _gen.append(gen)
    end = time.time()
    times.append(end - start)
    lu.print_and_log(f'Time taken: {int((end - start) / 60)}:{str(int(end - start) % 60).zfill(2)} min\n')
    lu.print_and_log(f'-END OF {i + 1}th RUN-\n')
    lu.print_and_log('-------------------------------------------------')
  avg_time = np.array(times).mean()
  avg_fitness = np.array(_fitnesses).mean()
  avg_gen = np.array(_gen).mean()
  lu.print_and_log(f'\n\nAverage generation before hitting terminated condition over 10 runs is {avg_gen}')
  lu.print_and_log(f'Average runtime over 10 runs is {int((avg_time) / 60)}:{str(int(avg_time) % 60).zfill(2)} min')
  lu.print_and_log(f'Average fitness over 10 runs is {avg_fitness}')
  lu.print_and_log(f'Average error over 10 runs is {1-avg_fitness}')
