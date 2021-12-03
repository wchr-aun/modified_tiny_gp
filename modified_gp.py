### THIS CODE IS MODIFIED FROM https://github.com/moshesipper/tiny_gp BY moshesipper

import math
import numpy as np
from random import random, randint
from statistics import mean
from copy import deepcopy
import time

POP_SIZE        = 120
MIN_DEPTH       = 2
MAX_DEPTH       = 5
GENERATIONS     = 300
TOURNAMENT_SIZE = 10
XO_RATE         = 0.7
PROB_MUTATION   = 0.3
FILE_NAME       = './cw1-q4-ds.txt' # File name
LOG             = True # Allow logging the output
LOG_FILE_NAME   = './question4_log.txt' # Log output file name
ITERATIONS      = 1 # Amount of iterations to run. For 1 time run, set this value to 1

def add(x, y): return x + y
def mul(x, y): return x * y
def sub(x, y): return x - y
def div(x,y):
  if math.isclose(y, 0):
    return 1
  return x/y
def sin(x, y): return math.sin(x*y)
def cos(x, y): return math.cos(x*y)
def exp(x):
  try:
    return math.exp(x)
  except:
    return 0
def abso(x): return abs(x)
def power(x, n):
  if n != 1 and n != 2 and n != 3 and n != 4:
    return 0
  try:
    return math.pow(x, n)
  except:
    return 0
def powerth(x, n):
  if n != 1 and n != 2 and n != 3 and n != 4:
    return 0
  try:
    return math.pow(x, 1/n)
  except:
    return 0

OPERATIONS = [add, sub, mul, div, sin, cos, exp, abso, power, powerth]
FUNCTIONS = [exp, abso]
TERMINALS = ['x', 'y', 1, 2, 3, 4, 10, 100, math.pi]

def clearLog():
  if LOG:
    f = open(LOG_FILE_NAME, 'w')
    f.write('')
    f.close()

def log(s):
  if LOG:
    f = open(LOG_FILE_NAME, 'a')
    f.write(s +'\n')
    f.close()

def print_and_log(s):
  print(s)
  log(s)

def generate_dataset():
  f = open(FILE_NAME, 'r')
  ds = f.read()
  f.close()
  return [list(map(float, data.split())) for data in ds.split('\n')][:-1]

def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

def symbols(opt, x='', y=''):
  x = 'pi' if is_number(x) and math.isclose(float(x), math.pi) else x
  y = 'pi' if is_number(y) and math.isclose(float(y), math.pi) else y
  if opt == 'add': return f'{x}+{y}'
  elif opt == 'mul': return f'{x}*{y}'
  elif opt == 'sub': return f'{x}-{y}'
  elif opt == 'div': return f'{x}/{y}'
  elif opt == 'power': return f'{x}^{y}'
  elif opt == 'powerth': return f'{x}^(1/{y})'
  elif opt == 'cos': return f'cos(({x})*({y}))'
  elif opt == 'sin': return f'sin(({x})*({y}))'
  elif opt == 'abso': return f'abs({x})'
  elif opt == 'exp': return f'exp({x})'
  else: return opt

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

def fitness(individual, dataset): # inverse mean absolute error over dataset normalized to [0,1]
  return 1 / (1 + mean([abs(individual.compute_tree(ds[0], ds[1]) - ds[2]) + 0.01*individual.size() for ds in dataset]))

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
  restart = True
  restart_count = 0
  while restart:
    restart = False
    dataset = generate_dataset()
    population = init_population() 
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]

    # go evolution!
    for gen in range(GENERATIONS):        
      nextgen_population=[]
      for i in range(POP_SIZE):
        parent1 = selection(population, fitnesses)
        parent2 = selection(population, fitnesses)
        parent1.crossover(parent2)
        parent1.mutation()
        nextgen_population.append(parent1)
      population=nextgen_population
      fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
      if max(fitnesses) > best_of_run_f:
        best_of_run_f = max(fitnesses)
        best_of_run_gen = gen
        best_of_run = deepcopy(population[fitnesses.index(best_of_run_f)])
        raw_fitness = 1 / (1 + mean([abs(best_of_run.compute_tree(ds[0], ds[1]) - ds[2]) for ds in dataset]))
        print('________________________')
        print(f'gen: {gen} best_of_run_f: {round(best_of_run_f,3)} raw_fitness: {round(raw_fitness, 3)}')
        print(f'Formula: {best_of_run.formula()}')
      if raw_fitness > 0.9 and gen - best_of_run_gen > 50: break # Let GP tries to simplify the formula for another 50 generations
      if (gen > 50 and best_of_run_f < 0.6) or (gen > 200 and best_of_run_f < 0.7): # If the fitness value does not reach some value at certain point, restart the run
      # if (gen > 50 and best_of_run_f < 0.6): # This condition will definitely take less (or equal) time to the above condition, but may not find the closest formula
        restart_count += 1
        restart = True
        break

  raw_fitness = 1 / (1 + mean([abs(best_of_run.compute_tree(ds[0], ds[1]) - ds[2]) for ds in dataset]))
  # print('-------------------------------------------------')
  print(f'-END OF RUN-\n')
  print_and_log(f'best_of_run attained at generation {best_of_run_gen}th of {restart_count}th restart.')
  print_and_log(f'Have the fitness value of {round(best_of_run_f, 3)} and raw fitness value of {round(raw_fitness, 3)}')
  print_and_log(f'Final Formula: {best_of_run.formula()}')
  return raw_fitness

if __name__== '__main__':
  clearLog()
  times = []
  _fitnesses = []
  for i in range(ITERATIONS):
    print_and_log('-------------------------------------------------')
    print_and_log(f'-START OF {i + 1}th RUN-\n')
    start = time.time()
    _fitnesses.append(main())
    end = time.time()
    print_and_log(f'Time taken: {int((end - start) / 60)}:{str(int(end - start) % 60).zfill(2)} min\n')
    times.append(end - start)
    print_and_log(f'-END OF {i + 1}th RUN-\n')
    print_and_log('-------------------------------------------------')
  avg_time = np.array(times).mean()
  avg_fitness = np.array(_fitnesses).mean()
  print_and_log(f'\n\nAverage runtime over {ITERATIONS} runs is {int((avg_time) / 60)}:{str(int(avg_time) % 60).zfill(2)} min')
  print_and_log(f'Average fitness over {ITERATIONS} runs is {avg_fitness}')
  print_and_log(f'Average error over {ITERATIONS} runs is {1-avg_fitness}')
