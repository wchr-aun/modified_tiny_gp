import math
from statistics import mean

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
  elif opt == 'cos' and y != '': return f'cos(({x})*({y}))'
  elif opt == 'cos' and y == '': return f'cos({x})'
  elif opt == 'sin': return f'sin(({x})*({y}))'
  elif opt == 'abso': return f'abs({x})'
  elif opt == 'exp': return f'exp({x})'
  else: return opt

class logUtil:
  def __init__(self, allow_log, log_file_name):
    self.allow_log  = allow_log
    self.log_file_name  = log_file_name

  def clearLog(self):
    if self.allow_log:
      f = open(self.log_file_name, 'w')
      f.write('')
      f.close()

  def log(self, s):
    if self.allow_log:
      f = open(self.log_file_name, 'a')
      f.write(s +'\n')
      f.close()

  def print_and_log(self, s):
    print(s)
    if self.allow_log:
      self.log(s)

def fitness(individual, dataset): # inverse mean absolute error over dataset normalized to [0,1]
  return 1 / (1 + mean([abs(individual.compute_tree(ds[0], ds[1]) - ds[2]) + 0.01*individual.size() + 0.02*individual.depth() for ds in dataset]))
