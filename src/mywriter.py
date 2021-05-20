class MyWriter:
  def __init__(self, writer):  
    self.step = 0
    self.scalars = {}
    self.writer = writer
  
  def set_step(self, step):
    self.scalars.clear()
    self.step = step

  def add_scalar(self, key, value):
    self.scalars[key] = value

  def write(self):
    #print("writing!!!")
    #writer.add_scalars('scalars', {'g_loss': self.scalars['g_loss'], 'd_loss': self.scalars['d_loss']}, step) 
    #print("self.scalars "  + str(self.scalars))
    #writer.add_scalars('scalars', {'g_loss': self.scalars['g_loss'], 'd_loss': self.scalars['d_loss']}, step) 
    self.writer.add_scalars('scalars', self.scalars, self.step)

