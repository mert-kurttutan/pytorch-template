
from torch import optim

def get_optimizer(model, type="sgd"):
  return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)