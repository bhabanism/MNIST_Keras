import numpy as np
import matplotlib.pyplot as plt

def plot_preds(preds):   
  order = list(range(len(preds[0])))
  #bar_preds = [pr[2] for pr in preds]
  bar_preds = preds[0]
  plt.barh(order, bar_preds)
  plt.xlabel('Probability')  
  plt.tight_layout()
  plt.show()
