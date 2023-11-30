import matplotlib.pyplot as plt
import pickle

fig = plt.plot(0, 0)
fig.show()
pickle.dump(fig, open('FigureObject.obj', 'wb'))


figx = pickle.load(open('FigureObject.obj', 'rb'))

figx.show()
