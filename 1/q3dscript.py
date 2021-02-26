from A1 import plot_data, error
import pickle

with open('dataA1Q3.pickle','rb') as f:
    dataTrain,dataTest = pickle.load(f)

plot_a, plot_b = plot_data(dataTrain[0],dataTrain[1])
print("Value of a for the fitted line is: " + str(plot_a))
print("Value of b for the fitted line is: " + str(plot_b))
print("Training error is: " + str(error(plot_a,plot_b, dataTrain[0], dataTrain[1])))
print("Test error is: " + str(error(plot_a,plot_b, dataTest[0], dataTest[1])))