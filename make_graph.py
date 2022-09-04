from matplotlib import pyplot
import csv

def get_data(file_name, delimiter):
    x = []
    y = []
    with open(file_name, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter = delimiter)
        reader.__next__()
        for row in reader:
            x.append(int(row[0]))
            y.append(float(row[1]))
        
        return x, y


pyplot.plot(*get_data("build/blis.csv", ','), label='BLIS')
pyplot.plot(*get_data("build/jpi_ji_packed.csv", ','), label='My_GEMM')
pyplot.legend()
pyplot.savefig("img/graph.png")
pyplot.show()