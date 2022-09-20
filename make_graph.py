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
            # y.append(float(row[1]))
            y.append(float(row[2]))
        
        return x, y


pyplot.plot(*get_data("build/blis.csv", ','), label='BLIS')
pyplot.plot(*get_data("build/jpi_ji_packed.csv", ','), label='My_GEMM')
pyplot.ylabel("GFLOPS")
pyplot.xlabel("Matrix dimension m=k=n")
pyplot.legend()
pyplot.savefig("img/graph.png")
pyplot.show()

# pyplot.plot(*get_data("build/blis_mt.csv", ','), label='BLIS_MT')
# pyplot.plot(*get_data("build/jpi_ji_packed_mt2.csv", ','), label='Loop 2')
# pyplot.plot(*get_data("build/jpi_ji_packed_mt3.csv", ','), label='Loop 3')
# pyplot.plot(*get_data("build/jpi_ji_packed_mt5.csv", ','), label='Loop 5')
# pyplot.ylabel("GFLOPS")
# pyplot.xlabel("Matrix dimension m=k=n")
# pyplot.legend()
# pyplot.savefig("img/graph_mt.png")
# pyplot.show()

# pyplot.plot(*get_data("build/blis_mt.csv", ','), label='BLIS_MT')
# pyplot.plot(*get_data("build/jpi_ji_packed_mt2.csv", ','), label='Loop 2')
# pyplot.plot(*get_data("build/jpi_ji_packed_mt3.csv", ','), label='Loop 3')
# pyplot.plot(*get_data("build/jpi_ji_packed_mt5.csv", ','), label='Loop 5')
# pyplot.ylabel("GFLOPS/thread")
# pyplot.xlabel("Matrix dimension m=k=n")
# pyplot.legend()
# pyplot.savefig("img/graph_mt_per_thread.png")
# pyplot.show()