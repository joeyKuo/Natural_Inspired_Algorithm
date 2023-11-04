import matplotlib.pyplot as plt
#from util import City, read_cities, write_cities_and_return_them, generate_cities, path_cost
import random
import math
import pandas as pd
import geopandas
import time
global runpath
global runtime

runpath=[]
runtime=[]


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        return math.hypot(self.x - city.x, self.y - city.y)

    def __repr__(self):
        return f"({self.x}, {self.y})"


def read_cities(size):
    cities = []
    
    with open(f'test_data/cities_{size}.data', 'r') as handle:
        lines = handle.readlines()
        for line in lines:
            x, y = map(float, line.split())
            cities.append(City(x, y))
    
    return cities


'''def write_cities_and_return_them(size):
    cities = generate_cities(size)
    with open(f'test_data/cities_{size}.data', 'w+') as handle:
        for city in cities:
            handle.write(f'{city.x} {city.y}\n')
    return cities'''


'''def generate_cities(size):
    return [City(x=int(random.random() * 1000), y=int(random.random() * 1000)) for _ in range(size)]
'''

def path_cost(route):
    return sum([city.distance(route[index - 1]) for index, city in enumerate(route)])


def visualize_tsp(title, cities):
    fig = plt.figure()
    fig.suptitle(title)
    x_list, y_list = [], []
    for city in cities:
        x_list.append(city.x)
        y_list.append(city.y)
    x_list.append(cities[0].x)
    y_list.append(cities[0].y)

    plt.plot(x_list, y_list, 'ro')
    plt.plot(x_list, y_list, '.k',markersize=2.5)
    plt.show(block=True)

class Greedy:
    def __init__(self, cities):
        self.unvisited = cities[1:]
        self.route = [cities[0]]

    def run(self, plot):
        if plot:
            plt.ion()
            plt.show(block=False)
            self.init_plot()
        while len(self.unvisited):
            index, nearest_city = min(enumerate(self.unvisited),
                                      key=lambda item: item[1].distance(self.route[-1]))
            self.route.append(nearest_city)
            del self.unvisited[index]
            self.plot_interactive(False)
        self.route.append(self.route[0])
        self.plot_interactive(False)
        self.route.pop()
        return path_cost(self.route)

    def plot_interactive(self, block):
        x1, y1, x2, y2 = self.route[-2].x, self.route[-2].y, self.route[-1].x, self.route[-1].y
        #plt.plot([x1, x2], [y1, y2], '-k',linewidth=0.2,color='gray')
        plt.plot([x1, x2], [y1, y2], '-k',linewidth=1.0,color='gray')
        plt.draw()
        plt.pause(0.01)
        plt.show(block=block)

    def init_plot(self):
        # fig = plt.figure(0,figsize=(6,10))
        # fig.suptitle('Greedy TSP')
        # uk=geopandas.read_file("uk_regions.geojson")
        # uk.plot(figsize=(6,8))       
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        ax = world[world.name == 'United Kingdom'].plot(
         color='lightblue', edgecolor='black',figsize=(6,8))
        plt.title('Top 300 cities in UK based on population',fontsize=12)
        plt.suptitle('Greedy TSP',fontsize=18)
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')

        x_list, y_list = [], []
        for city in [*self.route, *self.unvisited]:
            x_list.append(city.x)
            y_list.append(city.y)
        x_list.append(self.route[0].x)
        y_list.append(self.route[0].y)
        plt.plot(x_list, y_list, '.k',markersize=2.5)

        #London location
        x=-0.1
        y=51.52
        plt.plot(x,y,'ro',label='London',markersize=5) #lonodn
        #plt.annotate('London', xy=(x,y)) #london
        x=-1.91
        y=52.48
        plt.plot(x,y,'go',label='Birmingham',markersize=5)#Birmingham
        x=-4.27
        y=55.87
        plt.plot(x,y,'yo',label='Glassgow',markersize=5) #Glassgow
        plt.legend()
        plt.show(block=False)
        


if __name__ == "__main__":
     start_time=time.time()
     cities = read_cities(51)
     #print(cities)
     greedy = Greedy(cities)

     print('Result length of the path:',greedy.run(plot=True))
     runpath.append(greedy.run)
     #print('greedy route',greedy.route)
     end_time=time.time()
     run_time=end_time-start_time
     print('result run time of finding the path:',run_time)
     plt.show(block=True)
     plt.pause(1)
     #plt.show(block=True)


