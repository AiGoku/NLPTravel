import requests
import jsonpath
import json
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def get_distance(origin,destination):
    lat1,lon1 = origin
    lat2,lon2 = destination
    radius = 6371

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2-lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def get_station_distance(station1,station2):
    return get_distance(station_info[station1],station_info[station2])

api = 'http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json'
response =requests.get(api)
data = response.text

json_obj = json.loads(data)
#print(type(json_obj))

#print(json_obj['l'])
station_info ={}
line_info = {}
for line in json_obj['l']:
    #print(line["ln"])
    line_name = line['ln']
    line_info[line_name] = [st['n'] for st in line['st']]
    for station in line["st"]:
        name = station['n']
        lon,lat = station['sl'].split(r',')
        #print(lon,lat)
        #print(station['n'])   #name
        #print(station['sl'])  #经纬度
        station_info[name] = (float(lon),float(lat))

#print(station_info)
#print(line_info)


def build_connection(station_info):
    station_connection = defaultdict(list)
    stations = list(station_info.keys())
    for station in stations:
        line_values = line_info.values()
        for line_stations in line_values:
            for i,name in enumerate(line_stations):
             if name == station:
                 if i == 0:
                    station_connection[station].append(line_stations[1])
                 elif i == len(line_stations)-1:
                    station_connection[station].append(line_stations[-2])
                 else:
                    station_connection[station].append(line_stations[i-1])
                    station_connection[station].append(line_stations[i+1])
    return station_connection

station_connection = build_connection(station_info)

#print(station_connection)
station_connection_graph = nx.Graph(station_connection)
nx.draw(station_connection_graph,station_info,with_labels=True,node_size=10)


def bfs_search1(graph, start, destination):
    pathes = [[start]]
    visited = set()

    while pathes:
        path = pathes.pop(0)
        froniter = path[-1]

        if froniter in visited: continue

        successsors = graph[froniter]

        for station in successsors:
            if station in path: continue  # check loop

            new_path = path + [station]

            pathes.append(new_path)  # bfs
            # pathes = [new_path] + pathes #dfs

            if station == destination:
                return new_path
        visited.add(froniter)

result = bfs_search1(station_connection,"西直门","上地")
print(result)


def bfs_search2(graph, start, destination, search_strategy):
    pathes = [[start]]
    # visited = set()
    while pathes:
        path = pathes.pop(0)
        froniter = path[-1]
        # if froniter in visited : continue
        # if froniter == destination:
        #    return path
        successsors = graph[froniter]

        for station in successsors:
            if station in path: continue  # check loop

            new_path = path + [station]

            pathes.append(new_path)  # bfs

        pathes = search_strategy(pathes)
        # visited.add(froniter)
        if pathes and (destination == pathes[0][-1]):
            return pathes[0]

def sort_by_distance(pathes):
    def get_distance_of_path(path):
        distance = 0
        for i,_ in enumerate(path[:-1]):
            distance += get_station_distance(path[i],path[i+1])
        return distance
    return sorted(pathes,key=get_distance_of_path)

def get_distance_of_path(path):
    distance = 0
    for i,_ in enumerate(path[:-1]):
        distance += get_station_distance(path[i],path[i+1])
    return distance

print(bfs_search1(station_connection,"上地","北京南站"))
print(bfs_search2(station_connection, "上地", "北京南站", search_strategy = lambda x:x))
print(bfs_search2(station_connection, "上地", "北京南站", search_strategy = sort_by_distance))