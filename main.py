import random
import numpy as np
import time
from scipy.interpolate import interp1d
import random
import copy
import asyncio
import websockets
import argparse
import os

INPUT_DATA_LOCATION = './data/'

attempts = 40
time_to_eat = 1

population = ['g', 's', 'b']
g_r = 1
max_nodes = 20
sim_min_range = -150
sim_max_range = 150
x_1 = 1
y_1 = 1
min_distance_food = 1
max_readius_slime_connection = 5


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-100):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

def find_close_node(node, nodes):
    selection = []
    for i, cand in enumerate(nodes):
        if np.linalg.norm(cand['cord'] - node['cord']) < max_readius_slime_connection and np.linalg.norm(cand['cord'] - node['cord']) != 0:
            #return cand
            selection.append(i)
    if len(selection) > 0:
        return nodes[selection[random.randint(0, len(selection)-1)]]
    return False

def plot_values(nodes, t, color, final_csv, recent_node):
    for node in nodes:
        node_2 = find_close_node(node, nodes)
        if node_2:
            final_csv += str(node['cord'][0]) + ',' + str(node['cord'][1]) + ',' + str(node['cord'][2]) + ',' + str(node_2['cord'][0]) + ',' + str(node_2['cord'][1]) + ',' + str(node_2['cord'][2]) + ',' + str(t) + ',' + 'line' +  ',' + color + '\n'
    return final_csv, ""
    
    '''if recent_node == False:
        node = random.choice(nodes)
    else:
        node = recent_node
    closest_node = find_close_node(node, nodes)
    if closest_node == False:
        return final_csv, recent_node
    else:
        node_2 = closest_node
        final_csv += str(node['cord'][0]) + ',' + str(node['cord'][1]) + ',' + str(node['cord'][2]) + ',' + str(node_2['cord'][0]) + ',' + str(node_2['cord'][1]) + ',' + str(node_2['cord'][2]) + ',' + str(t) + ',' + 'line' +  ',' + color + '\n'
        return final_csv, node_2'''

def plot_food(food, t, color, final_food_csv):
    for f in food:
        final_food_csv += str(f['cord'][0]) + ',' + str(f['cord'][1]) + ',' + str(f['cord'][2]) + ',' + str(f['r']) + ',' + str(t) + ',' + 'dot' + ',' + color +',' +'\n'
    return final_food_csv


def check_node_near_food(node, food):
    for i in range(len(food)):
        if np.linalg.norm(node['cord'] - np.array(food[i]['cord'])) <= food[i]['r']:
            food[i]['count'] = food[i]['count'] + 1
            type_ret = food[i]['type']
            if (food[i]['count'] >= time_to_eat):
                del food[i]
            return [True, type_ret, food]
    return [False, False, food]


def find_closest_food(node, food, rand):
    rand = random.randint(0, len(food))
    if rand == 1:
        if len(food) == 1:
            return food[0]
        elif len(food) == 0:
            return False
        else:
            return food[random.randint(0, len(food))-1]
    max_val = float('inf')
    closest_food = {'cord': np.array(
        [0, 0, 0]), 'r': 1, 'type': True, 'count': 0}
    for i in range(len(food)):
        if np.linalg.norm(node['cord'] - np.array(food[i]['cord'])) < min_distance_food:
            return food[i]
        if np.linalg.norm(node['cord'] - np.array(food[i]['cord'])) < max_val:
            max_val = np.linalg.norm(node['cord'] - np.array(food[i]['cord']))
            closest_food = food[i]
    return closest_food


def init_points(n, init_r, factor):
    nodes = []
    for i in range(n):
        theta = 2*np.pi*random.random()
        phi = 2*np.pi*random.random()
        r = init_r*random.random()
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = r*np.cos(phi)
        #z = 0
        random_point = np.array([x, y, z]) + factor
        nodes.append({'cord': random_point, 'state': 'g'})
    return nodes

def rescale_points(values):
    values_to_return = []
    for i, dim in enumerate(values):
        values_to_return.append([])
        r_max = max(dim)
        r_min = min(dim)
        t_max = (r_max - r_min)/2
        t_min = -(r_max - r_min)/2
        for m in dim:
            values_to_return[i].append(((m - r_min)*(t_max-t_min)/(r_max-r_min)) + t_min)
            #values_to_return[i].append(m - r_min)
    return values_to_return


def init_food(n, min_bound, max_bound, r_range):
    food = []
    for i in range(n):
        r = r_range*random.random()+3
        x = random.uniform(min_bound, max_bound)
        y = random.uniform(min_bound, max_bound)
        z = random.uniform(min_bound, max_bound)
        #z = 0
        food.append({'cord': np.array([x, y, z]),
                    'count': 0, 'r': r, 'type': True})
    return food

def generate_food_from_file(filename, r_range):
    values = [[],[],[]]
    food = []
    with open(filename) as file:
        for line in file:
            tmp_food = line.rstrip().split(",")
            print(tmp_food)
            tmp_food = [float(i) for i in tmp_food]
            for i, val in enumerate(tmp_food):
                values[i].append(val)
    food_scaled = rescale_points(values)
    for i in range(len(food_scaled[0])):
        tmp_food = [food_scaled[0][i], food_scaled[1][i], food_scaled[2][i]]
        food.append({'cord': np.array([tmp_food[0],tmp_food[1], tmp_food[2]]), 'count': 0, 'r': r_range, 'type': True})
    return food


def generate_grid_food(n_layers, min_bound, max_bound, r_range):
    food = []
    for k in range(n_layers):
        i = min_bound
        while i < max_bound:
            j = min_bound
            while j < max_bound:
                r = r_range
                x = j
                y = i
                z = k
                food.append(
                    {'cord': np.array([80*x, 80*y, 80*z]), 'count': 0, 'r': r, 'type': True})
                j += 1
            i += 1
    return food


def subtract_intersecting_food(negative_food, food):
    food_to_pop = []
    for i, f in enumerate(food):
        for j, n_f in enumerate(negative_food):
            tmp_dist = np.linalg.norm(n_f['cord'] - f['cord'])
            if tmp_dist <= f['r'] or tmp_dist <= n_f['r']:
                food_to_pop.append(i)
    food_to_pop.sort(reverse=False)
    shift = 0
    for i in food_to_pop:
        food.pop(i-shift)
        shift += 1
    return food


def find_intersect_food(negative_food, vector):
    for i in negative_food:
        if np.linalg.norm(vector - np.array(i['cord'])) <= i['r']:
            return i
    return False


def check_if_intersect(negative_food, vector):
    for i in negative_food:
        if np.linalg.norm(vector - np.array(i['cord'])) <= i['r']:
            return True
    return False


def generate_one_cycle(nodes,food, p_pos, p_neg, p_neu, negative_food):
    new_nodes = []
    for i, node in enumerate(nodes):
        #time.sleep(0.01)
        is_near_food = check_node_near_food(nodes[i], food)
        food = is_near_food[2]
        if is_near_food[0] and is_near_food[1]:
            weights = p_pos
        elif is_near_food[0] and (is_near_food[1] == False):
            weights = p_neg
        else:
            weights = p_neu
        new_state = random.choices(population, weights=weights, k=1)[0]
        node['state'] = new_state
        closest_food = find_closest_food(node, food, False)
        #random_food = find_closest_food(node, food, True)
        if closest_food == False:
            break
        if new_state == 'g':
            litmus = False
            counter = 0
            new_direction = node['cord'] + g_r * (-node['cord'] + closest_food['cord'])
            '''while check_if_intersect(negative_food, new_direction):
                n_f = find_intersect_food(negative_food, new_direction)
                if n_f != False:
                    #print('expansion grow')
                    # Define plane
                    norm_n_f = -n_f['cord'] + node['cord']
                    planeNormal = norm_n_f
                    # Any point on the plane
                    planePoint = (
                        n_f['r']*norm_n_f/np.linalg.norm(norm_n_f)) + n_f['cord']
                    # Define ray
                    rayDirection = -node['cord'] + n_f['cord']
                    rayPoint = n_f['cord']  # Any point along the ray
                    psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
                    new_direction = -rayPoint + psi + planeNormal
                counter += 1
                if counter <= attempts:
                    litmus = True
                    break
            if not litmus:
                nodes[i]['cord'] = new_direction'''
            nodes[i]['cord'] = new_direction
        if new_state == 'b' and len(nodes) < max_nodes:
            cord_1 = node['cord'] + g_r * (-node['cord'] + closest_food['cord'])
            litmus = False
            counter = 0
            '''while check_if_intersect(negative_food, cord_1):
                n_f = find_intersect_food(negative_food, cord_1)
                if n_f != False:
                    #print('branch grow')
                    cord_1 = node['cord'] + g_r * (-node['cord'] + closest_food['cord'])
                    norm_n_f = -n_f['cord'] + node['cord']
                    # Define plane
                    planeNormal = norm_n_f
                    # Any point on the plane
                    planePoint = (n_f['r']*norm_n_f/np.linalg.norm(norm_n_f)) + n_f['cord']
                    # Define ray
                    rayDirection = -node['cord'] + n_f['cord']
                    rayPoint = n_f['cord']  # Any point along the ray
                    psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
                    cord_1 = -rayPoint + psi + planeNormal
                counter += 1
                if counter <= attempts:
                    litmus = True
                break
            if not litmus:
                new_node_1 = {'cord': cord_1, 'state': 'g'}
                new_nodes.append(new_node_1)'''
            new_node_1 = {'cord': cord_1, 'state': 'g'}
            new_nodes.append(new_node_1)
    for i, node in enumerate(nodes):
        if node['state'] == 's':
            del nodes[i]
    print(len(food))
    print(len(nodes))
    print('----')
    return nodes, new_nodes


def generate_cycles(nodes, food, p_neg, p_pos, p_neu):
    final_csv = ''
    final_food_csv = ''
    nodes_to_return = []
    for i, nodes_set in enumerate(nodes):
        negative_food = []
        for j, item in enumerate(food):
            if j != i:
                negative_food += item
        nodes, new_nodes = generate_one_cycle(nodes_set, food[i], p_neg, p_pos, p_neu, negative_food)
        nodes = nodes + new_nodes
        nodes_to_return.append(nodes)
        #final_csv = plot_values(nodes, generation, final_csv)
        #final_food_csv = plot_food(food, generation, final_food_csv)
    text_file = open("res.txt", "w")
    n = text_file.write(final_csv)
    text_file.close()
    
    text_file = open("food_res.txt", "w")
    n = text_file.write(final_food_csv)
    text_file.close()
    return nodes_to_return, food

async def run_model(websocket):
    # ['g', 's', 'b']
    communication = True
    p_neg = [0.3, 0.1, 0.6]
    p_pos = [0.3, 0.1, 0.6]
    p_neu = [0.3, 0.1, 0.6]


    food = []
    color = []
    nodes = []
    dir_name = INPUT_DATA_LOCATION+args.dataset
    for filename in os.listdir(dir_name):
        if filename.endswith(".txt"):
            food.append(generate_food_from_file(dir_name+'/'+filename, 0.5))
            color.append(filename[:-4])
            nodes.append(init_points(5, 1, 0))
    final_nodes = ""
    food_csv_string = ""
    nodes_csv_string = ""
    nodes_2_csv_String = ""

    generation = 0
    recent_node = [False, False, False, False, False, False, False]
    while communication:
        empty_food = False
        nodes, food = generate_cycles(nodes, food, p_pos, p_neg, p_neu)
        food_string = ""
        nodes_string = ""
        for i, nodes_set in enumerate(nodes):
            if color[i] == 'b':
                print(recent_node[i])
            tmp_nodes_string, tmp = plot_values(nodes_set, generation, color[i], "", recent_node[i])
            recent_node[i] = copy.deepcopy(tmp)
            if color[i] == 'b':
                print(recent_node[i])
            print("----")
            nodes_string += tmp_nodes_string
        for i, food_set in enumerate(food):
            food_string += plot_food(food_set,generation, color[i], "")
        food_csv_string += food_string
        nodes_csv_string += nodes_string

        #what to send
        if len(food_string) == 0:
            communication = False
        await websocket.send(nodes_string+food_string)
        

        for food_set in food:
            if len(food_set) == 0:
                empty_food = True
        
        generation += 1
    '''text_file = open("res.txt", "w")
    n = text_file.write(nodes_csv_string)
    text_file.close()

    text_file = open("food_res.txt", "w")
    n = text_file.write(food_csv_string)
    text_file.close()'''
    

parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset', 
    required=True,
    type=str,
    help='Obj name'
) 

async def main():
    async with websockets.serve(run_model, "", 8001):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(main())


    