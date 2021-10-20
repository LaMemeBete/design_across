import random
import numpy as np
import serial
import time
from scipy.interpolate import interp1d
import socket
import random
UDP_IP = "127.0.0.1"
UDP_PORT = 6400
MESSAGE = ""
attempts = 10
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet socket.SOCK_DGRAM) # UDP
time_to_eat = 1
population = ['g', 's', 'b']
g_r = 0.4
max_nodes = 10
sim_min_range = -400
sim_max_range = 400
x_1 = 1
y_1 = 1

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-100):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        print('enters')
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

def plot_values(nodes, t, final_csv):
    for node in nodes:
        node_2 = random.randint(0, len(nodes)-1)
        node_3 = random.randint(0, len(nodes)-1)
        node_4 = random.randint(0, len(nodes)-1)
        final_csv += str(node['cord'][0]) + ','+ str(node['cord'][1]) + ','+ str(node['cord'][2]) + ','+ str(nodes[node_2]['cord'][0]) + ',' + str(nodes[node_2]['cord'][1]) + ',' + str(nodes[node_2]['cord'][2]) + ',' + str(t) + ',' + 'line' + '\n'
        final_csv += str(node['cord'][0]) + ','+ str(node['cord'][1]) + ','+ str(node['cord'][2]) + ','+ str(nodes[node_3]['cord'][0]) + ',' + str(nodes[node_3]['cord'][1]) + ',' + str(nodes[node_3]['cord'][2]) + ',' + str(t) + ',' + 'line' + '\n'
        #final_csv += str(node['cord'][0]) + ','+ str(node['cord'][1]) + ','+ str(node['cord'][2]) + ','+ str(nodes[node_4]['cord'][0]) + ',' + str(nodes[node_4]['cord'][1]) + ',' + str(nodes[node_4]['cord'][2]) + ','+ str(t) + 'line' + '\n'
    return final_csv

def plot_food(food, t, final_food_csv):
    for f in food:
        final_food_csv += str(f['cord'][0]) + ',' + str(f['cord'][1]) + ',' + str(f['cord'][2]) + ',' + str(f['r']) + ',' + str(t) + ',' + 'dot' + '\n'
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
    closest_food = {'cord':np.array([0,0,0]), 'r':1, 'type':True, 'count': 0}
    for i in range(len(food)):
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
        #z = r*np.cos(phi)
        z = 0
        random_point = np.array([x, y, z]) + factor
        nodes.append({'cord': random_point, 'state': 'g'})
    return nodes

def init_food(n, min_bound, max_bound, r_range):
    food = []
    for i in range(n):
        r = r_range*random.random()+3
        x = random.uniform(min_bound, max_bound)
        y = random.uniform(min_bound, max_bound)
        #z = random.uniform(min_bound, max_bound)
        z = 0
        food.append({'cord': np.array([x, y, z]), 'count': 0, 'r': r, 'type': True})
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
                food.append({'cord': np.array([80*x, 80*y, 80*z]), 'count': 0, 'r': r, 'type': True})
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

def generate_cycles(nodes, nodes_2, food, cycles, p_neg, p_pos, p_neu, b_pos, b_neg, b_neu, generation, negative_food):
    final_csv = ''
    final_food_csv = ''
    for t in range(cycles):
        time.sleep(0.1)
        '''ser = serial.Serial('/dev/cu.usbserial-01937934', 9800, timeout=1)
        m = interp1d([0,1023],[sim_min_range,sim_max_range])
        l = 0
        while l < 5:
            line = ser.readline()
            if line:
                line = line.decode()
                if line.split(" ")[0] != "0" and line.split(" ")[1] != "0" and line.split(" ")[2] != "0":
                    x = m(float(line.split(" ")[0]))
                    y = m(float(line.split(" ")[1]))
                    z = m(float(line.split(" ")[2]))
                    food.append({'cord': np.array([x, y, z]), 'count': 0, 'r': 10*random.random()+3, 'type': True})
                    print("this is the new amount of food" + str(len(food)))
            l += 1
        ser.close()'''
        new_nodes = []
        for i, node in enumerate(nodes):
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
            random_food = find_closest_food(node, food, True)
            if closest_food==False or random_food == False:
                break
            if new_state == 'g':
                litmus = False
                counter = 0
                new_direction = node['cord'] + g_r*(-node['cord'] + closest_food['cord']) #+ 2*random.random()
                while check_if_intersect(negative_food, new_direction):
                    n_f = find_intersect_food(negative_food, new_direction)
                    if n_f != False:
                        print('expansion grow')
                        #Define plane
                        norm_n_f = -n_f['cord'] + node['cord']
                        planeNormal = norm_n_f
                        planePoint = (n_f['r']*norm_n_f/np.linalg.norm(norm_n_f)) + n_f['cord'] #Any point on the plane
                        #Define ray
                        rayDirection = -node['cord'] + n_f['cord']
                        rayPoint = n_f['cord'] #Any point along the ray
                        psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
                        new_direction = -rayPoint + psi + planeNormal
                    counter += 1
                    if counter <= attempts:
                        litmus = True
                        break
                if not litmus:
                    nodes[i]['cord'] = new_direction
            if new_state == 'b' and len(nodes) < max_nodes:
                cord_1 = node['cord'] + g_r*(-node['cord'] + closest_food['cord']) 
                litmus = False
                counter = 0
                while check_if_intersect(negative_food, cord_1):
                    n_f = find_intersect_food(negative_food, cord_1)
                    if n_f != False:
                        print('branch grow')
                        cord_1 = node['cord'] + g_r*(-node['cord'] + closest_food['cord'])
                        norm_n_f = -n_f['cord'] + node['cord']
                        #Define plane
                        planeNormal = norm_n_f
                        planePoint = (n_f['r']*norm_n_f/np.linalg.norm(norm_n_f)) + n_f['cord'] #Any point on the plane
                        #Define ray
                        rayDirection = -node['cord'] + n_f['cord']
                        rayPoint = n_f['cord'] #Any point along the ray
                        psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
                        cord_1 = -rayPoint + psi + planeNormal
                    counter += 1
                    if counter <= attempts:
                        litmus = True
                    break
                if not litmus:
                    new_node_1 = {'cord': cord_1, 'state': 'g'}
                    new_nodes.append(new_node_1)
        for i, node in enumerate(nodes):
            if node['state'] == 's':
                del nodes[i]
        print(len(food))
        print(len(nodes))
        print('----')
        nodes = nodes + new_nodes

        '''new_nodes = []
        for i, node in enumerate(nodes_2):
            is_near_food = check_node_near_food(nodes_2[i], food)
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
            random_food = find_closest_food(node, food, True)
            if closest_food==False or random_food == False:
                break
            if new_state == 'g':
                nodes_2[i]['cord'] = node['cord'] + g_r*(-node['cord'] + closest_food['cord']) + 2*random.random()
                nodes_2[i]['cord'] = node['cord'] + g_r*(-node['cord'] + random_food['cord']) + 2*random.random()
            if new_state == 'b' and len(nodes_2) < max_nodes:
                cord_1 = node['cord'] + g_r*(-node['cord'] + closest_food['cord']) + 2*random.random()
                cord_2 = node['cord'] + g_r*(-node['cord'] + random_food['cord']) + 2*random.random()
                new_node_1 = {'cord': cord_1, 'state': 'g'}
                new_node_2 = {'cord': cord_2, 'state': 'g'}
                new_nodes.append(new_node_1)
                new_nodes.append(new_node_2)
        for i, node in enumerate(nodes_2):
            if node['state'] == 's':
                del nodes_2[i]
        print(len(food))
        print(len(nodes_2))
        print('----')
        nodes_2 = nodes_2 + new_nodes'''

        final_csv = plot_values(nodes, generation, final_csv)
        final_food_csv = plot_food(food, generation, final_food_csv)
    text_file = open("res.txt", "w")
    n = text_file.write(final_csv)
    text_file.close()
    
    text_file = open("food_res.txt", "w")
    n = text_file.write(final_food_csv)
    text_file.close()
    return nodes, nodes_2, food

if __name__ == "__main__":
    cycles = 1
    #['g', 's', 'b']
    p_neg = [0.3, 0.1, 0.6]
    p_pos = [0.3, 0.1, 0.6]
    p_neu = [0.3, 0.1, 0.6]
    b_pos = 1
    b_neg = 1
    b_neu = 1
    nodes = init_points(5, 1, -110)
    nodes_2 = init_points(10,1, 0)
    
    food = init_food(10, sim_min_range, sim_max_range, 5)
    #food = generate_grid_food(2, -5, 5, 10)

    negative_food = [{'cord': np.array([0, 0, 0]), 'r': 100, 'type': True}, {'cord': np.array([180, 180, 0]), 'r': 70, 'type': True}]
    food = subtract_intersecting_food(negative_food, food)
    print(food)
    final_nodes = ""

    food_csv_string = ""
    nodes_csv_string = ""
    nodes_2_csv_String = ""


    generation = 0
    while True:
        nodes, nodes_2, food = generate_cycles(nodes, nodes_2, food, cycles, p_neg, p_pos, p_neu, b_pos, b_neg, b_neu, generation, negative_food)
        nodes_string = plot_values(nodes, generation, "")
        nodes_2_string = plot_values(nodes_2, generation, "")
        food_string = plot_food(food, generation, "")
        food_csv_string += food_string
        nodes_csv_string += nodes_string
        nodes_2_csv_String += nodes_2_string
        '''tmp_food = food_string.split("\n")
        for i in range(len(tmp_food)):
            sock.sendto(str.encode(tmp_food[i]+"\n"), (UDP_IP, UDP_PORT))'''
        sock.sendto(str.encode(nodes_string+nodes_2_string+food_string), (UDP_IP, UDP_PORT))
        if len(food) == 0:
            break
        generation += 1
    text_file = open("res.txt", "w")
    n = text_file.write(nodes_csv_string)
    text_file.close()
    
    text_file = open("food_res.txt", "w")
    n = text_file.write(food_csv_string)
    text_file.close()
    # make sure the 'COM#' is set according the Windows Device Manager
    