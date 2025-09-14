import pandas as pd
from math import sqrt
from pyomo.environ import ConcreteModel, Set, RangeSet, Param, Var, Binary, NonNegativeIntegers, NonNegativeReals, Constraint, Objective, minimize, SolverFactory, value

R1045 = pd.read_csv("../Dataset/ParsedData/R104-5.csv")
R1045v = pd.read_csv("../Dataset/ParsedData/R104-5 vehicle.csv")


## Define quantities

n = len(R1045)
c = n - 1

# Define a function to compute Euclidean distance

def Euclid_dist(x1, y1, x2, y2):
    return round(sqrt((x2 - x1)**2 + (y2 - y1)**2), 5)

# Define Distance Matrix

DistM = []

for i in range(n):
    row_dist = []
    for j in range(n):
        if i == j:
            row_dist.append(0)
        else:
            x1 = R1045['x'][i]
            y1 = R1045['y'][i]
            x2 = R1045['x'][j]
            y2 = R1045['y'][j]
            row_dist.append(Euclid_dist(x1, y1, x2, y2))
    DistM.append(row_dist)

max_dist = max(max(row) for row in DistM)

max_cap = R1045v['load_capacity'][0]
max_cap_b = R1045v['load_capacity'][1]
max_fuel = R1045v['fuel_capacity'][0]
max_fuel_b = R1045v['fuel_capacity'][1]
swap_time = R1045v['swap_time'][0]

demand = dict(R1045['demand'][1:])
ready_time = dict(R1045['ready_time'][1:])
due_date = dict(R1045['due_date'][1:])
service_time = dict(R1045['service_time'][1:])

max_due_date = R1045['due_date'][0]

# Convert lists into a dictionary

Dist = {(i, j): DistM[i][j] for j in range(n) for i in range(n)}

## Perform arc pruning

E_arcs = list(Dist.keys())
B_arcs = list(Dist.keys())

for j in range(n):
    for k in range(n):
        if j == k:
            E_arcs.remove((j,k))
            B_arcs.remove((j,k))
        if Dist[(j,k)] > max_fuel and j!=k:
            E_arcs.remove((j,k))
        if Dist[(j,k)] > max_fuel_b and j!= k:
            B_arcs.remove((j,k))
        if k != 0:
            if (Dist[(0,j)] + Dist[(j,k)]) > due_date[k]:
                E_arcs.remove((j,k))
                B_arcs.remove((j,k))
            
for k in range(n):
    if k != 0:
        if Dist[(0,k)] > due_date[k]:
            E_arcs.remove((0,k))
            B_arcs.remove((0,k))
        if Dist[(k,0)] > max_fuel:
            E_arcs.remove((k,0))
        if Dist[(k,0)] > max_fuel_b:
            B_arcs.remove((k,0))

## Define model, variables and parameters

model = ConcreteModel()

# Define sets

model.Nodes = RangeSet(0, c)
model.Customers = RangeSet(1, c)
model.E_arcs = Set(initialize = E_arcs, dimen = 2)
model.B_arcs = Set(initialize = B_arcs, dimen = 2)
model.ECVs = RangeSet(1, c)
model.BSVs = RangeSet(1, c)

# Define constant parameters

model.max_cap = Param(initialize = max_cap)
model.max_cap_b = Param(initialize = max_cap_b)
model.max_fuel = Param(initialize = max_fuel)
model.max_fuel_b = Param(initialize = max_fuel_b)
model.swap_time = Param(initialize = swap_time)

# Define remaining parameters

model.Dist = Param(model.Nodes, model.Nodes, initialize = Dist)
model.demand = Param(model.Customers, initialize = demand)
model.ready_time = Param(model.Customers, initialize = ready_time)
model.due_date = Param(model.Customers, initialize = due_date)
model.service_time = Param(model.Customers, initialize = service_time)

# Define Big M values

model.M_cap = Param(initialize = max(max_cap, max_cap_b))
model.M_fuel = Param(initialize = max(max_fuel, max_fuel_b) + max_dist)
model.M_time = Param(initialize = max_due_date + max_dist + max(service_time.values())+ swap_time)

# Define real variables

model.ta = Var(model.ECVs, model.Nodes, within = NonNegativeReals)
model.td = Var(model.ECVs, model.Nodes, within = NonNegativeReals)
model.tba = Var(model.BSVs, model.Nodes, within = NonNegativeReals)
model.tbd = Var(model.BSVs, model.Nodes, within = NonNegativeReals)
model.ea = Var(model.ECVs, model.Nodes, within = NonNegativeReals, bounds = (0, max_fuel))
model.ed = Var(model.ECVs, model.Nodes, within = NonNegativeReals, bounds = (0, max_fuel))
model.eba = Var(model.BSVs, model.Nodes, within = NonNegativeReals, bounds = (0, max_fuel_b))
model.ebd = Var(model.BSVs, model.Nodes, within = NonNegativeReals, bounds = (0, max_fuel_b))

# Define integer variables

model.ca = Var(model.ECVs, model.Customers, within = NonNegativeIntegers, bounds = (0, max_cap))
model.cd = Var(model.ECVs, model.Nodes, within = NonNegativeIntegers, bounds = (0, max_cap))
model.cba = Var(model.BSVs, model.Customers, within = NonNegativeIntegers, bounds = (0, max_cap_b))
model.cbd = Var(model.BSVs, model.Nodes, within = NonNegativeIntegers, bounds = (0, max_cap_b))
model.u = Var(model.ECVs, model.Customers, within = NonNegativeIntegers, bounds = (0, c))
model.ub = Var(model.BSVs, model.Customers, within = NonNegativeIntegers, bounds = (0, c))

# Define binary variables

model.x = Var(model.ECVs, model.E_arcs, within = Binary)
model.xb = Var(model.BSVs, model.B_arcs, within = Binary)
model.y = Var(model.ECVs, within = Binary)
model.yb = Var(model.BSVs, within = Binary)
model.z = Var(model.ECVs, model.BSVs, model.Customers, within = Binary)

## Define constraints

# Flow of vehicles

model.one_ECV_per_node = Constraint(model.Customers, rule = lambda m, k: sum(m.x[i,j,k1] for i in m.ECVs for j,k1 in m.E_arcs if k1 == k) == 1)
model.one_BSV_per_node = Constraint(model.Customers, rule = lambda m, k: sum(m.xb[b,j,k1] for b in m.BSVs for j,k1 in m.B_arcs if k1 == k) <= 1)

model.max_one_arc_per_ECV = Constraint(model.ECVs, model.Customers, rule= lambda m, i, k: sum(m.x[i,j,k1] for j,k1 in m.E_arcs if k1 == k) <= 1)
model.max_one_arc_per_BSV = Constraint(model.BSVs, model.Customers, rule= lambda m, b, k: sum(m.xb[b,j,k1] for j,k1 in m.B_arcs if k1 == k) <= 1)

model.ECV_flow = Constraint(model.ECVs, model.Nodes, rule= lambda m, i, k: sum(m.x[i,j,k1] for j,k1 in m.E_arcs if k1 == k) == sum(m.x[i,k2,l] for k2,l in m.E_arcs if k2 == k))
model.BSV_flow = Constraint(model.BSVs, model.Nodes, rule= lambda m, b, k: sum(m.xb[b,j,k1] for j,k1 in m.B_arcs if k1 == k) == sum(m.xb[b,k2,l] for k2,l in m.B_arcs if k2 == k))

model.BSV_iff_ECV = Constraint(model.Customers, rule= lambda m, k: sum(m.xb[b,j,k1] for b in m.BSVs for j,k1 in m.B_arcs if k1 == k) <= sum(m.x[i,j,k2] for i in m.ECVs for j,k2 in m.E_arcs if k2 == k))


# Usage Vs Travel Vs Swap

model.ECV_order = Constraint(model.ECVs, rule= lambda m, i: Constraint.Skip if i == 1 else m.y[i] <= m.y[i-1])
model.BSV_order = Constraint(model.BSVs, rule= lambda m, b: Constraint.Skip if b == 1 else m.yb[b] <= m.yb[b-1])

model.ECV_depot_and_y = Constraint(model.ECVs, rule= lambda m, i: sum(m.x[i,j,k] for j,k in m.E_arcs if (j == 0 and k in m.Customers)) == m.y[i])
model.BSV_depot_and_yb = Constraint(model.BSVs, rule= lambda m, b: sum(m.xb[b,j,k] for j,k in m.B_arcs if (j == 0 and k in m.Customers)) == m.yb[b])

model.BSV_iff_swap = Constraint(model.BSVs, model.Customers, rule= lambda m, b, k: sum(m.xb[b,j,k1] for j,k1 in m.B_arcs if k1 == k) <= sum(m.z[i,b,k] for i in m.ECVs))
model.one_vehicle_per_swap = Constraint(model.Customers, rule= lambda m, k: sum(m.z[i,b,k] for i in m.ECVs for b in m.BSVs) <= 1)

model.z_y = Constraint(model.ECVs, model.BSVs, model.Customers, rule= lambda m, i, b, k: m.z[i,b,k] <= m.y[i])
model.z_yb = Constraint(model.ECVs, model.BSVs, model.Customers, rule= lambda m, i, b, k: m.z[i,b,k] <= m.yb[b])

model.z_x = Constraint(model.ECVs, model.BSVs, model.Customers, rule= lambda m, i, b, k: m.z[i,b,k] <= sum(m.x[i,j,k1] for j,k1 in m.E_arcs if k1 == k))
model.z_xb = Constraint(model.ECVs, model.BSVs, model.Customers, rule= lambda m, i, b, k: m.z[i,b,k] <= sum(m.xb[b,j,k1] for j,k1 in m.B_arcs if k1 == k))


# MTZ subtour elimination

model.u_LB = Constraint(model.ECVs, model.Customers, rule= lambda m, i, j: m.u[i,j] >= sum(m.x[i,j1,k] for j1,k in m.E_arcs if j1 == j))
model.u_UB = Constraint(model.ECVs, model.Customers, rule= lambda m, i, j: m.u[i,j] <= c*sum(m.x[i,j1,k] for j1,k in m.E_arcs if j1 == j))

model.ub_LB = Constraint(model.BSVs, model.Customers, rule= lambda m, b, j: m.ub[b,j] >= sum(m.xb[b,j1,k] for j1,k in m.B_arcs if j1 == j))
model.ub_UB = Constraint(model.BSVs, model.Customers, rule= lambda m, b, j: m.ub[b,j] <= c*sum(m.xb[b,j1,k] for j1,k in m.B_arcs if j1 == j))

model.ECV_subtour = Constraint(model.ECVs, model.Customers, model.Customers, rule= lambda m, i, j, k: Constraint.Skip if (j,k) not in m.E_arcs else m.u[i,j] - m.u[i,k] + c*m.x[i,j,k] <= c - 1)
model.BSV_subtour = Constraint(model.BSVs, model.Customers, model.Customers, rule= lambda m, b, j, k: Constraint.Skip if (j,k) not in m.B_arcs else m.ub[b,j] - m.ub[b,k] + c*m.xb[b,j,k] <= c - 1)


# Capacity constraints

model_ECV_demand_max_cap = Constraint(model.ECVs, rule= lambda m, i: sum(m.demand[k] * sum(m.x[i,j,k] for j,k in m.E_arcs if j != k) for k in m.Customers) <= m.max_cap)
model_ECV_max_cap = Constraint(model.ECVs, rule= lambda m, i: m.cd[i,0] == m.max_cap * m.y[i])
model.ECV_cap_update = Constraint(model.ECVs, model.Customers, rule= lambda m, i, k: m.cd[i,k] == m.ca[i,k] - m.demand[k] * sum(m.x[i,j,k1] for j,k1 in m.E_arcs if k1 == k))
model.ECV_cap_1 = Constraint(model.ECVs, model.Nodes, model.Customers, rule= lambda m, i, j, k: Constraint.Skip if (j,k) not in m.E_arcs else m.ca[i,k] - m.cd[i,j] <= m.M_cap * (1 - m.x[i,j,k]))
model.ECV_cap_2 = Constraint(model.ECVs, model.Nodes, model.Customers, rule= lambda m, i, j, k: Constraint.Skip if (j,k) not in m.E_arcs else m.ca[i,k] - m.cd[i,j] >= -m.M_cap * (1 - m.x[i,j,k]))

model_BSV_max_cap = Constraint(model.BSVs, rule= lambda m, b: m.cbd[b,0] == m.max_cap_b * m.yb[b])
model.BSV_cap_1 = Constraint(model.BSVs, model.Nodes, model.Customers, rule= lambda m, b, j, k: Constraint.Skip if (j,k) not in m.B_arcs else m.cba[b,k] - m.cbd[b,j] <= m.M_cap * (1 - m.xb[b,j,k]))
model.BSV_cap_2 = Constraint(model.BSVs, model.Nodes, model.Customers, rule= lambda m, b, j, k: Constraint.Skip if (j,k) not in m.B_arcs else m.cba[b,k] - m.cbd[b,j] >= -m.M_cap * (1 - m.xb[b,j,k]))


# Fuel constraints

model.ECV_start_max_fuel = Constraint(model.ECVs, rule= lambda m, i: m.ed[i,0] == m.max_fuel)
model.ECV_arr_dep_fuel = Constraint(model.ECVs, model.Customers, rule= lambda m, i, k: m.ed[i,k] >= m.ea[i,k])
model.ECV_fuel_1 = Constraint(model.ECVs, model.E_arcs, rule= lambda m, i, j, k: m.ea[i,k] - m.ed[i,j] + m.Dist[j,k] <= m.M_fuel * (1 - m.x[i,j,k]))
model.ECV_fuel_2 = Constraint(model.ECVs, model.E_arcs, rule= lambda m, i, j, k: m.ea[i,k] - m.ed[i,j] + m.Dist[j,k] >= -m.M_fuel * (1 - m.x[i,j,k]))

model.BSV_start_max_fuel = Constraint(model.BSVs, rule= lambda m, b: m.ebd[b,0] == m.max_fuel_b)
model.BSV_arr_dep_fuel = Constraint(model.BSVs, model.Customers, rule= lambda m, b, k: m.ebd[b,k] == m.eba[b,k])
model.BSV_fuel_1 = Constraint(model.BSVs, model.B_arcs, rule= lambda m, b, j, k: m.eba[b,k] - m.ebd[b,j] + m.Dist[j,k] <= m.M_fuel * (1 - m.xb[b,j,k]))
model.BSV_fuel_2 = Constraint(model.BSVs, model.B_arcs, rule= lambda m, b, j, k: m.eba[b,k] - m.ebd[b,j] + m.Dist[j,k] >= -m.M_fuel * (1 - m.xb[b,j,k]))


# Time constraints

model.ECV_depot_dep_time = Constraint(model.ECVs, rule= lambda m, i: m.td[i,0] == 0)

model.ECV_time_1 = Constraint(model.ECVs, model.Customers, rule= lambda m, i, k: m.td[i,k] <= m.ta[i,k] + m.service_time[k] + m.M_time * (1 - sum(m.x[i,j,k1] for j,k1 in m.E_arcs if k1 == k)) + m.swap_time * sum(m.z[i,b,k] for b in m.BSVs))
model.ECV_time_2 = Constraint(model.ECVs, model.Customers, rule= lambda m, i, k: m.td[i,k] >= m.ta[i,k] + m.service_time[k] - m.M_time * (1 - sum(m.x[i,j,k1] for j,k1 in m.E_arcs if k1 == k)) + m.swap_time * sum(m.z[i,b,k] for b in m.BSVs))
model.BSV_time = Constraint(model.BSVs, model.Customers, rule= lambda m, b, k: m.tbd[b,k] >= m.tba[b,k] + m.swap_time * sum(m.z[i,b,k] for i in m.ECVs) - m.M_time * (1 - sum(m.xb[b,j,k1] for j,k1 in m.B_arcs if k1 == k)))

model.ECV_arr_dep_time = Constraint(model.ECVs, model.E_arcs, rule= lambda m, i, j, k: m.ta[i,k] - m.td[i,j] - m.Dist[j,k] >= -m.M_time * (1 - m.x[i,j,k]))
model.BSV_arr_dep_time = Constraint(model.BSVs, model.B_arcs, rule= lambda m, b, j, k: m.tba[b,k] - m.tbd[b,j] - m.Dist[j,k] >= -m.M_time * (1 - m.xb[b,j,k]))

model.ECV_tw_1 = Constraint(model.ECVs, model.Customers, rule= lambda m, i, k: m.ta[i,k] >= m.ready_time[k] - m.M_time * (1 - sum(m.x[i,j,k1] for j,k1 in m.E_arcs if k1 == k)))
model.ECV_tw_2 = Constraint(model.ECVs, model.Customers, rule= lambda m, i, k: m.ta[i,k] <= m.due_date[k] + m.M_time * (1 - sum(m.x[i,j,k1] for j,k1 in m.E_arcs if k1 == k)))
model.BSV_tw_1 = Constraint(model.BSVs, model.Customers, rule= lambda m, b, k: m.tba[b,k] >= m.ready_time[k] - m.M_time * (1 - sum(m.z[i,b,k] for i in m.ECVs)))
model.BSV_tw_2 = Constraint(model.BSVs, model.Customers, rule= lambda m, b, k: m.tba[b,k] <= m.due_date[k] + m.M_time * (1 - sum(m.z[i,b,k] for i in m.ECVs)))


# Battery swap constraints

model.ECV_swap_1 = Constraint(model.ECVs, model.Customers, rule= lambda m, i, k: m.ed[i,k] - m.ea[i,k] <= m.M_fuel * sum(m.z[i,b,k] for b in m.BSVs))
model.ECV_swap_2 = Constraint(model.ECVs, model.Customers, rule= lambda m, i, k: m.ed[i,k] >= m.max_fuel * sum(m.z[i,b,k] for b in m.BSVs))

model.BSV_swap = Constraint(model.BSVs, model.Customers, rule= lambda m, b, k: m.cbd[b,k] == m.cba[b,k] - sum(m.z[i,b,k] for i in m.ECVs))

#%%

## Create a Greedy Algorithm

unserved = list(range(1,n))
greedy_ecv_routes = []
greedy_bsv_routes = []
swaps = []
infos = []
y = 0
yb = 0
while len(unserved):
    y += 1
    route = [0]
    current = 0
    fuel = max_fuel
    load = max_cap
    time = 0
    
    while True:
        cand = []
        for k in unserved:
            dist = Dist[(current, k)]
            arrival = time + dist
            start_service = max(arrival, ready_time[k])
            if (start_service <= due_date[k]) and (dist <= fuel) and (demand[k] <= load):
                cand.append((k, dist, start_service))
        if not cand:
            break
        
        cand.sort(key=lambda x:(due_date[x[0]], x[1]))
        k, travel, start_service = cand[0]
        route.append(k)
        unserved.remove(k)
        
        time = start_service + service_time[k]
        fuel -= dist
        load -= demand[k]
        current = k
        
        infos.append({'ECV' : y, 'customer' : k, 'ta' : start_service, 'td' : time, 'ea' : (fuel + dist), 'ed' : fuel, 'ca' : (load + demand[k]), 'cd' : load})
        
        depot_back = Dist[(current,0)]
        if depot_back > fuel:
            swaps.append({'ECV' : y, 'customer' : k, 'time' : time})
            break
        
    route.append(0)
    greedy_ecv_routes.append(route)

    if swaps:
        swaps_sorted = sorted(swaps, key=lambda s: s['time'])
        br = [0]
        cur = 0
        for s in swaps_sorted:
            k = s['customer']
            br.append(k)
            cur = k
        br.append(0)
        greedy_bsv_routes.append(br)     

## Filter ECV routes

remove_ecv_route = []
for i,r in enumerate(greedy_ecv_routes):
    for j in range(len(r) - 1):
        if (r[j], r[j+1]) not in E_arcs:
            remove_ecv_route.append([i,(r[j], r[j+1])])

for r in remove_ecv_route:
    j,k = r[1]
    greedy_ecv_routes[r[0]].remove(j)
    greedy_ecv_routes[r[0]].remove(k)
    
    
## Filter BSV routes

bsv_routes_temp = []
for r in greedy_bsv_routes:
    if r not in bsv_routes_temp:
        bsv_routes_temp.append(r)

remove = []
for i in range(len(bsv_routes_temp)-1):
    if set(bsv_routes_temp[i]) <= set(bsv_routes_temp[i+1]):
        remove.append(bsv_routes_temp[i])

for r in remove:
    bsv_routes_temp.remove(r)

greedy_bsv_routes = bsv_routes_temp

remove_bsv_route = []
for b,r in enumerate(greedy_bsv_routes):
    for j in range(len(r) - 1):
        if (r[j], r[j+1]) not in B_arcs:
            remove_bsv_route.append([b,(r[j], r[j+1])])

for r in remove_bsv_route:
    j,k = r[1]
    greedy_bsv_routes[r[0]].remove(j)
    greedy_bsv_routes[r[0]].remove(k)

## Print values 

greedy_ecvs = y
print("Greedy Algorithm Solution:")

greedy_ecv_dist = 0
greedy_bsv_dist = 0
for i,route in enumerate(greedy_ecv_routes, start = 1):
    for r in range(len(route) - 1):
        greedy_ecv_dist += model.Dist[(route[r], route[r+1])]

for b,route in enumerate(greedy_bsv_routes, start = 1):
    for r in range(len(route) - 1):
        greedy_bsv_dist += model.Dist[(route[r], route[r+1])]

def objective(ecv_dist, bsv_dist, Necvs, Nbsvs, Nswaps):
    return 2*ecv_dist + bsv_dist + 1.5*Necvs + Nbsvs + Nswaps  

greedy_cost = objective(greedy_ecv_dist, greedy_bsv_dist, greedy_ecvs, 1, len(swaps))

print(f"Total cost = {greedy_cost}")
print(f"Total Number of Vehicles = {greedy_ecvs+1}")
print(f"Total Distance = {greedy_ecv_dist + greedy_bsv_dist}")
print(f"Total ECVs = {greedy_ecvs}")
print(f"Total ECVs Distance = {greedy_ecv_dist}")
print("Total BSVs = 1")
print(f"Total BSVs Distance = {greedy_bsv_dist}")
print(f"Total number of Swapping Services Requested = {len(swaps)}")
print("ECV Routes:")
for route in greedy_ecv_routes:
    print(f"{' -> '.join(map(str, route))}")

print("BSV Routes:")
for route in greedy_bsv_routes:
    print(f"{' -> '.join(map(str, route))}")

infeasible = {}
infeasible['yes?'] = 0
for info in infos:
    if info['ea'] < 0:
        infeasible = {'yes?' : 1, 'ECV' : info['ECV'], 'k' : info['customer'], 'd/a' : 'a', 'fuel' : info['ea']}

    if info['ed'] < 0:
        infeasible = {'yes?' : 1, 'ECV' : info['ECV'], 'k' : info['customer'], 'd/a' : 'd', 'fuel' : info['ed']}

if infeasible['yes?'] == 1:
    if infeasible['d/a'] == 'd':
        print(f"Greedy infeasible, negative departure fuel of {infeasible['fuel'] : .2f} at node {infeasible['k']} for ECV {infeasible['ECV']}")
    
    if infeasible['d/a'] == 'a':
        print(f"Greedy infeasible, negative arrival fuel of {infeasible['fuel'] : .2f} at node {infeasible['k']} for ECV {infeasible['ECV']}")
          
else:
    print("Gready feasible")
 

#%%

model.obj = Objective(
    rule=lambda m: objective(sum(m.Dist[j, k] * m.x[i, j, k] for i in m.ECVs for j,k in m.E_arcs),
                 sum(m.Dist[j, k] * m.xb[b, j, k] for b in m.BSVs for j,k in m.B_arcs),
                 sum(m.z[i, b, k] for i in m.ECVs for b in m.BSVs for k in m.Customers),
                 sum(m.y[i] for i in m.ECVs),
                 sum(m.yb[b] for b in m.BSVs)),
    sense=minimize
)


solver = SolverFactory("gurobi")
results = solver.solve(model)

ecv_routes = []
bsv_routes = []
print("Optimization model solution:")
ecvs = 0
bsvs = 0
swaps = int(sum(value(model.z[i, b, k]) for i in model.ECVs for b in model.BSVs for k in model.Customers))
for i in model.ECVs:
    if value(model.y[i]) > 0.5:
        ecvs += 1
        # Build route
        route = [0]
        current = 0
        visited = set()
        while True:
            next_nodes = [k for j, k in model.E_arcs if (j == current and value(model.x[i, j, k]) > 0.5)]
            if not next_nodes:
                break
            nxt = next_nodes[0]
            route.append(nxt)
            if nxt == 0:
                break
            if nxt in visited:
                route.append("!!subtour!!")
                break
            visited.add(nxt)
            current = nxt
        ecv_routes.append(route)                

for b in model.BSVs:
    if value(model.yb[b]) > 0.5:
        bsvs += 1
        route = [0]
        current = 0
        visited = set()
        while True:
            next_nodes = [k for j,k in model.B_arcs if (j == current and value(model.xb[b, j, k]) > 0.5)]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            route.append(next_node)
            if next_node == 0:
                break
            if next_node in visited:
                route.append("!!subtour!!")
                break
            visited.add(next_node)
            current = next_node
        bsv_routes.append(route)

ecv_total_dist = sum((value(model.Dist[j, k]) * value(model.x[i, j, k])) for i in model.ECVs for j,k in model.E_arcs if value(model.x[i,j,k] > 0.5))
bsv_total_dist = sum((value(model.Dist[j, k]) * value(model.xb[b, j, k])) for b in model.BSVs for j,k in model.B_arcs if value(model.xb[b,j,k] > 0.5))         
total_dist = ecv_total_dist + bsv_total_dist

print(f"Total Cost = {value(model.obj)}")
print(f"Total Number of Vehicles = {ecvs + bsvs}")
print(f"Total Distance = {total_dist}")
print(f"Total ECVs = {ecvs}")
print(f"Total ECVs Distance = {ecv_total_dist}")
print(f"Total BSVs = {bsvs}")
print(f"Total BSVs Distance = {bsv_total_dist}")
print(f"Total number of Swapping Services Requested = {swaps}")
print("ECV Routes:")
for route in ecv_routes:
    print(f"{' -> '.join(map(str, route))}")

print("BSV Routes:")
for route in bsv_routes:
    print(f"{' -> '.join(map(str, route))}")

