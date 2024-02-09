x=[]
for i in range(4):
    x.append(int(input("enter the elements:")))
     
def manhattan_distance(x):
   
    manhattan_dist = abs(x[0]-x[2])+abs(x[1]-x[3])
    
    return manhattan_dist

def euclidean_dist(x):
   
    return ((x[0]-x[2])**2 + (x[1]-x[3])**2)**0.5
result=euclidean_dist(x)
result1=manhattan_distance(x)
print(result)
print(result1)

#question2
def euclidean_dist(point1, point2):
   
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
def KNN(k, points):
    distances = []

    for i in range(1, len(points)):
        calculated_distance = euclidean_distance(points[0], points[i])
        distances.append((calculated_distance, points[i][2]))
    
    for j in range(len(distances)):
        for l in range(0, len(distances) - j - 1):
            if distances[l][0] > distances[l + 1][0]: 
                temp = distances[l]
                distances[l] = distances[l + 1]
                distances[l + 1] = temp
        
    k_nearest = distances[:k]

    frequency1 = 0
    for distance, label in k_nearest: 
        if label == 1:
            frequency1 += 1
    
    frequency2 = k - frequency1

    if frequency1 > frequency2: 
        return "Belongs to the first class"
    else: 
        return "Belongs to the second class"
    
#question3
def label_encode(labels):
    unique_labels_set = set(labels)
    label_to_code_map = {}
    code = 0

    for unique_label in unique_labels_set:
        label_to_code_map[unique_label] = code
        code += 1
    
    encoded_labels = []
    for label in labels:
        encoded_labels.append(label_to_code_map[label])
    
    return encoded_labels, label_to_code_map
#question4
def hot_encode(labels):
    unique_labels_sorted = sorted(set(labels)) 
    one_hot_encoding = {}
    for label in unique_labels_sorted:
        one_hot_encoding[label] = [0] * len(unique_labels_sorted)
   
    for i, label in enumerate(unique_labels_sorted):
        one_hot_encoding[label][i] = 1
    
    encoded_labels = []
    for label in labels:
        encoded_labels.append(one_hot_encoding[label])
    
    return encoded_labels, one_hot_encoding

def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    attributes = []
    data = []
    labels = []

    read_data = False
    for line in lines:
        line = line.strip()

        if not line or line.startswith('%'):
            continue
        
        if line.lower().startswith('@attribute'):
            attribute_name = line.split()[1]
            attributes.append(attribute_name)
        
        elif line.lower().startswith('@data'):
            read_data = True
        
        elif read_data:
            data.append(line.split(','))
        
        elif line.lower().startswith('@relation'):
            relation_name = line.split()[1]
        
        elif line.startswith('{'):
            labels.extend(line.strip('{}').split(','))
    
    return data, attributes, labels

file_path = 'stackex_coffee.arff'
data, attributes, labels = read_file(file_path)

coordinates = []
for row in data:
    features = [float(value) for value in row[:-1]]
    label = int(row[-1])
    coordinates.append((features, label))

k = int(input("Enter a certain value of k: "))

result = KNN(k, coordinates)
print("Classification result:", result)