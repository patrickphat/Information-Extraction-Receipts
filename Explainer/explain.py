def explain(
    model,
    A_s,
    V_s,
    IMG_PATHS,
    IDX = 50,
    NODE = 1,
    LR = 1e-2,
    PRINT_EVERY = 50,
    EPOCHS = 300,
    THRESH = 0.15,
    BASE_LINE_THICKNESS = 2,
    BASE_BOX_THICKNESS = 2,
    ):
    
    explainer = Explainer(torch.Tensor([A_s[IDX]]),
                      torch.Tensor([V_s[IDX]]),
                      model=model,
                      num_nodes=160,
                      num_edges=5,
                      args={"use_mask_bias":False,
                     "init_strategy":"normal"},
                   coeffs={"ent_loss":2,
                      "size_loss": 0.0005})#.cuda

    explainer.explain(NODE,epochs=EPOCHS,lr=LR,print_every = PRINT_EVERY)

    with torch.no_grad():
        mask = torch.nn.Sigmoid()(explainer.edge_mask[:,:,:-1]).cpu().numpy()
    important_edges = np.array(np.where(mask>THRESH)).T
    edges_proba = []

    for edge in important_edges:
        origin,target,channel = edge
        proba = mask[origin,target,channel]
        edges_proba.append(proba)

    # Draw on image
    img_path = IMG_PATHS[IDX]
    img = cv2.imread(img_path)
    text = texts_test[IDX]
    max_len = len(text)
    values = []
    result = inference(model,IDX)
    predict_values = result[0].cpu().numpy()
    predict_values_type = []

    for i in range(max_len):
        value = predict_values[i]
        if value == 0:
            predict_values_type.append("Company")
        elif value == 1:
            predict_values_type.append("Date")
        elif value == 2:
            predict_values_type.append("Address")
        elif value == 3:
            predict_values_type.append("Price")
        else:
            predict_values_type.append("Other")



    # Draw edges
    for edge,proba in zip(important_edges,edges_proba):
        origin,target,channel = edge

        # Draw boxes
        try:
            x_min,y_min,x_max,y_max = get_coords(coords_test[IDX][origin])
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0), BASE_BOX_THICKNESS)
            x_min,y_min,x_max,y_max = get_coords(coords_test[IDX][target])
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0), BASE_BOX_THICKNESS)
        except:
            continue


    # Draw chosen box
    coord = coords_test[IDX][NODE]
    x_min = coord["x_min"]
    x_max = coord["x_max"]
    y_min = coord["y_min"]
    y_max = coord["y_max"]
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (36,255,12), 3)
    cv2.putText(img,predict_values_type[NODE], (x_max, y_max-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Draw connection line
    for edge,proba in zip(important_edges,edges_proba):
        origin,target,channel = edge
        try:
            mid_origin = find_mid_point(coords_test[IDX][origin])
            mid_target = find_mid_point(coords_test[IDX][target])
        except:
            continue  
      # Draw line
        cv2.line(img, mid_origin, mid_target, (0,0,255), BASE_LINE_THICKNESS)
    return img
    
def checkin(point, coords):
    # Check if a point is in a bbox (represented with 4 coords x_min,x_max,y_min,y_max)\
    
    x_min,y_min,x_max,y_max = get_coords(coords)

    if point[0] > x_min and point[0] < x_max and point[1] > y_min and point[1] < y_max:
        return True
    else: 
        return False

def findidx(point,coords):
    # Find idx of a point given a list of coords
    for i,coord in enumerate(coords):
        if checkin(point,coord):
            return i
    return False
    
def inference(model,idx):
    # Forward
    with torch.no_grad():
        model.eval()
        result = model.forward(torch.Tensor([A_s_test[idx]]).float(),torch.tensor([V_s_test[idx]]).float())
    return result.argmax(-1)

def find_mid_point(coord):
    # Find mid point of a bounding given 4 corresponding coords x_min,x_max,y_min,y_max
    x_min,x_max,y_min,y_max = coord['x_min'],coord['x_max'],coord['y_min'],coord['y_max']
    x_mid = int((x_min + x_max)/2)
    y_mid = int((y_min + y_max)/2)
    return (x_mid,y_mid)

def get_coords(coord):
    x_min,y_min,x_max,y_max = coord["x_min"],coord["y_min"],coord["x_max"],coord["y_max"]
    return x_min,y_min,x_max,y_max
    
