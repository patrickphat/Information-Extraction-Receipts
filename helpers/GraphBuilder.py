import numpy as np

def is_range(range_):

  # If left range is bigger or equal to right of the range
  if range_[0] >= range_[1]:
    # Return False since not a range
    return False

  return True

def intersect_range(range1, range2):
  x1 = max(range1[0],range2[0])
  x2 = min(range1[1],range2[1])

  if is_range((x1,x2)):
    return (x1,x2)
  else:
    return False 

def is_intersect_range(range1,range2):
  if intersect_range(range1, range2):
    return True
  
  return False

def out_bound_under(src_bbox,dest_bbox):

  return dest_bbox["x_max"] < src_bbox["x_min"] or dest_bbox["x_min"] > src_bbox["x_max"]

def block_sight_under(src_bbox,dest_bbox,mid_bbox):
  # Check if mid bbox doesn't block src_Bbox

  if out_bound_under(src_bbox,mid_bbox):
    return False
  mid_under_src = src_bbox["y_max"] <= mid_bbox["y_min"]
  dest_under_mid = mid_bbox["y_max"] <= dest_bbox["y_min"]
  return mid_under_src and dest_under_mid


def direct_sight_under(src_bbox,dest_bbox,list_bbox,src_idx,dest_idx):

  # If bottom edge of src doesn't appear out of bound of src
  if not out_bound_under(src_bbox,dest_bbox):

    # Look through all the bbox
    for i,mid_bbox in enumerate(list_bbox):

      if i == src_idx or i == dest_idx: 
        continue
     
      #If mid block sight of src_bbox to dest_bbox
      if block_sight_under(src_bbox,dest_bbox,mid_bbox):
        return False

  else:
     return False

  return True

def out_bound_right(src_bbox,dest_bbox,epsilon=2):

  return dest_bbox["y_min"]  > src_bbox["y_max"] - epsilon or dest_bbox["y_max"] - epsilon < src_bbox["y_min"]

def block_sight_right(src_bbox,dest_bbox,mid_bbox):
  # Check if mid bbox doesn't block src_Bbox

  if out_bound_right(src_bbox,mid_bbox):
    return False

  mid_right_src = src_bbox["x_max"] <= mid_bbox["x_min"]
  dest_right_mid = mid_bbox["x_max"] <= dest_bbox["x_min"]
  return mid_right_src and dest_right_mid



def direct_sight_right(src_bbox,dest_bbox,list_bbox,src_idx,dest_idx):

  # If bottom edge of src doesn't appear out of bound of src
  if not out_bound_right(src_bbox,dest_bbox):

    # Look through all the bbox
    for i,mid_bbox in enumerate(list_bbox):

      if i == src_idx or i == dest_idx: 
        # print("\ndest_bbox:",list_labels[dest_idx])
        # print("mid: ",list_labels[i])
        continue
     
      #If mid block sight of src_bbox to dest_bbox
      if block_sight_right(src_bbox,dest_bbox,mid_bbox):
        return False

  else:
     return False

  return True

def build_graph_from_coords(list_coords):

  # Number of bboxes in img
  N = len(list_coords)

  # Adjacency matrix in 5 different sectors: up down left right and self (identity matrix)
  adjacency_tensor = np.zeros((N,N,5))

  # The third one is the identity matrix
  adjacency_tensor[:,:,4] = np.identity(N)

  for src_idx,src_bbox in enumerate(list_coords):

    for dest_idx,dest_bbox in enumerate(list_coords):

      # Once look, never look back
      if dest_idx <= src_idx:
        continue
      
      if (direct_sight_under(src_bbox,dest_bbox,list_coords,src_idx,dest_idx)):

        # Update up adj matrix
        adjacency_tensor[:,:,0][src_idx][dest_idx] = 1

        # Update down adj matrix
        adjacency_tensor[:,:,1][dest_idx][src_idx] = 1

      if (direct_sight_right(src_bbox,dest_bbox,list_coords,src_idx,dest_idx)):

        # Update left adj matrix
        adjacency_tensor[:,:,2][src_idx][dest_idx] = 1

        # Update right adj matrix
        adjacency_tensor[:,:,3][dest_idx][src_idx] = 1

  return adjacency_tensor


