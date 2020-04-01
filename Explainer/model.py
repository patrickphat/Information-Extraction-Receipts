class Explainer(nn.Module):
    def __init__(self,
               A,
               V,
               model,
               num_nodes,
               num_edges,
               args={"use_mask_bias":True,
                     "init_strategy":"normal"},
               coeffs={"ent_loss":1,
                      "size_loss": 0.005,}):
        super(Explainer, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.args = args
        self.coeffs = coeffs

        self.A = A # Adjacency matrix
        self.V = V # Node features

        self.edge_mask, self.mask_bias = self.construct_edge_mask(num_nodes,num_edges)


        # Freeze all model parameters
        for parameter in model.parameters():
            parameter.requires_grad = False

        self.model = model # GCN Multi-edges model

    def explain(self,node_idx,epochs=30,lr=1e-3,print_every=20):

        optimizer = torch.optim.Adam(self.parameters(),lr=lr)

        V_true = self.prepare_target(node_idx)

        for t in range(epochs):    

            # Forward pass: Compute predicted y by passing x to the model
            V_pred = self.forward(node_idx)

            # Compute and print loss

            loss,pred_loss,mask_ent_loss,size_loss = self.loss(V_true,V_pred)

            if t%print_every == print_every-1:
                print("epochs {} loss {} pred_loss {} mask_ent_loss {} size_loss {}".format(t+1,loss,pred_loss,mask_ent_loss,size_loss))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
      
    

    def construct_edge_mask(self, num_nodes,num_edges, const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes,num_edges))
        mask[:,:,-1].require_grads = False

        if self.args['init_strategy'] == "normal":
            # std = nn.init.calculate_gain("relu") * math.sqrt(
            #     2.0 / (num_nodes + num_nodes + num_edges)
            # )
            # with torch.no_grad():
            #     mask.normal_(1.0, std)
            #     # mask.clamp_(0.0, 1.0)
            nn.init.xavier_normal(mask)


        elif self.args['init_strategy'] == "const":
            nn.init.constant_(mask, const_val)

        if self.args['use_mask_bias']:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes,num_edges))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = 0
        
        
        return mask, mask_bias
  
    def forward(self,node_idx):
        
        # Apply mask to adjacency tensor
        new_A = self.A*self.edge_mask + self.mask_bias
        
        # Forward with new masked adjacency tensors
        V = self.model.forward(new_A,self.V)
        
        return V[0][node_idx] # return prediction of target node
  
    def loss(self,V_true,V_pred):

        # Turn into 2 probability distribution
        V_true = torch.nn.Softmax()(V_true) + 1e-3
        V_pred = torch.nn.Softmax()(V_pred) + 1e-3


        # Cross entropy on 2 probability distribution
        pred_loss = -torch.mean(V_true*torch.log(V_pred))
        
        # Apply sigmoid to force range (0,1)
        mask = torch.sigmoid(self.edge_mask) + 1e-3
        
        # Entropy loss to measure uncertainty of the mask
        mask_ent = -mask*torch.log(mask) -(1-mask)*torch.log(1-mask)
        mask_ent_loss = self.coeffs["ent_loss"]*torch.mean(mask_ent)
           
        # Size loss will penalize large masks with lots of edges
        size_loss = self.coeffs["size_loss"] * torch.sum(mask)
        
        # The ultimate loss is the sum of loss
        loss = pred_loss + mask_ent_loss + size_loss

        return loss, pred_loss,mask_ent_loss,size_loss
  
  ## all other part became -100 for crossentropy
    def prepare_target(self,node_idx):
        target = self.model(self.A,self.V)[0] # (N,F)
        return target[node_idx] # (F)
