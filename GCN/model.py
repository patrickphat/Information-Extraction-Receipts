class GraphConv(nn.Module):
    def __init__(self, input_dim,output_dim,num_edges,use_act=None, use_bn = False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(GraphConv, self).__init__()

        self.C = output_dim
        self.L = num_edges
        self.F = input_dim
        
        
        self.filter_params = nn.Parameter(torch.FloatTensor(self.F * self.L,self.C))#.cuda()
        self.bias = nn.Parameter(torch.FloatTensor(self.C))#.cuda()
        self.use_bn = use_bn
        if use_act == "relu":
          self.act = torch.nn.ReLU()
        elif use_act =="softmax":
          self.act = torch.nn.Softmax(dim=-1)
        else:
          self.act = None
        #self.my_parameters = nn.ModuleList([self.filter_params.parameters(),self.bias.parameters()])
        nn.init.xavier_normal(self.filter_params)
        nn.init.normal(self.bias, mean=0.0001,std=0.0005)

    def apply_bn(self,x):
      batch_norm_module = nn.BatchNorm1d(x.size()[1])
      if next(self.parameters()).is_cuda:
        batch_norm_module.cuda()
      return batch_norm_module(x)

    def forward(self, A_s,V_s):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        
        # Get number of batches
        B = list(A_s.size())[0]

        # Get number of features
        N = list(A_s.size())[1]

        # Give up first dimension
        A_s = A_s.view(B*N,N,self.L)

        A_s = A_s.transpose(1,2).reshape(-1,(self.L)*N,N)
       
        new_V = torch.matmul(A_s,V_s.view(-1,N,self.F)).view(-1,N,self.L*self.F)
       
        V_out = torch.matmul(new_V,self.filter_params) + self.bias.unsqueeze(0)

        #return V_out
        if self.act:
          V_out = self.act(V_out)
        
        V_out = V_out.view(B,N,self.C)

        #print(V_out.shape)
        if self.use_bn:
          return self.apply_bn(V_out)#).view(B,N,self.C))
        else:
          return V_out



class GCN_Classifier(nn.Module):

  def __init__(self, input_size=768, num_classes = 5,hidden_size=256, num_edges = 5,use_bn=True):
    super(GCN_Classifier, self).__init__()
    self.input_size = 768
    self.num_classes = num_classes
    self.hidden_size = hidden_size
    self.num_edges = num_edges
    self.use_bn = use_bn

    self.linemb1 = LinearEmbedding(input_size,self.hidden_size,use_act="relu")
    self.dropout1 = torch.nn.modules.Dropout(p=0.5)
    
    self.gcn1 = GraphConv(self.hidden_size,self.hidden_size,num_edges,use_act='relu')
    self.dropout2 = torch.nn.modules.Dropout(p=0.5)

    self.gcn2 = GraphConv(self.hidden_size,self.hidden_size,num_edges,use_act='relu')
    self.dropout3 = torch.nn.modules.Dropout(p=0.5)

    self.gcn3 = GraphConv(self.hidden_size,self.hidden_size*2,num_edges,use_act='relu')
    self.dropout4 = torch.nn.modules.Dropout(p=0.5)

    self.linemb2  =  LinearEmbedding(self.hidden_size*2,int(self.hidden_size/2),use_act='relu')
    self.dropout5 = torch.nn.modules.Dropout(p=0.5)

    self.gcn4 = GraphConv(int(self.hidden_size/2),int(self.hidden_size/2),num_edges,use_act='relu')
    self.dropout6 = torch.nn.modules.Dropout(p=0.5)

    self.gcn5 = GraphConv(int(self.hidden_size/2),int(self.hidden_size/2),num_edges,use_act='relu')

    self.out_linemb  =  LinearEmbedding(int(self.hidden_size/2),num_classes,use_act=None)
    self.dropout7 = torch.nn.modules.Dropout(p=0.5)


  def forward(self,A_s,V_s):
    V1 = self.dropout2(self.gcn1(A_s,self.dropout1(self.linemb1(V_s))))
    V2 = self.dropout3(self.gcn2(A_s,V1))
    V3 = self.dropout4(self.gcn3(A_s,V2))
    V4 = self.dropout6(self.gcn4(A_s,self.dropout5(self.linemb2(V3))))
    V5 = self.out_linemb(self.gcn5(A_s,V4))#self.dropout7(self.out_linemb(self.gcn5(A_s,V4)))
    return V5

class LinearEmbedding(nn.Module):
  def __init__(self,input_size,output_size,use_bias=True,use_act="relu"):
    super(LinearEmbedding,self).__init__()
    self.C = output_size
    self.F = input_size
    
    self.W = nn.Parameter(torch.FloatTensor(self.F,self.C))
    self.B = nn.Parameter(torch.FloatTensor(self.C))

    if use_act == "relu":
      self.act = torch.nn.ReLU()
    elif use_act =="softmax":
      self.act = torch.nn.Softmax(dim=-1)
    else:
      self.act = None
    
    nn.init.xavier_normal_(self.W)
    nn.init.normal(self.B,mean=1e-4,std=1e-5)
  
  def forward(self,V):
    # V shape B,N,F

    V_out = torch.matmul(V,self.W) + self.B

    if self.act:
      V_out = self.act(V_out)
    
    return V_out

