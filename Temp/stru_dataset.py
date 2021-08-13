"""Dataset for Graph Isomorphism Network(GIN)
(chen jun): Used for compacted graph kernel dataset in GIN

Data sets include:

MUTAG, COLLAB, IMDBBINARY, IMDBMULTI, NCI1, PROTEINS, PTC, REDDITBINARY, REDDITMULTI5K
https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip
"""




from dgl.graph import DGLGraph



class STRDataset(object):
    """Datasets for Graph Isomorphism Network (GIN)
    Adapted from https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip.

    The dataset contains the compact format of popular graph kernel datasets, which includes: 
    MUTAG, COLLAB, IMDBBINARY, IMDBMULTI, NCI1, PROTEINS, PTC, REDDITBINARY, REDDITMULTI5K

    This datset class processes all data sets listed above. For more graph kernel datasets, 
    see :class:`TUDataset`

    Paramters
    ---------
    name: str
        dataset name, one of below -
        ('MUTAG', 'COLLAB', \
        'IMDBBINARY', 'IMDBMULTI', \
        'NCI1', 'PROTEINS', 'PTC', \
        'REDDITBINARY', 'REDDITMULTI5K')
    self_loop: boolean
        add self to self edge if true
    degree_as_nlabel: boolean
        take node degree as label and feature if true

    """

    def __init__(self, strus, features, targets, self_loop, degree_as_nlabel=False):
        """Initialize the dataset."""

       
     
        self.strus = strus
        self.features = features
        self.targets = targets
        
        self.self_loop = self_loop

        self.graphs = []
        self.labels = []

        # relabel
        self.glabel_dict = {}
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}

        # global num
        self.N = 0  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        # global num of classes
        self.gclasses = 0
        self.nclasses = 0
        self.eclasses = 0
        self.dim_nfeats = 0

        # flags
        self.degree_as_nlabel = degree_as_nlabel
        self.nattrs_flag = False
        self.nlabels_flag = False
        self.verbosity = False

        # calc all values
        self._load()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]



    def _load(self):
        """ Loads input dataset from dataset/NAME/NAME.txt file

        """


        #print('loading data...')
        
        self.N = len(self.strus)
        
        for i in range(self.N):
            
            stru = self.strus[i]
            
            n_nodes = len(stru)
            
            glabel = self.targets[i]         
      
            self.labels.append(glabel)

            g = DGLGraph()           
            
            g.add_nodes(n_nodes)
        
            
            #nlabels1 = []  # node labels
            #nlabels2 = []  # node labels
            
            #nattrs1 = []  # node attributes if it has
            #nattrs2 = []  # node attributes if it has
            
            m_edges = 0 

        
            for j in range(n_nodes):
                
                row = stru[j]
                
                neig = ((row == 1).nonzero())
                neig = neig[neig!=j]
                
                m_edges+=len(neig)
             
                if len(neig)>0:
                    g.add_edges(j, neig)
               
                # add self loop
                if self.self_loop:
                    m_edges += 1
                    g.add_edge(j, j)  
                   
                    
                    
            #print('Graph: %d' %i)
            #print('n_nodes:%d' %n_nodes)
            #print(self.features[i])        
            g.ndata['attr'] = self.features[i]
            
            self.graphs.append(g)
           

        self.gclasses = len(self.glabel_dict)
        self.nclasses = len(self.features[0][0])
        self.eclasses = len(self.elabel_dict)
        self.dim_nfeats = len(self.graphs[0].ndata['attr'][0])       


        '''print('Done.')
        print(
            """
            -------- Data Statistics --------'
            #Graphs: %d
            #Graph Classes: %d
            #Nodes: %d
            #Node Classes: %d
            #Node Features Dim: %d
            #Edges: %d
            #Edge Classes: %d
            Avg. of #Nodes: %.2f
            Avg. of #Edges: %.2f
            Graph Relabeled: %s
            Node Relabeled: %s
            Degree Relabeled(If degree_as_nlabel=True): %s \n """ % (
                self.N, self.gclasses, self.n, self.nclasses,
                self.dim_nfeats, self.m, self.eclasses,
                self.n / self.N, self.m / self.N, self.glabel_dict,
                self.nlabel_dict, self.ndegree_dict)) '''                 
        
