# interaction_model
code for interacting  
  
1.GlobalGraph.py->A2A and attention layer from laneGCN  
  instead of transformer, use the distance feature  
  dist = agt_ctrs[hi] - ctx_ctrs[wi] # why use the (x1,y1)-(x2,y2) as distance feature?  
  each batch includes agent feature = [A,D]  
