#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:27:03 2025

@author: chandler
"""

################
#NAME: generalConstraintDecomp
#CREATED: Chandler
#MODIFY DATE: 3/24/2025
#DESC: Ingests a networkx graph object, with edge weights and node concentrations as 'weight' and 'conc'.
#   Computes each of (Johnson and Sasson) terms decomposed from Burt's (1992) constraint measure. 
#NOTES
#   1/ Should force dropping of isolates, or skip them?
#   2/ Handles self-loops as influencing pijs (via totaActivity)
#   3/ Presumes no duplicates, including self-loops (though networkx doesn't support them anyway?)
#   4/ There's some inefficiency in the quadratic calculations. First, we traverse every graph twice: once
#       in each direction. Second, once the code is validated, it's not necessary to calculate both IR and CC. 
#       One is calculable from Ci and the other terms. 
#   5/ edgeDeleteThreshold deletes all edges with pij values < x. This allows replicating Burt (2008),
#       who deletes edges with pij < 0.02.
#   6/ Add multiprocessing at the i-level (outer loop)?
#   7/ Delete weight-0 edges prior to creating missing reciprocal edges?
#  
#   8/ 3/24/2025: Added an option to recompute indirect terms based on i's ego network, as per Everett and Borgatti.
#       Note that it's not clear how to handle an alter's tie back to ego in this case. Is i included in her own 
#       ego network? Probably not. But w/o that, the ego network logic can even more dramatically inflate alters' j
#       connections w/in i's ego network. 
#   9/ Changed quadratic computation to prioritize IR ("not in") rather than CC. This allows computing CC as residual of C
#       and everything else. Since CC tends to be much larger than IR, this should save maximum time. 
#   10/ 3/28/2025: Added an option to preemptively cull self-loops. If enabled, this elimintes self-loops prior 
#       to computing pijs, thus weakly increasing pijs. This weak increase in pij values affects Burt's filtering, 
#       potentially including ties that might have otherwise been excluded. 
##############

import math
import networkx as nx
import numpy as np


#Define the function to decompose constraint to its terms
def constraintDecomp(net: nx.Graph, edgeDeleteThreshold: float, normalizeP: bool, 
        egoNetworkMeasure: bool, cullSelfLoops: bool):
    # Ensure graph is directed
    net = nx.DiGraph(net)
    
    #############
    #OPTIONAL: Cull self-loops before entering decomposition
    #   Move this into the decomposition code as an option
    #############
    if cullSelfLoops:
        print('preemptively eliminating self-loops')
        for i in list(net.nodes):
            if net.has_edge(i, i):
                net.remove_edge(i, i)

    
    #Remove all edges with pij = 0? Missing reciprocated edges would be added below. But this would avoid creating a reciprocal
        #edge based on a weight-0 tie. 
    #edges_to_remove = [(u, v) for u, v, d in net.edges(data=True) if d.get('pij', 0) == 0]
    #net.remove_edges_from(edges_to_remove)

    # Add missing reverse edges, assigning them weight 0. pij values exist when ij exists but ji does not.
        #So we create ji ties with weight = 0 in order to produce correct pij values.
    for i, j in list(net.edges):
        if not net.has_edge(j, i):
            net.add_edge(j, i, weight=0.0)

    # Initialize node attributes
    #for i in net.nodes:
    #    net.nodes[i]['output'] = sum(net[i][j]['weight'] for j in net.successors(i))
    #    net.nodes[i]['input'] = sum(net[j][i]['weight'] for j in net.predecessors(i))

    # Compute direct elements
    for i in list(net.nodes):
        net.nodes[i]['output'] = sum(net[i][j]['weight'] for j in net.successors(i))
        net.nodes[i]['input'] = sum(net[j][i]['weight'] for j in net.predecessors(i))
        net.nodes[i]['degree'] = 0
        net.nodes[i]['shared_alters'] = 0
        net.nodes[i]['Ci'] = 0
        net.nodes[i]['Herfindahl_conc_wtd'] = 0.0
        net.nodes[i]['alter_agg_conc'] = 0.0
        net.nodes[i]['degree_effect'] = 0.0
        net.nodes[i]['degree_effect_ex_pii'] = 0.0
        net.nodes[i]['degree_effect_ex'] = 0.0
        net.nodes[i]['ts_var_effect'] = 0.0
        net.nodes[i]['CovAlterConcSqTieStr'] = 0.0
        net.nodes[i]['pex'] = 0.0
        net.nodes[i]['TriadBalance'] = 0.0
        net.nodes[i]['IndirectDep'] = 0.0
        net.nodes[i]['CC'] = 0.0
        net.nodes[i]['IR'] = 0.0
        
        ts = []  # List of pij values
        pij_sq = []  # List of pij squared values
        conc_vals = []  # List of conc_j values

        #Compute totActivty, excluding an instance of self-loop weights, if they exist
        net.nodes[i]['totActivity'] = (
            net.nodes[i]['input'] + net.nodes[i]['output'] - 
            (net.edges[i, i]['weight'] if net.has_edge(i, i) else 0))

        #store the self-referencing pij (before deleting the self-loop). For nodes w/o self-loops, assign 0.
            #This is robust to the case where no self-loop exists (i.e., no KeyError)
        net.nodes[i]['self_pij'] = net[i].get(i, {}).get('weight', 0.0) / net.nodes[i]['totActivity']
        net.nodes[i]['pex'] += net.nodes[i]['self_pij']

        #Now that totActivity[i] is calculated, delete any self-loop. We don't want it to affect subsequent
            #calculations. This means we will not have any self-loop pijs (pii)
        if net.has_edge(i, i):
            net.remove_edge(i, i)

        #Compute directed pij values; assemble some lists that we'll want for subsequent calculations of 
            #constraint's terms; compute the Herfindahl (Burt's first term, "size"); and compute some rolling
            #sums for subsequent calculations
        for j in list(net.successors(i)):
            #Avoid KeyErrors from deleting edges (below)
            pij = ((net[i][j]['weight'] if net.has_edge(i, j) else 0) +
                 (net[j][i]['weight'] if net.has_edge(j, i) else 0))/ net.nodes[i]['totActivity']
            net.edges[i, j]['pij_egoNet'] = 0.0
            #pij = (net.edges[i, j].get('weight', 0) + net.edges[j, i].get('weight', 0)) / net.nodes[i]['totActivity']
            #process edges above the deletion threshold, but delete edges below (as per Burt)
            if pij >= edgeDeleteThreshold:
                net.edges[i, j]['pij'] = pij
                ts.append(pij)
                pij_sq.append(pij**2)
                conc_vals.append(net.nodes[j].get('conc', 1.0))  # Get conc_j, default = 1.0
                #Herfindahl will be only partial in the case where some pijs have been forced = 0 from edgeDeleteThreshold. 
                    #Can reconstruct the "actual" Herfindahl as the sum of our 3 parts, which is correct. 
                net.nodes[i]['Herfindahl_conc_wtd'] += pij**2 * net.nodes[j].get('conc', 1.0)
                net.nodes[i]['alter_agg_conc'] += net.nodes[j].get('conc', 1.0)
                net.nodes[i]['degree'] +=1
            #remove edge, or just set weight = 0?
                #deleting edges in this loops removes those edges weights, and thus affects subsequent edges' pijs
                #delete edges only after processing all edges, and setting empty edges weights = 0
            else: 
                #net.remove_edge(i, j)
                net.nodes[i]['pex'] += pij
                net.edges[i, j]['pij'] = 0.0
                #successors = list(net.successors(i)) #why was this here?

        #Normalize pij to the remaining weights if the relevant parameter is passed. It's necessary to 
            #recompute all of the direct components here, or to execute this before computing direct components.
            #Could avoid calculating total_p, and just use 1 - pex.
            #Doesn't require first removing weight-0 edges if using total_p based on (1-pex). 
            #Could just always normalize. If no exclusions, then no effect.
        if normalizeP:
            
            #empty the lists with tie strengths, etc., as these will change after normalizing
            ts = []  # List of pij values
            pij_sq = []  # List of pij squared values
            conc_vals = []  # List of conc_j values
            
            #Reset Herfindahl and alter_agg_conc
            net.nodes[i]['Herfindahl_conc_wtd'] = 0.0
            net.nodes[i]['alter_agg_conc'] = 0.0
            
            # Normalize edge weights
            total_p = 1.0 - net.nodes[i].get('pex', 0) 
                #sum(net.edges[i, j].get('pij', 0) for j in net.successors(i))
            net.nodes[i]['total_p'] = total_p
    
            # Normalize p, and archive original values. Only update direct constraint terms when pij > 0 (skipping 
                #ties zeroed out above)
            if total_p > 0:
                for j in (j for j in net.successors(i) if net.edges[i, j].get('pij', 0) > 0):
                #for j in list(net.successors(i)):
                    net.edges[i, j]['pij_archived'] = net.edges[i, j].get('pij', 0)
                    net.edges[i, j]['pij'] /= total_p
                    
                    #Compute the direct constraint components
                    ts.append(net.edges[i, j]['pij'])
                    pij_sq.append(net.edges[i, j]['pij']**2)
                    conc_vals.append(net.nodes[j].get('conc', 1.0))  # Get conc_j, default = 1.0
                    #Herfindahl will be only partial in the case where some pijs have been forced = 0 from edgeDeleteThreshold. 
                        #Can reconstruct the "actual" Herfindahl as the sum of our 3 parts, which is correct. 
                    net.nodes[i]['Herfindahl_conc_wtd'] += net.edges[i, j]['pij']**2 * net.nodes[j].get('conc', 1.0)
                    net.nodes[i]['alter_agg_conc'] += net.nodes[j].get('conc', 1.0)
                    #net.nodes[i]['degree'] +=1 #shouldn't need to update degree, as this already conditioned on 
                        #non-0 weight
                
        else:
            # Behavior when normalizeP is False
            print("Normalization is disabled")

        #Compute constraint's direct elements (i.e., those based strictly on ego's ties)
        net.nodes[i]['varTS'] = np.var(ts, ddof=0) if len(ts) > 1 else 0.0
        net.nodes[i]['ts_var_effect'] = net.nodes[i]['varTS'] * net.nodes[i]['alter_agg_conc']
        net.nodes[i]['degree_effect'] = (1 / max(1, net.nodes[i]['degree']**2)) * sum(net.nodes[j].get('conc', 1.0) 
            for j in net.successors(i) if net.edges[i, j].get('pij', 0) > 0)
        #sum(net.nodes[j].get('conc', 1.0) for j in net.successors(i))
        net.nodes[i]['degree_effect_ex_pii'] = (((1-net.nodes[i]['self_pij']) / max(1, net.nodes[i]['degree']))**2) * sum(net.nodes[j].get('conc', 1.0) 
            for j in net.successors(i) if net.edges[i, j].get('pij', 0) > 0)
        net.nodes[i]['degree_effect_ex'] = (((1-net.nodes[i]['pex']) / max(1, net.nodes[i]['degree']))**2) * sum(net.nodes[j].get('conc', 1.0) 
            for j in net.successors(i) if net.edges[i, j].get('pij', 0) > 0)
        #* sum(net.nodes[j].get('conc', 1.0) for j in net.successors(i))
        
        # Compute covariance of (conc_j, pij^2)
        if len(conc_vals) > 1:  
            cov_value = np.cov(conc_vals, pij_sq, ddof=0)[0, 1]  # Compute covariance
        else:
            cov_value = 0.0  # No covariance if only one alter
        net.nodes[i]['CovAlterConcSqTieStr'] = net.nodes[i]['degree'] * cov_value

    #Remove all edges with pij = 0
    edges_to_remove = [(u, v) for u, v, d in net.edges(data=True) if d.get('pij', 0) == 0]
    net.remove_edges_from(edges_to_remove)
    
    #Add an optional step here, to compute different pijs based on E&B's "ego network" definition.
    #NOTE: This cannot occur w/in the indirect loop, as that loop depends on pqj values already being calculated. Nesting this ego network
    #   replication w/in the indirect loop creates divide-by-zero errors, as some pij_egoNet values are not yet calculated when 
    #   they're called to compute piqpqj.
    if egoNetworkMeasure:
        print('egoNetworkMeasure invoked')
        
        for i in list(net.nodes):
        
            #Loop over i's directed ties.
            #NOTE: Should this be neighbors rather than successors? Almost certainly not predecessors. 
            for j in net.successors(i):
            
                #Find j's total p associated with i's alters. We'll use this to rescale pqj values
                #when computing indirect components. 
                #NOTE: pij_egoNet refers to j's ties with i's alters. Later, "j" will be "q". So we'll
                #   interpret this as the total p that q has with i's alters, and inflate those pijs accordingly.
                #NOTE: Some pij_egoNet values will = 0. This happens when i and j are not connected to any 
                #   common alters (depending on the successor/neighbor question below). This should not product 
                #   DIV!0 errors, as these ties will never be retrieved, since the edges don't exist. 
                #NOTE: Direction matters here. Should q be a successor or j? A neighbor? A predecessor? Later,
                #   we'll compute piq*pqj, and we'll factor pqj by these pij_egoNet values, effectively swapping
                #   the q/j indices. I'm not sure if Everett and Borgatti ever thought of this, but it seems
                #   successors is correct. Eventually, j should be a successor of q. Since we'll swap indexing,
                #   q should not be a successor of j.
                for q in net.successors(j): 
                    if net.has_edge(i, q):  # Ensure [i, q] exists. 
                        net.edges[i, j]['pij_egoNet'] += net.edges[j, q].get('pij', 0)
                #should we also add pji? Otherwise, if j is connected to i, we'll inadvertently inflate
                    #every pjq. This question is "is i in its own immediate/ego network?" Note that 
                    #including pji will not eliminate all zeroes. If filtering, it's possible that pji 
                    #does not exist though pij does. 
                if net.has_edge(j, i):
                    net.edges[i, j]['pij_egoNet'] += net.edges[j, i].get('pij', 0)
                        
    else: 
        print('not using egoNetwork measure')
        nx.set_edge_attributes(net, 1.0, "pij_egoNet")

    # Compute indirect constraint components
    #   NOTE: this depends on all of the pijs already being computed, hence again looping of every node and edge
    for i in list(net.nodes):
        net.nodes[i].update({'Ci': 1.0 if net.nodes[i]['degree'] == 0 else 0.0, 'TriadBalance': 0.0, 'IndirectDep': 0.0, 'IR': 0.0, 'CC': 0.0})

        #Compute triadic components 
        for j in net.successors(i):

            net.edges[i, j]['aggIndirect'] = 0.0
            for q in (set(net.successors(i))&set(net.predecessors(j))):
                #piqpqj = net.edges[i,q]['pij']*net.edges[q,j]['pij']
                #print(i, j, q, net.edges[i,q]['pij_egoNet'])
                piqpqj = net.edges[i,q]['pij']*net.edges[q,j]['pij']/net.edges[i,q]['pij_egoNet']
                net.nodes[i]['TriadBalance'] += 2*piqpqj*net.edges[i, j]['pij']*net.nodes[j].get('conc', 1.0)
                net.edges[i, j]['aggIndirect'] += piqpqj
                net.nodes[i]['IndirectDep'] += (piqpqj**2)*net.nodes[j].get('conc', 1.0) #swapping (Johnson & Sasson)'s indexing of (q, j):
                    #SUM(Oq*(pij*pjq)^2), which will yield the same result. This is more efficient. 

                #Compute quadratic components
                for k in (set(net.successors(i)) & set(net.predecessors(j)) - {q}):
                    #print('k is ', k)
                    #pikpkj = net.edges[i,k]['pij']*net.edges[k,j]['pij']
                    pikpkj = net.edges[i,k]['pij']*net.edges[k,j]['pij']/net.edges[i,k]['pij_egoNet']
                    #Deal with open quadriads via the 'not in' clause. These are E&B (2020)'s "shadow ego networks"
                    if k not in set(net.predecessors(q)):
                        net.nodes[i]['IR'] += piqpqj*pikpkj*net.nodes[j].get('conc', 1.0)
                        #print(i, j, q, k, piqpqj, pikpkj, net.nodes[j].get('conc', 1.0), ' CC is ', net.nodes[i]['CC'])
                        
                    #Otherwise, we've got closed quadriads. The math is the same, but we allocate to a different bucket. Use this
                    #   if wanting to calculate both quadratic pieces, rather than inferring CC from the other terms and C.
                    #else:
                    #    net.nodes[i]['CC'] +=  piqpqj*pikpkj*net.nodes[j].get('conc', 1.0)
                    #    #print(i, j, q, k, piqpqj, pikpkj, net.edges[i,k]['pij_egoNet'])
                        
            #Accumulate constraint
            net.nodes[i]['Ci'] += ((net.edges[i, j]['pij'] + net.edges[i, j]['aggIndirect'])**2) * net.nodes[j].get('conc', 1.0)

        """ 
        for j in list(net.successors(i)):
            net.edges[i, j]['aggIndirect'] = 0.0
            ai = set(net.successors(i))
            aj = set(net.successors(j))
            sharedAlters = ai & aj
            print('for successor ', j, ' shared alters are', list(sharedAlters))
            successors = list(net.successors(i))
            successors = list(net.successors(j))
            net.nodes[i]['shared_alters'] =+ len(sharedAlters) #closed triads; used to replicate E&B effective size, k (2020)

            #find (i, j) common alters, and compute metrics through those ij*jq and iq*qj paths
            for q in sharedAlters:
                #This fails if the q->j edge might has been deleted (hence setting = 0), even if q is still a successor of 
                    #both i and j, as q can be a successor but not a predecessor. 
                    #Fix by setting weight = 0 for edges below the threshold? Inefficient. Or by only CONTINUE-ing if q a 
                    #predessor of j? That violates current logic, as IndirectDep doesn't require q to be a j predecessor. Perhaps
                    #could change, such that we find i's successors that are j's 
                    #predecessors? 
                    #As-is, Same issue will occur below, in IR and CC calculations
                net.edges[i, j]['aggIndirect'] += net.edges[i, q]['pij'] * net.edges[q, j]['pij']
                net.nodes[i]['IndirectDep'] += (net.edges[i, j]['pij'] * net.edges[j, q]['pij'])**2 * net.nodes[q].get('conc', 1.0)

                #And compute the quadratic terms
                
                #First, compute the unclosed quadriad term, E&B's "shadow ego" situation
                    #Identify (i, j) alters not connected to q, excluding q itself
                aq = set(net.successors(q))
                print('for ', q,' in sharedAlters, aq is ', aq)
                shared_q_bridges = sharedAlters - aq  - {q}
                print('for ', q,' in sharedAlters, shared_q_bridges is ', aq)

                for k in shared_q_bridges:
                    #multiplying by 2 was a code conversion error. The prior method iterated through a list, avoiding processing
                        #the same cycle twice, but requiring doubling. This doesn't yet have that efficiency, so 
                        #should not be doubled. 
                    net.nodes[i]['IR'] += net.nodes[j].get('conc', 1.0) * net.edges[i, q]['pij'] * net.edges[i, k]['pij'] * net.edges[q, j]['pij'] * net.edges[k, j]['pij']
                    #net.nodes[i]['IR'] += 2 * net.nodes[j].get('conc', 1.0) * net.edges[i, q]['pij'] * net.edges[i, k]['pij'] * net.edges[q, j]['pij'] * net.edges[k, j]['pij']


                #For testing, verify that "closed communities" are correctly computed. For effiiciency, eventually disable and
                    #instead rely on: QS = CC + IR, and QS = term3 - ID
                closed_quads = sharedAlters & aq
                print('for ', q,' in sharedAlters, closed_quads is ', closed_quads)
                for k in closed_quads:
                    net.nodes[i]['CC'] += net.nodes[j].get('conc', 1.0)*net.edges[i, q]['pij']*net.edges[i, k]['pij']*net.edges[q, j]['pij']*net.edges[k, j]['pij']

            net.nodes[i]['Ci'] += (net.edges[i, j]['pij'] + net.edges[i, j]['aggIndirect'])**2 * net.nodes[j].get('conc', 1.0)
            #TriadBalance is Burt's "density"
            net.nodes[i]['TriadBalance'] += 2 * net.edges[i, j]['pij'] * net.edges[i, j]['aggIndirect'] * net.nodes[j].get('conc', 1.0)
        """
            
        #Include some checks to validate decomposition implementation
        #net.nodes[i]['CC_check'] = net.nodes[i]['Ci'] - (net.nodes[i]['Herfindahl_conc_wtd'] + net.nodes[i]['TriadBalance'] 
        #    + net.nodes[i]['IndirectDep'] + net.nodes[i]['IR'] + net.nodes[i]['CC'])

        #Compute CC as constraint minus everything else, leveraging the decomposition to make that fast. Set min = 0.
        net.nodes[i]['CC_inferred'] = max(net.nodes[i]['Ci'] - (net.nodes[i]['Herfindahl_conc_wtd'] + net.nodes[i]['TriadBalance'] 
            + net.nodes[i]['IndirectDep'] + net.nodes[i]['IR']), 0)
        net.nodes[i]['Constraint_check'] = net.nodes[i]['Ci'] - (net.nodes[i]['degree_effect']  
            + net.nodes[i]['ts_var_effect'] + net.nodes[i]['CovAlterConcSqTieStr'] + net.nodes[i]['TriadBalance'] 
            + net.nodes[i]['IndirectDep'] + net.nodes[i]['IR'] + net.nodes[i]['CC_inferred'])
        net.nodes[i]['Constraint_check_ex'] = net.nodes[i]['Ci'] - (net.nodes[i]['degree_effect_ex']  
            + net.nodes[i]['ts_var_effect'] + net.nodes[i]['CovAlterConcSqTieStr'] + net.nodes[i]['TriadBalance'] 
            + net.nodes[i]['IndirectDep'] + net.nodes[i]['IR'] + net.nodes[i]['CC_inferred'])
        net.nodes[i]['Herf_check'] = net.nodes[i]['Herfindahl_conc_wtd'] - (net.nodes[i]['degree_effect']  
            + net.nodes[i]['ts_var_effect'] + net.nodes[i]['CovAlterConcSqTieStr'])
        net.nodes[i]['Herf_check_ex'] = net.nodes[i]['Herfindahl_conc_wtd'] - (net.nodes[i]['degree_effect_ex']  
            + net.nodes[i]['ts_var_effect'] + net.nodes[i]['CovAlterConcSqTieStr'])


    return net
