# Convex optimization approach for 1-bit compression

# Imports
# --------------
import cvxpy as cp
import numpy as np
import scipy as sp
from scipy import stats



def noise_model(y):
    flip_prob = 1/4 # prob must be less than 1/2 or else unable to recover
    bern_dist = np.ones(len(y)) - 2*sp.stats.bernoulli.rvs(flip_prob, size = len(y))
    return bern_dist


def one_bit_convex(y,A,n,s):
    x = cp.Variable(n)

    objective = cp.Maximize((np.transpose(y)*(A*x)))
    constraints = [cp.norm(x,p=1)<= np.sqrt(s), cp.norm(x,p=2)<=1]

    prob = cp.Problem(objective, constraints)

    result = prob.solve()

    return x.value

# give orhonormal basis E of R^d with column vectors as ortho basis
def one_bit_dc(E, oracle, eps, delta, noise):
    transpose_E = np.transpose(E)
    while(len(transpose_E) > 1):
        print( "transpose_E: ", transpose_E)
        print("transpose_E length: ", len(transpose_E))
        new_element = one_bit_dc2(transpose_E[0],transpose_E[1],eps,delta,oracle, noise)
        print("new_element: ",new_element)
        for i in range(0,2):
            transpose_E = np.delete(transpose_E,0,0)
            print("del_1,del_2",transpose_E )
        transpose_E = np.insert(transpose_E,0, new_element,0)

    print("transpose_E: ", transpose_E)
    return transpose_E[0]

# orthonormal vectors e1,e2
# estimation error ep
# delta s.t.
# success probability s_prob >= 1- delta

def one_bit_dc2(e1, e2, eps, delta, oracle, noise):
    # p_0(h) is uniform on h in S^1
    p = [0.5,0.5] # vector p stores the probability on each piecewise segment on circle
    angl = [1,1] # u stores the segments from [0,2] in radian prop s.t. each entry is the length of the segments in order from 0 to 2*pi

    orth_ang = 0.5  # populated later zero for now

    #T_eps_delta = 20*(int(np.log(1/eps)+np.log(1/delta)) ) # need to determine how to define this later, just take c = 20 i guess
    #print("T_eps_delta: ",T_eps_delta)

    # calculate T_eps_delta
#     for m in range(0, T_eps_delta+1):
    for m in range(0, 3): # check for small values if work
        # find vector xm solution to \int sgn(<x,h>)p_{m-1}(h) dh = 0
        # find the dividing plane
        
        print("Iteration " + str(m))
        print("--------------------------------")

        #print("probability mat: ", p)
        #print("angle matrix: ", angl)

        p_sum = 0 # probability sum for 0 to 1pi cut 
        angl_sum = 0 #angle sum
        
        # Find the initial sum for the red line
        ind = -1 # index counter
        while (angl_sum < 1):
            ind += 1
            angl_sum += angl[ind]
            p_sum += p[ind]

        # residual angle
        a_res = angl_sum - 1
        left_side = 0 # left side slice angle initial
        right_side = 1 # right side slice endpt initial
        
        #print("first angle residual: ", a_res)
        #print("p_sum before res subtract", p_sum)

        p_sum = p_sum - (a_res/angl[ind])*p[ind] # adjust slice measure 
        
        #print("starting p_sum after angle subtract: ", p_sum)

        ind_left = 0
        ind_right = ind
        

        a_res_left = angl[0]
        if(a_res == 0 ):
            ind_right += 1
            a_res_right = angl[ind_right]
        else:
            a_res_right = a_res
            
        #print("starting left index: ", ind_left)
        #print("starting right index: ", ind_right)

        # solve on segments ie rotating between intervals
        while(p_sum != 1/2 ):
            #print("p: ",p)
            #print("angl: ",angl)
            b = p[ind_right] # prob density for right interval
            a = p[ind_left] # prob density for left interval


            min_res = min(a_res_left, a_res_right) # constraints on theta, rotation is bounded by the minimum angle possible to rotate clockwise without jumping out of density bounds
            #print("p_sum: ", p_sum)
    

            if (b == a ):
                theta = min_res
            else:
                theta = (1/2-p_sum)/(b/angl[ind_left]-a/angl[ind_right]) # solution to p_sum + b*theta/2 - a* theta/2 = 1/2

            #print("angle solution theta: ", theta )
            #print("b: ",b)
            #print("a: ",a)
            #print("a_res_left: ", a_res_left)
            #print("a_res_right: ", a_res_right)
            #print("min_res: " , min_res)
            
            
            # Case where the slice is found for measure 1/2
            # Add the two new points for the segments as well as update the probability
            if ( theta != None and 0 <= theta < min_res ):
                #print("see if enters the loop")
                print("theta: ", theta)
                orth_ang = (left_side+right_side)/2 + theta # orthogonal vector angle

                # Add the left and right endpoints to the angle vec and note that the measure on either side is initially 1/2
                dis_left = a_res_left - theta # to understand draw picture, angle is subtracted clockwise
                angl[ind_left] = angl[ind_left] - dis_left
                #print("first angle change: ",angl)
                angl.insert((ind_left+1)%len(angl), dis_left)
                #print("first angle insert: ",angl)

                # update left side probability
                tot_left_prob = p[ind_left]  # probability of the current segment on left side
                p[ind_left] = (angl[ind_left]/(angl[ind_left]+angl[(ind_left+1)%len(p)]))* tot_left_prob
                #print("p set left once: ", p)
                p.insert((ind_left+1)%len(p), (angl[(ind_left+1)%len(angl)]/(angl[ind_left]+angl[(ind_left+1)%len(p)])) * tot_left_prob )
                #print("p insert other half left: ", p)
                
                # set the new indices for left
                ind_left = (ind_left+1)%(len(p)-1)

                # Correction after left insertion
                if((ind_left+1)%(len(angl)-1) <= ind_right ):
                    ind_right = ind_right+1
                
                # Right side, add the angle clockwise
                dis_right = a_res_right - theta
                angl[ind_right] = angl[ind_right] - dis_right
                #print("right angle change: ",angl)
                angl.insert((ind_right + 1)%len(angl), dis_right)
                #print("right angle insert: ",angl)
                    
                # update right side probability
                tot_right_prob = p[ind_right]  # probability of the current segment on right side
                #print("tot_right_prob: ", tot_right_prob)
                #print("insert: ", angl[ind_right])
                #print("updated 1 denom: ", angl[ind_right]+ angl[ind_right+1] )
                p[ind_right] = (angl[ind_right]/(angl[ind_right]+angl[(ind_right+1)%len(p)]))* tot_right_prob
                #print("p insert once right", p)
                p.insert(ind_right+1, (angl[(ind_right+1)%len(p)]/(angl[ind_right]+angl[(ind_right+1)%len(p)])) * tot_right_prob )
                #print("p insert the other half right", p)


#                 p.remove(0) # remove useless segments
#                 angl.remove(0)
                #print("p after removing useless segment: ",p)
                #print("angl after removing useless segment: ",angl)

                # set the new indices for right
                ind_right = (ind_right+ 1)%(len(p)-1)
                
                break

            else: # this is when  the measure isnt found and rotations need to be implemented
                #print("1/2 1/2 split not found and other loop used ")
                if(min_res == a_res_left):
                    #print("left side rot is max")
                    ind_left = (ind_left+ 1)%(len(p))
                    a_res_left = angl[ind_left]
                    a_res_right -= min_res 
                elif(min_res == a_res_right):
                    #print("right side rot is max")
                    ind_right = (ind_right +1)%(len(p))
                    a_res_right = angl[ind_right]
                    #print("new right side residual: ", a_res_right)
                    a_res_left -= min_res
                    #print("new left side residual: ", a_res_left)

                
                # Update the new left and right bounds 
                left_side += min_res
                right_side += min_res
            
            
            p_sum = p_sum + b*min_res - a*min_res # update the rotated probability 


        # Construct  xm after the repeated divisions
        print("orthangle: ", orth_ang)
        xm = np.cos(orth_ang*np.pi)*e1/np.linalg.norm(e1) + np.sin(orth_ang*np.pi)*e2/np.linalg.norm(e2)
        # Ask oracle for value of sgn(<xm,h*>)
        sgn = oracle(xm)
        # update distribution p_{m-1} -> p_m

        #print("p before update: ", p)
        if (sgn == 1):
            #print("index left: ",ind_left)
            #print("index right: ",ind_right)
            if(ind_right >= ind_left):
                for updateind in range(0,len(p)):
                    #print("update_index", updateind)
                    if updateind in range(ind_left, ind_right):
                        #print("in range")
                        p[updateind] = 2*(1-noise)*p[updateind]
                    else:
                        #print("not in range")
                        p[updateind] = 2*noise*p[updateind]
            else:
                for updateind in range(ind_left, len(p)):
                    #print("update_index", updateind)
                    #print("in range")
                    p[updateind] = 2*(1-noise)*p[updateind]
                for updateind in range(0,ind_right):
                    #print("update_index", updateind)
                    #print("in range")
                    p[updateind] = 2*(1-noise)*p[updateind]
                for updateind in range(ind_right,ind_left):
                    #print("update_index", updateind)
                    #print("not in range")
                    p[updateind] = 2*noise*p[updateind]
                    
            
                    

        if (sgn == -1):
            #print("index left: ",ind_left)
            #print("index right: ",ind_right)
            if(ind_right >= ind_left):
                for updateind in range(0,len(p)):
                    #print("update_index", updateind)
                    if updateind in range(ind_left, ind_right):
                        #print("in range")
                        p[updateind] = 2*noise*p[updateind]
                    else:
                        #print("not in range")
                        p[updateind] = 2*(1-noise)*p[updateind]
            else:
                for updateind in range(ind_left, len(p)):
                    #print("update_index", updateind)
                    #print("in range")
                    p[updateind] = 2*noise*p[updateind]
                for updateind in range(0,ind_right):
                    #print("update_index", updateind)
                    #print("in range")
                    p[updateind] = 2*noise*p[updateind]
                for updateind in range(ind_right,ind_left):
                    #print("update_index", updateind)
                    #print("not in range")
                    p[updateind] = 2*(1-noise)*p[updateind]


    max_ind = np.argmax(p)
    max_angle = 0
    for z in range(0, max_ind+1):
        max_angle += angl[z]
    max_vecs = np.cos((max_angle-angl[max_ind]/2)*np.pi)*e1/np.linalg.norm(e1) + np.sin((max_angle-angl[max_ind]/2)*np.pi)*e2/np.linalg.norm(e2)
    
    return max_vecs# vector which is argmax_h p_T(h)

def test_oracle(x):
    #print("x transpose in oracle: ", np.transpose(x))
    n = len(x)
    h = np.array([10,3,-10,7])
    #print("h in oracle: ",h)
    #print("the dot product: ", np.matmul(np.transpose(x), h) )
    #print("the sign: ", np.sign( np.matmul(np.transpose(x), h) ))
    return np.sign( np.matmul(np.transpose(x), h) )

# Testing 
# ---------------

x = np.array([1,0,1,0,1,1,1]) # 1 bit results

A = np.eye(7) # Measurement matrix

y = np.sign(np.matmul(A,x))

test = noise_model(y)
y_noisy = np.multiply(y, test)

#print("original x: ", x)
#print("y_noisy: ",y_noisy)
#print("convex approach: ", one_bit_convex(y_noisy,A,7,5))



# Testing 1 bit
#e1 = np.array( [1,0] )
#e2 = np.array([0,1])
eps = 0.1
delta = 0.1
noise = 0.00000001

# testing dc on multi dims
n = 4
E = 10*np.eye(n)

#print(one_bit_dc2(e1, e2, eps, delta, test_oracle, 0.0001))
print(one_bit_dc(E,test_oracle,eps,delta,noise))
