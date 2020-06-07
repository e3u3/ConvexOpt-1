# Convex optimization approach for 1-bit compression

# Imports
# --------------
import cvxpy as cp
import numpy as np
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt


# Implement Random bit flip noise model
def noise_model(y):
    flip_prob = 0.1 # prob must be less than 1/2 or else unable to recover
    bern_dist = np.ones(len(y)) - 2*sp.stats.bernoulli.rvs(flip_prob, size = len(y))
    return bern_dist

# Solver for the convex algorithm solution
def one_bit_convex(y,A,n,s):
    x = cp.Variable(n)

    objective = cp.Maximize((np.transpose(y)*(A*x)))
    constraints = [cp.norm(x,p=1)<= np.sqrt(s), cp.norm(x,p=2)<=1]

    prob = cp.Problem(objective, constraints)

    result = prob.solve()

    return x.value

# Dimension Coupling Algorithm, relies on recursion on the DC-2 below
def one_bit_dc(E, oracle, eps, delta, noise):
    transpose_E = np.transpose(E)
    while(len(transpose_E) > 1):
        new_element = one_bit_dc2(transpose_E[0],transpose_E[1],eps,delta,oracle, noise)
        for i in range(0,2):
            transpose_E = np.delete(transpose_E,0,0)
        transpose_E = np.insert(transpose_E,0, new_element,0)

    return transpose_E[0]



# 2 dimensional Dimensional Coupling algorithm (DC-2)

# orthonormal vectors e1,e2
# estimation error ep
# delta s.t.
# success probability s_prob >= 1- delta

def one_bit_dc2(e1, e2, eps, delta, oracle, noise):
    
    # p_0(h) is uniform on h in S^1
    p = [0.5,0.5] # vector p stores the probability on each piecewise segment on circle
    angl = [1,1] # u stores the segments from [0,2] in radian prop s.t. each entry is the length of the segments in order from 0 to 2*pi

    orth_ang = 0.5  # populated later zero for now

    # calculate T_eps_delta an upper bound to the number of iterations for an accurate orthogonal vector
    #T_eps_delta = 20*(int(np.log(1/eps)+np.log(1/delta)) ) 
#     for m in range(0, T_eps_delta+1):

    # In reality, using the above upper bound, there will be numerical issues calculating the solution to the rotation problem 
    # after a certain number of iterations. ie. it will loop infinitely
    # Therefore manually tuning the  number of iteration will allow convergence of a solution in practice
    for m in range(0, 7): # check for small values if work
        # find vector xm solution to \int sgn(<x,h>)p_{m-1}(h) dh = 0
        # find the dividing plane
       
        # For clarity print the iterations for fine tuning of converging solutions 
        print("Iteration " + str(m))
        print("--------------------------------")

        print("probability mat: ", p)
        print("angle matrix: ", angl)

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
        

        p_sum = p_sum - (a_res/angl[ind])*p[ind] # adjust slice measure 
        

        ind_left = 0
        ind_right = ind
        

        a_res_left = angl[0]
        if(a_res == 0 ):
            ind_right = ind_right + 1
            a_res_right = angl[ind_right]
        else:
            a_res_right = a_res
            

        # solve on segments ie rotating between intervals
        while(p_sum != 1/2 ):
            b = p[ind_right] # prob density for right interval
            a = p[ind_left] # prob density for left interval


            min_res = min(a_res_left, a_res_right) # constraints on theta, rotation is bounded by the minimum angle possible to rotate clockwise without jumping out of density bounds
    

            if (b == a or (b/angl[ind_left]-a/angl[ind_right] == 0) ):
                theta = min_res
            else:
                theta = (1/2-p_sum)/(b/angl[ind_left]-a/angl[ind_right]) # solution to p_sum + b*theta/2 - a* theta/2 = 1/2

            
            # Case where the slice is found for measure 1/2
            # Add the two new points for the segments as well as update the probability
            if ( 0 <= theta <  min_res  ):
                orth_ang = (left_side+right_side)/2 + theta # orthogonal vector angle

                # Add the left and right endpoints to the angle vec and note that the measure on either side is initially 1/2
                dis_left = a_res_left - theta # to understand draw picture, angle is subtracted clockwise
                angl[ind_left] = angl[ind_left] - dis_left
                angl.insert((ind_left+1)%len(angl), dis_left)

                # update left side probability
                tot_left_prob = p[ind_left]  # probability of the current segment on left side
                p[ind_left] = (angl[ind_left]/(angl[ind_left]+angl[(ind_left+1)%len(p)]))* tot_left_prob
                p.insert((ind_left+1)%len(p), (angl[(ind_left+1)%len(angl)]/(angl[ind_left]+angl[(ind_left+1)%len(p)])) * tot_left_prob )
                
                # set the new indices for left
                ind_left = (ind_left+1)%(len(p)-1)

                # Correction after left insertion
                if((ind_left+1)%(len(angl)-1) <= ind_right ):
                    ind_right = ind_right+1
                
                # Right side, add the angle clockwise
                dis_right = a_res_right - theta
                angl[ind_right] = angl[ind_right] - dis_right
                angl.insert((ind_right + 1)%len(angl), dis_right)
                    
                # update right side probability
                tot_right_prob = p[ind_right]  # probability of the current segment on right side
                p[ind_right] = (angl[ind_right]/(angl[ind_right]+angl[(ind_right+1)%len(p)]))* tot_right_prob
                p.insert(ind_right+1, (angl[(ind_right+1)%len(p)]/(angl[ind_right]+angl[(ind_right+1)%len(p)])) * tot_right_prob )



                # set the new indices for right
                ind_right = ind_right+ 1

                if (ind_left >= ind_right):
                    ind_left = ind_left+1

                break

            else: # this is when  the measure isnt found and rotations need to be implemented
                #print("1/2 1/2 split not found and other loop used ")
                if(min_res == a_res_left):
                    ind_left = (ind_left+ 1)%(len(p))
                    a_res_left = angl[ind_left]
                    a_res_right -= min_res 
                elif(min_res == a_res_right):
                    ind_right = (ind_right +1)%(len(p))
                    a_res_right = angl[ind_right]
                    a_res_left -= min_res

                
                # Update the new left and right bounds 
                left_side += min_res
                right_side += min_res
            
            
            p_sum = p_sum + b*min_res - a*min_res # update the rotated probability 


        # Construct  xm after the repeated divisions
        print("orthangle: ", orth_ang)
        xm = np.cos(orth_ang*np.pi)*e1/np.linalg.norm(e1) + np.sin(orth_ang*np.pi)*e2/np.linalg.norm(e2)
        # Ask oracle for value of sgn(<xm,h*>)
        sgn = oracle(xm)
        print("sign: ",sgn)
        
        # update distribution p_{m-1} -> p_m
        if (sgn == 1):
            if(ind_right >= ind_left):
                for updateind in range(0,len(p)):
                    if updateind in range(ind_left, ind_right):
                        p[updateind] = 2*(1-noise)*p[updateind]
                    else:
                        p[updateind] = 2*noise*p[updateind]
            else:
                for updateind in range(ind_left, len(p)):
                    p[updateind] = 2*(1-noise)*p[updateind]
                for updateind in range(0,ind_right):
                    p[updateind] = 2*(1-noise)*p[updateind]
                for updateind in range(ind_right,ind_left):
                    p[updateind] = 2*noise*p[updateind]
                    

        if (sgn == -1):
            if(ind_right >= ind_left):
                for updateind in range(0,len(p)):
                    if updateind in range(ind_left, ind_right):
                        p[updateind] = 2*noise*p[updateind]
                    else:
                        p[updateind] = 2*(1-noise)*p[updateind]
            else:
                for updateind in range(ind_left, len(p)):
                    p[updateind] = 2*noise*p[updateind]
                for updateind in range(0,ind_right):
                    p[updateind] = 2*noise*p[updateind]
                for updateind in range(ind_right,ind_left):
                    p[updateind] = 2*(1-noise)*p[updateind]

    max_ind = np.argmax(p)
    max_angle = 0
    for z in range(0, max_ind+1):
        max_angle += angl[z]
    max_vecs = np.cos((max_angle-angl[max_ind]/2)*np.pi)*e1/np.linalg.norm(e1) + np.sin((max_angle-angl[max_ind]/2)*np.pi)*e2/np.linalg.norm(e2)
    
    return max_vecs# vector which is argmax_h p_T(h)

# Testing 
# ---------------

# Determine the noise
eps = 0.1
delta = 0.1
noise = 0.1


# Plot the norm difference between the recovered signal and the true signal
# Last part of the project

# Plot for dim from 3 to 8
for n in range(3,9):
    n_convex = []
    n_div = []
    for s in range(1, n+1):
        density = s/n
        x =  sp.sparse.random(n,1,density=density).todense()
        x = np.array(x).flatten()
        x = x/np.linalg.norm(x)
        A = np.eye(n)
        y = np.sign(np.matmul(A,x))

        noisy = noise_model(y)

        y_noisy = np.multiply(y,noisy)

        val_conv = one_bit_convex(y_noisy,A,n,s)

        conv_norm = np.linalg.norm(x-val_conv) # find the euclidean distance

        # Oracle with noise 0.1

        def test_oracle(y):
            h = x
            true = np.sign( np.matmul(np.transpose(x), h) )
            noise = 0.1
            flips = sp.stats.bernoulli.rvs(noise, size=1)
            if (flips ==1):
                true = - true
            return true



        divide_and_c = one_bit_dc(A, test_oracle,eps,delta,noise)

        div_c_norm = np.linalg.norm(x-divide_and_c)

        n_convex.append(conv_norm)
        n_div.append(div_c_norm)

    title = "Plot for n = "+str(n)

    plt.suptitle(title)
    plt.plot(n_convex, label = "convex norm difference")
    plt.plot(n_div, label = "dc norm difference")
    plt.xlabel("sparsity")
    plt.ylabel("norm difference between expected signal and true signal")
    plt.legend()
    plt.show()






