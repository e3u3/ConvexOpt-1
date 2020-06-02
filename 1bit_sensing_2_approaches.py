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

# give orhonormal basis E of R^d
#def one_bit_dc(E, oracle):
#    for j in range(0, d-1):
#        1_bit_dc2(E[i],E[j],eps,delta,oracle)
#    return E[0]

# orthonormal vectors e1,e2
# estimation error ep
# delta s.t.
# success probability s_prob >= 1- delta

# returns unit vector e, an estimate for normalized orthogonal projection of h^* onto span(e1,e2)
def one_bit_dc2(e1, e2, eps, delta, oracle):
    # p_0(h) is uniform on h in S^1
    p = [1] # vector p stores the probability on each piecewise segment on circle
    angl = [2] # u stores the segments from [0,2] in radian prop s.t. each entry is the length of the segments in order from 0 to 2*pi

    # calculate T_eps_delta
    for m in range(0, T_eps_delta+1):
        # find vector xm solution to \int sgn(<x,h>)p_{m-1}(h) dh = 0
        # find the dividing plane

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
        rightside = 1 # right side slice endpt initial

        p_sum = psum - (a_res/angl[ind])*p[ind] # adjust slice measure 

        ind_left = 0
        ind_right = ind

        a_res_left = a[0]
        a_res_right = a_res

        # solve on segments ie rotating between intervals
        while(p_sum != 1/2):
            b = 1/ p[ind_right] # prob density for right interval
            a = 1/p[inf_left] # prob density for left interval
            theta = (1-2*p_sum)/(b-a) # solution to p_sum + b*theta/2 - a* theta/2 = 1/2

            min_res = min(a_res_left, a_res_right) # constraints on theta, rotation is bounded by the minimum angle possible to rotate clockwise without jumping out of density bounds

            # Case where the slice is found for measure 1/2
            # Add the two new points for the segments as well as update the probability
            if (0 <= theta <= min_res ):
                orth_ang = (b+a)/2 + theta # orthogonal vector angle

                # Add the left and right endpoints to the angle vec and note that the measure on either side is initially 1/2
                dis_left = a_res_left - theta # to understand draw picture, angle is subtracted clockwise
                a[ind_left] = a[ind_left] - dis_left
                a.insert(ind_left+1, dis_left)

                # update left side probability
                tot_left_prob = p[ind_left]  # probability of the current segment on left side
                p[ind_left] = (a[ind_left]/(a[ind_left]+a[ind_left+1]))* tot_left_prob
                p.insert(ind_left+1, (a[ind_left+1]/(a[ind_left]+a[ind_left+1])) * tot_left_prob )

                # Right side, add the angle clockwise
                dis_right = a_res_right - theta
                a[ind_right] = a[ind_right] - dis_right
                a.insert(ind_right + 1, dis_right)
                    
                # update right side probability
                tot_right_prob = p[ind_right]  # probability of the current segment on left side
                p[ind_right] = (a[ind_right]/(a[ind_right]+a[ind_right+1]))* tot_right_prob
                p.insert(ind_right+1, (a[ind_right+1]/(a[ind_right]+a[ind_right+1])) * tot_right_prob )
                
                break # break out of calculation loop if solved
            else: # this is when  the measure isnt found and rotations need to be implemented








        # Ask oracle for value of sgn(<xm,h*>)
        sgn = oracle(xm)
        # update distribution p_{m-1} -> p_m
        if (sgn == 1):
        if (sgn == -1):
    return

def test_oracle(x):
    n = len(x)
    h = np.array(np.ones(n))
    return np.sign( np.transpose(x,h) )

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
