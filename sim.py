# Simulation of Gaussian mean width of w(S_ns)

# Imports
# -------------
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import cvxpy as cp

# As per requirement fixing s

# Gaussian mean width is calculated as E[sup<g,x>], supremum over x in S_ns-S_ns,
# Expectation over standard Gaussian random vector g, K is Euclidean ball in R^n

# By lemma 2.3 w(S_ns) = E[max_|T|=s ||g_T||_2]


# Define the constraints for the sparse signal set
#S_ns_constraints = [cp.norm(x,0) <= s , cp.norm(x,2)<= 1]


def gaus_mean_width_S_ns(n,s):
    # by roman vershynin 3.5.3
    # with high probability w(K) is about w(K,g) = sup <g,u>, u in K-K
    g = sp.stats.multivariate_normal.rvs(np.zeros(n))
    inner_array = []
    density = s/n
    rvs = sp.stats.uniform(loc=-50, scale = 100 ).rvs
    for j in range(0, 50*n):
        u_1 =  sp.sparse.random(n,1,density=density,data_rvs=rvs).todense()
        u_2 = sp.sparse.random(n,1,density=density, data_rvs=rvs).todense()
        inner = np.transpose(g)*(u_1/np.linalg.norm(u_1) - u_2/np.linalg.norm(u_2)) # u in S_ns - S_ns
        inner_array.append( inner )
    return max(inner_array)


# plot w(S_ns) versus n with its upper bound and lower bound given by Lemma 2.3
# bounds cs log(2n/s) =< w^2(S_ns) =< Cs log(2n/s)

# returns the lower bound and then upper bound constant not with c and C
def gaus_mw_low_up_bd(n,s):
    return np.sqrt( np.log(2*n/s) )



# Below are the conditions for S^1_ns, S^2_ns, S^3_ns

def gaus_mean_width_S_ns_1(n,s):
    g = sp.stats.multivariate_normal.rvs(np.zeros(n))
    inner_array = []
    density = s/n
    rvs = sp.stats.uniform(loc=-50, scale = 100 ).rvs
    for j in range(0, 50*n):
        # change u_1 and u_2 to satisfy the restrictions of xi >= 0
        u_1 =  sp.sparse.random(n,1,density=density, data_rvs=rvs).todense()
        u_1 = abs(u_1)
        u_2 = sp.sparse.random(n,1,density=density, data_rvs=rvs).todense()
        u_2 = abs(u_2)
        inner = np.transpose(g)*(u_1/np.linalg.norm(u_1) - u_2/np.linalg.norm(u_2)) # u in S_ns - S_ns
        inner_array.append( inner )
    return max(inner_array)



# Check this to see if it is really what is wanted later and confirm
def gaus_mean_width_S_ns_2(n,s):
    g = sp.stats.multivariate_normal.rvs(np.zeros(n))
    inner_array = []
    density = s/n
    rvs = sp.stats.uniform(loc= np.random.randint(low=-50,high=100) , scale = 0 ).rvs
    for j in range(0, 50*n):
        u_1 =  sp.sparse.random(n,1,density=density, data_rvs=rvs).todense()
        u_2 = sp.sparse.random(n,1,density=density, data_rvs = rvs).todense()
        inner = np.transpose(g)*(u_1/np.linalg.norm(u_1) - u_2/np.linalg.norm(u_2)) # u in S_ns - S_ns
        inner_array.append( inner )
    return max(inner_array)



def gaus_mean_width_S_ns_3(n,s):
    g = sp.stats.multivariate_normal.rvs(np.zeros(n))
    inner_array = []
    density = s/n
    rvs = sp.stats.uniform(loc= np.random.randint(low=-50,high=100) , scale = 0 ).rvs
    for j in range(0, 50*n):
        u_1 =  sp.sparse.random(n,1,density=density,data_rvs=rvs).todense()
        m1 = np.random.randint(low= 0, high = n+1)
        idx1 = np.random.choice(n,m1, replace=False)
        u_1[idx1] = -u_1[idx1]


        u_2 = sp.sparse.random(n,1,density=density,data_rvs=rvs).todense()
        m2 = np.random.randint(low= 1, high = n+1)
        idx2 = np.random.choice(n,m2, replace=False)
        u_2[idx2] = -u_2[idx2]
        inner = np.transpose(g)*(u_1/np.linalg.norm(u_1) - u_2/np.linalg.norm(u_2)) # u in S_ns - S_ns
        inner_array.append( inner )
    return max(inner_array)


s = 5
n_array = [i for i in range(s,40)]
bound =[]
g_mean_Sn5 = []
g_mean_Sn5_1 = []
g_mean_Sn5_2 = []
g_mean_Sn5_3 = []

for i in n_array:
    bound.append(gaus_mw_low_up_bd(i,s))
    g_mean_Sn5.append(np.float(gaus_mean_width_S_ns(i,s)) )
    g_mean_Sn5_1.append(np.float(gaus_mean_width_S_ns_1(i,s)))
    g_mean_Sn5_2.append(np.float(gaus_mean_width_S_ns_2(i,s)))
    g_mean_Sn5_3.append(np.float(gaus_mean_width_S_ns_3(i,s)))

plt.plot(n_array,bound, label='bound_coeff')
plt.plot(n_array,g_mean_Sn5, label='S_n5')
plt.plot(n_array,g_mean_Sn5_1, label='S_n5_1')
plt.plot(n_array,g_mean_Sn5_2, label='S_n5_2')
plt.plot(n_array,g_mean_Sn5_3, label='S_n5_3')
plt.xlabel('dimension n')
plt.ylabel('values of w(K)')
plt.title(" dimension vs w(S_ns)")
plt.legend()
plt.show()

