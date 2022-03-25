import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from functools import partial
import random
from evoalgos.algo import CommaEA
#Evo-algorithms libraries
from evoalgos.individual import CMSAIndividual
from optproblems import Problem

@jit(nopython=True) 
def holza2015_uniaxial(parameters,l1,dh=1e-6,lon=True):
    """ Numerical evaluation of uniaxial stress of the Holzapfel(2015) Model, based on uniform deformation gradient.
        We use newton raphson method to determine lambda2, in order to, obtain the Cauchy stress in the principal direction.
    
    Parameters:
    l1: ndarray
        Array containing the uniaxial stretches
        
    l2: ndarray
        Array for initialization of the lambda2 stretches. Remember, that this stretch is obtained through
        the newton raphson method. Recommended value = 1
        
    parameters: tuple
        Models parameters of Holzapfel 2015 (u1,A,B,k1,k2,fi)
        Note that fi is the angle defined in the (longitudinal-cirfunferential) plane in an artery. 
        
    dh: float
        delta numérico para newton raphson para calcular lambda2.
    
    lon: Bool
        If it's longitudinal of circunferential
    
    Return:
    ----------
    (l2,1./(l1*l2),s11,dw1,dw4,dwn): Tupla
    
    l2: ndarray
        lambda 2 obtained through newton raphson.
        
    l3: ndarray
        lambda 3 derived from incompressibility condition.
        
    s11: ndarray
        uniaxial cauchy stress.
        
    dw1:ndarray
        partial derivative of W faction with respect to I1
        
    dw4:ndarray
        partial derivative of W faction with respect to I4
        
    dw6:ndarray
        partial derivative of W faction with respect to I6
        
    dwn:ndarray
        partial derivative of W faction with respect to In        
        
    
    """
    l2 = l1*0+1.
    
    u1,A,B,k1,k2,fi = parameters
    
    # If its lon, then its circumferential
    if lon == True:
        fi = np.abs(fi-np.pi/2.)
        
    for i in range(15):
        l3=(1./(l1*l2))
        I1=l1**2.+l2**2.+l3**2.
        I4=(l1*np.cos(fi))**2.+(l2*np.sin(fi))**2.
        Ei=A*I1+B*I4+(1.-3.*A-B)*l3**2.-1.0
        dEi=k1*Ei*np.exp(k2*Ei**2.0)
        dw1=u1/2.0+2.*A*dEi
        dw4=2.*B*dEi
        dwn=2.*(1.-3.*A-B)*dEi
        p=2.*(l3**2)*(dw1+dwn)
        f=2.0*(l2**2.0)*dw1+2.0*(l2*np.sin(fi))**2.0*dw4-p
        l2=l2+dh
        l3=(1/(l1*l2))
        I1=l1**2.+l2**2.+l3**2.
        I4=(l1*np.cos(fi))**2.+(l2*np.sin(fi))**2.
        Ei=A*I1+B*I4+(1.-3.*A-B)*l3**2.-1.0
        dEi=k1*Ei*np.exp(k2*Ei**2.0)
        dw1=u1/2.0+2.*A*dEi
        dw4=2.*B*dEi
        dwn=2.*(1.-3.*A-B)*dEi
        p=2*(l3**2)*(dw1+dwn)
        f1=2.0*(l2**2.0)*dw1+2.0*(l2*np.sin(fi))**2.0*dw4-p
        l2=l2-dh;
        l2=l2-f/((f1-f)/dh)
    s11=2.0*l1**2.0*dw1+2.0*(l1*np.cos(fi))**2.0*dw4-p
    return (l2,1./(l1*l2),s11,dw1,dw4,dwn)

def holzapfel2015_uniaxial_objective(phenome,lon_curv,cir_curv,lower_bound,upper_bound):
    
    #Grid boundary
    for i in range(len(lower_bound)):
        if phenome[i]<=lower_bound[i] or phenome[i]>=upper_bound[i]:
            return 1e14
    
    #Numerical step for derivative approximation
    dh = 1e-7 
    
    #Gasser 2015 parameters
    parameters = np.array(phenome,dtype=np.float64)
    (u,A,B,k1,k2,fi) = parameters  # [u,A,B,k1,k2,fi,dh]

    #Lambda 2 and 3, Uniaxial long stress, partial derivates of invariants.
    ll2, ll3, sl1, dw1l, dw4l, dwnl = holza2015_uniaxial(parameters,lon_curv[:,0])
    
    if np.isnan(sl1).any():
        return 1e11
    
    #Partial derivatives of invariants depending on the principal stretch (lambda1).    
    _ , _, _, ddw1l, ddw4l, ddwnl = holza2015_uniaxial(parameters,lon_curv[:,0],dh=dh)
    
    # Psi, term that determines the proportion between the isotropy and anisotropy (please refer to Canales (2020))
    psil = (dw1l + dw4l*np.sin(fi)**2.0 + dwnl)/(dw1l+dwnl)
    
    # Derivative of PSI
    dpsil = (psil - (ddw1l + ddw4l*np.sin(fi)**2.0 + ddwnl)/(ddw1l+ddwnl))/dh
    
    #Stability condition
    condition_l = dpsil - (2.*psil/lon_curv[:,0])
    
    #Penalization of the objective function if the condition isn´t met. 
    for i in condition_l:
        if i > 0:
            return 1e9

    #Lambda 2 and 3, Uniaxial long stress, partial derivates of invariants.
    lc2,lc3, sc1, dw1c, dw4c, dw6c  = holza2015_uniaxial(parameters,cir_curv[:,0],lon=False,dh=dh)
    
    if np.isnan(sc1).any():
        return 1e12

    error = np.sqrt(np.square(sl1-lon_curv[:,1]).sum()) + np.sqrt(np.square(sc1-cir_curv[:,1]).sum())
    
    return error

def holzapfel2015_lon_cir_plotter(plot_info,save_fig=False,experimental_data=None):
    """ This function plot the longitudinal and circumferential stress curves of
    Holzapfel 2015, with a homogeneous deformation gradient. It is possible to save
    the figures and also to plot the experimental data.
    
    plot_info: dict
        this gives all the plotting information
        
        example:
        plot_info = {"longitudinal_stretch": np.linspace(1,2.4,100),
             "circumferential_stretch": np.linspace(1,2.4,100),
             "x_limits": [1, 2.4],
             "y_limits" : [0, 300],
             "parameters" : [10.84,0.265,0.1343,39.73, 32.31,56.168*np.pi/180.],
             "figsize": [10, 5]
            }
            
    save_fig: bool
        if it is true, then the figure will be saved.
        
    experimental_data: list or tuple
        Is a list with the longitudinal and circumferential information
        [longitudinal_data, circumferential_data]
        
        longitudinal or circumferential data, has a different number of rows(number of points)
        but they have only two columns (stretch, Cauchy stress).
    
    """
    
    if experimental_data != None:
        curve_l = experimental_data[0]
        curve_c = experimental_data[1]
    
    ll1, lc1 = plot_info["longitudinal_stretch"], plot_info["circumferential_stretch"]
    parameters = np.array(plot_info["parameters"],dtype=np.float64) 
    ll2,ll3,sl1,_,_,_ = holza2015_uniaxial(parameters,ll1)
    lc2,lc3,sc1,_,_,_ = holza2015_uniaxial(parameters,lc1,lon=False)
    
    #Setting up plots
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(*plot_info["figsize"])
    ax1.set_xlim(*plot_info["x_limits"])
    ax1.set_ylim(*plot_info["y_limits"])
    
    #Stress plot
    ax1.set_xlabel('Stretch [mm/mm]')
    ax1.set_title('Incompressible uniaxial Test')
    ax1.plot(ll1,sl1,'k-',label='Longitudinal Fit')
    ax1.plot(lc1,sc1,'c-',label='Circumferential Fit')
    
    if experimental_data != None:
        ax1.plot(curve_l[:,0],curve_l[:,1],'kv',label='Longitudinal Data')
        ax1.plot(curve_c[:,0],curve_c[:,1],'co',label='Circumferential Data')
        
    ax1.set_ylabel('Cauchy Stress [kPa]',size=10)
    ax1.legend(loc='upper left')
    
    #Transversal stretches plot
    ax2.set_xlabel('Stretch [mm/mm]')
    ax2.set_title('Transversal stretches')
    ax2.plot(ll1,ll3,'k-',label='Longitudinal $\lambda_3$')
    ax2.plot(lc1,lc3,'c-',label='Circumferential $\lambda_3$')
    ax2.plot(ll1,ll2,'k--',label='Longitudinal $\lambda_2$')
    ax2.plot(lc1,lc2,'c--',label='Circumferential $\lambda_2$')    
    ax2.legend(loc='upper right')
    
    if save_fig == True:
        fig.savefig(plot_info["fig_name"])
    
    
def holzapfel2015_uniaxial_evo_characterization(optimization_info):
    #Setting up boundaries and experimental data
    lower_bound = optimization_info["lower_bound"] #[u,A,B,k1,k2,fi]
    upper_bound = optimization_info["upper_bound"] #
    lon_curv = optimization_info["experimental_data"][0]
    cir_curv = optimization_info["experimental_data"][1]

    # Setting up the objective function
    objective = partial(holzapfel2015_uniaxial_objective,lon_curv=lon_curv,
                        cir_curv=cir_curv,lower_bound=lower_bound,upper_bound=upper_bound)
    
    # Defining the population information
    dim = 6
    popsize = optimization_info["popsize"]
    num_offspring = 4 * popsize
    generations = optimization_info["generations"] 
    population = []
    problem = Problem(objective,num_objectives=1,max_evaluations=(num_offspring+popsize)*generations)

    #Initialization of algorithm
    for _ in range(popsize):
        ind = CMSAIndividual(num_parents=popsize, num_offspring=num_offspring)
        ind.genome = [random.uniform(lower_bound[i],upper_bound[i]) for i in range(dim)]
        ind.genome = ind.genome
        population.append(ind)

    ea = CommaEA(problem, population, popsize, num_offspring,verbosity=1)
    #Running the EA algorithm
    ea.run()
    #Printing some usefull info
    print("Objective value:",ea.population[0].objective_values)
    print("Consumed iterations:", problem.consumed_evaluations)
    return ea.population[0].phenome    
