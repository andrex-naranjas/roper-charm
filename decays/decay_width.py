"""
---------------------------------------------------------------
  Authors: A. Ramirez-Morales (andres.ramirez@fisica.uaz.edu.mx)
           H. Garcia-Tecocoatzi
---------------------------------------------------------------
"""
from decays.decay_wrapper import decay
import decays.decay_utils as du
import numpy as np


class DecayWidths:
    """
    Class that administrates the decay width calculations of the hevay baryon widths done by the C++ class
    The class calls the python wrapper and feeds the functions with the needed quatumn numbers
    and masses. 

    baryon FLAG: 1 -> omega, 2->cascade_6, 3->sigma,# 4 -> lambda, 5-> cascade_3
    ModEx  FLAG: 0-> ground(grd), 1 -> lambda(lam), 2->rho, 3->rop_lam(rpl), 4->rop_rho(rpr), 5->mix  Excitation
    decPr  FLAG: 3->Xi'+Pi, 5->Sigma+K  decayProduct Flag
    """
    def __init__(self, bootstrap=False, baryons='', workpath="."):
        self.m_width = decay(workpath)
        self.fetch_decay_masses(bootstrap)
        self.set_gamma_val(bootstrap)
        self.channel_widths_vector = []

    def load_average_mass(self, mass_avg=0):
        self.MassA_avg = mass_avg/1000.0 # central value of the bootstrap distribution
        
    def total_decay_width(self, baryons, tau, massA, SA_val, L_val, JA_val, SL_val, ModEx_val, bootstrap=False, gamma_val=None, md1=0, md2=0, md3=0, mc=0):
        """
        Method that calls the wrapper and sums the individual decay widths
        """
        MassA = massA/1000.0
        MassA_avg = self.MassA_avg
        SA_qm = SA_val
        LA_qm = L_val
        JA_qm = JA_val
        SL_qm = SL_val
        baryon= self.baryon_flag(baryons)
        ModEx = self.ModEx_flag(ModEx_val)
        nChannels = self.n_channels(baryons)
        m_mu = self.reduced_masses(baryons, md1, md2, md3, mc)
        channel_widths = ([])
        
        alpha_lam = self.alphas(tau, m_mu)
        print(alpha_lam)
        alpha_rho = 1

        if gamma_val is None:
            gamma = self.gamma_fitted(bootstrap)
        else:
            gamma = gamma_val

        for i in range(nChannels):
            decPr = i+1
            MassB,MassC = self.decay_masses(bootstrap, baryons, decPr)
            MassB_avg,MassC_avg = self.decay_masses(False, baryons, decPr)
            single_decay_value = self.m_width.decay_width(MassA_avg, MassB_avg, MassC_avg, MassA, MassB, MassC,
                                                          gamma, SA_qm,
                                                          LA_qm, JA_qm, SL_qm, alpha_lam, alpha_rho,
                                                          baryon, ModEx, decPr)
            if (MassA_avg<MassB_avg+MassC_avg): # energy conservation
                single_decay_value = 0.
            channel_widths = np.append(channel_widths, single_decay_value)
            baryon_name, ModEx_name, decPr_name = du.state_labels(baryon, ModEx, decPr, LA_qm)
            if not bootstrap:
                print('%6s |  %10s | %12s |  %5.3f |  %5.3f | %5.3f |  %5.1f |  %5.1f |  %5.1f |  %5.1f | %5.6f '
                      %(baryon_name, ModEx_name, decPr_name, MassA, MassB, MassC, JA_qm, LA_qm, SA_qm, SL_qm, single_decay_value))
                    
        # sum the individual width to obtain total width
        total_decay_width = np.sum(channel_widths)
        # print(alpha_lam,alpha_rho)
        if not bootstrap:
            print('          ******************   TOTAL WIDTH FOR', baryons, ModEx_name, round(total_decay_width,4), '   ******************')
            print('-------------------------------------------------------------------------------------------------------------')
            
        self.channel_widths_vector.append(channel_widths) # for individual decay tables, this is a list of arrays!
        return total_decay_width


    def gamma_fitted(self, bootstrap=False):
        """
        gamma=17.25/np.sqrt(7) # old value
        """
        if not bootstrap: return self.gamma_fit
        else:
            return np.random.choice(self.gauss_gamma, size=None)
        
    def baryon_flag(self, baryons):
        """
        Method to parse the baryons names to integers
        """
        if(baryons=='omegas'):           return 1
        elif(baryons=='cascades'):       return 2
        elif(baryons=='sigmas'):         return 3
        elif(baryons=='lambdas'):        return 4
        elif(baryons=='cascades_anti3'): return 5
        
    def ModEx_flag(self, ModEx_val):
        """
        Method to parse the h.o mode to integers
        grd=0, lam =1 , rho=2, rpl=3, rpr=4, mix=5
        """
        if(ModEx_val=='grd'):   return 0
        elif(ModEx_val=='lam'): return 1
        elif(ModEx_val=='rho'): return 2
        elif(ModEx_val=='rpl'): return 3
        elif(ModEx_val=='rpr'): return 4
        elif(ModEx_val=='mix'): return 5

    def n_channels(self, baryons):
        """
        Method to set number of decay channels has each baryon
        """
        if(baryons=='omegas'):           return 1
        elif(baryons=='cascades'):       return 7
        elif(baryons=='sigmas'):         return 1
        elif(baryons=='lambdas'):        return 1
        elif(baryons=='cascades_anti3'): return 6        
        
    def reduced_masses(self, baryons, m1_input, m2_input, m3_input, mc_input):
        """
        Method to calculate reduced masses of the harmonic oscillator
        """
        m_mu=0
        if(baryons=='omegas'):
            m_mu = (m1_input*mc_input)/(m1_input+mc_input)
        elif(baryons=='cascades' or baryons =='cascades_anti3'):
            m_mu = (m2_input*mc_input)/(m2_input+mc_input)
        elif(baryons=='sigmas' or baryons=='lambdas'):
            m_mu = (m3_input*mc_input)/(m3_input+mc_input)
             
        return m_mu
                    
    def alphas(self, tau, m_mu):
        """
        Method to calculate the decay alphas
        """
        value = (4*m_mu*tau)/(3*pow(np.pi, 0.5))
        return pow(value/1000., -1) # transform from GeV -> MeV
            
    def decay_masses(self, bootstrap, baryons, decPr):
        """
        Method to fetch mass of the decay products
        """
        if(baryons=='cascades'):
            if(decPr==1):
                if not bootstrap: return self.lambda_mass, self.kaon_mass
                else: return np.random.choice(self.gauss_lambda, size=None), np.random.choice(self.gauss_kaon, size=None)
            elif(decPr==2):
                if not bootstrap: return self.xi_mass,     self.pion_mass
                else: return np.random.choice(self.gauss_xi, size=None), np.random.choice(self.gauss_pion, size=None)
            elif(decPr==3):
                if not bootstrap: return self.xi_p_mass,   self.pion_mass
                else: return np.random.choice(self.gauss_xi_p, size=None), np.random.choice(self.gauss_pion, size=None)
            elif(decPr==4):
                if not bootstrap: return self.xi_s_mass,   self.pion_mass
                else: return np.random.choice(self.gauss_xi_s, size=None), np.random.choice(self.gauss_pion, size=None)
            elif(decPr==5):
                if not bootstrap: return self.sigma_mass,  self.kaon_mass
                else: return np.random.choice(self.gauss_sigma, size=None), np.random.choice(self.gauss_kaon, size=None)
            elif(decPr==6):
                if not bootstrap: return self.sigma_s_mass,self.kaon_mass
                else: return np.random.choice(self.gauss_sigma_s, size=None), np.random.choice(self.gauss_kaon, size=None)
            elif(decPr==7):
                if not bootstrap: return self.xi_mass,     self.eta_mass
                else: return np.random.choice(self.gauss_xi, size=None), np.random.choice(self.gauss_eta, size=None)         
        elif(baryons=='cascades_anti3'):
            if(decPr==1):
                if not bootstrap: return self.lambda_mass, self.kaon_mass
                else: return np.random.choice(self.gauss_lambda, size=None), np.random.choice(self.gauss_kaon, size=None)
            elif(decPr==2):
                if not bootstrap: return self.xi_mass,     self.pion_mass
                else: return np.random.choice(self.gauss_xi, size=None), np.random.choice(self.gauss_pion, size=None)
            elif(decPr==3):
                if not bootstrap: return self.xi_p_mass,   self.pion_mass
                else: return np.random.choice(self.gauss_xi_p, size=None), np.random.choice(self.gauss_pion, size=None)
            elif(decPr==4):
                if not bootstrap: return self.xi_s_mass,   self.pion_mass
                else: return np.random.choice(self.gauss_xi_s, size=None), np.random.choice(self.gauss_pion, size=None)
            elif(decPr==5):
                if not bootstrap: return self.sigma_mass,  self.kaon_mass
                else: return np.random.choice(self.gauss_sigma, size=None), np.random.choice(self.gauss_kaon, size=None)
            elif(decPr==6):
                if not bootstrap: return self.sigma_s_mass,self.kaon_mass
                else: return np.random.choice(self.gauss_sigma_s, size=None), np.random.choice(self.gauss_kaon, size=None)

    def fetch_decay_masses(self, bootstrap):
        self.pion_mass        = 0.13725
        self.kaon_mass        = 0.49564
        self.eta_mass         = 0.54786
        # Charm hadrons
        self.lambda_mass      = 2.28646
        self.xi_mass          = 2.46908
        self.xi_p_mass        = 2.57850
        self.xi_s_mass        = 2.64563
        self.sigma_mass       = 2.45350
        self.sigma_s_mass     = 2.51813

        if(bootstrap):
            self.gauss_pion        = np.random.normal(0.13725, 0.00295, 10000)
            self.gauss_kaon        = np.random.normal(0.49564, 0.00279, 10000)
            self.gauss_eta         = np.random.normal(0.54786, 0.00002, 10000)
            # Charm baryons
            self.gauss_lambda      = np.random.normal(2.28646, 0.00014, 10000)
            self.gauss_xi          = np.random.normal(2.46908, 0.00158, 10000)
            self.gauss_xi_p        = np.random.normal(2.57850, 0.00100, 10000)
            self.gauss_xi_s        = np.random.normal(2.64563, 0.00100, 10000)
            self.gauss_sigma       = np.random.normal(2.45350, 0.00090, 10000)
            self.gauss_sigma_s     = np.random.normal(2.51813, 0.00280, 10000)
            
            
    def set_gamma_val(self, bootstrap):
        
        self.gamma_fit = 20.8 # this reproduces an specific decay width
        self.gauss_gamma = np.random.normal(20.8, np.power( 1.47**2 + 4.44**2, 0.5) , 10000)
        if(bootstrap):
            self.gauss_gamma = np.random.normal(20.8, np.power( 1.5**2 + 2.0**2, 0.5) , 10000) # error 2, is arbitrary
