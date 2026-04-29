"""
---------------------------------------------------------------
  Authors: A. Ramirez-Morales (andres.ramirez.morales@cern.ch)
           H. Garcia-Tecocoatzi
---------------------------------------------------------------
"""
from decays.decay_wrapper import decay
import decays.decay_utils_em as du
import numpy as np


class ElectroWidths:
    """
    Class that administrates the decay width calculations of the hevay baryon widths done by the C++ class
    The class calls the python wrapper and feeds the functions with the needed quatumn numbers
    and masses

    baryon FLAG: 1 -> omega, 2->cascade_6, 3->sigma,# 4 -> lambda, 5-> cascade_3
    ModEx  FLAG: 0 -> ground(grd), 1 -> lambda(lam), 2->rho, rpl->3, rpr->4, mix->5
    decPr  FLAG: 0 -> ...  decayProduct Flag
    """
    def __init__(self, bootstrap=False, baryons='', workpath="."):
        self.m_width = decay(workpath)
        #self.fetch_decay_masses(bootstrap)
        self.fetch_decay_masses_lambda(bootstrap) # lambdas
        self.fetch_decay_masses_cascade_anti3(bootstrap) # cascades_anti3
        self.fetch_decay_masses_sigma(bootstrap) # sigmas
        self.fetch_decay_masses_cascade_prime(bootstrap) # cascades
        self.fetch_decay_masses_omega(bootstrap) # omegas
        self.channel_widths_vector_swave = []
        self.channel_widths_vector_pwave = []
        self.channel_widths_vector_dwave = []

    def total_decay_width(self, baryons, k_prim, massA, SA_val, JA_val, LA_val, SlA_val,
                          ModEx_val, bootstrap=False, m1=0, m2=0, m3=0):
        """
        Method that calls the wrapper and sums the individual decay widths
        """
        mc = m1
        ms = m2
        ml = m3

        # ml = 284 #299.0
        # ms = 455 #465.0
        # mc = 1606 #4928.0
        # k_prim = 5044.799302252 #check this number
        # # massA = 6.354 * 1000

        MassA = massA/1000.0
        mcharm  = mc/1000.0
        mupdown  = ml/1000.0
        mstrange = ms/1000.0

        SA_qm = SA_val
        JA_qm = JA_val
        LA_qm = LA_val
        SlA_qm = SlA_val
        LlA_qm, LrA_qm = self.orbital_projections(ModEx_val, LA_val)
            
        baryon = self.baryon_flag(baryons)
        ModEx = self.ModEx_flag(ModEx_val)
        
        nChannels_Pwave = self.n_channels_Pwave(baryons)
        
        m_lam, m_rho = self.reduced_masses(baryons, mcharm*1000, mupdown*1000, mstrange*1000)
        
        alpha_lam = self.alphas(k_prim, m_lam)
        alpha_rho = self.alphas(k_prim, m_rho)
        
        if LA_val<=1: # loop P-wave (por el momento tambien ground)
            if (ModEx == 0 or ModEx == 1 or ModEx == 2): # GS, P_lambda and P_rho initial states
                nChannels_Swave = self.n_channels_Swave(baryons)
                channel_widths = ([])
                for i in range(nChannels_Swave):
                    decPr = i + 100 + 1
                    MassB = self.decay_mass(bootstrap, baryons, decPr)               
                    single_decay_value = self.m_width.electro_width(MassA, SA_qm, JA_qm, LA_qm, SlA_qm, LlA_qm, LrA_qm,
                                                                    MassB,
                                                                    alpha_lam, alpha_rho,
                                                                    mcharm, mupdown, mstrange,
                                                                    baryon, ModEx, decPr)
                    channel_widths = np.append(channel_widths, single_decay_value)
                    baryon_name, ModEx_name, decPr_name =  du.state_labels(baryon, ModEx, decPr, LA_qm)
                    if not bootstrap:
                        print('%6s |  %10s | %12s | %3s %5.3f | %3s %5.3f | %3s%5.1f | %3s%5.1f | %3s%5.1f | %3s%5.1f | %5.6f '
                              %(baryon_name, ModEx_name, decPr_name, "MA=", MassA, "MB=", MassB, "JA=", JA_qm, "LA=", LA_qm, "SA=", SA_qm, "SlA=", SlA_qm, single_decay_value))

                # sum the individual width to obtain total width
                total_decay_width = np.sum(channel_widths)
                # print(alpha_lam,alpha_rho)
                if not bootstrap:
                    print(' ******************   TOTAL ELECTROMAGNETIC WIDTH FOR', baryons, ModEx_name, round(total_decay_width,4), '   ******************')
                    print('-------------------------------------------------------------------------------------------------------------')
            
                self.channel_widths_vector_pwave.append(channel_widths) # for individual decay tables, this is a list of arrays!
            else:
                # mixed/radial -> ground
                nChannels_Swave = self.n_channels_Swave(baryons)
                channel_widths = ([])             
                for i in range(nChannels_Swave):
                    decPr = i + 100 + 1
                    MassB = self.decay_mass(bootstrap, baryons, decPr)
                    single_decay_value = self.m_width.electro_width(MassA, SA_qm, JA_qm, LA_qm, SlA_qm, LlA_qm, LrA_qm,
                                                                    MassB,
                                                                    alpha_lam, alpha_rho,
                                                                    mcharm, mupdown, mstrange,
                                                                    baryon, ModEx, decPr)
                    
                    channel_widths = np.append(channel_widths, single_decay_value)
                    baryon_name, ModEx_name, decPr_name =  du.state_labels(baryon, ModEx, decPr, LA_qm)
                    if not bootstrap:
                        print('%6s |  %10s | %12s | %3s %5.3f | %3s %5.3f | %3s%5.1f | %3s%5.1f | %3s%5.1f | %3s%5.1f | %5.6f '
                              %(baryon_name, ModEx_name, decPr_name, "MA=", MassA, "MB=", MassB, "JA=", JA_qm, "LA=", LA_qm, "SA=", SA_qm, "SlA=", SlA_qm, single_decay_value))

                nChannels_Pwave = self.n_channels_Pwave(baryons)
                for i in range(nChannels_Pwave):
                    decPr = i + 200 + 1
                    MassB = self.decay_mass(bootstrap, baryons, decPr)
                    single_decay_value = self.m_width.electro_width(MassA, SA_qm, JA_qm, LA_qm, SlA_qm, LlA_qm, LrA_qm,
                                                                    MassB,
                                                                    alpha_lam, alpha_rho,
                                                                    mcharm, mupdown, mstrange,
                                                                    baryon, ModEx, decPr)
                    channel_widths = np.append(channel_widths, single_decay_value)
                    baryon_name, ModEx_name, decPr_name =  du.state_labels(baryon, ModEx, decPr, LA_qm)
                    if not bootstrap:
                        print('%6s |  %10s | %12s | %3s %5.3f | %3s %5.3f | %3s%5.1f | %3s%5.1f | %3s%5.1f | %3s%5.1f | %5.6f '
                              %(baryon_name, ModEx_name, decPr_name, "MA=", MassA, "MB=", MassB, "JA=", JA_qm, "LA=", LA_qm, "SA=", SA_qm, "SlA=", SlA_qm, single_decay_value))
                    
                # sum the individual width to obtain total width
                total_decay_width = np.sum(channel_widths)
                # print(alpha_lam,alpha_rho)
                if not bootstrap:
                    print(' ******************   TOTAL ELECTROMAGNETIC WIDTH FOR', baryons, ModEx_name, round(total_decay_width,4), '   ******************')
                    print('-------------------------------------------------------------------------------------------------------------')
            
                self.channel_widths_vector_pwave.append(channel_widths) # for individual decay tables, this is a list of arrays!

        if LA_val==2: # loop D-wave
            nChannels_Swave = self.n_channels_Swave(baryons)
            channel_widths = ([])
            for i in range(nChannels_Swave):
                decPr = i + 100 + 1
                MassB = self.decay_mass(bootstrap, baryons, decPr)
                single_decay_value = self.m_width.electro_width(MassA, SA_qm, JA_qm, LA_qm, SlA_qm, LlA_qm, LrA_qm,
                                                                MassB,
                                                                alpha_lam, alpha_rho,
                                                                mcharm, mupdown, mstrange,
                                                                baryon, ModEx, decPr)
                    
                channel_widths = np.append(channel_widths, single_decay_value)
                baryon_name, ModEx_name, decPr_name =  du.state_labels(baryon, ModEx, decPr, LA_qm)
                if not bootstrap:
                    print('%6s |  %10s | %12s | %3s %5.3f | %3s %5.3f | %3s%5.1f | %3s%5.1f | %3s%5.1f | %3s%5.1f | %5.6f '
                              %(baryon_name, ModEx_name, decPr_name, "MA=", MassA, "MB=", MassB, "JA=", JA_qm, "LA=", LA_qm, "SA=", SA_qm, "SlA=", SlA_qm, single_decay_value))
                    

            nChannels_Pwave = self.n_channels_Pwave(baryons)
            for i in range(nChannels_Pwave):
                decPr = i + 200 + 1
                MassB = self.decay_mass(bootstrap, baryons, decPr)
                single_decay_value = self.m_width.electro_width(MassA, SA_qm, JA_qm, LA_qm, SlA_qm, LlA_qm, LrA_qm,
                                                                MassB,
                                                                alpha_lam, alpha_rho,
                                                                mcharm, mupdown, mstrange,
                                                                baryon, ModEx, decPr)
                channel_widths = np.append(channel_widths, single_decay_value)
                baryon_name, ModEx_name, decPr_name =  du.state_labels(baryon, ModEx, decPr, LA_qm)
                if not bootstrap:
                    print('%6s |  %10s | %12s | %3s %5.3f | %3s %5.3f | %3s%5.1f | %3s%5.1f | %3s%5.1f | %3s%5.1f | %3s%5.1f |%3s%5.1f |%5.6f '
                              %(baryon_name, ModEx_name, decPr_name, "MA=", MassA, "MB=", MassB, "JA=", JA_qm, "LA=", LA_qm, "SA=", SA_qm, "SlA=", SlA_qm, "LlA=", LlA_qm, "LrA=", LrA_qm, single_decay_value))
                    
            # sum the individual width to obtain total width
            total_decay_width = np.sum(channel_widths)
            # print(alpha_lam,alpha_rho)
            if not bootstrap:
                print(' ******************   TOTAL ELECTROMAGNETIC WIDTH FOR', baryons, ModEx_name, round(total_decay_width,4), '   ******************')
                print('-------------------------------------------------------------------------------------------------------------')
            
            self.channel_widths_vector_dwave.append(channel_widths) # for individual decay tables, this is a list of arrays!

            
        return total_decay_width

    
    def orbital_projections(self, ModEx_val, LA_val):
        """
        Method to fecth the orbital projection (up to D-wave "new")
        """
        if(ModEx_val=="grd"):
            LlA = LA_val
            LrA = LA_val
        elif(ModEx_val=="lam"):
            LlA = LA_val
            LrA = 0
        elif(ModEx_val=="rho"):
            LlA = 0
            LrA = LA_val
        elif(ModEx_val=="rpl"):
            LlA = 0
            LrA = 0
        elif(ModEx_val=="rpr"):            
            LlA = 0
            LrA = 0
        elif(ModEx_val=="mix"):
            LlA = 1
            LrA = 1
        else:
            LlA = -1
            LrA = -1
        return LlA, LrA

    def baryon_flag(self, baryons):
        """
        Method to parse the baryons names to integers
        """
        if(baryons=='omegas'):           return 1
        elif(baryons=='cascades'):       return 2
        elif(baryons=='sigmas'):         return 3
        elif(baryons=='lambdas'):        return 4
        elif(baryons=='cascades_anti3'): return 5

    def reduced_masses(self, baryons, mc_input, ml_input, ms_input):
        """
        Method to calculate reduced masses of the harmonic oscillator
        """
        m_lam, m_rho=0,0
        if(baryons=='omegas'):
            m_rho = ms_input
            m_lam = (3*ms_input*mc_input)/(2*ms_input+mc_input)
        elif(baryons=='cascades' or baryons =='cascades_anti3'):
            m_rho = (ml_input+ms_input)/2
            m_lam = (1.5*(ml_input+ms_input)*mc_input)/(mc_input+ml_input+ms_input)
        elif(baryons=='sigmas' or baryons=='lambdas'):
             m_rho = ml_input
             m_lam = (3*ml_input*mc_input)/(2*ml_input+mc_input)
        return m_lam, m_rho

    def alphas(self, k_prim, m_lam_rho):
        """
        Method to calculate the decay alphas
        """
        value1 = (np.sqrt(3./m_lam_rho)) * k_prim
        value2 = value1*m_lam_rho
        return np.sqrt(value2)/1000. # transform from MeV -> GeV

    def ModEx_flag(self, ModEx_val):
        """
        Method to parse the h.o mode to integers
        grd=0, lam =1, rho=2, rpl=3, rpr=4, mix=5
        """
        if(ModEx_val=='grd'):   return 0
        elif(ModEx_val=='lam'): return 1
        elif(ModEx_val=='rho'): return 2
        elif(ModEx_val=='rpl'): return 3
        elif(ModEx_val=='rpr'): return 4
        elif(ModEx_val=='mix'): return 5

    def n_channels_Swave(self, baryons): 
        """
        Method to set number of decay channels has each baryon
        Defines the number of Ground states decays for any initial baryon
        """
        if(baryons=='omegas'):           return 2
        elif(baryons=='cascades'):       return 6
        elif(baryons=='sigmas'):         return 7
        elif(baryons=='lambdas'):        return 3
        elif(baryons=='cascades_anti3'): return 6

    def n_channels_Pwave(self, baryons):
        """
        Method to set number of decay channels has each baryon
        Defines the number of P wave states decays for any initial baryon
        """
        if(baryons=='omegas'):           return 7  #2
        elif(baryons=='cascades'):       return 28 #6
        elif(baryons=='sigmas'):         return 28 #7
        elif(baryons=='lambdas'):        return 14 #3
        elif(baryons=='cascades_anti3'): return 28 #6

    def decay_mass(self, bootstrap, baryons, decPr):
        """
        Method to fetch mass of the decay products
        """
        if(baryons=='omegas'):
            if(decPr==101):
                if not bootstrap:  return self.omega_1_decay
                else: return np.random.choice(self.gauss_omega_1_decay, size=None)
            elif(decPr==102):
                if not bootstrap:  return self.omega_2_decay
                else: return np.random.choice(self.gauss_omega_2_decay, size=None)
            # omega
            elif(decPr==201):
                if not bootstrap: return self.omega_3_decay
                else: return np.random.choice(self.gauss_omega_3_decay, size=None)
            elif(decPr==202):
                if not bootstrap: return self.omega_4_decay
                else: return np.random.choice(self.gauss_omega_4_decay, size=None)
            elif(decPr==203):
                if not bootstrap: return self.omega_5_decay
                else: return np.random.choice(self.gauss_omega_5_decay, size=None)
            elif(decPr==204):
                if not bootstrap: return self.omega_6_decay
                else: return np.random.choice(self.gauss_omega_6_decay, size=None)
            elif(decPr==205):
                if not bootstrap: return self.omega_7_decay
                else: return np.random.choice(self.gauss_omega_7_decay, size=None)
            elif(decPr==206):
                if not bootstrap: return self.omega_8_decay
                else: return np.random.choice(self.gauss_omega_8_decay, size=None)
            elif(decPr==207):
                if not bootstrap: return self.omega_9_decay
                else: return np.random.choice(self.gauss_omega_9_decay, size=None)
            #else: 
            #    return self.omega_2_decay # testing    
        elif(baryons=='cascades' or baryons=='cascades_anti3'):
            if(decPr==101):
                if not bootstrap: return self.xi_1_decay
                else: return np.random.choice(self.gauss_xi_1_decay, size=None)
            elif(decPr==102):
                if not bootstrap: return self.xi_1_decay
                else: return np.random.choice(self.gauss_xi_1_decay, size=None)
            elif(decPr==103):
                if not bootstrap: return self.xi_p_1_decay
                else: return np.random.choice(self.gauss_xi_p_1_decay, size=None)
            elif(decPr==104):
                if not bootstrap: return self.xi_p_2_decay
                else: return np.random.choice(self.gauss_xi_p_2_decay, size=None)
            elif(decPr==105):
                if not bootstrap: return self.xi_p_1_decay
                else: return np.random.choice(self.gauss_xi_p_1_decay, size=None)
            elif(decPr==106):
                if not bootstrap: return self.xi_p_2_decay
                else: return np.random.choice(self.gauss_xi_p_2_decay, size=None)
            # cascade anti_triplet
            elif(decPr==201 or decPr==208):
                if not bootstrap: return self.xi_2_decay
                else: return np.random.choice(self.gauss_xi_2_decay, size=None)
            elif(decPr==202 or decPr==209):
                if not bootstrap: return self.xi_3_decay
                else: return np.random.choice(self.gauss_xi_3_decay, size=None)
            elif(decPr==203 or decPr==210):
                if not bootstrap: return self.xi_4_decay
                else: return np.random.choice(self.gauss_xi_4_decay, size=None)
            elif(decPr==204 or decPr==211):
                if not bootstrap: return self.xi_5_decay
                else: return np.random.choice(self.gauss_xi_5_decay, size=None)
            elif(decPr==205 or decPr==212):
                if not bootstrap: return self.xi_6_decay
                else: return np.random.choice(self.gauss_xi_6_decay, size=None)
            elif(decPr==206 or decPr==213):
                if not bootstrap: return self.xi_7_decay
                else: return np.random.choice(self.gauss_xi_7_decay, size=None)
            elif(decPr==207 or decPr==214):
                if not bootstrap: return self.xi_8_decay
                else: return np.random.choice(self.gauss_xi_8_decay, size=None)
            # cascade prime
            elif(decPr==215 or decPr==222):
                if not bootstrap: return self.xi_p_3_decay
                else: return np.random.choice(self.gauss_xi_p_3_decay, size=None)
            elif(decPr==216 or decPr==223):
                if not bootstrap: return self.xi_p_4_decay
                else: return np.random.choice(self.gauss_xi_p_4_decay, size=None)
            elif(decPr==217 or decPr==224):
                if not bootstrap: return self.xi_p_5_decay
                else: return np.random.choice(self.gauss_xi_p_5_decay, size=None)
            elif(decPr==218 or decPr==225):
                if not bootstrap: return self.xi_p_6_decay
                else: return np.random.choice(self.gauss_xi_p_6_decay, size=None)
            elif(decPr==219 or decPr==226):
                if not bootstrap: return self.xi_p_7_decay
                else: return np.random.choice(self.gauss_xi_p_7_decay, size=None)
            elif(decPr==220 or decPr==227):
                if not bootstrap: return self.xi_p_8_decay
                else: return np.random.choice(self.gauss_xi_p_8_decay, size=None)
            elif(decPr==221 or decPr==228):
                if not bootstrap: return self.xi_p_9_decay
                else: return np.random.choice(self.gauss_xi_p_9_decay, size=None)
            else:
                return self.xi_p_2_decay # test
                        
        elif(baryons=='sigmas'):
            if(decPr==101):
                if not bootstrap: return self.sigma_1_decay
                else: return np.random.choice(self.gauss_sigma_1_decay, size=None)
            elif(decPr==102):
                if not bootstrap: return self.sigma_2_decay
                else: return np.random.choice(self.gauss_sigma_2_decay, size=None)
            elif(decPr==103):
                if not bootstrap: return self.sigma_1_decay
                else: return np.random.choice(self.gauss_sigma_1_decay, size=None)
            elif(decPr==104):
                if not bootstrap: return self.sigma_2_decay
                else: return np.random.choice(self.gauss_sigma_2_decay, size=None)
            elif(decPr==105):
                if not bootstrap: return self.sigma_1_decay
                else: return np.random.choice(self.gauss_sigma_1_decay, size=None)
            elif(decPr==106):
                if not bootstrap: return self.sigma_2_decay
                else: return np.random.choice(self.gauss_sigma_2_decay, size=None)
            elif(decPr==107):
                if not bootstrap: return self.lambda_1_decay
                else: return np.random.choice(self.gauss_lambda_1_decay, size=None)
            # sigmas
            elif(decPr==201 or decPr==208 or decPr==215):
                if not bootstrap: return self.sigma_3_decay
                else: return np.random.choice(self.gauss_sigma_3_decay, size=None)
            elif(decPr==202 or decPr==209 or decPr==216):
                if not bootstrap: return self.sigma_4_decay
                else: return np.random.choice(self.gauss_sigma_4_decay, size=None)
            elif(decPr==203 or decPr==210 or decPr==217):
                if not bootstrap: return self.sigma_5_decay
                else: return np.random.choice(self.gauss_sigma_5_decay, size=None)
            elif(decPr==204 or decPr==211 or decPr==218):
                if not bootstrap: return self.sigma_6_decay
                else: return np.random.choice(self.gauss_sigma_6_decay, size=None)
            elif(decPr==205 or decPr==212 or decPr==219):
                if not bootstrap: return self.sigma_7_decay
                else: return np.random.choice(self.gauss_sigma_7_decay, size=None)
            elif(decPr==206 or decPr==213 or decPr==220):
                if not bootstrap: return self.sigma_8_decay
                else: return np.random.choice(self.gauss_sigma_8_decay, size=None)
            elif(decPr==207 or decPr==214 or decPr==221):
                if not bootstrap: return self.sigma_9_decay
                else: return np.random.choice(self.gauss_sigma_9_decay, size=None)
            # lambdas
            elif(decPr==222):
                if not bootstrap: return self.lambda_2_decay
                else: return np.random.choice(self.gauss_lambda_2_decay, size=None)
            elif(decPr==223):
                if not bootstrap: return self.lambda_3_decay
                else: return np.random.choice(self.gauss_lambda_3_decay, size=None)
            elif(decPr==224):
                if not bootstrap: return self.lambda_4_decay
                else: return np.random.choice(self.gauss_lambda_4_decay, size=None)
            elif(decPr==225):
                if not bootstrap: return self.lambda_5_decay
                else: return np.random.choice(self.gauss_lambda_5_decay, size=None)
            elif(decPr==226):
                if not bootstrap: return self.lambda_6_decay
                else: return np.random.choice(self.gauss_lambda_6_decay, size=None)
            elif(decPr==227):
                if not bootstrap: return self.lambda_7_decay
                else: return np.random.choice(self.gauss_lambda_7_decay, size=None)
            elif(decPr==228):
                if not bootstrap: return self.lambda_8_decay
                else: return np.random.choice(self.gauss_lambda_8_decay, size=None)

        elif(baryons=='lambdas'):
            if(decPr==101):
                if not bootstrap: return self.lambda_1_decay
                else: return np.random.choice(self.gauss_lambda_1_decay, size=None)
            elif(decPr==102):
                if not bootstrap: return self.sigma_1_decay
                else: return np.random.choice(self.gauss_sigma_1_decay, size=None)
            elif(decPr==103):
                if not bootstrap: return self.sigma_2_decay
                else: return np.random.choice(self.gauss_sigma_2_decay, size=None)
            # lambdas
            elif(decPr==201):
                if not bootstrap: return self.lambda_2_decay
                else: return np.random.choice(self.gauss_lambda_2_decay, size=None)
            elif(decPr==202):
                if not bootstrap: return self.lambda_3_decay
                else: return np.random.choice(self.gauss_lambda_3_decay, size=None)
            elif(decPr==203):
                if not bootstrap: return self.lambda_4_decay
                else: return np.random.choice(self.gauss_lambda_4_decay, size=None)
            elif(decPr==204):
                if not bootstrap: return self.lambda_5_decay
                else: return np.random.choice(self.gauss_lambda_5_decay, size=None)
            elif(decPr==205):
                if not bootstrap: return self.lambda_6_decay
                else: return np.random.choice(self.gauss_lambda_6_decay, size=None)
            elif(decPr==206):
                if not bootstrap: return self.lambda_7_decay
                else: return np.random.choice(self.gauss_lambda_7_decay, size=None)
            elif(decPr==207):
                if not bootstrap: return self.lambda_8_decay
                else: return np.random.choice(self.gauss_lambda_8_decay, size=None)
            # sigmas0
            elif(decPr==208):
                if not bootstrap: return self.sigma_3_decay
                else: return np.random.choice(self.gauss_sigma_3_decay, size=None)
            elif(decPr==209):
                if not bootstrap: return self.sigma_4_decay
                else: return np.random.choice(self.gauss_sigma_4_decay, size=None)
            elif(decPr==210):
                if not bootstrap: return self.sigma_5_decay
                else: return np.random.choice(self.gauss_sigma_5_decay, size=None)
            elif(decPr==211):
                if not bootstrap: return self.sigma_6_decay
                else: return np.random.choice(self.gauss_sigma_6_decay, size=None)
            elif(decPr==212):
                if not bootstrap: return self.sigma_7_decay
                else: return np.random.choice(self.gauss_sigma_7_decay, size=None)
            elif(decPr==213):
                if not bootstrap: return self.sigma_8_decay
                else: return np.random.choice(self.gauss_sigma_8_decay, size=None)
            elif(decPr==214):
                if not bootstrap: return self.sigma_9_decay
                else: return np.random.choice(self.gauss_sigma_9_decay, size=None)
        
    #def fetch_decay_masses(self, bootstrap):
        '''
        Method to fetch the decay products coming from our fit (mA)
        '''
    #    self.omega_mass    = 6.06400
    #    self.omega_s_mass  = 6.09300
    #    self.sigma_mass    = 5.80500
    #    self.sigma_s_mass  = 5.83400
    #    self.xi_p_mass     = 5.92500
    #    self.xi_p_s_mass   = 5.95500
    #    self.xi_mass       = 5.80600
    #    self.lambda_mass   = 5.61400
       
    #    if(bootstrap):
    #        self.gauss_omega    = np.random.normal(6.06400, 0.00600, 10000)
    #        self.gauss_omega_s  = np.random.normal(6.09300, 0.00700, 10000)
    #        self.gauss_sigma    = np.random.normal(5.80500, 0.00600, 10000)
    #        self.gauss_sigma_s  = np.random.normal(5.83400, 0.00700, 10000)
    #        self.gauss_xi_p     = np.random.normal(5.92500, 0.00500, 10000)
    #        self.gauss_xi_p_s   = np.random.normal(5.95500, 0.00500, 10000)
    #        self.gauss_xi       = np.random.normal(5.80600, 0.00700, 10000)
    #        self.gauss_lambda   = np.random.normal(5.61400, 0.00700, 10000)


    def fetch_decay_masses_lambda(self, bootstrap):
        '''
        Method to fetch the decay products coming from our fit (mA)
        '''
        # decay to lambda
        self.lambda_1_decay = 2.261   # 0 ground
        self.lambda_2_decay = 2.616   # 0 2p1/2-lam 
        self.lambda_3_decay = 2.643   # 0 2p3/2-lam
        self.lambda_4_decay = 2.799   # 0 2p1/2-rho
        self.lambda_5_decay = 2.841   # 0 4p1/2-rho
        self.lambda_6_decay = 2.826   # 0 2p3/2-rho
        self.lambda_7_decay = 2.868   # 0 4p3/2-rho
        self.lambda_8_decay = 2.913   # 0 4p5/2-rho
        if(bootstrap):
            self.gauss_lambda_1_decay  = np.random.normal(2.26100, 0.011, 10000) # 0 ground   
            self.gauss_lambda_2_decay  = np.random.normal(2.61600, 0.010, 10000) # 0 2p1/2-lam
            self.gauss_lambda_3_decay  = np.random.normal(2.64300, 0.010, 10000) # 0 2p3/2-lam
            self.gauss_lambda_4_decay  = np.random.normal(2.799  , 0.015, 10000) # 0 2p1/2-rho
            self.gauss_lambda_5_decay  = np.random.normal(2.841  , 0.023, 10000) # 0 4p1/2-rho
            self.gauss_lambda_6_decay  = np.random.normal(2.826  , 0.015, 10000) # 0 2p3/2-rho
            self.gauss_lambda_7_decay  = np.random.normal(2.868  , 0.020, 10000) # 0 4p3/2-rho
            self.gauss_lambda_8_decay  = np.random.normal(2.913  , 0.021, 10000) # 0 4p5/2-rho


    def fetch_decay_masses_cascade_anti3(self, bootstrap):
        '''
        Method to fetch the decay products coming from our fit (mA)
        '''
        # decay to cascades anti-triplet
        self.xi_1_decay = 2.46600 # 0 ground
        self.xi_2_decay = 2.78800 # 0 2p1/2-lam 
        self.xi_3_decay = 2.81500 # 0 2p3/2-lam
        self.xi_4_decay = 2.935   # 0 2p1/2-rho
        self.xi_5_decay = 2.977   # 0 4p1/2-rho
        self.xi_6_decay = 2.962   # 0 2p3/2-rho
        self.xi_7_decay = 3.004   # 0 4p3/2-rho
        self.xi_8_decay = 3.049   # 0 4p5/2-rho
        if(bootstrap):
            self.gauss_xi_1_decay  = np.random.normal(2.46600, 0.010, 10000) # 0 ground   
            self.gauss_xi_2_decay  = np.random.normal(2.78800, 0.010, 10000) # 0 2p1/2-lam
            self.gauss_xi_3_decay  = np.random.normal(2.81500, 0.010, 10000) # 0 2p3/2-lam
            self.gauss_xi_4_decay  = np.random.normal(2.935  , 0.012, 10000) # 0 2p1/2-rho
            self.gauss_xi_5_decay  = np.random.normal(2.977  , 0.020, 10000) # 0 4p1/2-rho
            self.gauss_xi_6_decay  = np.random.normal(2.962  , 0.012, 10000) # 0 2p3/2-rho
            self.gauss_xi_7_decay  = np.random.normal(3.004  , 0.017, 10000) # 0 4p3/2-rho
            self.gauss_xi_8_decay  = np.random.normal(3.049  , 0.019, 10000) # 0 4p5/2-rho


    def fetch_decay_masses_sigma(self, bootstrap):
        '''
        Method to fetch the decay products coming from our fit (mA)
        '''
        # decay to sigma
        self.sigma_1_decay = 2.456   # 0 ground
        self.sigma_2_decay = 2.525   # 0 ground
        self.sigma_3_decay = 2.811   # 0 2p1/2-lam 
        self.sigma_4_decay = 2.853   # 0 4p1/2-lam
        self.sigma_5_decay = 2.838   # 0 2p3/2-lam
        self.sigma_6_decay = 2.880   # 0 4p3/2-lam
        self.sigma_7_decay = 2.925   # 0 4p5/2-lam
        self.sigma_8_decay = 2.994   # 0 2p1/2-rho
        self.sigma_9_decay = 3.021   # 0 2p3/2-rho
        if(bootstrap):
            self.gauss_sigma_1_decay  = np.random.normal(2.456, 0.011, 10000) # 0 ground
            self.gauss_sigma_2_decay  = np.random.normal(2.525, 0.011, 10000) # 0 ground   
            self.gauss_sigma_3_decay  = np.random.normal(2.811, 0.012, 10000) # 0 2p1/2-lam
            self.gauss_sigma_4_decay  = np.random.normal(2.853, 0.017, 10000) # 0 4p1/2-lam
            self.gauss_sigma_5_decay  = np.random.normal(2.838, 0.013, 10000) # 0 2p3/2-lam
            self.gauss_sigma_6_decay  = np.random.normal(2.880, 0.013, 10000) # 0 4p3/2-lam
            self.gauss_sigma_7_decay  = np.random.normal(2.925, 0.016, 10000) # 0 4p5/2-lam
            self.gauss_sigma_8_decay  = np.random.normal(2.994, 0.017, 10000) # 0 2p1/2-rho
            self.gauss_sigma_9_decay  = np.random.normal(3.021, 0.017, 10000) # 0 2p3/2-rho

            
    def fetch_decay_masses_cascade_prime(self, bootstrap):
        '''
        Method to fetch the decay products coming from our fit (mA)
        '''
        # decay to cascades prime
        self.xi_p_1_decay = 2.571   # 0 ground
        self.xi_p_2_decay = 2.640   # 0 ground
        self.xi_p_3_decay = 2.893   # 0 2p1/2-lam 
        self.xi_p_4_decay = 2.935   # 0 4p1/2-lam
        self.xi_p_5_decay = 2.920   # 0 2p3/2-lam
        self.xi_p_6_decay = 2.962   # 0 4p3/2-lam
        self.xi_p_7_decay = 3.007   # 0 4p5/2-lam
        self.xi_p_8_decay = 3.040   # 0 2p1/2-rho
        self.xi_p_9_decay = 3.067   # 0 2p3/2-rho
        if(bootstrap):
            self.gauss_xi_p_1_decay  = np.random.normal(2.571, 0.008, 10000) # 0 ground
            self.gauss_xi_p_2_decay  = np.random.normal(2.640, 0.007, 10000) # 0 ground   
            self.gauss_xi_p_3_decay  = np.random.normal(2.893, 0.009, 10000) # 0 2p1/2-lam
            self.gauss_xi_p_4_decay  = np.random.normal(2.935, 0.015, 10000) # 0 4p1/2-lam
            self.gauss_xi_p_5_decay  = np.random.normal(2.920, 0.009, 10000) # 0 2p3/2-lam
            self.gauss_xi_p_6_decay  = np.random.normal(2.962, 0.009, 10000) # 0 4p3/2-lam
            self.gauss_xi_p_7_decay  = np.random.normal(3.007, 0.012, 10000) # 0 4p5/2-lam
            self.gauss_xi_p_8_decay  = np.random.normal(3.040, 0.010, 10000) # 0 2p1/2-rho
            self.gauss_xi_p_9_decay  = np.random.normal(3.067, 0.010, 10000) # 0 2p3/2-rho


    def fetch_decay_masses_omega(self, bootstrap):
        '''
        Method to fetch the decay products coming from our fit (mA)
        '''
        # decay to omegas
        self.omega_1_decay = 2.709   # 0 ground
        self.omega_2_decay = 2.778   # 0 ground
        self.omega_3_decay = 3.008   # 0 2p1/2-lam
        self.omega_4_decay = 3.050   # 0 4p1/2-lam
        self.omega_5_decay = 3.035   # 0 2p3/2-lam
        self.omega_6_decay = 3.077   # 0 4p3/2-lam
        self.omega_7_decay = 3.122   # 0 4p5/2-lam
        self.omega_8_decay = 3.129   # 0 2p1/2-rho
        self.omega_9_decay = 3.156   # 0 2p3/2-rho
        if(bootstrap):
            self.gauss_omega_1_decay  = np.random.normal(2.709, 0.010, 10000) # 0 ground
            self.gauss_omega_2_decay  = np.random.normal(2.778, 0.009, 10000) # 0 ground   
            self.gauss_omega_3_decay  = np.random.normal(3.008, 0.010, 10000) # 0 2p1/2-lam
            self.gauss_omega_4_decay  = np.random.normal(3.050, 0.015, 10000) # 0 4p1/2-lam
            self.gauss_omega_5_decay  = np.random.normal(3.035, 0.010, 10000) # 0 2p3/2-lam
            self.gauss_omega_6_decay  = np.random.normal(3.077, 0.009, 10000) # 0 4p3/2-lam
            self.gauss_omega_7_decay  = np.random.normal(3.122, 0.012, 10000) # 0 4p5/2-lam
            self.gauss_omega_8_decay  = np.random.normal(3.129, 0.009, 10000) # 0 2p1/2-rho
            self.gauss_omega_9_decay  = np.random.normal(3.156, 0.010, 10000) # 0 2p3/2-rho

        
