import numpy as np

import matplotlib.pyplot as plt

def calculate_HHG_Delta(omegas_DVR,omegas_approx,signal_DVR,signal_approx,omega,Ecutoff):
    positive_omegas_DVR=(omegas_DVR / omega) > 0.8
    positive_omegas_approx=(omegas_approx / omega) > 0.8
    omegas_DVR_pos=(omegas_DVR / omega)[positive_omegas_DVR]
    omegas_approx_pos=(omegas_approx / omega)[positive_omegas_approx]
    omegas_DVR_pos=omegas_DVR_pos[::6]
    omegas_approx_pos=omegas_approx_pos[::6]
    signal_of_interest_DVR=(signal_DVR[positive_omegas_DVR])[::6]
    signal_of_interest_approx=(signal_approx[positive_omegas_DVR])[::6]
    difference=np.log10(signal_of_interest_DVR)-np.log10(signal_of_interest_approx)
    cumulative_difference=abs(difference)
    N=int(Ecutoff/omega)
    twoN=int(2*Ecutoff/omega)
    if N%2==0:
        N+=1
    if twoN%2==0:
        twoN+=1
    toConsider_N=int((N+1)/2)
    toConsider_twoN=int((twoN+1)/2)
    print("Delta")
    print("N (%d): %.7f"%((N),(np.cumsum(cumulative_difference)[toConsider_N-1]/(N))))
    print("2N (%d): %.7f"%((twoN),(np.cumsum(cumulative_difference)[toConsider_twoN-1]/(twoN))))

def calculate_HHG_Upsilon(omegas_DVR,omegas_approx,signal_DVR,signal_approx,omega,Ecutoff):
    positive_omegas_DVR=(omegas_DVR / omega) > 0.8
    positive_omegas_approx=(omegas_approx / omega) > 0.8
    omegas_DVR_pos=(omegas_DVR / omega)[positive_omegas_DVR]
    omegas_approx_pos=(omegas_approx / omega)[positive_omegas_approx]
    omegas_DVR_pos=omegas_DVR_pos[::6]
    omegas_approx_pos=omegas_approx_pos[::6]
    signal_of_interest_DVR=(signal_DVR[positive_omegas_DVR])[::6]
    signal_of_interest_approx=(signal_approx[positive_omegas_DVR])[::6]
    difference=np.log10(signal_of_interest_DVR)-np.log10(signal_of_interest_approx)
    cumulative_difference=abs(difference)**2
    N=int(Ecutoff/omega)
    twoN=int(2*Ecutoff/omega)
    if N%2==0:
        N+=1
    if twoN%2==0:
        twoN+=1
    toConsider_N=int((N+1)/2)
    toConsider_twoN=int((twoN+1)/2)
    print("Upsilon")
    print("N (%d): %.7f"%((N),(np.cumsum(cumulative_difference)[toConsider_N-1]/(N))))
    print("2N (%d): %.7f"%((twoN),(np.cumsum(cumulative_difference)[toConsider_twoN-1]/(twoN))))

def calculate_HHG_D(omegas_DVR,omegas_approx,signal_DVR,signal_approx,omega,Ecutoff):
    positive_omegas_DVR=(omegas_DVR / omega) > 0.8
    positive_omegas_approx=(omegas_approx / omega) > 0.8
    omegas_DVR_pos=(omegas_DVR / omega)[positive_omegas_DVR]
    omegas_approx_pos=(omegas_approx / omega)[positive_omegas_approx]
    omegas_DVR_pos=omegas_DVR_pos
    omegas_approx_pos=omegas_approx_pos
    signal_of_interest_DVR=(signal_DVR[positive_omegas_DVR])
    signal_of_interest_approx=(signal_approx[positive_omegas_DVR])
    DVR_signal=np.log10(signal_of_interest_DVR)
    approx_signal=np.log10(signal_of_interest_approx)
    Dcorr_unsummed=DVR_signal*DVR_signal-approx_signal*DVR_signal
    N=int(Ecutoff/omega)
    twoN=int(2*Ecutoff/omega)
    if N%2==0:
        N+=1
    if twoN%2==0:
        twoN+=1
    toConsider_N=int((N+1)/2)
    toConsider_twoN=int((twoN+1)/2)
    #print(len(Dcorr_unsummed[omegas_approx_pos<N]))
    print("D corr ")
    print("N (%d): %.7f"%((N),np.sum(Dcorr_unsummed[omegas_approx_pos<N])))
    print("N (%d): %.7f"%((twoN),np.sum(Dcorr_unsummed[omegas_approx_pos<twoN])))
