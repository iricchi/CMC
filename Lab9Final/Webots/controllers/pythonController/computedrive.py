import numpy as np

def computefreqamp(drive, gradient):
  vsatbody=0.
  Rsatbody=0.
  cv1body=0.2
  cv0body=0.3
  cr1body=0.03
  cr0body=0.196
        
  vsatlimb=0.
  Rsatlimb=0.
        
  cv1limb=1
  cv0limb=0.
  cr1limb=1
  cr0limb=0.
        
  dlowbody=1.
  dlowlimb=1.
  dhighbody=5.
  dhighlimb=3.
        
  d = drive
        
        
  if d !=None:
    if d >=dlowbody and d <=dhighbody:
        # v = c1*d + c0
        vref_body = cv1body*d + cv0body
        Rrefbody=cr1body*d+cr0body    
    else:
        vref_body=vsatbody
        Rrefbody=Rsatbody
            
            
    if d >=dlowlimb and d <=dhighlimb:
        # v = c1*d + c0
        vref_limb = cv1limb*d + cv0limb
        Rreflimb=cr1limb*d+cr0limb    
    else:
        vref_limb=vsatlimb
        Rreflimb=Rsatlimb
                
 
    if gradient==False:          
      freqsbody = (np.ones((24,1))*vref_body)[:,0]
      freqslimb = (np.ones((4,1))*vref_limb)[:,0]
      
      frequencies=np.concatenate((freqsbody, freqslimb), axis=None)
      nominal_amplitude = np.ones(20) * Rrefbody
      amplitudes=np.concatenate((nominal_amplitude, np.ones(4)*Rreflimb), axis=None)    
      
  return(frequencies, amplitudes)