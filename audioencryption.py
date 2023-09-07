from IPython.core.async_helpers import inspect
import numpy as np
import wave
import struct
import secrets
import soundfile as sf
import math
import numpy as np
import random

def read_wav_file(filename):
    with wave.open(filename, 'rb') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        data = f.readframes(nframes)
    # Convert binary data to integers
    dtype = '<i' + str(sampwidth)
    audio = np.frombuffer(data, dtype=dtype)
    print(audio)
    return audio
    
def initial_seq(I3,SineCosineSequence,SequenceSize):
    arr=np.zeros(SequenceSize,dtype=np.bool)
    outputarr=np.zeros(SequenceSize,dtype='uint8')
    inc=1
    flag=0
    
    
    
    I3=float(I3)
    #print("SineCosineSequence[1]-->",SineCosineSequence[1])
    #value_data=int(I3*inc*SineCosineSequence[inc])
    #value_data=value_data/15
    #value_data=int(value_data)%13
    #print("value_data-->",value_data)
    SequenceSize1=SequenceSize+1
    #if SequenceSize==16 or SequenceSize==8 or SequenceSize==4 or SequenceSize==2:
        #SequenceSize1=SequenceSize+1
    

    while flag<SequenceSize:
        inc1=float(inc)
       
        value_data=int(I3*inc1*SineCosineSequence[inc])%SequenceSize1
        inc+=1
        #print("value_data-->",value_data)
        if value_data>(SequenceSize-1):
            continue
        #print("flag-->",flag)
        if arr[value_data]==False:
            outputarr[flag]=value_data
            flag+=1
            arr[value_data]=True
        

    return outputarr










def binary_to_dna(binary_arr):
    
    output = np.empty(len(binary_arr), dtype='U8')

    for i in range(len(binary_arr)):
        temp=binary_arr[i]
        #print("temp->",temp)
        #print("temp[0]->",temp[0])
        dna_seq = ''
        for j in range(0, 16, 2):
            
            if temp[j] == 0 and temp[j+1]==0:
                dna_seq += 'A'
                
            elif temp[j] == 0 and temp[j+1]==1:
                dna_seq += 'C'
                
            elif temp[j] == 1 and temp[j+1]==0:
                dna_seq += 'G'
                
            elif temp[j] == 1 and temp[j+1]==1:
                dna_seq += 'T'
                
            #output=np.concatenate((output,dna_seq))
        #print(dna_seq)
        output[i]=dna_seq
    
    return output

def dna_to_binary(binary_arr):
    
    output = np.empty(len(binary_arr), dtype='U16')

    for i in range(len(binary_arr)):
        temp=binary_arr[i]
        #print("temp->",temp)
        #print("temp[0]->",temp[0])
        dna_seq = ''
        for j in range(8):
            
            if temp[j] == 'A':
                dna_seq += '00'
                
            elif temp[j] == 'C':
                dna_seq += '01'
                
            elif temp[j] == 'G':
                dna_seq += '10'
                
            elif temp[j] == 'T':
                dna_seq += '11'
                
            #output=np.concatenate((output,dna_seq))
        #print(dna_seq)
        output[i]=dna_seq
    print("Output dna to binary-->",output)
    return output

def dna_addition(dna1,dna2):
    
    output = np.empty(len(dna1), dtype='U8')

    for i in range(len(dna1)):
        temp=dna1[i]
        temp2=dna2[i]
        #print("temp->",temp)
        #print("temp[0]->",temp[0])
        dna_seq = ''
        for j in range(8):
            
            if temp[j] == 'A' and temp2[j]=='A':
                dna_seq += 'A'
            elif temp[j] == 'A' and temp2[j]=='C':
                dna_seq += 'C'
            elif temp[j] == 'A' and temp2[j]=='G':
                dna_seq += 'G'    
            elif temp[j] == 'A' and temp2[j]=='T':
                dna_seq += 'T'

            elif temp[j] == 'C' and temp2[j]=='A':
                dna_seq += 'C'
            elif temp[j] == 'C' and temp2[j]=='C':
                dna_seq += 'G'
            elif temp[j] == 'C' and temp2[j]=='G':
                dna_seq += 'T'
            elif temp[j] == 'C' and temp2[j]=='T':
                dna_seq += 'A'

            elif temp[j] == 'G' and temp2[j]=='A':
                dna_seq += 'G'
            elif temp[j] == 'G' and temp2[j]=='C':
                dna_seq += 'T'
            elif temp[j] == 'G' and temp2[j]=='G':
                dna_seq += 'A'
            elif temp[j] == 'G' and temp2[j]=='T':
                dna_seq += 'C'

            elif temp[j] == 'T' and temp2[j]=='A':
                dna_seq += 'T'
            elif temp[j] == 'T' and temp2[j]=='C':
                dna_seq += 'A'
            elif temp[j] == 'T' and temp2[j]=='G':
                dna_seq += 'C'
            elif temp[j] == 'T' and temp2[j]=='T':
                dna_seq += 'G'
                
            #output=np.concatenate((output,dna_seq))
        #print(dna_seq)
        output[i]=dna_seq
    
    return output



def apply_dna(key, binary_audio, binary_val1, binary_val2):
    print("key-->",key)
    key=int(key)
    rule = key % 8
    
    # Convert binary sequences to DNA sequences
    dnaseq_audio = binary_to_dna( binary_audio)
    dnaseq_val1 = binary_to_dna( binary_val1)
    dnaseq_val2 = binary_to_dna(binary_val2)
    #print("dnaseq_audio-->",dnaseq_audio)
    #print("dnaseq_val1-->",dnaseq_val1)
    #print("dnaseq_val2-->",dnaseq_val2)
    # Perform DNA addition
    add_value1 = dna_addition(dnaseq_audio, dnaseq_val1)
    result = dna_addition(add_value1, dnaseq_val2)
    print("add_value1-->",add_value1)
    print("result-->",result)
    # Convert result to binary sequence
    dnaseq_output = dna_to_binary(result)
    
    return dnaseq_output


def Encryption(Audio, ShareKey, InitialPar):
    # Initialization phase
    Length = len(Audio)
    CipherVoice = np.zeros(Length, dtype='int16')
    InitialVector_hex = str(InitialPar)
    I0 = InitialVector_hex[0:4]
    I1 = InitialVector_hex[4:8]
    I2 = InitialVector_hex[8:12]
    I3 = (InitialVector_hex[12:])
    #print(int(I3)%16)
    val = int(I0,16) % 32
    ShareKey_hex = str(ShareKey)
    print("val-->",val)
    print("ShareKey_hex-->",ShareKey_hex)
    IntermediateKey1 = ShareKey_hex[val:val+32]
    print("IntermediateKey1-->",IntermediateKey1)
    
    def hw(num):
    
    #Calculates the hamming weight (number of non-zero bits) of a binary number.
                   
        count = 0
        
        while num:
            count += num & 1
            num >>= 1
        return count
    d = int(2 * (math.floor(hw(InitialPar)) /2)) + 1
    def circular_shift(arr, shift):
        n = len(arr)
        shift = shift % n  # Ensure shift is within range
        return arr[-shift:] + arr[:-shift]

    TmpKey = circular_shift(ShareKey_hex, d)
    print("TmpKey-->",TmpKey)
    val1 = int(I1,16) % 32
    print("val1-->",val1)
    IntermediateKey2 = TmpKey[val1:val1+32]
    print("IntermediateKey2-->",IntermediateKey2)
    IP_SC = 1 / (hw(ShareKey) + 1)
    print("IP_SC-->",IP_SC)
    CP_SC = 2.2 + (int(I2,16) ^ int(I3,16)) % 5
    print("CP_SC-->",CP_SC)

    IP_LSC = 1 / (hw(InitialPar) + 1)
    print("IP_LSC-->",IP_LSC)

    def hw1(num):
    
    #Calculates the hamming weight (number of non-zero bits) of a binary number.
                   
        count = 0
        num=int(I3)
        while num:
            count += num & 1
            num >>= 1
        return count
    CP_LSC = 1 / (hw1(I3) + 1)
    print("CP_LSC-->",CP_LSC)
    # Chaotic Map Generation
    

    def SineCosineChaoticMap(ip, cp, length):
        
        x = np.zeros(length)
        x[0] = ip
        r=cp
        Y=[]
        for i in range(1,length):
            x[i]=abs(abs(np.sin(-r*x[i-1] + (x[i-1])**3-r*np.sin(x[i-1]))) - abs(np.cos(-r*x[i-1] + (x[i-1])**3-r*np.sin(x[i-1]))))
        for i in range(1,length):
            x[i]=abs(abs(np.sin(-r*x[i-1] + (x[i-1])**3-r*np.sin(x[i-1]))) - abs(np.cos(-r*x[i-1] + (x[i-1])**3-r*np.sin(x[i-1]))))
        #Y.append(x)
        return x

    SineCosineSequence = SineCosineChaoticMap(IP_SC, CP_SC, Length)
    print("SineCosineSequence",SineCosineSequence)
   # print(len(SineCosineSequence))
    

    def LogisticSineCosine(ip, cp, length):
        x = np.zeros(length)
        x[0] = ip
        r=cp
        Y=[]
        for i in range(1,length):
            x[i]=(np.cos(math.pi*(4*r*x[i-1]*(1-x[i-1])+(1-r)*np.sin(math.pi*x[i-1])-0.5)))
        for i in range(1,length):
            x[i]=(np.cos(math.pi*(4*r*x[i-1]*(1-x[i-1])+(1-r)*np.sin(math.pi*x[i-1])-0.5)))
       
        return x

    LogisticSineCosineSequence = LogisticSineCosine(IP_LSC, CP_LSC, Length)
    print("LogisticSineCosineSequence-->",LogisticSineCosineSequence)
  
  
  # Permutation phase
    def Permutation(Audio):
        rem = len(Audio) % 16
        div = int(len(Audio) / 16)
        
        print("rem-->",rem)
        print("div-->",div)
        InitialSequence1 = initial_seq(I3,SineCosineSequence,16)
        
        print("InitialSequence1-->",InitialSequence1)
        
        #InitialSequence1 = [12,13,14,15,0,11,10,9,8,7,6,5,4,3,2,1]
        InitialSequence2 = initial_seq(I3,SineCosineSequence,rem)
        #InitialSequence2 = np.random.permutation(rem)
        print("InitialSequence2-->",InitialSequence2)
        audio_pos = 0
        output = np.array([], dtype=np.int16)
        PermutatedVal = np.zeros(16, dtype=np.int16)
        for i in range(div + 1):
            Il=((i+1) * int(I3) )% 16
            #print("Il-->",Il)
            
            Sequence = np.roll(InitialSequence1, Il)
            #Sequence=InitialSequence1
            #Sequence = InitialSequence1
            #print("Sequence-->",Sequence)
            Audio16Val = Audio[audio_pos:audio_pos+16]
            #print("Audio16Val-->",Audio16Val)
            #print("Sequence-->",Sequence)
            if len(Audio16Val) == 16:
                for j in range(16):
                    PermutatedVal[j] = Audio16Val[Sequence[j]]
                    
            else:
                break
            output = np.concatenate((output,PermutatedVal))
            #print("print(PermutatedVal)->",PermutatedVal)
            #print("output-->",output)
            audio_pos += 16
            
        Audio16Val = Audio[audio_pos:audio_pos+rem]
        PermutatedVal1 = np.zeros(rem, dtype=np.int16)
        for i in range(rem):
            PermutatedVal1[i] = Audio16Val[InitialSequence2[i]]
        output =np.concatenate((output,PermutatedVal1))
        print("Output-->",output)
        return output


    PermutatedAudio = Permutation(audio)
    
  # Diffusion phase
    print("IntermediateKey1-->",IntermediateKey1)
    #IntermediateKey1=2
    #IntermediateKey2=3
    #SineCosineSequence[:] = 10
    #LogisticSineCosineSequence[:] = 20
    print("IntermediateKey1-->",type(IntermediateKey1))
    IMK1=float(IntermediateKey1)
    CM1 = ((IMK1 * SineCosineSequence)/2**16)

    CM1 = list(map(int, CM1))
    print("CM1-->",CM1)
    IMK2=float(IntermediateKey2)
    CM2 = ((IMK2 * LogisticSineCosineSequence)/2**16 )
    CM2 = list(map(int, CM2))
    print("CM2-->",CM2)
    #PermutatedAudio = list(map(int, PermutatedAudio))
    PermutatedAudio = PermutatedAudio.astype(int)
    #print("IntermediateKey1-->",IntermediateKey1)
    #print("IntermediateKey2-->",IntermediateKey2)
    #print("CM1-->",CM1)
    #print("CM2-->",CM2)
    #print("SineCosineSequence->",SineCosineSequence)
    #print("LogisticSineCosineSequence->",LogisticSineCosineSequence)
    #print("PermutatedAudio-->",PermutatedAudio)
    
    #binary_val1 = np.binary_repr(CM1, width=16)
    binary_val1 = [[int(bit) for bit in bin(num)[2:].zfill(16)] for num in CM1]
    #binary_val2 = np.binary_repr(CM2, width=16)
    binary_val2 = [[int(bit) for bit in bin(num)[2:].zfill(16)] for num in CM2]
    #binary_audio = [[int(bit) for bit in bin(num)[2:].zfill(16)] for num in PermutatedAudio]
    binary_audio = np.array([list(np.binary_repr(num, width=16)) for num in PermutatedAudio]).astype(int)

    #print("binary_val1->",binary_val1[1])
    #print("binary_val2->",binary_val2[1])
    #print("binary_audio->",binary_audio[1])
    DNA_data = apply_dna(IntermediateKey2, binary_audio, binary_val1, binary_val2)
    print("DNA_data",DNA_data)
    ## Dynamic sequence Generation
    print(type(I1))
    print(type(SineCosineSequence))
    I1=int(I1)
    I01 = (I1 * SineCosineSequence)
    print("I01-->",I01)
    print(type(I2))
    I2=int(I2)
    I02 = (I2 * LogisticSineCosineSequence)
    DNAVoiceData = np.array(DNA_data, dtype=np.int16)
    print("I02-->",I02)
    print("DNAVoiceData-->",DNAVoiceData)
    for i in range(Length):
      I0=int(I0)
      val = (I0 * (i+1)) % 2
      if val == 0:
           CipherVoice[i] = (DNAVoiceData[i] + (I01[i] * (i+1))) % 2**16
      else:
          CipherVoice[i] = (DNAVoiceData[i] + (I02[i] * (i+1))) % 2**16

  ## Save the Cipher .wav file
    write_wav_file('Output_encrypted.wav', CipherVoice, framerate=44100)
    #sf.write("CipherVoice.wav", CipherVoice, SampleRate=44100)

  
# Load audio file
#audio = sf.read('in.wav')

# Initialize the ShareKey and InitialVector
ShareKey = secrets.randbits(128)

InitialPar = secrets.randbits(128)


def write_wav_file(filename, audio, framerate):
    # Convert integers to binary data
    dtype = '<i2' # 16-bit signed integer
    data = audio.astype(dtype).tobytes()
    # Write to .wav file
    with wave.open(filename, 'wb') as f:
        nchannels = 1 # mono
        sampwidth = 2 # 16-bit
        f.setparams((nchannels, sampwidth, framerate, len(audio), 'NONE', 'not compressed'))
        f.writeframes(data)
# Read audio file
audio = read_wav_file('in.wav')
# Encrypt audio
encrypted_audio = Encryption(audio, ShareKey, InitialPar)

audio_copy = audio.astype(np.float)

# Manipulate the copied array
audio_copy *= 0.5

# Convert the copied array back to integer type
audio_copy = audio_copy.astype(np.int16)

# Write modified audio to file
#write_wav_file('modified.wav', audio, framerate=44100)
