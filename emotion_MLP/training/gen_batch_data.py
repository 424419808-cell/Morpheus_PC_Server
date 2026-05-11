import numpy as np
import os
import sys

def get_base_bs(emotion):
    bs = np.zeros(52)
    if emotion == "Neutral":
        pass
    elif emotion == "Happy":
        bs[43]=1.0; bs[44]=1.0; bs[18]=0.6; bs[19]=0.6; bs[6]=0.5; bs[7]=0.5; bs[24]=0.1
    elif emotion == "Excitement":
        bs[43]=1.0; bs[44]=1.0; bs[20]=0.7; bs[21]=0.7; bs[2]=0.8; bs[24]=0.3
    elif emotion == "Humor":
        bs[44]=1.0; bs[28]=0.8; bs[19]=0.5; bs[38]=0.6; bs[4]=0.5
    elif emotion == "Pride":
        bs[43]=0.3; bs[44]=0.3; bs[0]=0.4; bs[1]=0.4; bs[10]=0.7; bs[11]=0.7; bs[47]=0.8
    elif emotion == "Trust":
        bs[43]=0.4; bs[44]=0.4; bs[18]=0.2; bs[19]=0.2; bs[2]=0.3
    elif emotion == "Love":
        bs[43]=0.7; bs[44]=0.7; bs[8]=0.3; bs[9]=0.3; bs[18]=0.8; bs[19]=0.8; bs[6]=0.6; bs[7]=0.6
    elif emotion == "Relief":
        bs[8]=0.9; bs[9]=0.9; bs[31]=0.4; bs[24]=0.1; bs[43]=0.2; bs[44]=0.2
    elif emotion == "Hope":
        bs[20]=0.6; bs[21]=0.6; bs[2]=0.9; bs[3]=0.7; bs[4]=0.7; bs[16]=0.8; bs[17]=0.8
    elif emotion == "Anger":
        bs[0]=1.0; bs[1]=1.0; bs[49]=0.8; bs[50]=0.8; bs[35]=0.8; bs[36]=0.8; bs[22]=0.5
    elif emotion == "Disgust":
        bs[49]=1.0; bs[50]=1.0; bs[29]=0.9; bs[30]=0.9; bs[33]=0.8; bs[41]=0.7
    elif emotion == "Fear":
        bs[20]=1.0; bs[21]=1.0; bs[24]=0.6; bs[2]=0.9; bs[0]=0.2; bs[1]=0.2; bs[31]=0.5
    elif emotion == "Vigilance":
        bs[18]=0.7; bs[19]=0.7; bs[0]=0.6; bs[1]=0.6; bs[32]=0.3; bs[12]=0.6
    elif emotion == "Sad":
        bs[0]=0.8; bs[1]=0.8; bs[2]=0.9; bs[29]=1.0; bs[30]=1.0; bs[39]=0.8; bs[40]=0.8
    elif emotion == "Loneliness":
        bs[29]=0.6; bs[30]=0.6; bs[10]=0.8; bs[11]=0.8; bs[8]=0.2; bs[9]=0.2
    elif emotion == "Guilt":
        bs[0]=0.9; bs[1]=0.9; bs[10]=1.0; bs[11]=1.0; bs[8]=0.4; bs[9]=0.4
    elif emotion == "Surprise":
        bs[2]=1.0; bs[3]=1.0; bs[4]=1.0; bs[24]=0.9; bs[20]=0.9; bs[21]=0.9; bs[31]=0.3
    elif emotion == "Confusion":
        bs[2]=1.0; bs[0]=0.8; bs[31]=0.6; bs[37]=0.5; bs[23]=0.4
    elif emotion == "Shyness":
        bs[26]=0.4; # mouthPressL (从0.7调低，避免嘴唇过度挤压)
        bs[27]=0.4; # mouthPressR
        bs[35]=0.4; # mouthStretchL (轻微向两侧拉动)
        bs[36]=0.4; # mouthStretchR
        bs[14]=0.5; # eyeLookDownL (眼神向下，不要低得太离谱)
        bs[15]=0.5; # eyeLookDownR
        bs[43]=0.2; # mouthSmileL (带一点点若有若无的微笑)
        bs[44]=0.2; # mouthSmileR
    return bs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen_batch_data.py [emotion_name]")
        sys.exit(1)
    
    emo_name = sys.argv[1]
    target_bs = get_base_bs(emo_name)
    frames = 60
    sequence = np.zeros((frames, 52))
    for i in range(frames):
        sequence[i] = target_bs * (i / (frames - 1))
    
    os.makedirs("./result/batch_data", exist_ok=True)
    np.save(f"./result/batch_data/{emo_name}.npy", sequence.astype(np.float32))
    print(f"Saved: ./result/batch_data/{emo_name}.npy")
