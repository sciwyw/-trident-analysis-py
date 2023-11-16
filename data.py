import os
import sys
sys.path.append("/lustre/neutrino/changqichao/trident-analysis-py/")
sys.path.append(os.getcwd())
sys.path.remove('/lustre/neutrino/hufan/trident-analysis-py')
import numpy as np
import pandas as pd
import uproot
import json

from trident import Hits, Track
from trident.reconstruct import Estimator, ResidualTimeSPE, LineFit
Hits.geotype = "penrose"
hor = 1.0
ver = 1.0

#file = '/lustre/neutrino/changqichao/data/100/9_1/mc_events_0.json'
file = '/lustre/neutrino/wangyingwei/data/shower/10_100TeV/batch0/data/mc_events.json'
with open(file,'r',encoding='utf8')as fp:
    json_data = json.load(fp)

npe = []
cos = []
ang = []
ang_po = []
trigger = []
energy = []
for k in range(1):
    #for i in np.arange(0,100,1):
        #file = '/lustre/neutrino/changqichao/data/100/9_1/job_'+str(i)+'/data/data.root'
        file = '/lustre/neutrino/wangyingwei/data/shower/10_100TeV/batch0/data/data.root'
        f1 = uproot.open(file)
        for j in range(1000):
            entry = j
            # hits = Hits.read_data(file, "DomHit", entry)
            hits, hits_merge, npe_ = Hits.read_hdom_hits(file, entry, hor, ver)
            npe.append(npe_)
            df1 = f1['Primary'].arrays(library='pd',entry_start=entry,entry_stop=entry+1)
            #if len(df1) != 0 and len(np.where(df1['PdgId']==13)[0]) !=0:
            if len(df1) != 0 and len(np.where(df1['PdgId']==11)[0]) !=0:
                px = json_data[j]["particles_in"][0]["px"]
                py = json_data[j]["particles_in"][0]["py"]
                pz = json_data[j]["particles_in"][0]["pz"]
                energy_ = np.sqrt(px**2 + py**2 + pz**2)
                track_true = Track(np.array([0,0,0,px/energy_,py/energy_,pz/energy_]))
                energy.append(energy_)
                cos.append(track_true.paras[5])
                ind = hits.data.index
                if npe_ >= 10 and np.max(ind)//20 != np.min(ind)//20:
                    track_guess = LineFit(hits_merge).guess()
                    re = Estimator(hits_merge)
                    track_estm = re.reconstruct(track_guess)
                    rec = ResidualTimeSPE(hits_merge,"sea_water_new")
                    track_rec = rec.reconstruct(track_estm)
                    angle_err = track_true.angle_error(track_rec)
                    ang.append(angle_err)

                    track_estm = re.reconstruct(track_guess,'Powell')
                    track_rec = rec.reconstruct(track_estm,'Powell')
                    angle_err = track_true.angle_error(track_rec)
                    ang_po.append(angle_err)

                    trigger.append(True)
                else:
                    ang.append(float('nan'))
                    ang_po.append(float('nan'))
                    trigger.append(False)
            else:
                energy.append(float('nan'))
                cos.append(float('nan'))
                ang.append(float('nan'))
                ang_po.append(float('nan'))
                trigger.append(False)
            #print(i,j)
            print(j)

data = []
for i in range(len(npe)):
    list = [npe[i],cos[i],ang[i],ang_po[i], trigger[i],energy[i]]
    data.append(list)

name=['npe','cos','angle','angle_po','trigger','energy']
test=pd.DataFrame(columns=name,data=data)
test.to_csv('./data1.csv')