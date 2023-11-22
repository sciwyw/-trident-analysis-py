import os
import sys
sys.path.append("/lustre/neutrino/changqichao/trident-analysis-py/")
sys.path.append(os.getcwd())
sys.path.remove('/lustre/neutrino/hufan/trident-analysis-py')
import numpy as np
import pandas as pd
import uproot
import json
import warnings

from trident import Hits, Track
from trident.reconstruct import Estimator, ResidualTimeSPE, LineFit
Hits.geotype = "penrose"
hor = 1.0
ver = 1.0

warnings.simplefilter(action='ignore', category=FutureWarning)
e_edge = ['1_10','10_100']
for k in range(len(e_edge)):
    for i in range(3,6,1):
        npe = []
        cos = []
        ang = []
        ang_po = []
        trigger = []
        energy = []
        json_file = f'/lustre/neutrino/wangyingwei/data/shower/{e_edge[k]}TeV/batch{i}/data/mc_events.json'
        print(json_file)
        with open(json_file,'r',encoding='utf8')as fp:
            json_data = json.load(fp)
        file = f'/lustre/neutrino/wangyingwei/data/shower/{e_edge[k]}TeV/batch{i}/data/data.root'
        f1 = uproot.open(file)
        for j in range(1000):
            entry = j
            hits, hits_merge, npe_ = Hits.read_hdom_hits(file, entry, hor, ver)
            npe.append(npe_)
            df1 = f1['Primary'].arrays(library='pd',entry_start=entry,entry_stop=entry+1)
            #if len(df1) != 0 and len(np.where(df1['PdgId']==13)[0]) !=0:
            if len(df1) != 0 and len(np.where(df1['PdgId']==11)[0]) !=0:
                px = json_data[j]["particles_in"][0]["px"]
                py = json_data[j]["particles_in"][0]["py"]
                pz = json_data[j]["particles_in"][0]["pz"]
                energy_ = np.sqrt(px**2 + py**2 + pz**2)
                #track_true = Track(np.array([0,0,0,px/energy_,py/energy_,pz/energy_]))
                energy.append(energy_)
                #cos.append(track_true.paras[5])
                cos.append(1.0)
                ind = hits.data.index
                if npe_ >= 15 and np.max(ind)//20 != np.min(ind)//20:
                    # track_guess = LineFit(hits_merge).guess()
                    # re = Estimator(hits_merge)
                    # track_estm = re.reconstruct(track_guess)
                    # rec = ResidualTimeSPE(hits_merge,"sea_water_new")
                    # track_rec = rec.reconstruct(track_estm)
                    # angle_err = track_true.angle_error(track_rec)
                    angle_err = 0.0
                    ang.append(angle_err)

                    # track_estm = re.reconstruct(track_guess,'Powell')
                    # track_rec = rec.reconstruct(track_estm,'Powell')
                    # angle_err = track_true.angle_error(track_rec)
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
            #print(j)

        data = []
        for n in range(len(npe)):
            list = [npe[n],cos[n],ang[n],ang_po[n], trigger[n],energy[n]]
            data.append(list)

        name=['npe','cos','angle','angle_po','trigger','energy']
        test=pd.DataFrame(columns=name,data=data)
        test.to_csv(f'./data{e_edge[k]}_batch{i}.csv')
        print(i,',',j)
