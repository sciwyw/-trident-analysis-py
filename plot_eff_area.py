import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from matplotlib import rcParams
color = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot():
    font = {'family': 'serif',
        'weight': 'normal', 'size': 12}
    plt.rc('font', **font)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    return None
    
    # eff area
vmin = 1e-1
vmax = 5e2
volume_pre = 1570796375900160.0 #half-height=250 radius=1000
volume_true = 1.9634954094772224e+16 #half-height=500 radius=2500
cos_zenith_edge = np.array([-1, 1])
cos_zenith_edge_2 = np.array([-0.5, 0])
cos_zenith_edge_3 = np.array([0, 0.5])
cos_zenith_bins = np.linspace(-1, 1, 11)
cos_zenith_bins_2 = np.linspace(-0.5, 0, 11)
cos_zenith_bins_3 = np.linspace(0, 0.5, 11)
log_energy_bins = np.linspace(3, 5, 11)
filename = 'eff_area'

def energy_zenith_weight():
    energy = []
    zenith = []
    weight_total = []
    weight_select = []
    num_data = 6
    e_edge = ['1_10','10_100']
    #e_edge = ['1_10']
    for k in range(len(e_edge)):
        #print(k)
        for i in range(0,num_data,1):
            #print(i)
            json_file = f'/lustre/neutrino/wangyingwei/data/shower/{e_edge[k]}TeV/batch{i}/data/mc_events.json'
            csv_file = f'/lustre/neutrino/wangyingwei/project/read_cascade/data{e_edge[k]}_batch{i}.csv'
            data = pd.read_csv(csv_file)
            angle = data['angle'].to_list()
            npe = data['npe'].to_list()
            trigger = data['trigger'].to_list()
            with open(json_file)as fp:
                json_data = json.load(fp)
            # particles = pd.json_normalize(json_data , record_path='particles_in', meta=['event_id']).set_index('event_id')
            # #print(particles)
            # #particles['energy'] = np.linage.norm(particles[['px','py','pz']], axis=1)
            # particles['energy'] = np.sqrt(particles[0].px**2 + particles[0].py**2 + particles[0].pz**2)
            # particles['costh'] = particles.pz / particles.energy
            # energy.append(particles.energy)
            # zenith.append(particles.costh)
            # particle_weights = pd.json_normalize(json_data , record_path='weights', meta=['event_id']).set_index('event_id')
            # weight_total.append(particle_weights.total)
            for j in range(len(json_data)):
                px = json_data[j]["particles_in"][0]["px"]
                py = json_data[j]["particles_in"][0]["py"]
                pz = json_data[j]["particles_in"][0]["pz"]
                energy_particle = np.sqrt(px**2 + py**2 + pz**2)
                zenith_particle = pz / energy_particle
                energy.append(energy_particle)
                zenith.append(zenith_particle)
                total = json_data[j]["weights"]["total"]/volume_pre * volume_true
                weight_total.append(total)
                weight_select.append(0)
                # if angle[i] >= 0:
                # weight_select[i] = 1
                if trigger[j] == True:
                    weight_select[j+((k*num_data+i)*len(json_data))] = 1
                    #weight_select[j+round(time)*len(json_data)] = 1
                    #print(j+((k*num_data+i)*len(json_data)))
    energy = np.array(energy)
    zenith = np.array(zenith)
    weight_total = np.array(weight_total)
    weight = weight_total * weight_select
    return energy, zenith, weight

def get_eff_area(cos_zenith_bins, log_energy_bins):
    energy, cos_zenith, weight = energy_zenith_weight()

    num_tot_weighted = np.histogram2d(
        x=cos_zenith, y=np.log10(energy), weights=weight,
        bins=[cos_zenith_bins, log_energy_bins])[0]
    bins_size = 2 * np.pi * \
        np.outer(-np.diff(cos_zenith_bins), 
                   -np.diff(np.power(10, log_energy_bins)))
    effarea = num_tot_weighted / bins_size / len(weight)
    log_energy = log_energy_bins
    return effarea, log_energy

def plot_eff_hist2d(vmin, vmax, cos_zenith_bins, log_energy_bins, filename: str = None):
    effarea, log_energy = get_eff_area(cos_zenith_bins, log_energy_bins)
    
    plot()
    fig = plt.figure(figsize=(8, 7.5), dpi=300)
    ax = fig.add_axes([0.15, 0.15, 0.65, 0.65])
    cax = fig.add_axes([0.83, 0.15, 0.03, 0.65])
    cmap_ = plt.get_cmap('plasma')
    norm = mpl.colors.LogNorm(vmin, vmax)
    im = ax.imshow(effarea / 1e4, cmap=cmap_, norm=norm, aspect="auto",
                    origin="lower", extent=[log_energy[0], log_energy[-1], -1, 1])
    ax.set_xlabel(r"log$_{10}$(E$_{\nu}$/GeV)")
    ax.set_ylabel(r"$\mathregular{cos(\theta_z)}$")
    ax.set_title(r"Effective Area [m$^2$]")
    color_bar = fig.colorbar(im, cax=cax)
    color_bar.orientation = 'vertical'
    # color_bar.set_label(r"Effective area [m$^2$]")
    if filename != None:
        fig.savefig(filename + '.jpg', bbox_inches='tight')
    plt.show()
    return

def get_eff_area_band(cos_zenith_edge, cos_zenith_bins, log_energy_bins):
    effarea, log_energy = get_eff_area(cos_zenith_bins, log_energy_bins)
    cos_zenith = cos_zenith_bins
    log_energy_center = (
        log_energy[1:] + log_energy[:-1]) / 2
    
    n_band = len(cos_zenith_edge) - 1
    eff_area_band_avg = np.zeros(shape=(n_band, len(log_energy)-1))
    idx_zenith_self = 1
    for idx_band in range(n_band):
        cos_zenith_up = cos_zenith_edge[idx_band+1]
        num_zenith_self = 0
        eff_area_in_band = np.zeros(len(log_energy)-1)
        while (idx_zenith_self < len(cos_zenith)) and (cos_zenith[idx_zenith_self] <= cos_zenith_up):
            eff_area_in_band += effarea[idx_zenith_self-1]
            num_zenith_self += 1
            idx_zenith_self += 1
        eff_area_band_avg[idx_band] = eff_area_in_band / num_zenith_self
    return log_energy, eff_area_band_avg, log_energy_center

def plot_eff_hist(vmin, vmax, cos_zenith_edge, cos_zenith_bins, log_energy_bins, filename: str = None):
    
    log_energy, eff_area_band_avg, log_energy_center = get_eff_area_band(cos_zenith_edge, cos_zenith_bins, log_energy_bins)
    log_energy_2, eff_area_band_avg_2, log_energy_center_2 = get_eff_area_band(cos_zenith_edge_2, cos_zenith_bins_2, log_energy_bins)
    log_energy_3, eff_area_band_avg_3, log_energy_center_3 = get_eff_area_band(cos_zenith_edge_3, cos_zenith_bins_3, log_energy_bins)
    style="step"
    eff_area_band = eff_area_band_avg
    eff_area_band_2 = eff_area_band_avg_2
    eff_area_band_3 = eff_area_band_avg_3
    #print(log_energy)
    plot()
    plt.subplots(figsize=(10,6.5), dpi=300)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i in range(len(cos_zenith_edge)-1):
        print(i)
        if style == "step":
            plt.step(log_energy[:-1], eff_area_band[i]/1e4,
                        where="post", c=colors[0],
                        label=f"{cos_zenith_edge[i]:.1f}" + r"$\leq \mathregular{cos(\theta_z)} <$"+f"{cos_zenith_edge[i+1]:.1f}")
            plt.step(log_energy[1:], eff_area_band[i]/1e4,
                        where="pre", c=colors[0])
            plt.step(log_energy_2[:-1], eff_area_band_2[i]/1e4,
                        where="post", c=colors[1],
                        label=f"{cos_zenith_edge_2[i]:.1f}" + r"$\leq \mathregular{cos(\theta_z)} <$"+f"{cos_zenith_edge_2[i+1]:.1f}")
            plt.step(log_energy_2[1:], eff_area_band_2[i]/1e4,
                        where="pre", c=colors[1])
            plt.step(log_energy_3[:-1], eff_area_band_3[i]/1e4,
                        where="post", c=colors[2],
                        label=f"{cos_zenith_edge_3[i]:.1f}" + r"$\leq \mathregular{cos(\theta_z)} <$"+f"{cos_zenith_edge_3[i+1]:.1f}")
            plt.step(log_energy_3[1:], eff_area_band_3[i]/1e4,
                        where="pre", c=colors[2])
        elif style == "line":
            plt.plot(log_energy_center, eff_area_band[i]/1e4, c=colors[0],
                        label=f"{cos_zenith_edge[i]:.1f}" + r"$\leq \cos(\theta_z)<$" f"{cos_zenith_edge[i+1]:.1f}")
            plt.plot(log_energy_center_2, eff_area_band_2[i]/1e4, c=colors[1],
                        label=f"{cos_zenith_edge_2[i]:.1f}" + r"$\leq \cos(\theta_z)<$" f"{cos_zenith_edge_2[i+1]:.1f}")
            plt.plot(log_energy_center_3, eff_area_band_3[i]/1e4, c=colors[2],
                        label=f"{cos_zenith_edge_3[i]:.1f}" + r"$\leq \cos(\theta_z)<$" f"{cos_zenith_edge_3[i+1]:.1f}")
    plt.xlim(log_energy[0], log_energy[-1])
    plt.xlabel(r"log$_{10}$(E$_{\nu}$/GeV)")
    plt.yscale("log")
    plt.ylim(vmin, vmax)
    plt.ylabel(r"Effective area [m$^2$]")
    plt.legend()
    plt.title('A_eff_electron')
    # plt.hold(True) 
    if filename != None:
        plt.savefig(filename + '.jpg', bbox_inches='tight')
    plt.show()
    return

plot_eff_hist2d(vmin, vmax, cos_zenith_bins, log_energy_bins, filename)
plot_eff_hist(vmin, vmax, cos_zenith_edge, cos_zenith_bins, log_energy_bins, filename)