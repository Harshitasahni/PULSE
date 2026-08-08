import os
import pickle
import random as _random
import re
import time
import warnings
import glob
import copy
import threading
import pickle
import os

import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance
import trajwrapper as TW
import EncodingGenerator as EG
import states as s

from openmm.app import PDBFile
from sklearn.decomposition import NMF
from openmm.unit import nanometer

from analysis_metadata import AnalysisMetadata, AnalysisResult
from global_state import global_state
from protein_io import *

warnings.filterwarnings("ignore")

# ---------------------------
# Change the name to a different protein, defined in protein_config.py
# ---------------------------
PROTEIN_NAME = "glob"



# ===================================================================
# SIMILARITY CHECKER CLASS
# ===================================================================
class SimpleSimilarity:
    """
    Simple similarity checker for hierarchical states
    Prevents creating redundant states

    """
    @staticmethod
    def check_if_similar(new_state, existing_states, threshold=0.7, verbose=False):
        if not existing_states:
            if verbose:
                print(f"[SIMILARITY] No existing states, creating first state")
            return False, None, 0.0

        parent_states = [s for s in existing_states if s.nmodels == new_state.nmodels - 1]

        if not parent_states:
            if verbose:
                print(f"[SIMILARITY] No parent states with {new_state.nmodels-1} models")
            return False, None, 0.0

        best_similarity = 0.0
        best_parent = None

        for parent_state in parent_states:
            similarity = SimpleSimilarity._calculate_similarity(parent_state, new_state)
            if verbose:
                print(f"[SIMILARITY] Comparing with State {parent_state.id}: {similarity:.3f}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_parent = parent_state

        is_similar = best_similarity > threshold

        if verbose:
            if is_similar:
                print(f"[SIMILARITY] TOO SIMILAR ({best_similarity:.3f}>{threshold}) - will reuse State {best_parent.id}")
            else:
                print(f"[SIMILARITY] Different enough ({best_similarity:.3f}<={threshold}) - creating new state")

        return is_similar, best_parent, best_similarity

    @staticmethod
    def _calculate_similarity(parent_state, child_state):

        shared_models = parent_state.nmodels
        similarities = []

        for model_idx in range(shared_models):
            parent_mean = parent_state.mean[model_idx]
            parent_std  = parent_state.stdev[model_idx]
            child_mean  = child_state.mean[model_idx]
            child_std   = child_state.stdev[model_idx]

            mean_diff       = abs(parent_mean - child_mean)
            max_mean        = max(abs(parent_mean), abs(child_mean), 0.001)
            mean_similarity = 1.0 / (1.0 + mean_diff / max_mean)

            std_diff       = abs(parent_std - child_std)
            max_std        = max(parent_std, child_std, 0.001)
            std_similarity = 1.0 / (1.0 + std_diff / max_std)

            model_similarity = 0.6 * mean_similarity + 0.3 * std_similarity
            similarities.append(model_similarity)

        return np.mean(similarities)

# ===================================================================
# VILLIN PDB SAVING FUNCTIONS
# ===================================================================

def villian(local_variables, fcurrent, traj_path, fmax, folder, traj_num, local_window):
    """
    Handles an uncertain window for villin:
      - Marks the window as uncertain in frames2states
      - Saves the max-error frame as a PDB using the villin amber topology
      - Saved under uncertain_descendants/
    """
    return save_uncertain_frame(
        PROTEIN_NAME,
        local_variables,
        fcurrent,
        traj_path,
        fmax,
        folder,
        traj_num,
        local_window,)
    


def villian_save_only(traj_path, fmax, folder, traj_num, local_window):
    """
    Save the fmax frame as a PDB WITHOUT modifying frames2states
    Used for forced-reuse frames (similar state exists but we want to save
    this frame for future exploration anyway).
    Saved under forced_reuse_descendants/ — separate folder
    from genuinely-uncertain frames for clean downstream analysis
    """
    return save_frame_only(
        PROTEIN_NAME,
        traj_path,
        fmax,
        folder,
        traj_num,
        local_window,)
    


def save_random_uncertain_frame(traj_path, folder, traj_num, local_window, fmax_frame,
                                 subfolder="uncertain_descendants"):
    """
    Saves a random frame (different from fmax_frame) under:
        <folder>/<subfolder>/random_frames/
    Uses the villin amber topology.
    """
    return save_random_uncertain_frameio(
        PROTEIN_NAME,
        traj_path,
        folder,
        traj_num,
        local_window,
        fmax_frame,)
    


# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

def ftest(s1, n1, s2, n2, alpha):
    alpha /= 2
    if s1 > s2:
        F_score        = s1 ** 2 / s2 ** 2
        critical_value = stats.f.ppf(1 - alpha, n1 - 1, n2 - 1)
    else:
        F_score        = s2 ** 2 / s1 ** 2
        critical_value = stats.f.ppf(1 - alpha, n2 - 1, n1 - 1)

    if F_score < critical_value:
        return True
    else:
        return False

#This function generates the windows for analysis 

def generate_data(generator, window, ret_shape=False):
    window_of_frames = []
    for k in range(window):
        x = next(generator)
        x = np.asarray(x)
        if ret_shape:
            x_shape = x.shape
        x = x.reshape((1, -1))
        window_of_frames.append(x[0])
    if ret_shape:
        return (window_of_frames, x_shape)
    else:
        return window_of_frames

def eval_reconstruction_loss(model, X):
    H   = model.components_
    W   = model.transform(X)
    out = np.dot(W, H)

    abs_loss          = out - X
    loss_square       = np.square(abs_loss)
    ssqloss_perframe  = np.sum(loss_square, axis=1)
    ssqloss_perres    = np.sum(loss_square, axis=0)
    absloss_perres    = np.sum(abs(abs_loss), axis=0)

    return ssqloss_perframe, ssqloss_perres, absloss_perres


def new_model_add(window_of_frames, frame, vplot, vsave):
    bestmodel = []
    top_nc    = 49
    min_nc    = 10
    nc        = int(min(min_nc, np.sqrt(window_of_frames[0].shape[0])))
    errs      = []
    x1        = []

    for components in range(nc, top_nc):
        model0 = NMF(n_components=components, init='nndsvd')
        model0.fit(window_of_frames)
        lossF, lossR, absR = eval_reconstruction_loss(model0, window_of_frames)
        err       = np.mean(lossF)
        bestmodel = model0
        numc      = components
        if vplot or vsave:
            errs.append(err)
            x1.append(components)
        if err < 0.004:
            break

    if (vplot or vsave) and (len(x1) > 1):
        fig = plt.figure(figsize=(4, 4))
        ax  = fig.add_subplot(1, 1, 1)
        ax.plot(x1, errs, color="blue")
        ax.set_xlabel("NMF Components", fontsize=12)
        ax.set_ylabel("Reconstruction Error", color="blue", fontsize=12)
        if vsave:
            plt.savefig(vsave + "_NMFC_" + str(frame) + ".pdf")
            plt.savefig(vsave + "_NMFC_" + str(frame) + ".png")
        if vplot:
            plt.show()

    return bestmodel

def prep_selection(alpha_CA, select_residues):
    if alpha_CA and select_residues == []:
        select_str = 'name CA'
    elif alpha_CA:
        cont = 0
        for r in select_residues:
            if cont == 0:
                select_str = "(name CA and resid {})".format(r)
                cont += 1
            else:
                select_str = select_str + " or (name CA and resid {})".format(r)
    elif select_residues != []:
        cont = 0
        for r in select_residues:
            if cont == 0:
                select_str = "(residue {}".format(r)
                cont += 1
            else:
                select_str = select_str + " or residue {}".format(r)
        select_str = select_str + ") and not element H"
    else:
        select_str = ""

    return select_str

def DIST(xyz, secondary_structure):
    num_res = len(xyz)
    dstd    = distance.squareform(distance.pdist(xyz, 'euclidean'))
    return dstd


def read_variables_local(filename, folder):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data
    else:
        return {
            "nmol":          -1,
            "frames2states": {},
            "data_plot":     {},
            "fcurrent":      0,
            "tica_files":    {},
            "NSTRIKES":      2,
            "change":        False,
            "models_c":      [],
            "just_added":    False,
            "mem_size_all":  [],
            "cpu_time_all":  [],
            "data_size_all": [],
            "strikes":       0,
            "maxf":          0,
            "old_State":     False,
            "last_state":    0,
            "my_states":     [],
            "active_states": [],
            "newTraj":       True,
            "explain":       [],
            "ci_x":          [],
            "CC":            dict()
        }

def save_variables_local(variables, filename):
    variables_to_save = {
        "frames2states": dict(variables["frames2states"]),
        "data_plot":     dict(variables["data_plot"]),
        "fcurrent":      variables["fcurrent"],
        "NSTRIKES":      variables["NSTRIKES"],
        "change":        variables["change"],
        "models_c":      list(variables["models_c"]),
        "just_added":    variables["just_added"],
        "strikes":       variables["strikes"],
        "maxf":          variables["maxf"],
        "old_State":     variables["old_State"],
        "last_state":    variables["last_state"],
        "my_states":     list(variables["my_states"]),
        "active_states": list(variables["active_states"]),
        "explain":       list(variables["explain"]),
        "ci_x":          list(variables["ci_x"]),
        "nmol":          variables['nmol'],
        "newTraj":       variables['newTraj'],
        "CC":            variables['CC']
    }

    with open(filename, 'wb') as file:
        pickle.dump(variables_to_save, file)
    file.close()

def extract_traj_window(path):
    pattern = r"traj_(\d+)_window_(\d+)"
    matches = re.findall(pattern, path)
    if matches:
        traj_num, window_num = matches[-1]
        return int(traj_num), int(window_num)
    else:
        return None, None

def eval_window_4all_old(window_of_frames, models, States, alpha, newTraj, trajnum, verbose=0):

    loss         = {}
    absE         = {}
    errR         = {}
    prob         = 0
    change       = True
    flag         = False
    last_err     = False
    explain_temp = [1000] * len(States)
    maxLastErr   = [0] * len(States)
    prob_temp    = []
    explain      = []
    flag         = True
    mode         = "obg"

    # key = States[-1].nmodels
    key = len(models)
    for nm in range(key):
        loss[nm], errR[nm], absE[nm] = eval_reconstruction_loss(models[nm], window_of_frames)
        if nm == 0:
            err_perR = np.zeros_like(absE[0])
            abs_perR = np.zeros_like(absE[0])
        if nm == 0 or max(errR[nm]) < max(err_perR_l):
            err_perR_l = errR[nm]
            abs_perR_l = absE[nm]

    fmax     = np.argmax(loss[key - 1])
    problist = [1000] * len(States)

    for i, state in enumerate(States):
        prob, avgloss  = state.evaluate_state(loss, mode, verbose)
        avgerr         = sum(absE[i]) / len(absE[i])
        maxLastErr[i]  = max(absE[i])
        if (not newTraj and avgloss < 1.0 and max(absE[i]) < 2.2) \
                or (not newTraj and max(absE[i]) < 2.5) \
                or (prob > 0.2):
            change          = False
            explain_temp[i] = sum(absE[i])
            problist[i]     = prob

    if sum(explain_temp) != 1000 * len(States):
        indices = sorted(range(len(explain_temp)), key=explain_temp.__getitem__)
        for j in indices:
            if explain_temp[j] < 1000 and j not in explain:
                explain.append(j)
                if flag:
                    err_perR = errR[j]
                    abs_perR = absE[j]
                    flag     = False

    return (loss, err_perR_l, abs_perR_l, fmax, maxLastErr, change, explain, explain_temp)

def update_counts(countsd,exp):
   countsd[exp] = countsd.get(exp, 0) + 1
   return countsd

# ===================================================================
# MAIN
# ===================================================================


def main(traj_path, top, residue_file, alpha_CA, folder, folder_traj, uncertainw=False):

# ===================================================================
# Change SIMILARITY_THRESHOLD to reduce the sensitivity or ENABLE_SIMILARITY_CHECK=False for frequent state generation
# alpha is the confidence interval for state creation check 
# ===================================================================
    SIMILARITY_THRESHOLD    = 0.5
    ENABLE_SIMILARITY_CHECK = True
    VERBOSE_SIMILARITY      = True

    window             = 50
    alpha              = 0.9
    verbose            = True
    vplot              = True
    vsave              = False
    profile            = False
    alpha_CA           = True
    globalPlot         = True
    newstate           = -1
    transform_function = DIST
    max_error          = 1

    traj_path          = os.path.join(folder, traj_path)
    traj_num, local_window = extract_traj_window(traj_path)
    local_fstart           = local_window * window
    local_file             = f"{folder}/variables_{traj_num}.pkl"

    local_variables         = read_variables_local(local_file, folder)
    local_variables['nmol'] += 1

    if uncertainw == True:
        local_variables["newTraj"] = False
        local_variables['nmol']    = traj_num
        local_variables['CC']      = {}

    if profile:
        with open(save_path + 'profile_withoutplot.txt', 'w') as profile_file:
            profile_file.write("Protein,Traj_Name,Model_Size,Data_Size,CPU_Time,Num_Windows\n")

    res_file        = residue_file
    select_residues = []
    with open(res_file, encoding='utf-8') as f:
        line      = f.read()
        residues  = line.split('[')[1].replace(']', '')
        residues  = residues.replace(' ', '')
        select_residues = residues.split(',')

    select_str = prep_selection(alpha_CA, select_residues)

    traj     = TW.TrajWrapper(traj_path, top)
    Encoding = EG.EncodingGenerator(traj, alpha_CA=alpha_CA, select=select_str, transform_function=transform_function)
    generator = iter(Encoding)



    window_of_frames = []
    fcurrent         = local_variables['fcurrent']

    if local_variables['newTraj'] == True:
        local_variables['nmol'] = traj_num
        local_variables['CC']   = {}

    if uncertainw == True:
        local_variables['newTraj'] = False
        local_variables['nmol']    = traj_num
        local_variables['CC']      = {}

    print(f"[DEBUG]----Fstart for this window is {fcurrent} and this is local to traj {traj_num}-------- {local_file}")

    while True:
        try:
            snapshot       = global_state.snapshot()
            localg_models  = snapshot['models']
            localg_states  = snapshot['states']
            data      = []
            new_losses = {}
            data, shape_data = generate_data(generator, window, True)

            startt = time.time()

            if int(fcurrent) == 0 and int(traj_num) == 0:
                print("[DEBUG]----Lets create our first model")
                local_variables['just_added'] = True
                local_variables['ci_x']       = [0]
                local_variables['newTraj']    = False

                model = new_model_add(data, fcurrent, False, False)
                new_losses[0], err_perR, abs_perR = eval_reconstruction_loss(model, data)

                st = s.State(0, 0, 1, new_losses, window, alpha)
                global_state.add_state(st, model)
                local_variables['my_states'].append(st.id)
                local_variables['models_c'] = [0]

                if vplot or vsave:
                    local_variables['frames2states'] = {0: [0]}
                    local_variables['explain']       = [0]

            else:
                new_losses, err_perR, abs_perR, fmax, maxLastErr, \
                    local_variables['change'], local_variables['explain'], good_fit = \
                    eval_window_4all_old(
                        data, localg_models, localg_states, alpha,
                        local_variables['newTraj'], traj_num, verbose)
                    

                myexplain = list(set(local_variables['my_states']) & set(local_variables['explain']))
                if myexplain:
                    local_variables['explain'] = myexplain.copy()
                if len(local_variables['explain']) >= 1:
                    local_variables['explain'] = [local_variables['explain'][0]]
                if local_variables['change'] or not local_variables['explain']:
                    local_variables['strikes'] += 1
                local_variables['ci_x'].append(fcurrent + window - 1)

                if (local_variables['strikes'] >= local_variables['NSTRIKES']) \
                        or (local_variables['newTraj'] and not len(local_variables['explain'])):

                    local_variables['NSTRIKES'] += 1
                    local_variables['strikes']   = 0
                    local_variables['just_added'] = True

                    nmodel    = len(localg_models)
                    new_model = new_model_add(data, fcurrent, False, False)
                    new_losses[nmodel], err_perR, abs_perR = eval_reconstruction_loss(new_model, data)
                    new_state = s.State(nmodel, fcurrent, nmodel + 1, new_losses, window, alpha)

                    if ENABLE_SIMILARITY_CHECK:
                        is_too_similar, similar_state, similarity_score = SimpleSimilarity.check_if_similar(
                            new_state, localg_states,
                            threshold=SIMILARITY_THRESHOLD,
                            verbose=VERBOSE_SIMILARITY
                        )

                        if is_too_similar:
                            print(f"[DECISION] Traj {traj_num}: Reusing existing State {similar_state.id} "
                                  f"(similarity: {similarity_score:.3f})")
                            local_variables['strikes'] = 0

                            def updater_fn(state):
                                state.add_window(new_losses, window)
                            global_state.update_state(similar_state.id, updater_fn)

                            if similar_state.id not in local_variables['my_states']:
                                local_variables['my_states'].append(similar_state.id)

                            local_variables['explain'] = [similar_state.id]

                            if 'similarity_stats' not in local_variables:
                                local_variables['similarity_stats'] = {'reused': 0, 'created': 0}
                            local_variables['similarity_stats']['reused'] += 1
                            if new_losses:
                                max_err_val = float(max(np.max(v) for v in new_losses.values()))
                            else:
                                max_err_val = 0.0
                            local_variables['frames2states'][fcurrent] = [similar_state.id]
                            local_variables['last_metadata'] = AnalysisMetadata.forced_reuse_explanation(
                                state_id=similar_state.id,
                                similarity=similarity_score,
                                max_err=max_err_val)                           

                        else:
                            global_state.add_state(new_state, new_model)
                            print(f"[DECISION] Traj {traj_num}: Creating new State {new_state.id} " f"(similarity: {similarity_score:.3f})")
                            local_variables['models_c'].append(fcurrent)
                            local_variables['my_states'].append(new_state.id)
                            local_variables['frames2states'][fcurrent] = [-2]
                            if 'similarity_stats' not in local_variables:
                                local_variables['similarity_stats'] = {'reused': 0, 'created': 0}
                            local_variables['similarity_stats']['created'] += 1

                            local_variables['last_metadata'] = AnalysisMetadata.new_state_explanation(new_state.id)
                            
                            if verbose:
                              print(f"[INFO]-------Change at {traj_num} at Window =, {fcurrent}")

                    else:
                        global_state.add_state(new_state, new_model)
                        print(f"[DECISION] Traj {traj_num}: Creating new State {new_state.id} "
                              f"(similarity check disabled)")
                        
                        local_variables['models_c'].append(fcurrent)
                        local_variables['my_states'].append(new_state.id)
                        local_variables['last_metadata'] = AnalysisMetadata.new_state_explanation(new_state.id)

                        if verbose:
                            print(f"[INFO]-------Change at {traj_num} at Window =, {fcurrent}")
                        if vplot or vsave:
                            local_variables['frames2states'][fcurrent] = [-2]
                else:
                    if local_variables['explain']:
                        for ex1 in local_variables['explain']:
                            if (local_variables['newTraj'] and maxLastErr[ex1] < max_error) \
                                    or not local_variables['newTraj']:
                                local_variables['strikes']  = 0
                                local_variables['newTraj']  = False

                                if maxLastErr[ex1] < max_error and ex1 in local_variables['my_states']:
                                    def updater_fn(state: s.State):
                                        state.add_window(new_losses, window)
                                    global_state.update_state(ex1, updater_fn)

                                if True:
                                    local_variables['CC'] = update_counts(local_variables['CC'], ex1)
                                    local_variables['frames2states'][fcurrent] = local_variables['explain']

                                local_variables['last_metadata'] = AnalysisMetadata.normal_explanation(
                                    state_id=ex1,
                                    max_err=maxLastErr[ex1],
                                    avg_err=maxLastErr[ex1]
                                )
                                local_variables['just_added'] = False
                                local_variables['old_State']  = False
                            else:
                                if local_variables['my_states']:
                                    def updater_fn(state: s.State):
                                        state.add_window(new_losses, window)
                                    ist = local_variables['my_states'][-1]
                                    global_state.update_state(ist, updater_fn)
                                if vplot or vsave:
                                    local_variables['frames2states'][fcurrent] = [-1]
                                    local_variables['last_metadata'] = AnalysisMetadata.uncertain(
                                        max_err=maxLastErr.get(ex1, 0.0)
                                    )
                    else:
                        output_fmax_pdb, uncertainty_name = villian(
                            local_variables, fcurrent, traj_path, fmax,
                            folder, traj_num, local_window)
                        local_variables['last_metadata'] = AnalysisMetadata.uncertain(max_err=fmax)
                        print(f"[INFO]-------[UNCERTAIN] Window {fcurrent}: No state can explain this window")
                        descendants_folder = os.path.join(folder, "uncertain_descendants")
                        os.makedirs(descendants_folder, exist_ok=True)
                        global_state.add_uncertain_frame(output_fmax_pdb, uncertainty_name)
                        print(f"[INFO]-------[UNCERTAIN] Max error frame saved: {uncertainty_name}")

                        save_random_uncertain_frame(
                            traj_path, folder, traj_num, local_window, fmax,
                            subfolder="uncertain_descendants"
                        )

            if profile:
                local_variables['endt'] = time.time()
                elapsed_time = (local_variables['endt'] - startt) * 1000
                local_variables['cpu_time_all'].append(elapsed_time)
                st_size   = total_size(localg_states)
                md_size   = total_size(localg_models[0]) * len(localg_models)
                data_size = (data[0].nbytes * window) / 1000
                mem_size  = (st_size + md_size) / 1000
                local_variables['mem_size_all'].append(mem_size)
                local_variables['data_size_all'].append(data_size)

            if vplot or vsave:
                local_variables['data_plot'][fcurrent] = new_losses
                for mm in new_losses.keys():
                    maxmm = max(new_losses[mm])
                    if maxmm > local_variables['maxf']:
                        local_variables['maxf'] = maxmm

            newstate = local_variables['frames2states'][fcurrent][0]

            # -----------------------------------------------------------------
            # FORCED REUSE: save uncertain frame for future exploration,
            #               in a SEPARATE folder from genuine uncertainty.
            #               Does NOT modify frames2states
            # -----------------------------------------------------------------
            # metadata = local_variables.get('last_metadata', AnalysisMetadata())

            # if metadata.should_save_frame() and metadata.forced_reuse:
            #     print(f"[FORCED-REUSE] Window {fcurrent}: Similarity forced reuse of State {metadata.reused_state_id}")
            #     print(f"[FORCED-REUSE] Saving uncertain frame for future exploration (separate folder)")

            #     output_fmax_pdb, uncertainty_name = villian_save_only(
            #         traj_path, fmax, folder, traj_num, local_window
            #     )

            #     global_state.add_uncertain_frame(output_fmax_pdb, uncertainty_name)
            #     print(f"[FORCED-REUSE] Frame saved: {uncertainty_name}")

            #     save_random_uncertain_frame(
            #         traj_path, folder, traj_num, local_window, fmax,
            #         subfolder="OM_paper_villian_forced_reuse_descendants"
            #     )

            local_variables['fcurrent'] = local_variables['fcurrent'] + window
            local_variables['newTraj']  = False

        except StopIteration:
            if ENABLE_SIMILARITY_CHECK and 'similarity_stats' in local_variables:
                stats_info = local_variables['similarity_stats']
                total      = stats_info['reused'] + stats_info['created']
                if total > 0:
                    reuse_rate = stats_info['reused'] / total * 100
                    print(f"[SIMILARITY STATS] Traj {traj_num}: "
                          f"{stats_info['reused']} reused, {stats_info['created']} created "
                          f"({reuse_rate:.1f}% reuse rate)")

            save_variables_local(local_variables, local_file)

            metadata = local_variables.get('last_metadata', AnalysisMetadata())
            return AnalysisResult(newstate, metadata)