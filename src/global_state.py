# global_state.py
from readerwriterlock import rwlock
import copy
import os
from instrumentation_scaling import logger
import time
class GlobalState:
    def __init__(self):
        self.models = []
        self.states = []
        self.num_models = 0
        self.uncertain_queue = []
        self.rw_lock = rwlock.RWLockFair()
        self._uncertain_seen = set()  # optional dedup by pdb_path

    def add_uncertain_frame(self, pdb_path, name):
        t_request = time.perf_counter()
        key = os.path.abspath(pdb_path)
        with self.rw_lock.gen_wlock():
            t_acquired = time.perf_counter()
            if key in self._uncertain_seen:
                t_released = time.perf_counter()
                return
            self._uncertain_seen.add(key)
            self.uncertain_queue.append((pdb_path, name))
            t_released = time.perf_counter()
            logger.log("gsm_lock",op="write", method="add_uncertain_frame", wait=t_acquired - t_request, hold=t_released - t_acquired)

    def get_uncertain_jobs(self):
        t_request = time.perf_counter()
        with self.rw_lock.gen_rlock():
            t_acquired = time.perf_counter()
            result = list(self.uncertain_queue)
            t_released = time.perf_counter()

            logger.log("gsm_lock",  op="read",  method="get_uncertain_jobs", wait=t_acquired - t_request, hold=t_released - t_acquired, count=len(result))
            return list(self.uncertain_queue)

    def pop_all_uncertain_jobs(self):
        """Consume-and-clear semantics: take current queue and empty it."""
        t_request = time.perf_counter()
        with self.rw_lock.gen_wlock():
            t_acquired = time.perf_counter()
            jobs = list(self.uncertain_queue)
            self.uncertain_queue.clear()
            t_released = time.perf_counter()
            logger.log("gsm_lock", op="write", method="pop_all_uncertain_jobs", wait=t_acquired - t_request, hold=t_released - t_acquired, count=len(jobs))
            return jobs

    def snapshot(self):
        t_request = time.perf_counter()
        with self.rw_lock.gen_rlock():
            t_acquired = time.perf_counter()
            snap = {
                "models": copy.deepcopy(self.models),
                "states": copy.deepcopy(self.states),
                "num_models": self.num_models
            }
            t_released = time.perf_counter()
            logger.log("gsm_lock", op="read", method="snapshot", wait=t_acquired - t_request, hold=t_released - t_acquired, n_states=len(snap["states"]), n_models=len(snap["models"]))
            return snap
          
    def add_state(self, state, model):
        t_request = time.perf_counter()
        with self.rw_lock.gen_wlock():
            t_acquired = time.perf_counter()
            state.id = len(self.states)
            self.states.append(state)
            self.models.append(model)
            self.num_models += 1
            t_released = time.perf_counter()
            logger.log("gsm_lock", op="write", method="add_state", wait=t_acquired - t_request, hold=t_released - t_acquired, new_state_id=state.id)
            return state.id


    def update_state(self, index, update_fn):
        t_request = time.perf_counter()
        with self.rw_lock.gen_wlock():
            t_acquired = time.perf_counter()
            if 0 <= index < len(self.states):
                update_fn(self.states[index])
            t_released = time.perf_counter()
            logger.log("gsm_lock", op="write", method="update_state", wait=t_acquired - t_request, hold=t_released - t_acquired, state_id=index)
          
    def restore(self, data):
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict from pickle, got {type(data)}")

        self.models = data.get("models", data.get("Models", []))
        self.states = data.get("States", data.get("states", []))
        self.num_models = data.get("Nmodels", data.get("num_models", len(self.models)))

        print(f"[RESTORE] Restored global_state: {len(self.states)} states, {len(self.models)} models")

global_state = GlobalState()
