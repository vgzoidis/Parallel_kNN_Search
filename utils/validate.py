import os
import struct
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def write_bin(filename, data):
    n, d = data.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', n))
        f.write(struct.pack('i', d))
        f.write(data.astype(np.float64).tobytes())

def read_idx_dst(idx_file, dst_file, n, k):
    with open(idx_file, 'rb') as f:
        idx = np.frombuffer(f.read(), dtype=np.int32).reshape(n, k)
    with open(dst_file, 'rb') as f:
        dst = np.frombuffer(f.read(), dtype=np.float64).reshape(n, k)
    return idx, dst

def clean_output_files(name):
    idx_out = f"outputs/idx_{name}.bin"
    dst_out = f"outputs/dst_{name}.bin"
    if os.path.exists(idx_out): os.remove(idx_out)
    if os.path.exists(dst_out): os.remove(dst_out)

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    n_C, n_Q, d, k = 10000, 1000, 10, 5
    print(f"Generating datasets:\n Corpus = {n_C}x{d}\n Queries = {n_Q}x{d}\n k = {k}\n")

    C = np.random.rand(n_C, d) * 100
    Q = np.random.rand(n_Q, d) * 100

    write_bin("data/C.bin", C)
    write_bin("data/Q.bin", Q)

    # 1. Baseline: scikit-learn (Brute force exact calculating)
    print("Running scikit-learn (Exact Brute Force)...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(C)
    
    start_time = time.time()
    dst_py, idx_py = nbrs.kneighbors(Q)
    sklearn_time = time.time() - start_time
    print(f"  Time: {sklearn_time:.4f}s\n")

    versions = [
        ("v0", "./bin/knn_v0", {}),
        ("v1_omp", "./bin/knn_v1_omp", {"OMP_NUM_THREADS": "4"}),
        ("v1_pth", "./bin/knn_v1_pth", {"PTH_NUM_THREADS": "4"})
    ]

    times = [sklearn_time]
    labels = ["scikit-learn"]

    t_serial = 1.0

    for name, cmd, env_vars in versions:
        idx_out = f"outputs/idx_{name}.bin"
        dst_out = f"outputs/dst_{name}.bin"

        env = os.environ.copy()
        env.update(env_vars)

        print(f"Running C implementation: {name}")
        if 'PTH_NUM_THREADS' in env_vars:
            print(f"  Threads Requested: {env_vars['PTH_NUM_THREADS']}")
        elif 'OMP_NUM_THREADS' in env_vars:
            print(f"  Threads Requested: {env_vars['OMP_NUM_THREADS']}")

        try:
            res = subprocess.run([cmd, "data/C.bin", "data/Q.bin", str(k), idx_out, dst_out],
                                 env=env, capture_output=True, text=True, check=True)
            
            # Retrieve time from C stdout output
            t_exec = float(res.stdout.strip().split('\n')[-1])
            times.append(t_exec)
            labels.append(name)
            
            print(f"  Time: {t_exec:.4f}s")
            
            if name == "v0":
                t_serial = t_exec
            else:
                speedup = t_serial / t_exec if t_exec > 0 else 0
                print(f"  Speedup vs Serial: {speedup:.2f}x")

            # Validate against scikit-learn
            if os.path.exists(idx_out) and os.path.exists(dst_out):
                idx_c, dst_c = read_idx_dst(idx_out, dst_out, n_Q, k)
                dist_err = np.max(np.abs(dst_c - dst_py))
                if dist_err < 1e-4:
                    print("  Validation: SUCCESS (Exact Distances Match)")
                else:
                    print(f"  Validation: FAILED (Max diff err: {dist_err})")
            else:
                print("  Validation: FAILED (Output files not generated)")

        except Exception as e:
            msg = e.stderr if hasattr(e,'stderr') else str(e)
            print(f"  Error running {name}: {msg}")
        
        print("-" * 40)

    # 4. Generate visualizer chart metrics
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=['gray', 'blue', 'orange', 'green'])
    plt.ylabel('Execution Time (sec)')
    plt.title(f'k-NN Execution Speedup Comparison (Queries: {n_Q}, Corpus: {n_C})')
    
    for idx, rect in enumerate(bars):
        height = rect.get_height()
        speedup_txt = "" 
        if idx > 1: # OpenMP or Pthreads speedup label calculation relative to v0 (which is idx 1)
             s_up = times[1] / times[idx]
             speedup_txt = f"\n({s_up:.1f}x speedup)"
        plt.text(rect.get_x() + rect.get_width()/2., height,
                 f"{height:.3f}s{speedup_txt}",
                 ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig("docs/speedup.png")
    print("Benchmarking visualization saved to docs/speedup.png!")

if __name__ == '__main__':
    main()

