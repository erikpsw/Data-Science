import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['text.usetex'] = False
plt.rc("font",family='MicroSoft YaHei')

def read_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            try:
                time = float(parts[0])
                value = float(parts[1])
                data.append((time, value))
            except ValueError:
                continue
    
    return data

def extract_sequences(data):
    sequences = []
    current_seq = []
    
    for time, value in data:
        if time == 0.0:
            if current_seq:
                sequences.append(current_seq)
            current_seq = [(time, value)]
        else:
            current_seq.append((time, value))
    
    if current_seq:
        sequences.append(current_seq)
    
    return sequences

def interpolate_sequence_to_integer_times(sequence):
    times, values = zip(*sequence)
    min_time = int(min(times))
    max_time = int(max(times)) + 1
    integer_times = list(range(min_time, max_time))
    interpolated_values = np.interp(integer_times, times, values)
    return list(zip(integer_times, interpolated_values))

def calculate_average_sequence(sequences):
    interpolated_sequences = [interpolate_sequence_to_integer_times(seq) for seq in sequences]
    min_time = min(int(min(time for time, _ in seq)) for seq in sequences)
    max_time = max(int(max(time for time, _ in seq)) for seq in sequences) + 1
    all_times = list(range(min_time, max_time))
    sum_values = [0.0] * len(all_times)
    counts = [0] * len(all_times)
    
    for seq in interpolated_sequences:
        seq_dict = dict(seq)
        for i, t in enumerate(all_times):
            if t in seq_dict:
                sum_values[i] += seq_dict[t]
                counts[i] += 1
    
    avg_values = []
    for i, t in enumerate(all_times):
        if counts[i] > 0:
            avg_values.append((t, sum_values[i] / counts[i]))
    
    return avg_values

def calculate_moving_average(data, window_size):
    times, values = zip(*data)
    result = []
    
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window = values[start_idx:i+1]
        ma_value = sum(window) / len(window)
        result.append((times[i], ma_value))
    
    return result

def visualize_sequences(sequences, avg_sequence):
    # Convert time values from arbitrary units to minutes for plotting
    avg_times_min = [t/60 for t, _ in avg_sequence]  # Convert to minutes
    avg_values = [v for _, v in avg_sequence]
    
    window_sizes = [5, 10, 20, 50]
    ma_styles = ['g--', 'b-.', 'm:', 'c-']
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_times_min, avg_values, 'r-', linewidth=3.5, label='平均序列')
    
    for idx, window_size in enumerate(window_sizes):
        ma_sequence = calculate_moving_average(avg_sequence, window_size)
        ma_times_min = [t/60 for t, _ in ma_sequence]  # Convert to minutes
        ma_values = [v for _, v in ma_sequence]
        plt.plot(ma_times_min, ma_values, ma_styles[idx], linewidth=2, 
                 label=f'移动平均-{window_size}')
    
    plt.xlabel('时间（分钟）')
    plt.ylabel('数量')
    plt.title('平均序列与移动平均')
    plt.legend()
    plt.grid(True)
    plt.savefig('moving_averages.png')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink']
    interpolated_sequences = [interpolate_sequence_to_integer_times(seq) for seq in sequences]
    
    for i, seq in enumerate(interpolated_sequences):
        times_min = [t/60 for t, _ in seq]  # Convert to minutes
        values = [v for _, v in seq]
        color = colors[i % len(colors)]
        plt.plot(times_min, values, color=color, alpha=0.7, label=f'序列 {i+1}')
    
    plt.plot(avg_times_min, avg_values, 'r-', linewidth=3.5, label='平均序列')
    plt.xlabel('时间（分钟）')
    plt.ylabel('数量')
    plt.title(f'原始序列（共 {len(sequences)} 条）')
    plt.legend()
    plt.grid(True)
    plt.savefig('original_sequences.png')
    plt.show()

def main():
    file_path = "Ex3/out.txt"
    
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    data = read_data(file_path)
    sequences = extract_sequences(data)
    print(f"Number of sequences found: {len(sequences)}")
    
    if not sequences:
        print("No sequences found starting from t=0.")
        return
    
    avg_sequence = calculate_average_sequence(sequences)
    visualize_sequences(sequences, avg_sequence)

if __name__ == "__main__":
    main()
