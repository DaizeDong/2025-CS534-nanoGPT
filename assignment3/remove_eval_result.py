import csv


def read_csv(file_path, has_header=True):
    """
    Read a CSV file and return its content.

    Args:
    - file_path (str): Path to the CSV file.
    - has_header (bool): Whether the CSV file has a header. Default is True.

    Returns:
    - list of list or dict: Content of the CSV file.
      If has_header is True, return a list of dictionaries;
      if has_header is False, return a list of lists.
    """
    data = []
    with open(file_path, newline='', encoding='utf-8') as f:
        if has_header:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                data.append(dict(row))
        else:
            csvreader = csv.reader(f)
            for row in csvreader:
                data.append(row)
    return data


files = [
    "assignment3/runs/losses_large_bs.csv",
    "assignment3/runs/losses_large_bs_mp2.csv",
    "assignment3/runs/losses_large_bs_mp4.csv",
    "assignment3/runs/losses_large_bs_mp6.csv",
]

for file in files:
    data = read_csv(file, has_header=True)
    new_data = []

    last_train_timepoint = 0.0
    accumulative_eval_time = 0.0
    all_deducted_eval_time = {}

    for row in data:
        if row['type'] == 'val_step':
            now_step = int(row['x'])
            this_eval_elapse = float(row['elapsed_s']) - last_train_timepoint
            accumulative_eval_time += this_eval_elapse
            all_deducted_eval_time[now_step] = accumulative_eval_time
            print(f"all_deducted_eval_time {now_step}: {accumulative_eval_time:.2f}s")
        elif row['type'] == 'iter':
            now_step = int(row['x'])
            last_train_timepoint = float(row['elapsed_s'])

    all_recorded_steps = sorted(list(all_deducted_eval_time.keys()))

    now_used_step = all_recorded_steps[0]
    now_deducted_eval_time = all_deducted_eval_time[now_used_step]
    all_recorded_steps = all_recorded_steps[1:]
    next_used_step = all_recorded_steps[0]

    for row in data:
        now_step = int(row['x'])
        if now_step > next_used_step:
            now_used_step = all_recorded_steps[0]
            now_deducted_eval_time = all_deducted_eval_time[now_used_step]
            all_recorded_steps = all_recorded_steps[1:]
            if all_recorded_steps:
                next_used_step = all_recorded_steps[0]
            else:
                next_used_step = float('inf')
        row['elapsed_s'] = float(row['elapsed_s']) - now_deducted_eval_time
        new_data.append(row)

    save_name = file.replace(".csv", "_no_eval.csv")
    with open(save_name, 'w', newline='', encoding='utf-8') as f:
        fieldnames = new_data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in new_data:
            writer.writerow(row)
        print(f"Saved adjusted data to {save_name}")
