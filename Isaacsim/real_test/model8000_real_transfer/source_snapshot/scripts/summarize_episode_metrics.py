"""Summarize completed episodes; never use successful steps as success rate."""
import argparse
import csv
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('csv_path')
args = parser.parse_args()
with open(args.csv_path, newline='') as stream:
    rows = list(csv.DictReader(stream))
n = len(rows)
result = {'completed_episodes': n}
for name in ('success_any', 'success_final', 'success_stable'):
    count = sum(int(row[name]) for row in rows)
    result[name] = {'count': count, 'rate': count/n if n else None}
# Report geometry/failure overlap explicitly, without changing the task label.
valid_final = sum(int(row['success_final']) and not (int(row.get('outside_workspace', 0)) or int(row.get('invalid_grasp_width', 0))) for row in rows)
result['success_final_no_failure'] = {'count': valid_final, 'rate': valid_final/n if n else None}
stable_final = sum(float(row['final_success_streak_s']) >= .5 for row in rows)
result['success_stable_final'] = {'count': stable_final, 'rate': stable_final/n if n else None}
for name in ('force_p95_N', 'force_peak_N', 'longest_success_s', 'action_saturation'):
    values = [float(row[name]) for row in rows]
    result[name] = {'mean_over_episodes': sum(values)/n if n else None,
                    'max_over_episodes': max(values) if values else None}
for name in ('time_out', 'outside_workspace', 'invalid_grasp_width'):
    result[name] = sum(int(row.get(name, 0)) for row in rows)
print(json.dumps(result, indent=2))
