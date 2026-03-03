
from csv import DictReader, writer
from ezpyzy.table import Table
from collections import Counter

if __name__ == '__main__':
    # for all predicted clusters that matched to some gold slot:
    #   Counter(predicted slot names)
    #   automatic_name = max(counter)
    #   get 5 examples (value, dialogue_context)

    induction_data = Table.of('results/induction_outputs/induction_data_mwoz2.4_valid_output.csv')
    sampled_induction_data = DictReader(open('results/induction_outputs/induction_data_mwoz2.4_valid_samples.csv'))
    cluster_samples = {}
    for item in sampled_induction_data:
        cluster_samples.setdefault(int(item['cluster_id']), []).append(item)
    clustered_data = induction_data[induction_data.cluster_id != -1]
    mapped_clusters = clustered_data[clustered_data.matched_slot != None]
    grouped_by_cluster = mapped_clusters().group(mapped_clusters.cluster_id)

    csv_writer = writer(open('results/induction_outputs/automatic_naming_evaluation.csv', 'w'))
    csv_writer.writerow(['slot name', 'slot value', 'context', 'suitable?'])
    for cluster_id, cluster_table in grouped_by_cluster.items():
        predicted_slot_name_counts = Counter(cluster_table.slot)
        mode_slot_name, mode_count = max(predicted_slot_name_counts.items(), key=lambda x: x[1])
        samples = [(mode_slot_name if i == 0 else '', item['value'], '\n'.join(item['context'].split('\n')[-3:]), '') for i, item in enumerate(cluster_samples[cluster_id])]
        csv_writer.writerows(samples)
    print(f"Wrote evaluation file for {len(grouped_by_cluster)} clusters")
