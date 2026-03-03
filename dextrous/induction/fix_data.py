
import dextrous.dst_data as dst


def fix_data(path):
    data = dst.Data(path)
    data.slot_values.value[:] = [', '.join(v) for v in data.slot_values.value]
    data.value_candidates.candidate_value[:] = [', '.join(v) for v in data.value_candidates.candidate_value]
    data.save(path)


if __name__ == '__main__':
    fix_data('data/silver_mwoz_filtered/valid')
    fix_data('data/silver_sgd_filtered/valid')
