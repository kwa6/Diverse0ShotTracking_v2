
import dextrous.dst_data as dst
import dextrous.preprocessing as pp
import pathlib as pl

unseen_domains = [
    'Messaging',
    'Payment',
    'Alarm',
    'Trains'
]

def filter_seen_domains(path):
    path = pl.Path(path)
    data = dst.Data(str(path))
    keep_domains = [d for d in data.slots.domain if any(x in d for x in unseen_domains)]
    pp.drop_domains(data, keep_domains, include_specified=True)
    data.save(path.parent / 'unseen')

def concat_unseen_domains(path):
    path = pl.Path(path)
    alarm_path = path / 'test_Alarm_1'
    messaging_path = path / 'test_Messaging_1'
    payment_path = path / 'test_Payment_1'
    trains_path = path / 'test_Trains_1'
    alarm_data = dst.Data(str(alarm_path))
    messaging_data = dst.Data(str(messaging_path))
    payment_data = dst.Data(str(payment_path))
    trains_data = dst.Data(str(trains_path))
    data = alarm_data
    data.slot_values += messaging_data.slot_values
    data.slot_values += payment_data.slot_values
    data.slot_values += trains_data.slot_values
    data.turns += messaging_data.turns
    data.turns += payment_data.turns
    data.turns += trains_data.turns
    data.save(path / 'unseen')

if __name__ == '__main__':
    concat_unseen_domains('data/sgd')