
import dextrous.dst_data as dst

mwoz = dst.Data('data/mwoz2.4/train')

domains = mwoz.slots().group(mwoz.slots.domain)

count = 0
for domain in mwoz.turns.domain[[x in {
    'taxi', 'train', 'restaurant', 'hotel', 'attraction', 'hospital', 'bus'
} for x in mwoz.turns.domain]]:
    count += len(domains[domain])

print(count)