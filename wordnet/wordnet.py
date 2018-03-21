from nltk.corpus import wordnet
from nltk.wsd import lesk
from nltk.corpus import wordnet_ic


#1
word = wordnet.synsets('port')
for s in word:
    print(s, s.definition())

#2

print('place:', (word[0], word[0].definition()))
print('dessert:', (word[1], word[1].definition()))

#3


place = 'We take a ship from the port today'
dessert = 'He drinks tasty port'

place_tokens = [i.strip(',’.') for i in place.split()]
dessert_tokens = [i.strip('.,') for i in dessert.split()]

print(lesk(place_tokens, 'port').definition())
print(lesk(dessert_tokens, 'port').definition())

#4
for m in word[0].hypernyms():
    print(m, m.definition())

#4.1

for m in word[1].hypernyms():
    print(m, m.definition())

#5

place = wordnet.synsets('place')
dessert = wordnet.synsets('dessert')

def get_dist_sim(ss1, lexeme):
    distances = []
    similarities = []
    for ss2 in lexeme:
        dist = ss1.shortest_path_distance(ss2)
        if dist is not None:
            distances.append(dist)
            sim = ss1.path_similarity(ss2)
            similarities.append(sim)
    return distances, similarities

dist1 = get_dist_sim(word[0], place)[0]
print('min d(port: "порт", place): {}'.format(min(dist1)))
print('closest lemma definition: {}\n'.format(place[dist1.index(min(dist1))].definition()))

dist2 = get_dist_sim(word[0], dessert)[0]
print('min d(port: "порт", dessert): {}'.format(min(dist2)))
print('closest lemma definition: {}\n'.format(dessert[dist2.index(min(dist2))].definition()))

dist3 = get_dist_sim(word[1], place)[0]
print('min d(port: "гавань", place): {}'.format(min(dist3)))
print('closest lemma definition: {}\n'.format(place[dist3.index(min(dist3))].definition()))

dist4 = get_dist_sim(word[1], dessert)[0]
print('min d(port: "гавань", dessert): {}'.format(min(dist4)))
print('closest lemma definition: {}\n'.format(dessert[dist4.index(min(dist4))].definition()))

print('min (d(port: "порт", place), d(port: "порт", dessert)): {}'.format(min(min(dist1), min(dist2))))
print('min (d(port: "гавань", place), d(port: "гавань", dessert)): {}'.format(min(min(dist3), min(dist4))))

#6
ic = wordnet_ic.ic('ic-brown.dat')

ball = wordnet.synsets('ball')
for ss in ball:
    print(ss, ss.definition())

##print('Leacock-Chodorow Similarity', word[0].lch_similarity(ball[0]))
##print('Jiang-Conrath Similarity', word[0].jcn_similarity(ball[0], ic=ic))

organism = wordnet.synsets('organism')
for ss in organism:
    print(ss, ss.definition())
    print(ss, ss.examples())

whole = wordnet.synsets('whole', 'n')
for ss in whole:
    print(ss, ss.definition())
    print(ss, ss.examples())

for ss1 in organism:
    print('"orginism" definition: ', ss1.definition())
    print()
    for ss2 in whole:
        print('"whole" definition: ', ss2.definition())
        print()
        print('Leacock-Chodorow Similarity: ', ss1.lch_similarity(ss2))
        print()
        print('Jiang-Conrath Similarity: ', ss1.jcn_similarity(ss2, ic=ic))
