import pandas as pd
import nltk
from nltk.collocations import *
from scipy.stats import spearmanr

df = pd.read_csv('court-V-N.csv', header=None)
rows = list(map(list, df.values)) 
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_documents(rows)

golden_stand = [(('МЕРА','ПРЕСЕЧЕНИЕ'), 1), 
                (('ВЫДАТЬ', 'САНКЦИЯ'), 2), 
                (('ПРИНЯТЬ', 'РЕШЕНИЕ'), 3), 
                (('УДОВЛЕТВОРИТЬ','ХОДАТАЙСТВО'), 4), 
                (('УДОВЛЕТВОРИТЬ','ИСК'), 5), 
                (('ВЫНЕСТИ','РЕШЕНИЕ'), 6), 
                (('ОСТАВИТЬ', 'СИЛА'), 7), 
                (('САНКЦИЯ', 'АРЕСТ'), 8), 
                (('ПРИЗНАТЬ','ВИНОВНАЯ'), 9), 
                (('ОТКАЗАТЬ', 'УДОВЛЕТВОРЕНИЕ'), 10)]

ranks_golden = [p[1] for p in golden_stand]

finder.apply_freq_filter(4)

list_with_scores_pmi_new = list(finder.score_ngrams(bigram_measures.pmi))
list_no_scores_pmi_new = [(x[0][0].strip(' '), x[0][1].strip(' ')) for x in list_with_scores_pmi_new]

list_with_scores_jaccard = list(finder.score_ngrams(bigram_measures.jaccard))
list_no_scores_jaccard = [(x[0][0].strip(' '), x[0][1].strip(' ')) for x in list_with_scores_jaccard]

ranks_pmi = [list_no_scores_pmi.index(i[0]) for i in golden_stand]
ranks_jaccard = [list_no_scores_jaccard.index(i[0]) for i in golden_stand]
print('gold', ranks_golden)
print('pmi', ranks_pmi)
print('jaccard', ranks_jaccard)
print('PMI & jaccard')
print(spearmanr(ranks_pmi, ranks_jaccard))
print('golden standart & jaccard')
print(spearmanr(ranks_golden, ranks_jaccard))
print('PMI & golden standart')
print(spearmanr(ranks_pmi, ranks_golden))
