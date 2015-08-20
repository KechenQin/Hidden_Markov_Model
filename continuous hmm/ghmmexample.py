import ghmm
from ghmm import *


''''example of discrete hmm'''

dna = ['A','C','G','T']
sigma = Alphabet(dna)
sigma = IntegerRange(1,7)
print sigma
A = [[0.95, 0.05], [0.15, 0.85]]
normal = [.4,.1,.1,.4]
island = [.2,.3,.3,.2]
B=[normal,island]
pi = [0.5] * 2
m=HMMFromMatrices(sigma,DiscreteDistribution(sigma),A,B,pi)
print m
 
seqList = [ char for char in 'GCGCATTAATCGTCGTCGTAGTTCCTT']
# seqList = [ char for char in 'TGCA']
train_seq = EmissionSequence(sigma, seqList)
print train_seq
print m.viterbi(train_seq)
print m.loglikelihood(train_seq)

''''example of continous hmm'''

F = ghmm.Float()   # emission domain of this model
# ld = LabelDomain(['1', '2', '3', '4', '5', '6'])
A = [[0.9, 0.1],
     [0.2, 0.8]]   # transition matrix

# Interpretation of B matrix for the mixture case (Example with three states and two components each):
#           B = [
#        [["mu111","mu112"],["sig1111","sig1112","sig1121","sig1122"],
#         ["mu121","mu122"],["sig1211","sig1212","sig1221","sig1222"],
#         ["w11","w12"] ],
#        [["mu211","mu212"],["sig2111","sig2112","sig2121","sig2122"],
#         ["mu221","mu222"],["sig2211","sig2212","sig2221","sig2222"],
#         ["w21","w22"] ],
#        [["mu311","mu312"],["sig3111","sig3112","sig3121","sig3122"],
#         ["mu321","mu322"],["sig3211","sig3212","sig3221","sig3222"],
#         ["w31","w32"] ],
#       ]
B = [
     [[5.0],[2.0],
      [6.0],[1.0],
      [0.3,0.7]],
     [[2.0],[1.0],
      [1.5],[0.4],
      [0.4, 0.7]]
     ]  # parameters of mixture models

pi = [0.1,0.1]  # initial probabilities per state
model = ghmm.HMMFromMatrices(F,ghmm.MultivariateGaussianDistribution(F), A, B, pi)
# modify model parameters (examples)
p = model.getInitial(0)
model.setInitial(0,0.5)
# re-set transition from state 0 to state 1
trans = model.getTransition(0,1)
model.setTransition(0,1,0.6)
# re-setting emission of state 0 component 1
model.setEmission(0,1,[[4.0],[2.0]])

model.normalize()   # re-normalize model parameters
print model
#    for 2 dimensional data, the data structure is like this [x11, x12, x21, x22, x31, x32...]
seq=EmissionSequence(F,[5.5, 0.1])
#  sample single sequence of length 50
seq = model.sampleSingle(50)
# # sample 10 sequences of length 2
# seq = model.sample(10,2,seed=3586662)
print seq
# get log P(seq | model)
logp = model.loglikelihood(seq)
print logp

# train model parameters, number of iterations is 100, threshold is 0.01.
print model.baumWelch(seq,100,0.01)
print model