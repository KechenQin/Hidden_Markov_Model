# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import numpy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from classifier import CodeBook

def Main():
    '''This function is used for test.'''
    pass
    
class HMM(Classifier):
    """A Hidden Markov Model classifier."""
    
    def __init__(self, model={}):
        super(HMM, self).__init__(model)
        self.states=()
        self.initial_prob=numpy.array([])
        self.emission_prob=numpy.array([])
        self.transition_prob=numpy.array([])
        self.forward=numpy.array([])
        self.viterbi=numpy.array([])
        self.path=numpy.array([])
        self.transit_record={}
        self.tagnum={}  # tagnum does not count the last tag
        self.tagnumall={}
        self.tagword={}
        self.firsttag={}
        self.labeldic=CodeBook('')
        self.datadic=CodeBook('')
        
        self.hmmdic={}
        self.hmmdic2={}

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)
    
    def _init_hmm(self,instance):
        """Initialize the value used in ice-cream example.
        Ice-cream example is used for testing the accuracy of my model"""
        
        self.states=instance['states']
        self.initial_prob=instance['initial_prob']
        self.emission_prob=instance['emission_prob']
        self.transition_prob=instance['transition_prob']
        self.hmmdic={'Hot':0, 'Cold':1, 1:0, 2:1, 3:2}
        self.hmmdic2={0:'Hot',1:'Cold'}
        
    def for_initialization(self,observations):
        """Initialization of forward value"""
        observation_index=self.hmmdic[observations[0]]
        self.forward[0]=self.initial_prob*self.emission_prob[observation_index] 
        
    def for_recursion(self,observations):    
        """Recursion of forward value"""
        for time_step in range(1,len(observations)):
            observation_index=self.hmmdic[observations[time_step]]
            self.forward[time_step]=numpy.sum((self.forward[time_step-1]*self.transition_prob),1)*self.emission_prob[observation_index]
    
    def for_termination(self,observations):
        """Termination of forward value"""
        return sum(self.forward[len(observations)-1])
        
    def likelihood(self,observations):
        """Calculate likelihood"""
        self.forward=numpy.zeros([len(observations.features()),len(self.states)])
        self.for_initialization(observations.features())
        self.for_recursion(observations.features())
        result=self.for_termination(observations.features())
        return result
    
    def vit_initialization(self,observations):
        """Initialization of viterbi value"""
        # I comment the code used for calculating ice-cream example, 
        
#        observation_index=self.hmmdic[observations[0]]
        if self.datadic.__contains__(observations[0]):
            observation_index=self.datadic.get(observations[0])
            self.viterbi[0]=self.initial_prob*self.emission_prob[observation_index]
        else:
            self.viterbi[0]=self.initial_prob
#            self.viterbi[0]=numpy.full((1,self.labeldic.__len__()),(1/self.datadic.__len__()))
        
    def vit_recursion(self,observations): 
        """Recursion of viterbi value"""
#        self.path=numpy.zeros([len(observations),len(self.states)])
        self.path=numpy.zeros([len(observations),self.labeldic.__len__()])
        for time_step in range(1,len(observations)):
#            observation_index=self.hmmdic[observations[time_step]]
            if self.datadic.__contains__(observations[time_step]):
                observation_index=self.datadic.get(observations[time_step])
                tmp_emisprob=numpy.reshape(self.emission_prob[observation_index],(self.labeldic.__len__(),1))
            else:
                tmp_emisprob=numpy.ones([self.labeldic.__len__(),1])
#                tmp_emisprob=numpy.full((self.labeldic.__len__(),1),(1/self.datadic.__len__()))
            tmp_vit=tmp_emisprob*self.viterbi[time_step-1]*self.transition_prob
            self.path[time_step-1]=numpy.argmax(tmp_vit, 1)
            self.viterbi[time_step]=numpy.max(tmp_vit, 1)
        
    def vit_termination(self, observations):
        """Termination of viterbi value"""
        max_path=[]
        max_path_index=numpy.argmax(self.viterbi[len(observations)-1])
        final=self.labeldic.name(max_path_index)
#        final=self.hmmdic2[max_path_index]
        max_path.append(final)
        for i in range(len(observations)-2,-1,-1):
            max_path.insert(0, self.labeldic.name(self.path[i][max_path_index]))
#            max_path.insert(0, self.hmmdic2[self.path[i][max_path_index]])
            max_path_index=self.path[i][max_path_index]
        return max_path
        
    def cal_viterbi(self,observations):
        """Calculate viterbi value"""
#        self.viterbi=numpy.zeros([len(observations),len(self.states)])
        self.viterbi=numpy.zeros([len(observations),self.labeldic.__len__()])
        self.vit_initialization(observations)
        self.vit_recursion(observations)
        return self.vit_termination(observations)
        
    def _init_hmmtag(self,instances):
        """Record the number of tags and labels so as to calculate transition matrix, emission matrix and initial probability."""
        for instance in instances:
            for i in range(0,len(instance.label)):
                # two Codebooks used for mapping data to index.
                self.labeldic.add(instance.label[i])
                self.datadic.add(instance.data[i])
                if self.tagnum.has_key(instance.label[i]):
                    self.tagnumall[instance.label[i]]+=1
                else:
                    self.tagnumall[instance.label[i]]=1
                if self.tagword.has_key((instance.label[i],instance.data[i])):
                    self.tagword[(instance.label[i],instance.data[i])]+=1
                else:
                    self.tagword[(instance.label[i],instance.data[i])]=1
                # record transitions
                if i<len(instance.label)-1:
                    if self.transit_record.has_key((instance.label[i],instance.label[i+1])):
                        self.transit_record[(instance.label[i],instance.label[i+1])]+=1
                    else:
                        self.transit_record[(instance.label[i],instance.label[i+1])]=1
                    if self.tagnum.has_key(instance.label[i]):
                        self.tagnum[instance.label[i]]+=1
                    else:
                        self.tagnum[instance.label[i]]=1
                # record the first tag.
                if i==0:
                    if self.firsttag.has_key(instance.label[i]):
                        self.firsttag[instance.label[i]]+=1
                    else:
                        self.firsttag[instance.label[i]]=1
        
        self.initial_prob=numpy.zeros([self.labeldic.__len__()])
        self.cal_intialprob(instances)
        self.transition_prob=numpy.zeros([self.labeldic.__len__(),self.labeldic.__len__()])
        self.cal_transitprob(instances)
        self.emission_prob=numpy.zeros([self.datadic.__len__(),self.labeldic.__len__()])
        self.cal_emisprob(instances)

    def cal_intialprob(self,instances):
        """Calculate the initial probability, using data from _init_hmmtag() method"""
        for label_index in range(self.labeldic.__len__()):
            tmplabel=self.labeldic.name(label_index)
            if tmplabel in self.firsttag:
                self.initial_prob[label_index]=self.firsttag[tmplabel]/float(len(instances))
            else:
                self.initial_prob[label_index]=0
                
    def cal_transitprob(self,instances):
        """Calculate the transition matrix, using data from _init_hmmtag() method"""
        for label_index1 in range(self.labeldic.__len__()):
            for label_index2 in range(self.labeldic.__len__()):
                tmplabel1=self.labeldic.name(label_index1)
                tmplabel2=self.labeldic.name(label_index2)
                if self.transit_record.has_key((tmplabel2,tmplabel1)):
                    self.transition_prob[label_index1][label_index2]=self.transit_record[(tmplabel2,tmplabel1)]/float(self.tagnum[tmplabel2])
                else:
                    self.transition_prob[label_index1][label_index2]=0
    
    def cal_emisprob(self,instances):
        """Calculate the emission matrix, using data from _init_hmmtag() method"""
        for voc_index in range(self.datadic.__len__()):
            for label_index in range(self.labeldic.__len__()):
                tmpvoc=self.datadic.name(voc_index)
                tmplabel=self.labeldic.name(label_index)
                if self.tagword.has_key((tmplabel,tmpvoc)):
                    self.emission_prob[voc_index][label_index]=self.tagword[(tmplabel,tmpvoc)]/float(self.tagnumall[tmplabel])
                else:
                    self.emission_prob[voc_index][label_index]=0
            
    def train(self, instances):
        self._init_hmmtag(instances)
#        self._init_hmm(instances)

    def classify(self, instance):
        result=self.cal_viterbi(instance.data)
#        result=self.cal_viterbi(instance.features())
        return result

    if __name__ == '__main__':
        Main()