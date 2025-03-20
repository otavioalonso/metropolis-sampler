from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object
import numpy as np
from copy import copy

class MCMC(object):
    def __init__(self, start, posterior, covariance, quiet=False,
        tuning_frequency=-1, tuning_grace=np.inf, tuning_end=np.inf, 
        scaling=2.4,
        exponential_probability=0.33333,
        n_drag=0):
        """
        posterior should return a PipelineResults object
        """
        #Set up basic variables
        self.posterior = lambda x: {'post': posterior(x), 'vector': x}
        self.p = np.array(start)
        self.ndim = len(self.p)
        self.quiet=quiet
        #Run the pipeline for the first time, on the 
        #starting point
        self.Lp = self.posterior(self.p)

        #Proposal
        self.covariance = covariance
        cholesky = np.linalg.cholesky(covariance)
        self.scaling = scaling
        self.exponential_probability = exponential_probability
        self.n_drag = n_drag
        self.fast_indices = None
        self.slow_indices = None
        self.oversampling = None
        self.fast_slow_is_ready = False
        self.rng = np.random.default_rng()

        if self.n_drag > 0 :
            raise ValueError('You must set use_cobaya to have n_drag>0')

        self.proposal = Proposal(cholesky, scaling=scaling, exponential_probability=exponential_probability)

        #For adaptive sampling
        self.last_covariance_estimate = covariance.copy()       
        self.covariance_estimate = covariance.copy()
        self.chain = []
        self.n_cov_fail = 0

        #self.covariance_estimate.copy()
        self.mean_estimate = start.copy()
        self.tuning_frequency = tuning_frequency
        self.original_tuning_frequency = tuning_frequency
        self.tuning_grace = tuning_grace
        self.tuning_end = tuning_end

        #Set up instance variables storing samples, etc.
        self.samples = []
        self.iterations = 0
        self.accepted = 0
        self.accepted_since_tuning = 0
        self.iterations_since_tuning = 0


    def sample(self, n):
        samples = []

        sample_method = (
            self._sample_dragging 
            if (self.fast_slow_is_ready and self.n_drag) 
            else self._sample_metropolis)
        #Take n sample mcmc steps

        samples = []
        for i in range(n):
            # update counts
            self.iterations += 1
            self.iterations_since_tuning += 1

            # generate a new sample
            s = sample_method()
            samples.append(s['vector'])
            # hack - should unify these
            self.chain.append(s['vector'])

            if self.should_tune_now():
                self.tune()

        return samples

    def _sample_metropolis(self):
        # proposal point and its likelihood
        q = self.proposal.propose(self.p)
        #assume two proposal subsets for now
        Lq = self.posterior(q)
        if not self.quiet:
            print("  ".join(str(x) for x in q))
        #acceptance test
        delta = Lq['post'] - self.Lp['post']
        if  accept(Lq['post'], self.Lp['post']):
            #update if accepted
            self.Lp = Lq
            self.p = q
            self.accepted += 1
            self.accepted_since_tuning += 1
            if not self.quiet:
                print("[Accept delta={:.3g}]\n".format(delta))
        elif not self.quiet:
            print("[Reject delta={:.3g}]\n".format(delta))

        return self.Lp


    def _sample_dragging(self):
        # get params with same fast params but different slow ones
        if not self.quiet:
            print("starting drag")
            print("Current post = ", self.Lp)
        start = self.p
        end = self.proposal.propose_slow(start)


        # posteriors and derived parameters etc.
        r_start = copy(self.Lp)
        r_end = self.posterior(end)
        if not self.quiet:
            print("slow proposal post = ", r_end)

        if not np.isfinite(r_end):
            if not self.quiet:
                print("[Reject: nan/-inf posterior]\n")
            return self.Lp


        # coordinates of current start and end
        p1 = copy(start)
        p2 = copy(end)

        # results for current start and end
        r1 = copy(r_start)
        r2 = copy(r_end)

        start_post = r1
        end_post = r2

        drag_accepts = 0

        for i in range(self.n_drag):
            delta_fast = self.proposal.propose_fast(p1) - p1
            if not self.quiet:
                print("delta fast", delta_fast)
            q1 = p1 + delta_fast
            q2 = p2 + delta_fast

            s1 = self.posterior(q1)

            if np.isfinite(s1):
                s2 = self.posterior(q2)
            else:
                s2 = -np.inf


            f = (1+i) /(1+self.n_drag)

            P1 = (1-f)*r1 + f*r2
            Q1 = (1-f)*s1 + f*s2

            accept_drag = accept(Q1, P1) and np.isfinite(s1) and np.isfinite(s2)

            if accept_drag:
                p1 = q1
                p2 = q2
                r1 = s1
                r2 = s2
                drag_accepts += 1
                if not self.quiet:
                    print("[Accept drag step delta={:.3g}]\n".format(Q1 - P1))
            elif not self.quiet:
                print("[Reject drag step delta={:.3g}]\n".format(Q1 - P1))

            start_post += r1
            end_post += r2

        if not self.quiet:
            print("[Accepted {}/{} drag steps]".format(drag_accepts,self.n_drag))

        start_post /= self.n_drag
        end_post /= self.n_drag
        accept_overall = accept(end_post, start_post)

        if not self.quiet:
            print("Done drag")
        if accept_overall:
            self.p = p2
            self.Lp = r_end
            self.accepted += 1
            self.accepted_since_tuning += 1
            if not self.quiet:
                print("[Accept delta={:.3g}]\n".format(end_post - start_post))
            return r2
        else:
            if not self.quiet:
                print("[Reject delta={:.3g}]\n".format(end_post - start_post))
            return self.Lp



    def should_tune_now(self):
        return (    
            self.tuning_frequency>0
        and self.iterations>self.tuning_grace 
        and self.iterations%self.tuning_frequency==0
        and self.iterations<self.tuning_end
        )


    def update_covariance_estimate(self):
        n = self.iterations
        self.mean_estimate = np.mean(self.chain, axis=0)
        C = np.cov(np.transpose(self.chain))
        if is_positive_definite(C):
            self.covariance_estimate = C
        else:
            print("Cov estimate not SPD.  If this keeps happening, be concerned.")
            # chain_outfile = 'joe_dump_chain_{}.txt'.format(self.n_cov_fail)
            # cov_outfile = 'joe_dump_cov_{}.txt'.format(self.n_cov_fail)
            # self.n_cov_fail += 1
            # np.savetxt(chain_outfile, self.chain)
            # np.savetxt(cov_outfile, np.transpose(C))
            # print("TEMPORARY (JOE - REMOVE LATER) - dumping to file")


    def set_fast_slow(self, fast_indices, slow_indices, oversampling):
        if self.n_drag:
            oversampling = 1
            print("Overriding oversampling parameter -> 1 since using dragging")
        self.fast_indices = fast_indices
        self.slow_indices = slow_indices
        self.oversampling = oversampling
        self.fast_slow_is_ready = True

        self.proposal = FastSlowProposal(self.covariance, fast_indices, slow_indices, oversampling, scaling=self.scaling, exponential_probability=self.exponential_probability)
        self.tuning_frequency = self.original_tuning_frequency * oversampling

    def tune(self):
        self.update_covariance_estimate()

        f = (self.covariance_estimate.diagonal()**0.5-self.last_covariance_estimate.diagonal()**0.5)/self.last_covariance_estimate.diagonal()**0.5
        i = abs(f).argmax()
        print("Largest parameter sigma fractional change = {:.1f}% for param {}".format(100*f[i], i))
        self.last_covariance_estimate = self.covariance_estimate.copy()

        print("Accepted since last tuning: {}%".format((100.*self.accepted_since_tuning)/self.iterations_since_tuning))
        self.accepted_since_tuning = 0
        self.iterations_since_tuning = 0


        if isinstance(self.proposal, FastSlowProposal):
            print("Tuning fast/slow sampler proposal.")
            self.proposal = FastSlowProposal(self.covariance_estimate, 
                self.fast_indices, self.slow_indices, self.oversampling,scaling=self.scaling, exponential_probability=self.exponential_probability)
        elif isinstance(self.proposal, Proposal):
            print("Tuning standard sampler proposal.")
            cholesky = np.linalg.cholesky(self.covariance_estimate)
            self.proposal = Proposal(cholesky, scaling=self.scaling, exponential_probability=self.exponential_probability)
        else:
            #unknown proposal type
            pass


def accept(post1, post0):
    return (post1 > post0) or (post1-post0 > np.log(np.random.uniform(0,1)))

def is_positive_definite(M):
    return np.all(np.linalg.eigvals(M) > 0)

class Proposal(object):
    def __init__(self, cholesky, scaling=2.4, exponential_probability=0.333333):
        self.iteration = 0
        self.ndim = len(cholesky)
        self.cholesky = cholesky 
        rotation = np.identity(self.ndim)
        self.proposal_axes = np.dot(self.cholesky, rotation)
        self.scaling = scaling
        self.exponential_probability = exponential_probability

    def proposal_distance(self, ndim, scaling):
        #from CosmoMC
        if np.random.uniform()<self.exponential_probability:
            r = np.random.exponential()
        else:
            n = min(ndim,2)
            r = (np.random.normal(size=n)**2).mean()**0.5
        return r * scaling

    def randomize_rotation(self):
        #After CosmoMC, we randomly rotate our proposal axes
        #to avoid doubling back on ourselves
        rotation = random_rotation_matrix(self.ndim)
        #All our proposals are done along the axes of our covariance
        #matrix
        self.proposal_axes = np.dot(self.cholesky, rotation)

    def propose(self, p):
        #Once we have cycled through our axes, re-randomize them
        i = self.iteration%self.ndim
        if i==0:
            self.randomize_rotation()
        self.iteration += 1
        #otherwise, propose along our defined axes
        return p + self.proposal_distance(self.ndim,self.scaling) * self.proposal_axes[:,i]



class FastSlowProposal(Proposal):
    def __init__(self, covariance, fast_indices, slow_indices, oversampling, scaling=2.4, exponential_probability=0.3333):
        self.ordering = np.concatenate([slow_indices, fast_indices])
        self.inverse_ordering = invert_ordering(self.ordering)
        self.nslow = len(slow_indices)
        self.oversampling = oversampling
        self.iteration = 0
        self.slow_iteration = 0
        reordered_covariance = covariance[:,self.ordering][self.ordering]
        reordered_cholesky = np.linalg.cholesky(reordered_covariance)

        #For the fast subspace we just use the original vanilla proposal.
        #The slow proposal must be a little different  - not a square matrix
        self.fast_proposal = Proposal(reordered_cholesky[self.nslow:, self.nslow:], exponential_probability=exponential_probability)
        self.slow_matrix = reordered_cholesky[:,:self.nslow]

        self.slow_rotation = np.identity(self.nslow)
        self.scaling = scaling
        self.exponential_probability = exponential_probability



    def propose(self, p):
        p = p[self.ordering]
        if self.iteration%(self.oversampling+1)==0:
            q = self.propose_slow(p)
        else:
            q = self.propose_fast(p)
        self.iteration += 1
        return q[self.inverse_ordering]

    def propose_fast(self, p):
        q = np.zeros_like(p)
        q[:self.nslow] = p[:self.nslow]
        q[self.nslow:] += self.fast_proposal.propose(p[self.nslow:])
        return q
        
    def propose_slow(self, p):
        i = self.slow_iteration%self.nslow
        if i==0:
            self.randomize_rotation()   

        #Following the notation in Lewis (2013)
        delta_s = self.slow_rotation[i] * self.proposal_distance(self.nslow,self.scaling)
        self.slow_iteration += 1

        q = p + np.dot(self.slow_matrix, delta_s)
        return q
        

    def randomize_rotation(self):
        self.slow_rotation = random_rotation_matrix(self.nslow)

#I've been copying this algorithm out of CosmoMC
#for the last decade.  Every time I need a new one
#I think to myself that this time I'll heed the warning
#in the CosmoMC code that, quote:
#!this is most certainly not the world's most efficient or 
#robust random rotation generator"
#unquote, and that I'll try and dig out a better one.
#And every time I spend about an hour looking, before
#coming back to this and translating it.
def random_rotation_matrix(n):
    R=np.identity(n)
    for j in range(n):
        while True:
            v = np.random.normal(size=n)
            for i in range(j):
                v -= R[i,:] * np.dot(v,R[i,:])
            L = np.dot(v,v)
            if (L>1e-3): break
        R[j,:] = v/L**0.5
    return R



def submatrix(M, x):
    """If x is an array of integer row/col numbers and M a matrix,
    extract the submatrix which is the all x'th rows and cols.
    i.e. A = submatrix(M,x) => A_ij = M_{x_i}{x_j}
    """
    return M[np.ix_(x,x)]


def invert_ordering(ordering):
    n = len(ordering)
    inverse = np.zeros_like(ordering)
    for i,j in enumerate(ordering):
        inverse[j] = i
    return inverse
