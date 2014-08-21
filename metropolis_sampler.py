from .. import ParallelSampler
import numpy as np
from . import metropolis
import os

#We need a global pipeline
#object for MPI to work properly
pipeline=None


METROPOLIS_INI_SECTION = "metropolis"

def posterior(p):
    return pipeline.posterior(p)


class MetropolisSampler(ParallelSampler):
    parallel_output = True
    sampler_outputs = [("like", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline
        self.samples = self.read_ini("samples", int, default=20000)
        random_start = self.read_ini("random_start", bool, default=False)
        self.Rconverge = None
        self.interrupted = False
        self.num_samples = 0
        #Any other options go here

        #start values from prior
        start = self.define_parameters(random_start)
        print "MCMC starting point:"
        for param, x in zip(self.pipeline.varied_params, start):
            print param, x
        self.n = self.read_ini("nstep", int, default=100)

        try:
            covmat = self.load_covariance_matrix()
        except IOError:
            covmat = None

        self.sampler = metropolis.MCMC(start, posterior, covmat)

    def worker(self):
        while not self.is_converged():
            self.execute()

    def execute(self):
        #Run the MCMC  sampler.
        try:
            samples = self.sampler.sample(self.n)
        except KeyboardInterrupt:
            self.interrupted=True
            return
        self.num_samples += self.n
        for vector, like in samples:
            self.output.parameters(vector, like)

        rate = self.sampler.accepted * 100.0 / self.sampler.iterations
        print "Accepted %d / %d samples (%.2f%%)" % \
            (self.sampler.accepted, self.sampler.iterations, rate)
        self.sampler.tune()

    def is_converged(self):
         # user has pressed Ctrl-C
        if self.interrupted:
            return True
        if self.num_samples >= self.samples:
            print "Full number of samples generated; sampling complete"
            return True
        elif self.num_samples > 0 and self.pool is not None and \
                self.Rconverge is not None:
            return np.all(self.analytics.gelman_rubin() <= self.Rconverge)
        else:
            return False



    def load_covariance_matrix(self):
        covmat_filename = self.read_ini("covmat", str, "").strip()
        if covmat_filename == "":
            return None
        if not os.path.exists(covmat_filename):
            raise ValueError(
            "Covariance matrix %s not found" % covmat_filename)
        covmat = np.loadtxt(covmat_filename)

        if covmat.ndim == 0:
            covmat = covmat.reshape((1, 1))
        elif covmat.ndim == 1:
            covmat = np.diag(covmat ** 2)

        nparams = len(self.pipeline.varied_params)
        if covmat.shape != (nparams, nparams):
            raise ValueError("The covariance matrix was shape (%d x %d), "
                    "but there are %d varied parameters." %
                    (covmat.shape[0], covmat.shape[1], nparams))
        return covmat



    def define_parameters(self, random_start):
        if random_start:
            return self.pipeline.randomized_start()
        else:
            return self.pipeline.start_vector()
