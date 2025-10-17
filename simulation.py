import numpy as np

### Constants ###
F = 96485 # C/mol, faraday's constant
R = 8.3145 # J/(mol·K), ideal gas constant
T = 298.15 # K, temperature (kelvin)
FRT = F / (R * T)

### Potential Waveforms ###
# Generates a sweep potential waveform for cyclic voltammetry (CV) simulation.
class Sweep:
    def __init__(self, params):
        # assign parameters
        self.Ei = params[0] # V, initial potential
        self.Ef = params[1] # V, final potential
        self.sr = params[2] # V/s, sweep rate
        self.dE = params[3] # V, potential step size
        self.N = params[4] # number of sweeps

        # sweep parameters
        Ewin = abs(self.Ef - self.Ei) # V, potential window
        tsw = Ewin / self.sr # s, time for one sweep
        nt = int(Ewin / self.dE) # number of steps per sweep

        # time array
        self.t = np.linspace(0, tsw * self.N, nt * self.N)

        # potential array construction
        E = []
        for n in range(1, self.N + 1):
            if n % 2 == 1: # forward sweep
                E.append(np.linspace(self.Ei, self.Ef, nt))
            else: # reverse sweep
                E.append(np.linspace(self.Ef, self.Ei, nt))
        self.E = np.concatenate(E)

# Generates a step potential waveform for chronoamperometry simulation
class Step:
    def __init__(self, params):
        # assign parameters
        self.Es = params[0] # V, step potential
        self.ts = params[1] # s. duration of the step
        self.dt = params[2] # s, time interval between points
        self.nt = int(self.ts / self.dt) # number of points in the time array

        # time array
        self.t = np.linspace(0, self.ts, self.nt)

        # potential array
        self.E = np.full(self.nt, self.Es)

### Initialize Environment ###
# Creates a 1D spatial grid for the PDE system
class Space:
    def __init__(self, wf, lamb=0.45):
        # dimensionless values for FDM
        self.lamb = lamb # dimensionless stability parameter
        self.nt = len(wf.t) # number of time elements
        dT = 1.0 / self.nt # dimensionless time step
        xmax = 6 * np.sqrt(self.nt * self.lamb) # cm, approx. maximum distance for the domain
        self.dx = np.sqrt(dT / self.lamb) # cm, spatial step size
        self.nx = int(xmax / self.dx) # number of spatial points

        # discretized spatial domain
        self.x = np.linspace(0, xmax, self.nx)

### MECHANISM ###
# Initializes electrochemical system and properties for simulation
class Mechanism:
    def __init__(self, wf, space, params):
        # spatial/time properties
        self.nt = space.nt # number of time points
        self.nx = space.nx # number of spatial points
        self.dx = space.dx # cm, spatial step size
        self.lamb = space.lamb # dimensionless stability parameter

        # reaction parameters
        self.E0 = params[0] # V, standard (formal) potential
        self.n = params[1] # mumber of electrons
        self.DO = params[2] # cm^2/s, diffusion coefficient (O)
        self.DR = params[3] # cm^2/s, diffusion coefficient (R)
        self.cOb = params[4] # M, bulk concentration (O)
        self.cRb = params[5] # M, bulk concentration (R)
        self.ks = params[6] # cm/s, heterogeneous rate constant
        self.alpha = params[7] # transfer coefficient
        self.BV = params[8] # kinetic model

        # derived properties
        self.delta = np.sqrt(self.DR * wf.t[-1])
        self.K0 = (self.ks * self.delta) / self.DR
        self.DOR = self.DO / self.DR

        # initialize concentration grids
        if self.cRb == 0: # If only O is present in the bulk
            # reduced species R = 0; oxidized species O = 1 (dimensionless)
            self.CR = np.zeros([self.nt, self.nx])
            self.CO = np.ones([self.nt, self.nx])
        else: # R is present in the bulk => fill with dimensionless R = 1
            self.CR = np.ones([self.nt, self.nx])
            # scale O by (cOb/cRb) if both are nonzero; else O=0 if cOb=0
            if self.cOb > 0:
                self.CO = np.ones([self.nt, self.nx]) * (self.cOb / self.cRb)
            else:
                self.CO = np.zeros([self.nt, self.nx])

### SIMULATION ###
# Simulates the electrochemical system in a 1D spatial domain
class Simulation:
    def __init__(self, wf, space, mec, Ageo=1):
        self.E = None
        self.t = None
        self.i = None
        self.cR = None
        self.cO = None
        self.x = None

        self.wf = wf
        self.space = space
        self.mec = mec
        self.Ageo = Ageo

        # dimensionless potential
        self.eps = (wf.E - mec.E0) * mec.n * FRT

        # concentration arrays
        self.CR = mec.CR
        self.CO = mec.CO

        # kinetics type
        self.BV = mec.BV

    # applies boundary conditions for R and O at the electrode surface (x=0)
    def apply_bc(self, CRS, COS, eps):
        dX = self.mec.dx
        K0 = self.mec.K0
        alpha = self.mec.alpha
        DOR = self.mec.DOR
        
        if self.BV == "QR":  # quasi-reversible (O <-> R)
            CR = (CRS + dX * K0 * np.exp(-alpha * eps) * (COS + CRS / DOR)) / (1 + dX * K0 * (np.exp((1 - alpha) * eps) + np.exp(-alpha * eps) / DOR))
            CO = COS + (CRS - CR) / DOR

        elif self.BV == "RO":  # irreversible reduction (R -> O)
            CR = CRS / (1 + dX * K0 * np.exp((1 - alpha) * eps))
            CO = COS + (CRS - CR) / DOR

        elif self.BV == "OR":  # irreversible oxidation (O -> R)
            CO = COS / (1 + dX * K0 * np.exp(-alpha * eps))
            CR = CRS + (COS - CO) / DOR

        else:
            raise ValueError(f"Invalid BV kinetics type: {self.BV}")

        return CR, CO

    # runs finite difference algo
    def calculate_fd(self):
        nt = self.space.nt
        nx = self.mec.nx

        for k in range(1, nt):
            # electrode-surface boundary condition at x=0, using the concentration at x=1 from previous time step k-1
            self.CR[k, 0], self.CO[k, 0] = self.apply_bc(self.CR[k - 1, 1], self.CO[k - 1, 1], self.eps[k])

            # interior points
            for j in range(1, nx - 1):
                # dimensionless update for R
                self.CR[k, j] = (self.CR[k - 1, j] + self.mec.lamb * (self.CR[k - 1, j + 1] - 2.0 * self.CR[k - 1, j] + self.CR[k - 1, j - 1]))
                # dimensionless update for O
                self.CO[k, j] = (self.CO[k - 1, j] + self.mec.DOR * self.mec.lamb * (self.CO[k - 1, j + 1] - 2.0 * self.CO[k - 1, j] + self.CO[k - 1, j - 1]))

            # far-boundary condition at x=nx-1
            self.CR[k, -1] = self.CR[k, -2]
            self.CO[k, -1] = self.CO[k, -2]

            # denorm
            self.denorm()

    # converts dimensionless results to physical units.
    def denorm(self):
        # decide which species is the reference, compute dimensionless flux at x = 0, and scale physical currents.
        if self.mec.cRb > 0:
            # R in bulk, dimensionless R = 1
            flux_dimless = (-self.CR[:, 2] + 4.0 * self.CR[:, 1] - 3.0 * self.CR[:, 0])
            D = self.mec.DR
            c_bulk = self.mec.cRb
            # physical concentrations
            cR = self.CR * self.mec.cRb
            if self.mec.cOb > 0:
                cO = self.CO * self.mec.cOb
            else:
                # cO dimensionless is 1 - CR, so scale by cRb
                cO = (1 - self.CR) * self.mec.cRb

        else:
            # only O in bulk, dimensionless O = 1
            flux_dimless = (self.CO[:, 2] - 4.0 * self.CO[:, 1] + 3.0 * self.CO[:, 0])
            D = self.mec.DO
            c_bulk = self.mec.cOb  # M
            # physical concentrations
            cO = self.CO * self.mec.cOb
            cR = (1 - self.CO) * self.mec.cOb

        # current in A:
        # i = n * F * A_geo * (D * c_bulk) * (dimensionless_flux) / (2·dx_dimless·delta)
        # where dx_dimless = self.space.dx (dimensionless), delta = mec.delta (cm),
        # and flux_dimless is the dimensionless derivative at x=0.
        # the factor of 2 in the denominator comes from the 3-point difference formula.
        dx_dimless = self.space.dx
        delta_cm = self.mec.delta

        i = (self.mec.n * F * self.Ageo * D * c_bulk * flux_dimless / (2.0 * dx_dimless * delta_cm))

        # physical x-array in cm
        x = self.space.x * delta_cm

        # store or update results
        self.E = self.wf.E
        self.t = self.wf.t
        self.i = i
        self.cR = cR
        self.cO = cO
        self.x = x
