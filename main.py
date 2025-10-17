from pathlib import Path
import sys

from PyQt5 import QtWidgets, uic
from pyqtgraph import PlotWidget
import numpy as np
import pyqtgraph as pg

from simulation import Sweep, Step, Space, Mechanism, Simulation


### PARAMETERS ###
# cyclic voltammetry waveform
Ei = -0.5 # V, initial potential
Ef = 0.5 # V, final potential
sr = 1 # V/s, sweep rate
dE = 0.01 # V, step size
N = 6 # number of sweeps

# chronoamperometry waveform
Es = 0.2 # V, constant potential
ts = 5 # s, step duration
dt = 0.01 # time interval between steps

# simulation properties
E0 = 0 # V, standard cell potential
n = 1 # number of electrons transferred
DO = 1e-5 # cm^2/s, diffusion coefficient of oxidized species (O)
DR = 1e-5 # cm^2/s, diffusion coefficient of reduced species (R)
cOb = 0 # M, bulk concentration of species O
cRb = 1e-6 # M, bulk concentration of species R
ks = 1e8 # cm/s, standard heterogeneous rate constant
a = 0.5 # transfer coefficient
Ageo = 1 # cm^2, geometric area of the electrode
BV = "QR" # kinetics type

### DISPLAY ###
# QT window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.wf = None
        self.space = None
        self.space = None
        self.mech = None
        self.sim = None
        ui_path = Path(__file__).resolve().parent / 'main.ui'
        uic.loadUi(str(ui_path), self)
        self.simulate()

    def simulate(self):
        # build waveforms
        CV_wf = Sweep([Ei, Ef, sr, dE, N])
        self.wf = CV_wf
        #CA_wf = Step([Es, ts, dt])
        #self.wf = Step(CA_params)
        print("Waveform initialized...")

        # build space, mechanism, simulation
        self.space = Space(self.wf)
        self.mech = Mechanism(self.wf, self.space, [E0, n, DO, DR, cOb, cRb, ks, a, BV])
        print("Environment initialized...")
        self.sim = Simulation(self.wf, self.space, self.mech, Ageo)
        self.sim.calculate_fd()
        print("Simulation complete...")
        self.plot(self.sim, -1)
        print("Plotting results...")

    def plot(self, sim, t):
        # potential vs. time
        self.vt_graph.setLabel('left', 'Potential', units='V')
        self.vt_graph.setLabel('bottom', 'Time', units='s')
        self.vt_graph.plot(sim.t[0:t], sim.E[0:t], pen=pg.mkPen('k', width=3), clear=True)

        # current vs. potential
        self.av_graph.setLabel('left', 'Current', units='A')
        self.av_graph.setLabel('bottom', 'Potential', units='V')
        self.av_graph.plot(sim.E[0:t], sim.i[0:t], pen=pg.mkPen('k', width=3), clear=True)

        # current vs. time
        self.at_graph.setLabel('left', 'Current', units='A')
        self.at_graph.setLabel('bottom', 'Time', units='s')
        self.at_graph.plot(sim.t[0:t], sim.i[0:t], pen=pg.mkPen('k', width=3), clear=True)

        # concentration vs. distance
        self.cd_graph.clear()
        self.cd_graph.addLegend()
        self.cd_graph.setLabel('left', 'Concentration', units='M')
        self.cd_graph.setLabel('bottom', 'Distance', units='m')
        self.cd_graph.plot(sim.x * 1e-2, sim.cR[t, :] * 1e3, pen=pg.mkPen('k', width=3), name='[R]')
        self.cd_graph.plot(sim.x * 1e-2, sim.cO[t, :] * 1e3, pen=pg.mkPen('r', width=3), name='[O]')

### MAIN ###
if __name__ == '__main__':
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
