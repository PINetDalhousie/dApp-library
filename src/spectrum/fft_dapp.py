#!/usr/bin/env python3
"""
dApp for Spectrum Sharing
"""

__author__ = "Conrado Boeira"

import multiprocessing
import time
import numpy as np
import threading
# np.set_printoptions(threshold=sys.maxsize)

from dapp.dapp import DApp
from e3interface.e3_logging import dapp_logger, LOG_DIR

class FFTDApp(DApp):

    ###  Configuration ###
    # gNB runs with BW = 40 MHz, with -E (3/4 sampling)
    # Center frequency = 3.6192 GHz
    # OAI sampling frequency 46.08e6
    # No SRS for now, hence in gNB config file set do_SRS = 0
    # gNB->frame_parms.ofdm_symbol_size = 1536
    # gNB->frame_parms.first_carrier_offset = 900
    # Noise floor threshold needs to be calibrated
    # We receive the symbols and average them over some frames, and do thresholding.

    def __init__(self, id: int = 1, noise_floor_threshold: int = 53, save_iqs: bool = False, control: bool = False, link: str = 'posix', transport:str = 'udc', **kwargs):
        super().__init__(link=link, transport=transport, id=int(id), **kwargs) 

        self.bw = 40.08e6  # Bandwidth in Hz
        self.center_freq = 3.6192e9 # Center frequency in Hz
        self.First_carrier_offset = 900
        self.Num_car_prb = 12
        self.prb_thrs = 75 # This avoids blacklisting PRBs where the BWP is scheduled (it’s a workaround bc the UE and gNB would not be able to communicate anymore, a cleaner fix is to move the BWP if needed or things like that)
        self.FFT_SIZE = 1536  
        self.Average_over_frames = 63
        self.noise_floor_threshold = noise_floor_threshold
        self.save_iqs = save_iqs
        self.e3_interface.add_callback(self.get_iqs_from_ran)
        if self.save_iqs:
            self.iq_save_file = open(f"{LOG_DIR}/iqs_{int(time.time())}.bin", "ab")
            self.save_counter = 0
            self.limit_per_file = 200
        self.control = control
        dapp_logger.info(f"Control is {'not ' if not self.control else ''}active")

        self.energyGui = kwargs.get('energyGui', False)
        self.iqPlotterGui = kwargs.get('iqPlotterGui', False)
        self.dashboard = kwargs.get('dashboard', False)

        self.control_count = 1
        self.abs_iq_av = np.zeros(self.FFT_SIZE)


        # Number of threads to be run to process the IQ samples
        # Simulates having multiple dApps running simultaneously
        self.n_threads = 1

        self.id = id
        print(f"ID {id}")

        if self.energyGui:
            from visualization.energy import EnergyPlotter
            self.sig_queue = multiprocessing.Queue() 
            self.energyPlotter = EnergyPlotter(self.FFT_SIZE, bw=self.bw, center_freq=self.center_freq) 

        if self.iqPlotterGui:
            from visualization.iq import IQPlotter
            self.iq_queue = multiprocessing.Queue() 
            iq_size = self.FFT_SIZE * 2 # double the size of ofdm_symbol_size since real and imaginary parts are interleaved
            self.iqPlotter = IQPlotter(buffer_size=500, iq_size=iq_size, bw=self.bw, center_freq=self.center_freq)    

        if self.dashboard:
            from visualization.dashboard import Dashboard
            self.demo_queue = multiprocessing.Queue()
            iq_size = self.FFT_SIZE * 2 # double the size of ofdm_symbol_size since real and imaginary parts are interleaved
            classifier = kwargs.get('classifier', None)
            self.demo = Dashboard(buffer_size=100, iq_size=iq_size, classifier=classifier) 

    def process_iqs(self, iq_comp, thread_id=0, seq_number=0):
        fft_result = np.fft.fft(iq_comp, n=self.FFT_SIZE)
        fft_magnitude = np.abs(fft_result)
        dapp_logger.info(f"FINISHED PROCESSING IQs | Thread {self.id} | Sequence Number {seq_number}")

        # Create the payload
        size = fft_magnitude.size.to_bytes(2,'little')
        prbs_to_send = fft_magnitude.tobytes(order="C")
        
        #size = b'4'
        #prbs_to_send = b'0000'

        # Schedule the delivery
        dapp_logger.info(f"FINISHED CREATING CONTROL | Thread {self.id} | Sequence Number {seq_number}")
        self.e3_interface.schedule_control(size+prbs_to_send, seq_number)

        if self.energyGui:
            self.sig_queue.put(fft_result)
        
        if self.dashboard:
            self.demo_queue.put(("fft_magnitude", fft_magnitude))

    def get_iqs_from_ran(self, data, seq_number):
        if self.save_iqs:
            dapp_logger.debug("I will write on the logfile iqs")
            self.save_counter += 1
            self.iq_save_file.write(data)
            self.iq_save_file.flush()
            if self.save_counter > self.limit_per_file:
                self.iq_save_file.close()
                self.iq_save_file = open(f"{LOG_DIR}/iqs_{int(time.time())}.bin", "ab")

        dapp_logger.info(f"PROCESSING IQs | Thread {self.id} | Sequence Number {seq_number}")

               
        iq_arr = np.frombuffer(data, dtype=np.int16)[:-2]
        
        if self.iqPlotterGui:
            self.iq_queue.put(iq_arr)

        if self.dashboard:
            self.demo_queue.put(("iq_data", iq_arr))

        if self.control:
            #dapp_logger.debug("Start control operations")
            iq_comp = iq_arr[::2] + iq_arr[1::2] * 1j
            #dapp_logger.debug(f"Shape of iq_comp {iq_comp.shape}")
            #abs_iq = np.abs(iq_comp).astype(float)
            #dapp_logger.debug(f"After iq division self.abs_iq_av: {self.abs_iq_av.shape} abs_iq: {abs_iq.shape}")
            #self.abs_iq_av += abs_iq
            self.control_count += 1
            #dapp_logger.debug(f"Control count is: {self.control_count}")

            if self.control_count == self.Average_over_frames:
                self.process_iqs(iq_comp, self.id, seq_number)

                # Reset the variables
                self.abs_iq_av = np.zeros(self.FFT_SIZE)
                self.control_count = 1  

    def _control_loop(self):
        if self.energyGui:
            abs_iq_av_db = self.sig_queue.get()
            self.energyPlotter.process_iq_data(abs_iq_av_db)

        if self.iqPlotterGui:
            iq_data = self.iq_queue.get()
            self.iqPlotter.process_iq_data(iq_data)

        if self.dashboard:
            message = self.demo_queue.get()
            self.demo.process_iq_data(message)

    def _stop(self):        
        if self.save_iqs:
            self.iq_save_file.close()
        
        if self.dashboard:
            self.demo.stop()

