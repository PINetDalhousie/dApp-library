#!/usr/bin/env python3
"""
dApp for Spectrum Sharing
"""

__author__ = "Conrado Boeira"

import multiprocessing
import time
import numpy as np
import threading
import os
import tensorflow as tf
# np.set_printoptions(threshold=sys.maxsize)

from dapp.dapp import DApp
from e3interface.e3_logging import dapp_logger, LOG_DIR
MODEL_PATH = '/users/grad/boeira/dApp/src/model_files/xcept_trained_model.keras'
NORMALIZATION_PARAMS_PATH = os.path.join(os.path.dirname(MODEL_PATH), 'xcept_normalization_params.npy')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Show more TF warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
#tf.compat.v1.disable_eager_execution()

class XceptDApp(DApp):

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
        self.iq_values = np.zeros(self.FFT_SIZE, dtype=np.complex128)

        tf.keras.backend.clear_session()

        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        self.model.make_predict_function()
        #self.graph = tf.get_default_graph()


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

    def create_spectrogram(self, iq_samples):
        """Convert a single chunk of IQ samples to a spectrogram."""
        from scipy import signal
        
        # No need to extract I and Q separately as we'll process the complex samples directly
        
        # Generate single spectrogram
        
        # Create spectrogram using scipy.signal
        f, t, Sxx = signal.spectrogram(iq_samples, nperseg=71, noverlap=64)
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
        
        # Normalize to [0, 1] range
        Sxx_norm = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db) + 1e-10)
        
        # Create 3-channel RGB image by duplicating the spectrogram
        spec_rgb = np.stack([Sxx_norm, Sxx_norm, Sxx_norm], axis=-1)
        
        # Return as a batch of one for model compatibility
        return np.array([spec_rgb])

    def process_iqs(self, thread_id=0, seq_number=0):
        norm_params = np.load(NORMALIZATION_PARAMS_PATH, allow_pickle=True).item()
        mean = norm_params['mean']
        std = norm_params['std']
        # Apply normalization
        #iq_comp = (self.iq_values- mean) / std
        #iq_comp = iq_comp.reshape(-1, self.FFT_SIZE)

        spectrum = self.create_spectrogram(self.iq_values)
        #spectrum = magnitude.reshape(-1, self.FFT_SIZE)
        #iq_comp = (iq_comp - mean) / std

        #print(f"Input dtype: {self.model.input.dtype}")
        #input_tensor = tf.convert_to_tensor(spectrum, dtype=tf.float32)
        predictions = self.model(spectrum, training=False).numpy()
    
        # Process predictions
        dapp_logger.info(f"FINISHED PROCESSING IQs | Thread {self.id} | Sequence Number {seq_number}")

        binary_predictions = (predictions > 0.5).astype(int)
        # Create the payload
        size = binary_predictions.size.to_bytes(2,'little')
        prbs_to_send = b'\x00' * binary_predictions.size
        
        size = b'\x00'
        #prbs_to_send = b'\x00\x00\x00\x00'

        # Schedule the delivery
        dapp_logger.info(f"FINISHED CREATING CONTROL | Thread {self.id} | Sequence Number {seq_number}")
        self.e3_interface.schedule_control(size+prbs_to_send)

        if self.energyGui:
            self.sig_queue.put(predictions)
        
        if self.dashboard:
            self.demo_queue.put(("binary_predictions", binary_predictions))

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
            iq_comp = iq_arr[::2] + iq_arr[1::2] * 1j
            #dapp_logger.debug("Start control operations")
            #dapp_logger.debug(f"Shape of iq_comp {iq_comp.shape}")
            #dapp_logger.debug(f"After iq division self.iq_values: {self.iq_values.shape} abs_iq: {abs_iq.shape}")
            #self.iq_values += samples
            #abs_iq = np.abs(iq_comp).astype(float)
            #dapp_logger.debug(f"After iq division self.abs_iq_av: {self.abs_iq_av.shape} abs_iq: {abs_iq.shape}")
            #self.iq_values += abs_iq
            self.iq_values += iq_comp
            self.control_count += 1
            #dapp_logger.debug(f"Control count is: {self.control_count}")

            if self.control_count == self.Average_over_frames:
                self.process_iqs(self.id, seq_number)

                # Reset the variables
                self.iq_values = np.zeros(self.FFT_SIZE, dtype=np.complex128)
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

