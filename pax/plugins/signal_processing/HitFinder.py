"""Hit finding plugin

If you get an error from either of the numba methods in this plugin (exception from native function blahblah)
Try commenting the @jit decorators, which will run a slow, pure-python version of the methods, allowing you to debug.
Don't forget to re-enable the @jit -- otherwise it will run quite slow!
"""


import numpy as np
import numba

# For diagnostic plotting:
import matplotlib.pyplot as plt
import os

from pax import plugin, datastructure, units


class FindHits(plugin.TransformPlugin):

    def startup(self):
        c = self.config

        self.reference_baseline = c.get('digitizer_reference_baseline', 16000)
        self.baseline_sample_length = c.get('baseline_sample_length', 40)
        self.max_hits_per_pulse = c['max_hits_per_pulse']

        self.make_diagnostic_plots = c.get('make_diagnostic_plots', 'never')
        self.make_diagnostic_plots_in = c.get('make_diagnostic_plots_in', 'small_pf_diagnostic_plots')
        if self.make_diagnostic_plots != 'never':
            if not os.path.exists(self.make_diagnostic_plots_in):
                os.makedirs(self.make_diagnostic_plots_in)

        self.dynamic_threshold_multiplier = self.config['dynamic_threshold_multiplier']
        self.base_threshold = self.config['base_threshold']
        self.detection_threshold_filter = self.config['detection_threshold_filter']
        self.bound_threshold = self.config['bound_threshold']

        self.detrend_baseline = self.config['detrend_baseline']

        # Conversion factor: multiply by this to convert from ADC counts above baseline -> electrons
        # Still has to be divided by PMT gain to go to photo-electrons (done below)
        self.adc_to_e = c['sample_duration'] * c['digitizer_voltage_range'] / (
            2 ** (c['digitizer_bits']) *
            c['pmt_circuit_load_resistor'] *
            c['external_amplification'] *
            units.electron_charge)

        # Keep track of how many times the "too many hits" warning has been shown
        self.too_many_hits_warnings_shown = 0

    def transform_event(self, event):
        # Allocate numpy arrays to hold numba peakfinder results
        # -1 is a placeholder for values that should never be used
        hits_buffer = -1 * np.ones((self.max_hits_per_pulse, 2), dtype=np.int64)
        argmaxes = -1 * np.zeros(self.max_hits_per_pulse, dtype=np.int64)
        areas = -1 * np.ones(self.max_hits_per_pulse)

        for pulse_i, pulse in enumerate(event.occurrences):
            start = pulse.left
            stop = pulse.right
            channel = pulse.channel

            # Don't consider dead channels
            if self.config['gains'][channel] == 0:
                continue

            # Retrieve the waveform, subtract ref baseline, invert
            w = self.reference_baseline - pulse.raw_data.astype(np.float64)

            # Correct & detrend baseline
            if not self.detrend_baseline or len(w) < 2 * self.baseline_sample_length + 2:
                # Use front baseline sample only
                pulse.baseline = baseline = baseline_intercept = np.median(w[:self.baseline_sample_length])
            else:
                baseline_front = np.median(w[:self.baseline_sample_length])
                baseline_rear = np.median(w[-self.baseline_sample_length:])
                pulse.baseline_slope = baseline_slope = (baseline_rear-baseline_front) \
                    / (len(w) - self.baseline_sample_length)
                pulse.baseline = baseline_intercept = baseline_front - \
                    baseline_slope * 0.5 * self.baseline_sample_length
                baseline = baseline_slope * np.arange(len(w)) + baseline_intercept
            w -= baseline

            # Determine the base noise level (- mean of samples < 0 in the initial baseline sample)
            # Median would be more robust against outliers... but we don't want robustness here:
            # If the noise is long-tailed / nongaussian, it means trouble!
            pulse.noise_level = noise_level = - np.mean(w[:self.baseline_sample_length][
                w[:self.baseline_sample_length] < 0])

            # Determine the detection threshold
            # base threshold * noise_level + multiplier * (w with all positive samples clipped to 0, smoothened)
            if len(w) < len(self.detection_threshold_filter):
                # Not enough samples for convolution, just take average
                detection_threshold = -np.mean(w[w < 0])*np.ones_like(w)
            else:
                detection_threshold = - self.convolve_padboundary(np.clip(w, float('-inf'), 0),
                                                                  self.detection_threshold_filter)
            detection_threshold *= self.dynamic_threshold_multiplier
            detection_threshold += self.base_threshold * noise_level

            # Call the numba hit finder -- see its documentation below
            # Results stored in hits_buffer
            n_hits_found = self._find_hits(w, detection_threshold, self.bound_threshold * noise_level, hits_buffer)
            n_hits_found = int(n_hits_found)    # see comments in numba hit finder below

            # Show too-many hits message
            if n_hits_found >= self.max_hits_per_pulse:
                if self.too_many_hits_warnings_shown > 3:
                    show_to = self.log.debug
                else:
                    show_to = self.log.info
                show_to("Pulse %s-%s in channel %s has more than %s hits. "
                        "This usually indicates a zero-length encoding breakdown after a very large S2. "
                        "Further hits in this pulse have been ignored." % (start, stop, channel,
                                                                           self.max_hits_per_pulse))
                self.too_many_hits_warnings_shown += 1
                if self.too_many_hits_warnings_shown == 3:
                    self.log.info('Further too-many hit messages will be suppressed!')

            # If no hits were found, this is a noise pulse: update the noise pulse count
            if n_hits_found == 0:
                event.noise_pulses_in[channel] += 1
                # Don't 'continue' to the next pulse! There's stuff left to do!

            # Only view the part of hits_buffer that contains peaks found in this event
            # The rest of hits_buffer contains zeros or random junk from previous pulses
            hits_found = hits_buffer[:n_hits_found]

            # TODO: compute and store pulse height

            # Compute area and max of each hit
            # Results stored in argmaxes, areas; declared outside loop, see above
            self._peak_argmax_and_area(w, hits_found, argmaxes, areas)

            # Store the found peaks in the datastructure
            # Convert area, noise_level and height from adc counts -> pe
            adc_to_pe = self.adc_to_e / self.config['gains'][channel]
            for i, hit in enumerate(hits_found):

                # Compute hit properties
                area = areas[i] * adc_to_pe
                height = w[hit[0] + argmaxes[i]]
                det_thr_at_max = detection_threshold[hit[0] + argmaxes[i]]
                if det_thr_at_max <= 0:
                    self.log.warning("Detection threshold at hit maximum is %s "
                                     "in pulse %d, channel %d, event %d! "
                                     "Indicates strange noise, hitfinder bug, or both." % (
                                         det_thr_at_max, pulse_i, channel, event.event_number))
                    height_over_threshold = float('inf')
                else:
                    height_over_threshold = height/detection_threshold[hit[0] + argmaxes[i]]
                height *= adc_to_pe
                left = start + hit[0]
                right = start + hit[1]
                max_idx = start + hit[0] + argmaxes[i]

                # Do sanity checks
                if not (0 <= left <= max_idx <= right) or not (0 <= height <= area):
                    raise RuntimeError("You found a hitfinder bug!\n"
                                       "Current hit %d-%d-%d, in event %s, channel %s, pulse %s.\n"
                                       "Indices in pulse: %s-%s-%s. Pulse bounds: %d-%d.\n"
                                       "Height is %s, noise sigma is %s, det.tr.at max %s; Area is %d.\n"
                                       "Please file a bug report!" % (
                                           left, max_idx, right, event.event_number, channel, pulse_i,
                                           hit[0], hit[0] + argmaxes[i], hit[1], start, stop,
                                           height, noise_level * adc_to_pe, det_thr_at_max, area))

                event.all_channel_peaks.append(datastructure.ChannelPeak({
                    'channel':               channel,
                    'left':                  left,
                    'index_of_maximum':      max_idx,
                    'right':                 right,
                    'area':                  area,
                    'height':                height,
                    'height_over_threshold': height_over_threshold,
                    'found_in_pulse':        pulse_i,
                }))

            # Diagnostic plotting
            # Can't more to plotting plugin: occurrence grouping of hits lost after clustering
            if self.make_diagnostic_plots == 'always' or \
               self.make_diagnostic_plots == 'no peaks' and not len(hits_found) or \
               self.make_diagnostic_plots == 'tricky cases' and (not len(hits_found) or
                                                                 len(hits_found) == 1 and
                                                                 event.all_channel_peaks[-1].height_over_threshold <
                                                                 1.2):
                plt.figure(figsize=(10, 7))
                plt.plot(w, drawstyle='steps', label='Data')
                if self.detrend_baseline:
                    plt.plot(w + baseline - baseline_intercept, drawstyle='steps', color='gray',
                             label='Data (not detrended)')
                plt.plot(detection_threshold, label='Detection threshold', color='red', drawstyle='steps')
                for hit in hits_found:
                    plt.axvspan(hit[0] - 1, hit[1], color='red', alpha=0.5)
                plt.plot(5 * np.ones_like(w) * noise_level, '--', label='5 * Noise level', color='orange')
                # TODO: don't draw another line, draw another y-axis!
                plt.plot(0.35 * np.ones_like(w) * self.config['gains'][channel] / self.adc_to_e,
                         '--', label='0.35 pe/sample', color='green')
                plt.legend()
                bla = (event.event_number, start, stop, channel)
                plt.title('Event %s, occurrence %d-%d, Channel %d' % bla)
                plt.xlabel("Sample number (%s ns)" % event.sample_duration)
                plt.ylabel("Amplitude (ADC counts above baseline)")
                plt.savefig(os.path.join(self.make_diagnostic_plots_in,
                                         'event%04d_occ%05d-%05d_ch%03d.png' % bla))
                plt.close()

        return event

    # TODO: Needs a test!
    @staticmethod
    @numba.jit(numba.void(numba.float64[:], numba.int64[:, :], numba.int64[:], numba.float64[:]), nopython=True)
    def _peak_argmax_and_area(w, raw_peaks, argmaxes, areas):
        """Finds the maximum index and area of peaks in w indicated by (left, right) bounds in raw_peaks.
        Will fill up argmaxes and areas with result.
            raw_peaks should be a numpy array of (left, right) bounds (inclusive)
        Returns nothing
        """
        for peak_i in range(len(raw_peaks)):
            current_max = -999.9
            current_argmax = -1
            current_area = 0
            for i, x in enumerate(w[raw_peaks[peak_i, 0]:raw_peaks[peak_i, 1]+1]):
                if x > current_max:
                    current_max = x
                    current_argmax = i
                current_area += x
            argmaxes[peak_i] = current_argmax
            areas[peak_i] = current_area

    @staticmethod
    @numba.jit(numba.float64(numba.float64[:],
                             numba.float64[:],
                             numba.float64,
                             numba.int64[:, :]), nopython=True)
    def _find_hits(w, detection_threshold, bound_threshold, hits_buffer):
        """Fills hits_buffer with left & right, inclusive indices of hits
        Hits = intervals > bound_threshold which somewhere cross detection_threshold.
         - detection threshold: numpy array of same size as w, contains threshold to use at each point
         - bound_threshold: float, hit ends when drops below this
         - hits_buffer: 2d numpy array of [-1,-1] pairs, will be filled by function.
        Returns: number of hits found
        Caution: will stop search after hits_buffer runs out!
        """
        assert len(w) == len(detection_threshold)
        in_candidate_interval = False
        current_interval_passed_test = False
        current_hit = 0
        max_n_hits = len(hits_buffer)      # First index which is outside hits buffer = max # of peaks to find
        max_idx = len(w) - 1
        current_candidate_interval_start = -1

        ##
        #   Hit finding
        ##

        for i, x in enumerate(w):

            if not in_candidate_interval and x > bound_threshold:
                # Start of candidate interval
                in_candidate_interval = True
                current_candidate_interval_start = i

            # This must be if, not else: an interval can cross threshold in start sample
            if in_candidate_interval:

                if x > detection_threshold[i]:
                    current_interval_passed_test = True

                if x < bound_threshold or i == max_idx:

                    # End of candidate interval
                    in_candidate_interval = False

                    if current_interval_passed_test:
                        # We've found a new peak!

                        # The interval ended just before this index
                        # unless, of course, we ended ONLY BECAUSE this is the last index
                        itv_end = i-1 if x < bound_threshold else i

                        # Add to hits_buffer
                        hits_buffer[current_hit, 0] = current_candidate_interval_start
                        hits_buffer[current_hit, 1] = itv_end

                        # Prepare for the next peak
                        current_hit += 1
                        current_interval_passed_test = False

                        # Check if we've reached the maximum # of peaks
                        # If we found more peaks than we have room in our result array,
                        # stop peakfinding immediately
                        if current_hit == max_n_hits:
                            break

        n_hits_found = current_hit

        # Return number of peaks found, baseline, noise sigma, and number of passes used
        # Convert ints to float, if you keep it int, it will sometimes be int32, sometimes int64 => numba crashes
        return float(n_hits_found)

    @staticmethod
    def convolve_padboundary(y, filt):
        """Returns convolution of y with filt with boundary values padded in the partially overlapping region"""
        if len(filt) % 2 == 0:
            raise ValueError("Filter length must be odd!")
        y_valid = np.convolve(y, filt, mode='valid')
        n_pad = (len(filt)-1)/2
        return np.concatenate((np.ones(n_pad)*y_valid[0], y_valid, y_valid[-1]*np.ones(n_pad)))
