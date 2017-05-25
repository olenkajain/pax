import numpy as np
from pax import plugin, exceptions, dsputils
from pax.dsputils import saturation_correction
import pandas as pd

# Must be run before 'BuildInteractions.BasicInteractionProperties'


class S2SpatialCorrection(plugin.TransformPlugin):
    """Compute S2 spatial(x,y) area correction
    """

    def startup(self):
        if 'xy_posrec_preference' not in self.config:
            raise ValueError('Configuration for %s must contain xy_posrec_preference' % self.name)
        self.s2_light_yield_map = self.processor.simulator.s2_light_yield_map

    def transform_event(self, event):

        for peak in event.peaks:
            # check that there is a position
            if not len(peak.reconstructed_positions):
                continue
            else:
                try:
                    # Get x,y position from peak
                    xy = peak.get_position_from_preferred_algorithm(self.config['xy_posrec_preference'])

                    # S2 area correction: divide by relative light yield at the position
                    peak.s2_spatial_correction /= self.s2_light_yield_map.get_value_at(xy)
                    if 'map_top'in self.s2_light_yield_map.map_names:
                        peak.s2_top_spatial_correction /= self.s2_light_yield_map.get_value_at(xy,
                                                                                               map_name='map_top')
                        peak.s2_bottom_spatial_correction /= self.s2_light_yield_map.get_value_at(xy,
                                                                                                  map_name='map_bottom')
                except ValueError:
                    self.log.debug("Could not find any position from the chosen algorithms")
        return event


class S2SaturationCorrection(plugin.TransformPlugin):
    """Compute S2 saturation(x,y,pmtpattern) area correction
    """

    def startup(self):
        self.s2_patterns = self.processor.simulator.s2_patterns
        self.zombie_pmts_s2 = np.array(self.config.get('zombie_pmts_s2', []))

    def transform_event(self, event):

        for peak in event.peaks:
            # check that there is a position
            if not len(peak.reconstructed_positions):
                continue
            try:
                # Get x,y position from peak
                xy = peak.get_position_from_preferred_algorithm(self.config['xy_posrec_preference'])
            except ValueError:
                self.log.debug("Could not find any position from the chosen algorithms")
                continue
            try:
                peak.s2_saturation_correction *= saturation_correction(
                    peak=peak,
                    channels_in_pattern=self.config['channels_top'],
                    expected_pattern=self.s2_patterns.expected_pattern((xy.x, xy.y)),
                    confused_channels=np.union1d(peak.saturated_channels, self.zombie_pmts_s2),
                    log=self.log)
            except exceptions.CoordinateOutOfRangeException:
                self.log.debug("Expected light pattern at coordinates "
                               "(%f, %f) consists of only zeros!" % (xy.x, xy.y))

        return event


class S2LinearityCorrection(plugin.TransformPlugin):
    """Inside a peak, the peak area is a sum of all hit area. This introduce a bias due to zero length suppression. In      this plugin, we consider to add all pulse values inside a peak to calculate the peak area, to test how much it affect the S2 linearity. 
    """
    def startup(self):
        self.reference_baseline = self.config['digitizer_reference_baseline']
    def transform_event(self, event):
        # Getting the pulses to construct a WF
        pulses = event.pulses
        p_left = np.array([p.left for p in pulses])
        p_right = np.array([p.right for p in pulses])
        p_channel = np.array([p.channel for p in pulses])
        p_baseline = np.array([p.baseline for p in pulses])
        p_id=np.array([pid for pid in range(0,len(pulses))])
        pulses_pd=pd.DataFrame({"p_id":p_id,"p_left":p_left,"p_right":p_right,"p_channel":p_channel, "p_baseline" : p_baseline})

        nPmts=248
        nPmtsTop = 127
        s2_sorted = list(sorted([p for p in event.peaks if p.type == 's2' and p.detector=='tpc'], key=lambda p: p.area, reverse = True))
        for peak in s2_sorted[:1]:
            if peak.type == 's2' and peak.area > 1000:
                start_sample=peak.left
                end_sample=peak.right

                wf_raw = np.zeros((nPmts, end_sample - start_sample))
                # select pulses inside this peak window
                pulses_pd = pulses_pd[(pulses_pd["p_left"] <= end_sample) & (pulses_pd["p_right"] >= start_sample)]
                # Start constructing the WFs
                
                for pmtid in range(0, nPmts):
                    index_all = pulses_pd[(pulses_pd.p_channel == pmtid)].p_id.values  # all pulses in this WF
                    adc_conversion = dsputils.adc_to_pe(self.config, channel=pmtid)

                    for index in index_all:
                        p = pulses[int(index)]
                        p_left = p.left
                        p_right = p.right
                        raw_data_adc = p.raw_data.astype(np.float64)
                        raw_data_adc = self.reference_baseline - raw_data_adc
                        raw_data_adc -= p.baseline
                        #print("TEST",raw_data_adc)
                        for sample in range(start_sample, end_sample):
                            if (p_left <= sample and p_right >= sample):
                                adc = raw_data_adc[sample - p_left]
                                wf_raw[pmtid, sample - start_sample] = adc * adc_conversion
                

                # Area per pmt w/o saturation correction
                area_pmt_after_correction = np.zeros(nPmts)
                areatop = 0.0

                for pmtid in range(0, nPmts):
                    for sample in range(start_sample, end_sample):
                        area_pmt_after_correction[pmtid] += wf_raw[pmtid, sample - start_sample]
                    peak.area_per_channel[pmtid] = area_pmt_after_correction[pmtid]

                    # Re compute area fraction on top
                    if (pmtid < nPmtsTop):
                        areatop += peak.area_per_channel[pmtid]
                
                print('area before correction: ', peak.area)
                area_tmp=peak.area
                peak.area = np.sum(peak.area_per_channel)
                print('area after correction: ', peak.area/area_tmp)

                if (peak.area > 0):
                    peak.area_fraction_top = areatop / peak.area

        return event
