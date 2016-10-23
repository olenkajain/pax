"""Write to Amazon dynamoDB
"""
import boto3
import decimal
from pax import plugin


class WriteDynamoDB(plugin.OutputPlugin):
    do_input_check = False
    do_output_check = False

    def startup(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table('reduced')

    def write_event(self, event):
        self.write_event_reduced(event)

    def write_event_full(self, event):
        doc = event.to_dict(convert_numpy_arrays_to='list',
                            nan_to_none=True,
                            fields_to_ignore=self.config['fields_to_ignore'],
                            use_decimal=True)
        
        doc['peaks'] = [peak for peak in doc['peaks'] if peak['area'] > 100]

        self.table.put_item(Item=doc)
    
    def write_event_reduced(self, event):

        # Template document with default values
        insert_doc = {            
            "event_number": None,
            "dataset_name": None,
            "run_number": int(self.config['run_number']),
            "cs1": None,
            "cs2": None,
            "s1": None,
            "s2": None,
            "s1_range_50p_area": None,
            "s2_range_50p_area": None,
            "s1_area_fraction_top": None,
            "s2_area_fraction_top": None,
            "s1_n_hits": None,
            "s2_n_hits": None,
            "s1_hit_time_mean": None,
            "s2_hit_time_mean": None,
            "event_time": None,
            "z": None,
            "largest_other_s1": None,
            "largest_other_s2": None,
            "s1_n_contributing_channels": None,
            "s2_n_contributing_channels": None,
            "s1_n_saturated_channels": None,
            "s2_n_saturated_channels": None,
            "event_length": None,
            "largest_other_s1_hit_time_mean": None,
            "largest_other_s2_hit_time_mean": None,
            "ns1": None,
            "ns2": None,
            "dt": None,
            "x": None,
            "y": None,
            "interactions": None,
        }

        s1s = event.s1s()
        s2s = event.s2s()
        peaks = event.peaks
        interactions = event.interactions
        
        insert_doc['ns1'] = len(s1s)
        insert_doc['ns2'] = len(s2s)
        insert_doc['event_time'] = event.start_time
        insert_doc['event_number'] = event.event_number
        insert_doc['event_length'] = event.stop_time - event.start_time
        insert_doc['interactions'] = len(interactions)
        insert_doc['dataset_name'] = event.dataset_name

        if len(interactions) > 0:
            interaction = interactions[0]
            s1 = peaks[interaction.s1]
            s2 = peaks[interaction.s2]

            insert_doc['s1'] = s1.area
            insert_doc['s2'] = s2.area
            insert_doc['cs1'] = insert_doc['s1']*interaction.s1_area_correction
            insert_doc['cs2'] = insert_doc['s2']*interaction.s2_area_correction
            insert_doc['x'] = interaction.x
            insert_doc['y'] = interaction.y
            insert_doc['z'] = interaction.z
            insert_doc['dt'] = interaction.drift_time
            insert_doc['s1_range_50p_area'] = s1.range_area_decile[5]
            insert_doc['s2_range_50p_area'] = s2.range_area_decile[5]
            insert_doc['s1_hit_time_mean'] = s1.hit_time_mean
            insert_doc['s2_hit_time_mean'] = s2.hit_time_mean
            insert_doc['s1_area_fraction_top'] = s1.area_fraction_top
            insert_doc['s2_area_fraction_top'] = s2.area_fraction_top
            insert_doc['s1_n_hits'] = s1.n_hits
            insert_doc['s2_n_hits'] = s2.n_hits
            insert_doc['s1_n_contributing_channels'] = s1.n_contributing_channels
            insert_doc['s2_n_contributing_channels'] = s2.n_contributing_channels
            insert_doc['s1_n_saturated_channels'] = s1.n_saturated_channels
            insert_doc['s2_n_saturated_channels'] = s2.n_saturated_channels

            # Now we want the largest other s1 and largest other s1 
            los1 = None
            alos1 = 0
            for s1 in s1s:
                if s1 == interaction.s1:
                    continue
                try:
                    if peaks[s1].area > alos1:
                        alos1 = peaks[s1].area
                        los1 = peaks[s1]
                except:
                    if s1.area > alos1:
                        alos1 = s1.area
                        los1 = s1

            los2 = None
            alos2 = 0
            for s2 in s2s:
                if s2 == interaction.s2:
                    continue
                try:
                    if peaks[s2].area > alos2:
                        alos2 = peaks[s2].area
                        los2 = peaks[s2]
                except:
                    if s2.area > alos2:
                        alos2 = s2.area
                        los2 = s2

            if los1 is not None:
                insert_doc['largest_other_s1'] = los1.area
                insert_doc['largest_other_s1_range_50p_area'] = los1.range_area_decile[5]
                insert_doc['largest_other_s1_hit_time_mean'] = los1.hit_time_mean
                insert_doc['largest_other_s1_n_contributing_channels'] = los1.n_contributing_channels
            if los2 is not None:
                insert_doc['largest_other_s2'] = los2.area
                insert_doc['largest_other_s2_range_50p_area'] = los2.range_area_decile[5]
                insert_doc['largest_other_s2_hit_time_mean'] = los2.hit_time_mean
                insert_doc['largest_other_s2_n_contributing_channels'] = los2.n_contributing_channels
        
        for key in insert_doc.keys():
            if isinstance(insert_doc[key], float):
                insert_doc[key] = decimal.Decimal("%f" % insert_doc[key])
            
    
        self.table.put_item(Item=insert_doc)


