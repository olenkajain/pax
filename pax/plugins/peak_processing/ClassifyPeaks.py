from pax import plugin, units
import numpy as np

# Parameters for classification function
switch1=30
switch2=150
slope1=0.25
slope2=0.05
slope3=0.07
offset=0.1

                  
# S1 height cut
def s1_height(s1area):
    x = s1area
    y =  ((x-offset)*slope1)*(x<switch1) + ((x>=switch1)&(x<switch2))*((switch1-offset)*slope1+slope2*(x-switch1)) + (x>switch2)*((x-switch2)*slope3+((switch1-offset)*slope1+slope2*(switch2-switch1)))
    return y

# S1 width cut
def s1_width(s1area):
    widthmax=100
    widthmin=20
    return widthmin + (widthmax-widthmin)* (1.0/(1+np.exp(-(s1area-60)/60))) 
    
# Qualify cut for S1 based on height and width
def peak_classification(area,width,height):
    if((width<=s1_width(area)) & (area>1.0) & (height>=s1_height(area))):
        return 's1'
    elif(area>3.5):
        return 's2'
    else:
        return 'unknown'


class AdHocClassification1T(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:
            # Don't work on noise and lone_hit
            if peak.type in ('noise', 'lone_hit'):
                continue

            area = peak.area
            width = peak.range_area_decile[5]
            height = peak.height

            peak.type = peak_classification(area,width,height)
        return event


class AdHocClassification(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:
            # Don't work on noise and lone_hit
            if peak.type in ('noise', 'lone_hit'):
                continue

            width = peak.range_area_decile[5]

            if peak.area > 50:
                # We don't have to worry about single electrons anymore
                if width < 100 * units.ns:
                    peak.type = 's1'
                elif width > 250 * units.ns:
                    peak.type = 's2'
            else:
                # Worry about SE-S1 identification.
                if width < 75 * units.ns:
                    peak.type = 's1'
                else:
                    if peak.area < 5:
                        peak.type = 'coincidence'
                    elif width > 100 * units.ns:
                        peak.type = 's2'

        return event
