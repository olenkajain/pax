"""Write ROOT class
"""

import gzip
import glob
import os
import pickle
import re

import ROOT
import numpy as np
from pax import plugin
from array import array
from pax import datastructure

class ROOTClass(plugin.OutputPlugin):
    # Lookup dictionary for converting python numpy types to
    # ROOT types, strings are handled seperately!
    root_type = {'float32': '/F',
                 'float64': '/D',
                 'int32': '/I',
                 'int64': '/L',
                 'bool': '/O',
                 'S': '/C'}

    numpy_type = {'F': np.float32,
                  'D': np.float64,
                  'I': np.int32,
                  'L': np.int64,
                  'O': np.bool,
                  'C': np.dtype('object')}

    def startup(self):
        # This line makes sure all TTree objects are NOT owned
        # by python, avoiding segfaults when garbage collecting
        ROOT.TTree.__init__._creates = False

        self.log.fatal('wtf')
        self.f = ROOT.TFile("output.root",
                            "RECREATE")
        self.t = None

    def setup_tree(self, event):
        """Use first event
        """

        self.cpp_string = """
#include "TTree.h"
#include "TFile.h"
#include "TRandom.h"
#include "TTree.h"
#include "TROOT.h"
#include <vector>
#include <string>
#include <iostream>

/*class ReconstructedPosition {
 public:
   Peak(){}


  ClassDef(Peak,1);
};*/

class Peak {
 public:
   Peak(){}
%s

  ClassDef(Peak,1);
};

#ifdef __MAKECINT__
//#pragma link C++ class Position+;
//#pragma link C++ class vector<Position>+;
#pragma link C++ class Peak+;
#pragma link C++ class vector<Peak>+;
#endif

vector <Peak> dummy;
"""
        peak_string = ""
        self.t = ROOT.TTree('T', 'event tree')
        self.log.debug("Creating event tree")
        self.branch_buffers = {}
        for name, value in event.get_fields_data():

            if type(value) == int:
                self.log.fatal(('yes', type(value)))
                self.branch_buffers[name] = np.zeros(1, np.int32)
                self.t.Branch(name,
                              self.branch_buffers[name],
                                                 name + '/I')
            elif type(value) == float:
                self.log.fatal(('yes', type(value)))
                self.branch_buffers[name] = np.zeros(1, np.float32)
                self.t.Branch(name,
                              self.branch_buffers[name],
                                                 name + '/F')
            if name == 'peaks':
                for name2, value2 in value[0].get_fields_data():
                    if isinstance(value2, float):
                        peak_string += "float " + name2 + ";\n"
                    elif isinstance(value2, int):
                        peak_string += "int " + name2 + ";\n"
                    elif isinstance(value2, str):
                        peak_string += "string " + name2 + ";\n"

                    #if name2 == 'reconstructedposition':

            else:
                self.log.fatal(('no', type(value)))

        ROOT.gROOT.ProcessLine(self.cpp_string % peak_string)
        self.peaks = ROOT.vector('Peak')()
        self.t.Branch('peaks', self.peaks)

    def write_event(self, event):
        if self.t == None:
            self.setup_tree(event)

        self.peaks.clear()

        for name, value in event.get_fields_data():
            if isinstance(value, (int, float)):
                self.branch_buffers[name][0] = value

            if name == 'peaks':
                for peak_pax in value:
                    peak_root = ROOT.Peak()
                    for name2, value2 in peak_pax.get_fields_data():
                        if isinstance(value2, (int, float, str)):
                            setattr(peak_root, name2, value2)

                    self.peaks.push_back(peak_root)

        self.t.Fill()
        self.log.debug('Writing event')

    def shutdown(self):
        self.t.Write()
        self.f.Close()
        pass