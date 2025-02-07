"""
Stream2segment processing+visualization module saving raw and pre-processed segments
from the database on the file system.

{{ PROCESS_PY_MAIN }}
"""
import os
# OrderedDict is a python dict that returns its keys in the order they are inserted
# (a normal python dict returns its keys in arbitrary order)
# Useful e.g. in  "main" if we want to control the *order* of the columns in the output csv
from collections import OrderedDict
from datetime import datetime, timedelta  # always useful
from math import factorial  # for savitzky_golay function

# import numpy for efficient computation:
import numpy as np
# import obspy core classes (when working with times, use obspy UTCDateTime when possible):
from obspy import Trace, Stream, UTCDateTime
from obspy.geodetics import degrees2kilometers as d2km
# decorators needed to setup this module @gui.sideplot, @gui.preprocess @gui.customplot:
from stream2segment.process import gui, SkipSegment
# strem2segment functions for processing obspy Traces. This is just a list of possible functions
# to show how to import them:
from stream2segment.process.funclib.traces import ampratio, bandpass, cumsumsq,\
    timeswhere, fft, maxabs, utcdatetime, ampspec, powspec, timeof
# stream2segment function for processing numpy arrays:
from stream2segment.process.funclib.ndarrays import triangsmooth, snr


def assert1trace(stream):
    """asserts the stream has only one trace, raising an Exception if it's not the case,
    as this is the pre-condition for all processing functions implemented here.
    Note that, due to the way we download data, a stream with more than one trace his
    most likely due to gaps / overlaps"""
    # stream.get_gaps() is slower as it does more than checking the stream length
    if len(stream) != 1:
        raise SkipSegment("%d traces (probably gaps/overlaps)" % len(stream))


def main(segment, config):
    """{{ PROCESS_PY_MAINFUNC | indent }}
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]  # work with the (surely) one trace now

    # discard saturated signals (according to the threshold set in the config file):
    amp_ratio = ampratio(trace)
    if amp_ratio >= config['amp_ratio_threshold']:
        raise SkipSegment('possibly saturated (amp. ratio exceeds)')

    original_trace = trace.copy()  # keep a track of the original mseed

    # bandpass the trace, according to the event magnitude.
    # WARNING: this modifies the segment.stream() permanently!
    # If you want to preserve the original stream, store trace.copy() beforehand.
    # Also, use a 'try catch': sometimes Inventories are corrupted and obspy raises
    # a TypeError, which would break the WHOLE processing execution.
    # Raising a SkipSegment will stop the execution of the currently processed
    # segment only (logging the error message):
    try:
        processed_trace = bandpass_remresp(segment, config)
    except TypeError as type_error:
        raise SkipSegment("Error in 'bandpass_remresp': %s" % str(type_error))

    stream_path = segment.sds_path(config['root_dir'])
    basedir = os.path.dirname(stream_path)
    path1 = stream_path + ".s2s.raw.mseed"
    path2 = stream_path + ".s2s.processed.mseed"
    # assure directories exists, making all intermediate paths if needed:
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    original_trace.write(path1, format='MSEED')
    processed_trace.write(path2, format='MSEED')


@gui.preprocess
def bandpass_remresp(segment, config):
    """{{ PROCESS_PY_BANDPASSFUNC | indent }}
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]

    inventory = segment.inventory()

    # define some parameters:
    evt = segment.event
    conf = config['preprocess']
    # note: bandpass here below copied the trace! important!
    trace = bandpass(trace, mag2freq(evt.magnitude), freq_max=conf['bandpass_freq_max'],
                     max_nyquist_ratio=conf['bandpass_max_nyquist_ratio'],
                     corners=conf['bandpass_corners'], copy=False)
    trace.remove_response(inventory=inventory, output=conf['remove_response_output'],
                          water_level=conf['remove_response_water_level'])
    return trace


def mag2freq(magnitude):
    '''returns a frequency in Hz from a given magnitude'''
    if magnitude <= 4.5:
        freq_min = 0.4
    elif magnitude <= 5.5:
        freq_min = 0.2
    elif magnitude <= 6.5:
        freq_min = 0.1
    else:
        freq_min = 0.05
    return freq_min
