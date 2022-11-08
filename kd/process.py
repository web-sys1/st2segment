"""
=========================================================================
Stream2segment processing+visualization module generating a segment-based
parametric table.
=========================================================================
A processing+visualization module implements the necessary code to process and
visualize downloaded data.
In the first case (data processing), edit this file and then, on the terminal:
- Run it as a script:
  `python <this_file_path>`
  (see section `if __name__ == "__main__"` at the end of the module)
- Run it within the `process` command: 
 `s2s process -p <this_file_path> -c <config_file_path>`
In the second case (data visualization), edit this file and then, to open the
graphical user interface (GUI) in your web browser, type on the terminal:
 `s2s show -p <this_file_path> -c <config_file_path>`
(`<config_file_path>` is the path of the associated a configuration file in YAML 
format. Optional with the `show` command).
You can also separate visualization and process routines in two different
Python modules, as long as in each single file the requirements described below 
are provided.
Processing
==========
When processing, Stream2segment will search for a so-called "processing function", i.e.
a function called "main":
```
def main(segment, config)
```
and execute the function on each selected segment (according to the 'segments_selection' 
parameter in the config). If you only need to run this module for processing (no
visualization), you can skip the remainder of this introduction and go to the
processing function documentation.
Visualization (web GUI)
=======================
When visualizing, Stream2segment will open a web page where the user can browse 
and visualize the data. When the `show` command is invoked with no argument, the page
will only show all database segments and their raw trace. Otherwise, Stream2segment 
will read the passed config and module, showing only selected segments (parameter 
'segments_selection' in the config) and searching for all module functions decorated with
either "@gui.preprocess" (pre-process function) or "@gui.plot" (plot functions).
IMPORTANT: any Exception raised  anywhere by any function will be caught and its message
displayed on the plot.
Pre-process function
--------------------
The function decorated with "@gui.preprocess", e.g.:
```
@gui.preprocess
def applybandpass(segment, config)
```
will be associated to a check-box in the GUI. By clicking the check-box,
all plots of the page will be re-calculated with the output of this function,
which **must thus return an ObsPy Stream or Trace object**.
All details on the segment object can be found here:
https://github.com/rizac/stream2segment/wiki/the-segment-object
Plot functions
--------------
The functions decorated with "@gui.plot", e.g.:
```
@gui.plot
def cumulative(segment, config)
```
will be associated to (i.e., its output will be displayed in) the plot below 
the main plot. All details on the segment object can be found here:
https://github.com/rizac/stream2segment/wiki/the-segment-object
You can also call @gui.plot with arguments, e.g.:
```
@gui.plot(position='r', xaxis={'type': 'log'}, yaxis={'type': 'log'})
def spectra(segment, config)
```
The 'position' argument controls where the plot will be placed in the GUI ('b' means 
bottom, the default, 'r' means next to the main plot, on its right) and the other two,
`xaxis` and `yaxis`, are dict (defaulting to the empty dict {}) controlling the x and y 
axis of the plot (for info, see: https://plot.ly/python/axes/).
When not given, axis types (e.g., date time vs numeric) will be inferred from the
function's returned value which *must* be a numeric sequence (y values) taken at 
successive equally spaced points (x values) in any of these forms:
- ObsPy Trace object
- ObsPy Stream object
- the tuple (x0, dx, y) or (x0, dx, y, label), where
    - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point.
      For time-series abscissas, UTCDateTime is quite flexible with several input
      formats. For info see:
      https://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html
    - dx (numeric or `timedelta`) is the sampling period. If x0 has been given as
      datetime or UTCDateTime object and 'dx' is numeric, its unit is in seconds
      (e.g. 45.67 = 45 seconds and 670000 microseconds). If `dx` is a timedelta object
      and x0 has been given as numeric, then x0 will be converted to UtcDateTime(x0).
    - y (numpy array or numeric list) are the sequence values, numeric
    - label (string, optional) is the sequence name to be displayed on the plot legend.
- a dict of any of the above types, where the keys (string) will denote each sequence
  name to be displayed on the plot legend (and will override the 'label' argument, if
  provided)
"""

from __future__ import division

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports). UNCOMMENT or REMOVE
# if you are working in Python3 (recommended):
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

import os

# From Python >= 3.6, dicts keys are returned (and thus, written to file) in the order
# they are inserted. Prior to that version, to preserve insertion order you needed to
# use OrderedDict:
from collections import OrderedDict
from datetime import datetime, timedelta  # always useful
from math import factorial  # for savitzky_golay function

from urllib.parse import urlparse

# import numpy for efficient computation:
import numpy as np
import pandas as pd
# import obspy core classes (when working with times, use obspy UTCDateTime when
# possible):
from obspy import Trace, Stream, UTCDateTime
from obspy.geodetics import degrees2kilometers as d2km
# decorators needed to setup this module @gui.preprocess @gui.plot:
from sdaas.core import trace_psd
from stream2segment.process import gui, SkipSegment, yaml_load, imap
# straem2segment functions for processing obspy Traces. This is just a list of possible
# functions to show how to import them:
from stream2segment.process.funclib.traces import ampratio, bandpass, cumsumsq,\
    timeswhere, fft, maxabs, utcdatetime, ampspec, powspec, timeof, sn_split
# stream2segment function for processing numpy arrays:
from stream2segment.process.funclib.ndarrays import triangsmooth, snr
from stream2segment.process.funclib.ndarrays import cumsumsq as _cumsumsq


################################
# Processing related functions #
################################


def assert1trace(stream):
    """Assert the stream has only one trace, raising an Exception if it's not the case,
    as this is the pre-condition for all processing functions implemented here.
    Note that, due to the way we download data, a stream with more than one trace his
    most likely due to gaps / overlaps
    """
    # stream.get_gaps() is slower as it does more than checking the stream length
    if len(stream) != 1:
        raise SkipSegment("%d traces (probably gaps/overlaps)" % len(stream))


def main(segment, config):
    """Main processing function. The user should implement here the processing for any
    given selected segment. Useful links:
    - Online tutorial (also available as Notebook locally with the command `s2s init`,
      useful for testing):
      https://github.com/rizac/stream2segment/wiki/using-stream2segment-in-your-python-code
    - `stream2segment.process.funclib.traces` (small processing library implemented in
       this program, most of its functions are imported here by default)
    - ObsPy Stream, Trace UTCDateTime objects (the latter is the object
      returned by all Trace and Stream datetime-based methods):
      https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html
      https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.html
      https://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html
    IMPORTANT: any exception raised here or from any sub-function will interrupt the
    whole processing routine with one special case: `stream2segment.process.SkipSegment`
    exceptions will be logged to file and the execution will resume from the next 
    segment. Raise them to programmatically skip a segment, e.g.:
    ```
    if segment.sample_rate < 60: 
        raise SkipSegment("segment sample rate too low")`
    ```
    Handling exceptions at any point of a time consuming processing is non trivial:
    some have to be skipped to save precious time, some must not be ignored and should
    interrupt the routine to fix critical errors.
    Therefore, we recommend to try to run your code on a smaller and possibly 
    heterogeneous dataset first: change temporarily the segment selection in the
    configuration file, and then analyze any exception raised, if you want to ignore 
    the exception, then you can wrap only  the part of code affected in a 
    "try ... catch" statement, and raise a `SkipSegment`.
    Also, please spend some time on refining the selection of segments: you might
    find that your code runs smoothly and faster by simply skipping certain segments in 
    the first place.
    :param: segment: the object describing a downloaded waveform segment and its metadata,
        with a full set of useful attributes and methods detailed here:
        https://github.com/rizac/stream2segment/wiki/the-segment-object
    :param: config: a dictionary representing the configuration parameters
        accessible globally by all processed segments. The purpose of the `config`
        is to encourage decoupling of code and configuration for better and more 
        maintainable code, avoiding, e.g., many similar processing functions differing 
        by few hard-coded parameters (this is one of the reasons why the config is
        given as separate YAML file to be passed to the `s2s process` command)
    :return: If the processing routine calling this function needs not to generate a
        file output, the returned value of this function, if given, will be ignored.
        Otherwise:
        * For CSV output, this function must return an iterable that will be written
          as a row of the resulting file (e.g. list, tuple, numpy array, dict. You must
          always return the same type of object, e.g. not lists or dicts conditionally).
          Returning None or nothing is also valid: in this case the segment will be
          silently skipped
          The CSV file will have a row header only if `dict`s are returned (the dict
          keys will be the CSV header columns). For Python version < 3.6, if you want
          to preserve in the CSV the order of the dict keys as the were inserted, use
          `OrderedDict`.
          A column with the segment database id (an integer uniquely identifying the
          segment) will be automatically inserted as first element of the iterable, 
          before writing it to file.
          SUPPORTED TYPES as elements of the returned iterable: any Python object, but
          we suggest to use only strings or numbers: any other object will be converted
          to string via `str(object)`: if this is not what you want, convert it to the
          numeric or string representation of your choice. E.g., for Python `datetime`s
          you might want to set `datetime.isoformat()` (string), for ObsPy `UTCDateTime`s
          `float(utcdatetime)` (numeric)
       * For HDF output, this function must return a dict, pandas Series or pandas
         DataFrame that will be written as a row of the resulting file (or rows, in case
         of DataFrame).
         Returning None or nothing is also valid: in this case the segment will be
         silently skipped.
         A column named 'segment_db_id' with the segment database id (an integer uniquely
         identifying the segment) will be automatically added to the dict / Series, or
         to each row of the DataFrame, before writing it to file.
         SUPPORTED TYPES as elements of the returned dict/Series/DataFrame: all types 
         supported by pandas: 
         https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes
         For info on hdf and the pandas library (included in the package), see:
         https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html
         https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-hdf5
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]  # work with the (surely) one trace now

    cml_max_diff_ = cml_max_diff(detrend(trace.copy()))  # copy trace (do not modify it)
    psd_periods = config['psd_periods']
    try:
       psd_values = trace_psd(trace, segment.inventory(), psd_periods)[0]
    except Exception as exc:
       raise SkipSegment('%s: %s' % (str(exc.__class__.__name__), str(exc)))

    # write stuff to csv:
    ret = OrderedDict()
    ret['snr'] = snr
    ret['PGA']
    ret['id'] = segment.id
    ret['cml_maxdiff'] = cml_max_diff_
    for per, val in zip(psd_periods, psd_values):
       ret['psd_%ss' % str(per)] = val
    ret['outlier'] = False  # change if labelled dataset (by default no labels)
    ret['evt_dist_deg'] = segment.event_distance_deg        # dist
    ret['duration_sec'] = trace.stats.endtime - trace.stats.starttime
    # ret['url'] = segment.data_center.dataselect_url
    ret['evt_id'] = segment.event.id
    ret['evt_url'] = segment.event.url
    ret['sta_id'] = segment.station.id
    ret['sta_url'] = segment.station.url
    ret['dc_id'] = segment.datacenter.id
    ret['dc_url'] = segment.datacenter.dataselect_url
    net, sta, loc, cha = segment.data_seed_id.split('.')
    ret['net'] = net
    ret['sta'] = net
    ret['loc'] = loc or ''
    ret['cha'] = cha
    ret['request_start'] = segment.request_start
    ret['request_end'] = segment.request_end
    # event metadata:
    # ret['ev_lat'] = segment.event.latitude
    # ret['ev_lon'] = segment.event.longitude
    # ret['ev_dep'] = segment.event.depth_km
    ret['mag'] = float(segment.event.magnitude)
    ret['mag_type'] = segment.event.mag_type

    return ret


def linreg(trace):
    # move import at module level if you use this function (not used anymore)
    from scipy.stats import linregress

    x = np.arange(0, trace.stats.npts, dtype=int)
    res = linregress(x, trace.data)
    return Trace(data=x*res.slope + res.intercept, header=trace.stats)


def cml_max_diff(trace):
    """
    Return a measure of how much is rough the cumulative of the given trace
    (lower number: cumulative smooth, higher numbers: rough).
    The idea is that either signal (i.e. with earthquake recorded) and noise
    traces have smooth cumulative. For abrupt changes (such as e.,g., spikes)
    the returned number should be high (tests revealed numbers < 120 for
    signal traces, ~=70 for noise traces, and > 300 for artificially created
    spikes).
    The returned number is given by:
    ```
        max(diff) / abs(mean(diff))
    ```
    (details here: https://stats.stackexchange.com/a/24610)
    where `diff` is the finite difference (approx. 1st derivative) of the
    cumulative of `trace`
    :param trace: the given trace
    :return: numeric number denoting the max diff of the cumulative
    """
    cml = _cumsumsq(trace.data, normalize=True)
    # cml = cumsumsq(trace, normalize=True, copy=False)
    # smoothness:  https://stats.stackexchange.com/a/24610
    diff = np.diff(cml)
    diff_mean = np.nanmean(diff)
    diff_abs_mean = np.abs(diff_mean)
    return np.nanmax(np.abs(diff)) / diff_abs_mean


def detrend(trace):
    return trace.detrend()


def append_instance(store, instance, evts, stas, dcs):
    evt_id, evt_url = instance['evt_id'], instance.pop('evt_url')
    mag, mag_type = instance.pop('mag'), instance.pop('mag_type')
    if evt_id not in evts:
        evts.add(evt_id)
        store.append('events', pd.DataFrame([{'id': evt_id, 'url': evt_url,
                                              'mag': mag,
                                              'mag_type': mag_type}]),
                     format='table', min_itemsize={'url': 100, 'mag_type': 5})

    sta_id, net, sta, sta_url = \
        instance['sta_id'], instance.pop('net'), instance.pop('sta'), instance.pop('sta_url')
    if sta_id not in stas:
        stas.add(sta_id)
        store.append('stations', pd.DataFrame([{'id': sta_id, 'net': net,
                                                'sta': sta, 'url': sta_url}]),
                     format='table', min_itemsize={'net': 2, 'sta': 5, 'url': 120})

    dc_id, dc_url = instance['dc_id'], instance.pop('dc_url')
    if dc_id not in dcs:
        dcs.add(dc_id)
        store.append('data_centers', pd.DataFrame([{'id': dc_id, 'url': dc_url}]),
                     format='table', min_itemsize={'url': 90})

    store.append('waveforms', pd.DataFrame([instance]),
                 format='table', min_itemsize={'loc': 2, 'cha': 3})


if __name__ == "__main__":
    # execute the code below only if this module is run as a script
    # (python <this_file_path>)
    root = os.path.abspath(os.path.dirname(__file__))

    # Example code TO BE EDITED before run
    # ------------------------------------
    config = yaml_load('kd/download.yaml')
    dburl = yaml_load('kd/download.yaml')['dburl']
    # segments to process (modify according to your needs). The variable
    # can also be a numeric list/numpy array of integers denoting the ID of
    # the segments to process. You can also read the selection from file or extract it
    # from the config above, if implemented therein
    segments_selection = config['segments_selection']

    # output file
    outfile = os.path.join(root, dburl[dburl.rfind('/')+1:] + '.hdf')
    # provide a log file path to track all skipped segment (SkipSegment exceptions).
    # Here we input the boolean True, which automatically creates a log file in the
    # same directory 'outfile' above. To skip logging, provide an empty string
    logfile = outfile + '.log'
    # show progressbar on the terminal and additional info
    verbose = True
    # overwrite existing outfile, if present. If True and outfile exists, already
    # processed segments will be skipped
    append = False
    # csv or hdf options. Type help(process) on terminal or notebook for details
    writer_options = {}
    # use sub-processes to speed up the routine
    multiprocess = True

    # from stream2segment.process import imap, process
    #
    # # run imap or process here. Example with process:
    # process(main, dburl, segments_selection=segments_selection, config=config,
    #         outfile=outfile, append=False, writer_options=writer_options,
    #         logfile=logfile, verbose=verbose, multi_process=multiprocess, chunksize=None)

    evts, stas, dcs = set(), set(), set()
    with pd.HDFStore(outfile, 'w') as store:
        try:
            for result in imap(main, dburl, segments_selection=segments_selection,
                               config=config,
                               logfile=logfile, verbose=verbose,
                               multi_process=multiprocess,
                               chunksize=None):
                append_instance(store, result, evts, stas, dcs)
        finally:
            store.close()


######################################
# GUI functions for displaying plots #
######################################


@gui.preprocess
def bandpass_remresp(segment, config):
    """Same as detrend(trace), used in the GUI as pre-processing.
    Modifies the Trace inplace
    :return: a Trace object (a Stream is also valid value for functions decorated with
        `@gui.preprocess`)
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]
    #  https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.detrend.html
    return detrend(trace)


@gui.plot('r', xaxis={'type': 'log'})  # , yaxis={'type': 'log'})
def spectra(segment, config):
    """
    Computes the signal and noise spectra, as dict of strings mapped to tuples (x0, dx, y).
    Does not modify the segment's stream or traces in-place
    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the tuples
    (f0, df, frequencies)
    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    from obspy.signal.spectral_estimation import get_nlnm, get_nhnm

    o_trace = segment.stream()[0]
    trace = o_trace  # divide_sensitivity(o_trace, segment.inventory())

    psd_periods = np.linspace(*config['psd_periods_gui'], endpoint=True)

    # compute psd values for both noise and signal:
    psd_s_y1 = psd(trace, segment.inventory(), psd_periods, obspy=True)
    psd_s_y2 = psd(trace, segment.inventory(), psd_periods)
    nlnm_x, nlnm_y = get_nlnm()
    nlnm_x, nlnm_y = nlnm_x[::-1], nlnm_y[::-1]
    nhnm_x, nhnm_y = get_nhnm()
    nhnm_x, nhnm_y = nhnm_x[::-1], nhnm_y[::-1]

    # sample at equally spaced periods. First get bounds:
    # period_min = 2.0 / trace.stats.sampling_rate
    # period_max = min(psd_n_x[-1] - psd_n_x[0], psd_s_x[-1] - psd_s_x[0])
    #
    # n_pts = config['num_psd_periods']  # 1024
    # periods = np.linspace(period_min, period_max, n_pts, endpoint=True)
    # psd_n_y = np.interp(np.log10(periods), np.log10(psd_n_x), psd_n_y)
    # psd_s_y = np.interp(np.log10(periods), np.log10(psd_s_x), psd_s_y)
    # nlnm_y = np.interp(np.log10(periods), np.log10(nlnm_x), nlnm_y)
    # nhnm_y = np.interp(np.log10(periods), np.log10(nhnm_x), nhnm_y)
    #

    x0, dx = psd_periods[0], psd_periods[1] - psd_periods[0]

    # replace NaNs with Nones:
    # psd_s_y1_nan = np.isnan(psd_s_y1)
    # if psd_s_y1_nan.any():
    #     psd_s_y1 = np.where(psd_s_y1_nan, None, psd_s_y1)
    # psd_s_y2_nan = np.isnan(psd_s_y2)
    # if psd_s_y2_nan.any():
    #     psd_s_y2 = np.where(psd_s_y2_nan, None, psd_s_y2)

    return {
        'PSD_obpsy': (x0, dx, psd_s_y1),
        'PSD_sdaas': (x0, dx, psd_s_y2),
        'nlnm': (x0, dx, np.interp(psd_periods, nlnm_x, nlnm_y)),
        'nhnm': (x0, dx, np.interp(psd_periods, nhnm_x, nhnm_y))
    }


def psd(tr, inventory, psd_periods, obspy=False):
    """Returns the tuple (psd_x, psd_y) values where the first argument is
    a numopy array of periods and the second argument is a numpy array of
    power spectrum values in Decibel
    """
    from obspy.signal.spectral_estimation import PPSD
    # tr = trace.trim(endtime=trace.stats.endtime + 60)
    if obspy:
        dt = (tr.stats.endtime.datetime - tr.stats.starttime.datetime).total_seconds()
        ppsd = PPSD(tr.stats, metadata=inventory, ppsd_length=int(dt))
        ppsd.add(tr)
        try:
            ppsd.psd_values[0]  # just a check (do we have 1 element?)
            val = np.interp(psd_periods, ppsd.period_bin_centers, ppsd.psd_values[0])
            val[psd_periods < ppsd.period_bin_centers[0]] = np.nan
            val[psd_periods > ppsd.period_bin_centers[-1]] = np.nan
            return val
        except IndexError:
            return []
        # return ppsd.period_bin_centers, ppsd.psd_values[0]
    else:
        return trace_psd(tr, inventory, psd_periods)[0]


def _cumulative_and_stats(trace):
    cml_max_diff_ = cml_max_diff(trace)

    # cum_sd_diff = np.nanstd(diff) / diff_abs_mean

    # linreg and MSE:
    # cml_min, cml_max = 0.1, 0.9
    # cml_trim = cml.copy().trim(*timeswhere(cml, cml_min, cml_max))
    # lreg = linreg(cml_trim)
    # mse = np.sum(np.abs(lreg.data - cml_trim.data)) / lreg.stats.npts

    # 2nd derivative https://stats.stackexchange.com/a/446201
    # diff2 = np.diff(diff)
    # diff2_mean = np.nanmean(diff2)
    # diff2_abs_mean = np.abs(diff2_mean)
    # cum_max_diff2 = np.nanmax(np.abs(diff2)) / diff2_abs_mean

    # return '%.2f_%.2f_%f' % (cum_max_diff, cum_sd_diff, cum2d_diff), cml
    return '{:,.2f}'.format(cml_max_diff_), cumsumsq(trace, normalize=True, copy=True)
    # return '{:,}'.format(int(cum_max_diff2)), cml
    # return '%.2f' % cum_max_diff2 , cml


@gui.plot
def detrend_(segment, config):
    from time import time
    data = {}
    for arg, kwargs in [
        ['simple', {}],
        ['linear', {}],
        ['constant', {}],
        ['polynomial', {'order': 2}],
        ['spline', {'order': 2, 'dspline': 500}]
    ]:
        tr = segment.stream()[0].copy()
        t = time()
        tr = tr.detrend(arg, **kwargs)
        t = time() - t
        data['%s %.5f' % (arg, t)] = tr
    return data


@gui.plot
def cumulative(segment, config):
    """Computes the cumulative of the squares of the segment's trace in the form of a Plot object.
    Modifies the segment's stream or traces in-place. Normalizes the returned trace values
    in [0,1]
    :return: an obspy.Trace
    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]
    ret = {}
    n = 10

    tra = trace.copy()
    caption, tra = _cumulative_and_stats(tra)
    ret['cum_'.rjust(n) + caption] = tra
    duration = (trace.stats.endtime - trace.stats.starttime) / 2

    from stream2segment.process.gui.webapp.mainapp import core
    # ret['cum'] = tra

    # tra = trace.copy()
    # tra.trim(starttime=trace.stats.starttime + duration)
    # tra.stats.starttime = trace.stats.starttime
    # caption, tra = _cumulative_and_stats(tra)
    # ret['cum2half_'.rjust(n) + caption] = tra
    #
    # tra = trace.copy()
    # tra.trim(endtime=trace.stats.endtime - duration)
    # caption, tra = _cumulative_and_stats(tra)
    # ret['cum1half_'.rjust(n) + caption] = tra
    #
    # leng = int(trace.stats.npts / 2)
    # tra = trace.copy()
    # tra.data[leng:] = trace.data[:trace.stats.npts-leng]
    # caption, tra = _cumulative_and_stats(tra)
    # ret['cumDouble_'.rjust(n) + caption] = tra
    #
    # tra = trace.copy()
    # tra.data *= 5
    # caption, tra = _cumulative_and_stats(tra)
    # ret['cum5times_'.rjust(n) + caption] = tra
    #
    # tra = trace.copy()
    # tra.data[int(tra.stats.npts / 2)] = 5 * tra.data.max()
    # caption, tra = _cumulative_and_stats(tra)
    # ret['cumSpike_'.rjust(n) + caption] = tra

    # tra = trace.copy()
    # tra.data = np.diff(_cumulative_and_stats(tra)[1].data)
    # ret['cum_1d'] = tra
    #
    # tra = trace.copy()
    # tra.data = np.diff(np.diff(tra.data))
    # ret['cum_2d'] = tra

    # tra = trace.copy()
    # tra.data[int(tra.stats.npts / 2)] *= 100
    # caption, tra = _cumulative_and_stats(tra)
    # ret['cumSpike_' + caption] = tra

    return ret