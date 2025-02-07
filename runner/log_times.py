#############################################
################## WARNING ##################
#############################################

# This is a sensitive script. Try to add as less as possible in it
# and just call it when needed. This runs a notebook by logging time
# for every cell in JSON file.
# If you try to do more things inside here, you may break things and it's
# subtle to see why. Examples:
# 
# 1) If you plan to put this code into a function and call it for multiple
#    notebooks.
#
# You have to reset an instance before running a new notebook. See what I'm doing below.
# I thought of creating a new instance and at the end destroying it,
# with InteractiveShell.instance(). But the doc is worrying:
# > This method create a new instance if none have previously been created
# > and returns a previously created instance is one already exists.
# So, we might get the same instance back. I guess we can `del` it at
# the end but let's not make our life hard. It's just that the following
# resets _everything_, so we have to import modules etc.
# ipython.reset(new_session=True, aggressive=True)
# ipython = get_ipython()
#
# 2) You have to be careful with the names you choose for variables and
#    functions. The namespace for this script is the same as the one
#    for the notebook that it runs since we're just call run_cel()
#    on the same IPython instance. So, be careful to use names that won't
#    clobber the NB namespace.
#
# If it is this sensitive, then why don't use %%time every cell in the notebooks
# and run it? See discussion for measuring time and handling exceptions below.

#### MEASURING TIME ####

# 1) %%time
# 2) time.perf_counter()
# 3) time.process_time()
# 4) timeit
#
# 2) and 3) are the most convenient. I don't think we care about super
# exact times here so I don't think it matters. 1) is a pain because
# you don't get all times in the same unit (maybe it'll report 2s or 2ms)
# and you have to extract them from a string.
# 4) is a problem because you have to remember to enable the GC and to
# not run it many times. I don't think it's meant for our case.
# If you use 1), you probably want to use %%capture and capture the output
# of the execution in an object. Then, regex the Wall time.
#
# I think what we want to measure is wall time. So, 3) won't do it because
# it reports total time (it's basically "total" in %%time). Then, we do either
# 1) or 2). 2) is more convenient so I'm going with that. Of course, we'll
# include the invocation of run_cell(), but I think it's fine.


#### Handling Exceptions ### 

# def custom_exc(shell, etype, evalue, tb, tb_offset=None):
#   print("Python hates me")

# ipython.set_custom_exc((Exception,), custom_exc)

# Apparently, this is not all easy. First, I was capturing the output
# of a cell, because we don't care about it. But if you do that,
# then surprise, there's apparently no way to catch the exceptions,
# except I guess if you manually inspect the output. But we can
# do sort of the same thing (although still we'll see output e.g., when exceptions
# occur) with silent=True in run_cell().
#
# Now, an alternative way is to register a custom exception-handler, like the
# commented code above. But this is inconvenient. Instead, you can save the
# result of run_cell() and check if `.success` is set. In both cases, for
# some reasons not all exceptions are logged. For example, check this:
# https://www.kaggle.com/code/addictedgt/credit-card-risk-analysis/
# This loads its data from /kaggle/input. Suppose that we don't place a copy
# of its data in /kaggle/input. There are two images before the read_csv(). These are supposed
# to be loaded from /kaggle/input. These raise an exception (try to run the notebook
# directlith `ipython <path to nb>`) but when we use run_cell(), res.success is set,
# so we don't stop (I guess such kinds of exceptions are handled by IPython?).
# That's fine because they don't impact the code
# and these images are not part of the data that we download for this notebook,
# so we couldn't do anything anyway. When we try to do the read_csv() though, then
# we do stop. And that's what we want.

from IPython import *
import json
import sys
import time
import os
import subprocess
import re

_IREWR_ipython = get_ipython()
if _IREWR_ipython is None:
    print("[ERROR]: Run with `ipython` not `python`")
    sys.exit(1)
assert isinstance(_IREWR_ipython, InteractiveShell)

_IREWR_run_config_filename = sys.argv[1]

_IREWR_f = open(_IREWR_run_config_filename)
try:
    _IREWR_run_config = json.load(_IREWR_f)
except:
    _IREWR_f.close()
    sys.exit(1)
_IREWR_f.close()

if _IREWR_run_config['less_replication']:
    os.environ["IREWR_LESS_REPLICATION"] = "True"

import bench_utils

_IREWR_error_file = _IREWR_run_config['error_file']
_IREWR_times_file = _IREWR_run_config['output_times_json']
_IREWR_measure_modin_mem = _IREWR_run_config['measure_modin_mem']
_IREWR_measure_data = _IREWR_run_config['measure_data']

if _IREWR_measure_data:
    import pandas as pd_xyz
    import modin.pandas as mpd_xyz


def _IREWR_err_txt(ctx):
    return \
        f"""
[START ERROR]
# Source cell idx: {ctx[0]}
# Source code:
{ctx[1]}
[END ERROR]
"""


_IREWR_ipython.run_line_magic('cd', _IREWR_run_config['src_dir'])

_IREWR_source_cells = _IREWR_run_config['cells']


def report_on_fail(_IREWR_ctx):
    global _IREWR_error_file
    bench_utils.write_to_file(_IREWR_error_file, _IREWR_err_txt(_IREWR_ctx))
    sys.exit(1)


_IREWR_cells = []
_IREWR_max_modin_mem = 0
_IREWR_max_modin_disk = 0
for _IREWR_cell_idx, _IREWR_cell in enumerate(_IREWR_source_cells):

    # This will not catch all failures. The caller has to check the stdout
    # of this script.
    _IREWR_ip_run_res = None
    _IREWR_start = time.perf_counter_ns()
    _IREWR_ip_run_res = _IREWR_ipython.run_cell(_IREWR_cell, silent=True)
    _IREWR_end = time.perf_counter_ns()
    _IREWR_diff_in_ns = _IREWR_end - _IREWR_start
    _IREWR_cell_stats = dict()
    _IREWR_cell_stats['raw'] = _IREWR_cell
    _IREWR_cell_stats['total-ns'] = _IREWR_diff_in_ns
    # Collect data for the ML model (1. the df/series variables and their sizes,
    # 2. their modin2pandas conversion times + vice versa)
    if _IREWR_measure_data:
        # Dictionaries to hold the results
        _IREWR_str_data = {}
        _IREWR_p2m_data = {}
        _IREWR_m2p_data = {}
        # Iterate through variables and check if they are df or series
        for _IREWR_k in list(_IREWR_ipython.user_ns.keys()):
            if (type(_IREWR_ipython.user_ns[_IREWR_k]) == pd_xyz.DataFrame) or (
                    type(_IREWR_ipython.user_ns[_IREWR_k]) == pd_xyz.Series):
                # Get the size of variables
                _IREWR_str_data[_IREWR_k] = sys.getsizeof(_IREWR_ipython.user_ns[_IREWR_k])
            # Collect modin conversion times if running modin
            modin_has_been_imported = "modin.pandas" in sys.modules
            # Check if modin is running
            if modin_has_been_imported:
                # If dataframe, else a series
                if isinstance(_IREWR_ipython.user_ns[_IREWR_k], mpd_xyz.DataFrame):
                    # Get the conversion time to pandas
                    _IREWR_m2p_start = time.perf_counter_ns()
                    _IREWR_ipython.user_ns[_IREWR_k] = _IREWR_ipython.user_ns[_IREWR_k]._to_pandas()
                    _IREWR_m2p_end = time.perf_counter_ns()
                    _IREWR_m2p_diff_in_ns = _IREWR_m2p_end - _IREWR_m2p_start
                    # Get the conversion time to modin
                    _IREWR_p2m_start = time.perf_counter_ns()
                    _IREWR_ipython.user_ns[_IREWR_k] = mpd_xyz.DataFrame(_IREWR_ipython.user_ns[_IREWR_k])
                    _IREWR_p2m_end = time.perf_counter_ns()
                    _IREWR_p2m_diff_in_ns = _IREWR_p2m_end - _IREWR_p2m_start
                    # Assign to dictionary
                    _IREWR_p2m_data[_IREWR_k] = _IREWR_p2m_diff_in_ns
                    _IREWR_m2p_data[_IREWR_k] = _IREWR_m2p_diff_in_ns
                elif isinstance(_IREWR_ipython.user_ns[_IREWR_k], mpd_xyz.Series):
                    # Get the conversion time to pandas
                    _IREWR_m2p_start = time.perf_counter_ns()
                    _IREWR_ipython.user_ns[_IREWR_k] = _IREWR_ipython.user_ns[_IREWR_k]._to_pandas()
                    _IREWR_m2p_end = time.perf_counter_ns()
                    _IREWR_m2p_diff_in_ns = _IREWR_m2p_end - _IREWR_m2p_start
                    # Get the conversion time to modin
                    _IREWR_p2m_start = time.perf_counter_ns()
                    _IREWR_ipython.user_ns[_IREWR_k] = mpd_xyz.Series(_IREWR_ipython.user_ns[_IREWR_k])
                    _IREWR_p2m_end = time.perf_counter_ns()
                    _IREWR_p2m_diff_in_ns = _IREWR_p2m_end - _IREWR_p2m_start
                    # Assign to dictionary
                    _IREWR_p2m_data[_IREWR_k] = _IREWR_p2m_diff_in_ns
                    _IREWR_m2p_data[_IREWR_k] = _IREWR_m2p_diff_in_ns
        # Assign to print
        _IREWR_cell_stats['vars'] = _IREWR_str_data
        _IREWR_cell_stats['p2m'] = _IREWR_p2m_data
        _IREWR_cell_stats['m2p'] = _IREWR_m2p_data

    if not _IREWR_ip_run_res.success:
        _IREWR_ctx = (_IREWR_cell_idx, _IREWR_cell)
        report_on_fail(_IREWR_ctx)

    modin_has_been_imported = "modin.pandas" in sys.modules
    if _IREWR_measure_modin_mem and modin_has_been_imported:
        ray_sample = subprocess.run(["ray", "memory", "--stats-only"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert ray_sample.returncode == 0
        ray_sample_out = ray_sample.stdout.decode()
        match_mem = re.search("Objects consumed by Ray tasks: (\d+) MiB", ray_sample_out)
        # Some files will contain nothing.
        if match_mem:
            _IREWR_max_modin_mem = max(_IREWR_max_modin_mem, int(match_mem.group(1)))

        match_disk = re.search("Spilled (\d+) MiB", ray_sample_out)
        if match_disk:
            _IREWR_max_modin_disk = max(_IREWR_max_modin_disk, match_disk.group(1))
    # END if _IREWR_measure_modin_mem #

    _IREWR_cells.append(_IREWR_cell_stats)

_IREWR_f = open(_IREWR_times_file, 'w')
_IREWR_json_d = dict()
if _IREWR_measure_modin_mem:
    _IREWR_json_d['max-mem-in-mb'] = _IREWR_max_modin_mem
    _IREWR_json_d['max-disk-in-mb'] = _IREWR_max_modin_disk
else:
    # Signify that _IREWR_measure_modin_mem should NOT be true
    # for time measurements. Only output times if it's not
    # enabled.
    _IREWR_json_d['cells'] = _IREWR_cells
json.dump(_IREWR_json_d, _IREWR_f, indent=2)
_IREWR_f.close()
