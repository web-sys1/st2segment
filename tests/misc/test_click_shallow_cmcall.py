"""
Created on May 23, 2017

@author: riccardo
"""
import os
from datetime import datetime, timedelta
import time
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from stream2segment.cli import cli
from stream2segment.process.inputvalidation import SEGMENT_SELECT_PARAM_NAMES
from stream2segment.resources import get_templates_fpaths, get_templates_fpath
from stream2segment.io import yaml_load
from stream2segment.process.inspectimport import load_source
from stream2segment.cli import copy_example_files as copyeexfiles


@pytest.fixture
def download_setup(pytestdir):
    basedir = pytestdir.makedir()

    def download_setup_func(filename, **params):
        yamldic = yaml_load(get_templates_fpath(filename))
        for key, val in params.items():
            if val is None:
                yamldic.pop(key, None)
            else:
                yamldic[key] = val
        path = os.path.join(basedir, filename)
        with open(path, 'w') as _opn:
            yaml.safe_dump(yamldic, _opn)
        return path, yamldic
    return download_setup_func


PKG = 'stream2segment.download'


@patch(PKG + ".main.configlog4download")
@patch(PKG + ".main.new_db_download")
@patch(PKG + ".inputvalidation.get_session")
@patch(PKG + ".main._run", return_value=0)
def test_click_download(mock_download, mock_create_sess, mock_new_db_download,
                        mock_configlog4download, download_setup):
    runner = CliRunner()
    # test normal case and arguments.
    (conffile, yamldic) = download_setup("download.yaml")
    result = runner.invoke(cli, ['download', '-c', conffile])
    dic = mock_download.call_args_list[0][1]
    assert dic['starttime'] == yamldic['starttime']
    assert dic['endtime'] == yamldic['endtime']
    mock_create_sess.assert_called_once_with(yamldic['dburl'])
    assert result.exit_code == 0

    # test by supplying an argument it is overridden
    mock_download.reset_mock()
    mock_create_sess.reset_mock()
    newdate = yamldic['starttime'] + timedelta(seconds=1)
    result = runner.invoke(cli, ['download', '-c', conffile, '--start', newdate])
    dic = mock_download.call_args_list[0][1]
    assert dic['starttime'] == newdate
    assert dic['endtime'] == yamldic['endtime']
    mock_create_sess.assert_called_once_with(yamldic['dburl'])
    # assert dic['dburl'] == yamldic['dburl']
    assert result.exit_code == 0

    # test by supplying the same argument as string instead of datetime (use end instead of
    # start this time)
    mock_download.reset_mock()
    mock_create_sess.reset_mock()
    result = runner.invoke(cli, ['download', '-c', conffile, '--end', newdate.isoformat()])
    dic = mock_download.call_args_list[0][1]
    assert dic['endtime'] == newdate
    assert dic['starttime'] == yamldic['starttime']
    mock_create_sess.assert_called_once_with(yamldic['dburl'])
    # assert dic['dburl'] == yamldic['dburl']
    assert result.exit_code == 0

    # test start and end given as integers
    # provide values IN THE YAML. Which means we need to write 'starttime' and 'endtime'
    # as those are the values stored in-there:
    (conffile, yamldic) = download_setup("download.yaml", starttime=1, endtime=0)
    mock_download.reset_mock()
    mock_create_sess.reset_mock()
    result = runner.invoke(cli, ['download', '-c', conffile])
    dic = mock_download.call_args_list[0][1]
    d = datetime.utcnow()
    startd = datetime(d.year, d.month, d.day) - timedelta(days=1)
    endd = datetime(d.year, d.month, d.day)
    assert dic['starttime'] == startd
    assert dic['endtime'] == endd
    mock_create_sess.assert_called_once_with(yamldic['dburl'])
    # assert dic['dburl'] == yamldic['dburl']
    assert result.exit_code == 0

    # test again start and end given as integers (30 days) FROM THE CLI
    (conffile, yamldic) = download_setup("download.yaml")
    mock_download.reset_mock()
    mock_create_sess.reset_mock()
    result = runner.invoke(cli, ['download', '-c', conffile, '-s', '30', '-e', '0'])
    dic = mock_download.call_args_list[0][1]
    d = datetime.utcnow()
    startd = datetime(d.year, d.month, d.day) - timedelta(days=30)
    endd = datetime(d.year, d.month, d.day)
    assert dic['starttime'] == startd
    assert dic['endtime'] == endd
    mock_create_sess.assert_called_once_with(yamldic['dburl'])
    # assert dic['dburl'] == yamldic['dburl']
    assert result.exit_code == 0

    # test removing an item in the config.yaml this item is not passed to download func
    (conffile, yamldic) = download_setup("download.yaml", inventory=None)
    mock_download.reset_mock()
    result = runner.invoke(cli, ['download', '-c', conffile])
    assert result.exit_code != 0
    assert not mock_download.called

    # test removing an item in the config.yaml this item is not passed to download func
    (conffile, yamldic) = download_setup("download.yaml", network=None)
    mock_download.reset_mock()
    result = runner.invoke(cli, ['download', '-c', conffile])
    dic = mock_download.call_args_list[0][1]
    assert 'networks' not in dic
    assert 'net' not in dic
    # we provided the default, with the parameter used in our functions:
    assert dic['network'] == []

    # test with an unknown argument
    (conffile, yamldic) = download_setup("download.yaml")
    mock_download.reset_mock()
    result = runner.invoke(cli, ['download', '--wtf_is_this_argument_#$%TGDAGRHNBGAWEtqevt3t', 5])
    assert result.exit_code != 0
    assert not mock_download.called


    mock_download.reset_mock()
    result = runner.invoke(cli, ['download', '--help'])
    assert result.exit_code == 0
    assert not mock_download.called


@patch("stream2segment.process.main.process", return_value=0)
def test_click_process(mock_process):
    runner = CliRunner()

    d_conffile, conffile, pyfile = \
        get_templates_fpaths("download.yaml", "paramtable.yaml", "paramtable.py")

    def dic2comparabletuple(dic):
        return [[_, dic[_]] for _ in sorted(dic)]

    segsel = dic2comparabletuple(yaml_load(conffile)[SEGMENT_SELECT_PARAM_NAMES[0]])


    # test no dburl supplied
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '-c', conffile, '-p', pyfile, 'c'])
    assert "Missing option" in result.output
    assert result.exc_info
    
    # test dburl supplied
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '-d', 'd', '-c', conffile, '-p', pyfile, 'c'])
    lst = list(mock_process.call_args_list[0][0])
    lst[2] = dic2comparabletuple(lst[2])  # convert segm. selection to a comparable tuple
    assert lst == [pyfile, 'd', segsel, conffile, 'c']
    assert result.exit_code == 0
    
    # test dburl supplied via config
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '-d', d_conffile , '-c', conffile, '-p', pyfile, 'c'])
    lst = list(mock_process.call_args_list[0][0])
    lst[2] = dic2comparabletuple(lst[2])  # convert segm. selection to a comparable tuple
    assert lst == [pyfile, yaml_load(d_conffile)['dburl'], segsel, conffile, 'c']
    assert result.exit_code == 0

    # test funcname supplied via cli. Wring (non existing) function:
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '--funcname', 'wat?', '-d', d_conffile,
                                 '-c', conffile, '-p', pyfile, 'c'])
    assert not mock_process.call_args_list
    assert 'Invalid value for "pyfile"' in result.output
    assert '"wat?"' in result.output
    assert result.exit_code != 0

    # test funcname supplied via cli. Provide a function in paramtabl.py that has the
    # expected signature:
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '--funcname', 'signal_noise_spectra', '-d', d_conffile ,
                                 '-c', conffile, '-p', pyfile, 'c'])
    lst = list(mock_process.call_args_list[0][0])
    lst[2] = dic2comparabletuple(lst[2])  # convert segm. selection to a comparable tuple
    assert lst == [pyfile + '::signal_noise_spectra', yaml_load(d_conffile)['dburl'], segsel, conffile, 'c']
    assert result.exit_code == 0

    # test an error in params: -dburl instead of --dburl:
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '-dburl', d_conffile , '-c', conffile,
                                 '-p', pyfile, 'c'])
    assert not mock_process.called
    assert result.exit_code != 0

    # assert help works:
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '--help'])
    assert not mock_process.called
    assert result.exit_code == 0


from stream2segment.process.gui.main import show_gui as orig_show


PKG = 'stream2segment.process.gui.main'


@patch(PKG + ".show_gui", side_effect=orig_show)
@patch(PKG + ".open_in_browser")
@patch(PKG + ".create_s2s_show_app")  # , return_value=mock.Mock())
def test_click_show(mock_create_s2s_show_app, mock_open_in_browser, mock_show):
    runner = CliRunner()
    d_conffile, conffile, pyfile = \
        get_templates_fpaths("download.yaml", "paramtable.yaml", "paramtable.py")

    # when asserting if we called open_in_browser, since tha latter is inside a thread which
    # executes with a delay of 1.5 seconds, we need to make our function here. Quite hacky,
    # but who cares
    # def assert_opened_in_browser(url=None):  # if None, assert
    #     time.sleep(2)  # to be safe
    #     mock_open_in_browser.assert_called_once
    #     args = mock_open_in_browser.call_args_list[0][0]
    #     assert len(args) == 1
    #     assert args[0].startswith('http://127.0.0.1:')

    # test no dburl supplied
    mock_show.reset_mock()
    mock_open_in_browser.reset_mock()
    result = runner.invoke(cli, ['show', '-c', conffile, '-p', pyfile])
    assert "Missing option" in result.output
    assert result.exc_info
    assert not mock_open_in_browser.called

    # test dburl supplied (wrong dburl)
    mock_show.reset_mock()
    mock_open_in_browser.reset_mock()
    result = runner.invoke(cli, ['show', '-d', 'd', '-c', conffile, '-p', pyfile])
    lst = list(mock_show.call_args_list[0][0])
    assert lst == ['d', pyfile, conffile]
    assert result.exit_code != 0
    assert 'Invalid value for "dburl"' in result.output
    # assert_opened_in_browser('d')

    # test dburl supplied via config (dburl ok)
    # INVALID: NOW the extrapolation is performed via input_validation
    # mock_show.reset_mock()
    # mock_open_in_browser.reset_mock()
    # result = runner.invoke(cli, ['show', '-d', d_conffile , '-c', conffile, '-p', pyfile])
    # lst = list(mock_show.call_args_list[0][0])
    # dburl = yaml_load(d_conffile)['dburl']
    # assert lst == [dburl, pyfile, conffile]
    # assert result.exit_code == 0
    # assert_opened_in_browser(dburl)

    # test an error in params: -dburl instead of --dburl:
    mock_show.reset_mock()
    mock_open_in_browser.reset_mock()
    result = runner.invoke(cli, ['show', '-dburl', d_conffile , '-c', conffile, '-p', pyfile])
    assert not mock_show.called
    assert result.exit_code != 0
    assert not mock_open_in_browser.called

    # assert help works:
    mock_show.reset_mock()
    mock_open_in_browser.reset_mock()
    result = runner.invoke(cli, ['show', '--help'])
    assert not mock_show.called
    assert result.exit_code == 0
    assert not mock_open_in_browser.called


@patch("stream2segment.cli.input")
@patch("stream2segment.cli.copy_example_files", side_effect=copyeexfiles)
def test_click_template(mock_main_init, mock_input, pytestdir):
    runner = CliRunner()
    # assert help works:
    result = runner.invoke(cli, ['init', '--help'])
    assert not mock_main_init.called
    assert result.exit_code == 0

    expected_files = ['download.yaml', 'paramtable.py', 'paramtable.yaml',
                      # 'save2fs.py', 'save2fs.yaml',  # <- NOT ANYMORE
                      'Using-Stream2segment-in-your-Python-code.ipynb',
                      'example.db.sqlite']
    non_python_files = [_ for _ in expected_files if os.path.splitext(_)[1]
                        not in ('.py', '.yaml')]

    dir_ = pytestdir.makedir()
    path = os.path.join(dir_, 'abc')

    def max_mod_time():
        return max(os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path))

    result = runner.invoke(cli, ['init', path])
    # FIXME: check how to mock os.path.isfile properly. This doesnot work:
    # assert mock_isfile.call_count == 5
    assert result.exit_code == 0
    assert mock_main_init.called
    files = os.listdir(path)
    assert sorted(files) == sorted(expected_files)
    assert not mock_input.called

    # assert we correctly wrote the files
    for fle in files:
        sourcepath = get_templates_fpath(fle)
        destpath = os.path.join(path, fle)
        if os.path.splitext(fle)[1] == '.yaml':
            # check loaded yaml, which also assures our templates are well formed:
            sourceconfig = yaml_load(sourcepath)
            destconfig = yaml_load(destpath)
            if os.path.basename(sourcepath) == 'download.yaml':
                assert sorted(sourceconfig.keys()) == sorted(destconfig.keys())
            else:
                # assert we have all keys. Note that 'advanced_settings' is not in
                # sourceconfig (it is added via jinja2 templating system):
                assert sorted(['advanced_settings'] + list(sourceconfig.keys())) \
                    == sorted(destconfig.keys())
            for key in sourceconfig.keys():
                assert type(sourceconfig[key]) == type(destconfig[key])
        elif os.path.splitext(fle)[1] == '.py':
            # check loaded Py modules, which also assures our templates are well formed:
            sourcepy = load_source(sourcepath)
            destpy = load_source(destpath)
            # compare module attrs except those starting and ending with '__':
            sourcekeys = [a for a in dir(sourcepy) if (a[:2] + a[-2:]) != "____"]
            destkeys = [a for a in dir(destpy) if (a[:2] + a[-2:]) != "____"]
            assert sorted(sourcekeys) == sorted(destkeys)
            for key in sourcekeys:
                assert type(getattr(sourcepy, key)) == type(getattr(destpy, key))
        elif fle not in non_python_files:
            raise ValueError('The file "%s" is not supposed to be copied by `init`' % fle)

    # try to write to the same dir (1)
    mock_input.reset_mock()
    mock_input.side_effect = lambda arg: '1'  # overwrite all files
    maxmodtime = max_mod_time()
    # we'll test that files are modified, but on mac timestamps are rounded to seconds
    # so wait 1 second to be safe
    time.sleep(1)
    result = runner.invoke(cli, ['init', path])
    assert mock_input.called
    assert max_mod_time() > maxmodtime
    assert '%d file(s) copied in' % len(expected_files) in result.output

    # try to write to the same dir (2)
    for click_prompt_ret_val in ('', '2'):
        # '' => skip overwrite
        # '2' => overwrite only non existing
        # in thus case, both the above returned values produce the same result
        mock_input.reset_mock()
        mock_input.side_effect = lambda arg: click_prompt_ret_val
        maxmodtime = max_mod_time()
        time.sleep(1)  # see comment above
        result = runner.invoke(cli, ['init', path])
        assert mock_input.called
        assert max_mod_time() == maxmodtime
        assert 'No file copied' in result.output

    os.remove(os.path.join(path, expected_files[0]))
    # try to write to the same dir (2)
    mock_input.reset_mock()
    mock_input.side_effect = lambda arg: '2'   # overwrite non-existing (1) file
    maxmodtime = max_mod_time()
    time.sleep(1)  # see comment above
    result = runner.invoke(cli, ['init', path])
    assert mock_input.called
    assert max_mod_time() > maxmodtime
    assert '1 file(s) copied in' in result.output


def test_click_template_realcopy(pytestdir):
    '''test a real example of copying files to a tempdir that will be removed'''
    runner = CliRunner()
    mydir = pytestdir.makedir()
    result = runner.invoke(cli, ['init', mydir])
    filez = os.listdir(mydir)
    assert "download.yaml" in filez
    assert "paramtable.yaml" in filez
    # assert "gui.yaml" in filez
    assert "paramtable.py" in filez
    # assert "gui.py" in filez


    # assert help works:
    assert result.exit_code == 0


@patch("stream2segment.download.db.inspection.main.stats", return_value=0)
def test_click_dstats(mock_da):

    prefix = ['dl', 'stats']
    runner = CliRunner()
    # assert help works:
    mock_da.reset_mock()
    result = runner.invoke(cli, prefix + ['--help'])
    assert not mock_da.called
    assert result.exit_code == 0

    # do a little test with variable length download ids

    result = runner.invoke(cli, prefix + ['-d', 'dburl', '-did', 1, '-did', 2])
    lst = list(mock_da.call_args_list[-1][0])
    assert lst == ['dburl', None, (1, 2), 0.5, False, None]
    assert result.exit_code == 0

    result = runner.invoke(cli, prefix + ['-d', 'dburl', '-1', '2'])
    lst = list(mock_da.call_args_list[-1][0])
    assert lst == ['dburl', ('-1', '2'), None, 0.5, False, None]
    assert result.exit_code == 0

    result = runner.invoke(cli, prefix + ['-d', 'dburl'])
    lst = list(mock_da.call_args_list[-1][0])
    assert lst == ['dburl', None, None, 0.5, False, None]
    assert result.exit_code == 0

    result = runner.invoke(cli, prefix + ['-d', 'dburl', '-g', 0.77, '--html', 'abc'])
    lst = list(mock_da.call_args_list[-1][0])
    assert lst == ['dburl', ('abc',), None, 0.77, True, None]
    assert result.exit_code == 0

    mock_da.reset_mock()
    result = runner.invoke(cli, prefix + ['-d', 'dburl', '-g', 'a'])
    assert not mock_da.called
    assert result.exit_code != 0

    mock_da.reset_mock()
    result = runner.invoke(cli, prefix + ['-d', 'dburl', '-did', 'a'])
    assert not mock_da.called
    assert result.exit_code != 0


pkg = "stream2segment.download.db.inspection.main"


# @patch("stream2segment.download.db.inspection.main.report", return_value=0)
@pytest.mark.parametrize('mocked_func_path', [
    pkg+'.config', pkg+'.log', pkg+'.summary'
])
def test_click_dreport(mocked_func_path):
    with patch(mocked_func_path, return_value=[]) as mock_da:
        prefix = ['dl', mocked_func_path.split('.')[-1]]
        runner = CliRunner()
        # assert help works:
        mock_da.reset_mock()
        result = runner.invoke(cli, prefix + ['--help'])
        assert not mock_da.called
        assert result.exit_code == 0

        is_summary_command = prefix[-1] == 'summary'
        default_when_missing = None # if is_summary_command else [-1]

        # do a little test with variable length download ids
        result = runner.invoke(cli, prefix + ['-d', 'dburl', '-did', 1, '-did', 2])
        lst = list(mock_da.call_args_list[-1][0])
        if is_summary_command:
            # summary command does not have a 4th argument (outfile). This argument is
            # always None for the other two commands because not used anymore. Hacky add:
            lst += [None]
        assert lst == ['dburl', default_when_missing, (1, 2), None]
        assert result.exit_code == 0
        # mock_da.reset_mock()

        # do a little test with variable length download ids
        result = runner.invoke(cli, prefix + ['-d', 'dburl',  '-1', '2'])
        lst = list(mock_da.call_args_list[-1][0])
        if is_summary_command:
            # summary command does not have a 4th argument (outfile). This argument is
            # always None for the other two commands because not used anymore. Hacky add:
            lst += [None]
        assert lst == ['dburl', ('-1', '2'), None, None]
        assert result.exit_code == 0
        # mock_da.reset_mock()

        # do a little test with variable length download ids
        result = runner.invoke(cli, prefix + ['-d', 'dburl', '-did', 1, '-did', 2,
                                              '-2', 'b'])
        lst = list(mock_da.call_args_list[-1][0])
        if is_summary_command:
            # summary command does not have a 4th argument (outfile). This argument is
            # always None for the other two commands because not used anymore. Hacky add:
            lst += [None]
        assert lst == ['dburl', ('-2', 'b'), (1, 2), None]
        assert result.exit_code == 0
        # mock_da.reset_mock()

        # Now try with no args and check the defaults:
        default_when_missing = None if is_summary_command else [-1]

        result = runner.invoke(cli, prefix + ['-d', 'dburl'])
        lst = list(mock_da.call_args_list[-1][0])
        if is_summary_command:
            # summary command does not have a 4th argument (outfile). This argument is
            # always None for the other two commands because not used anymore. Hacky add:
            lst += [None]
        assert lst == ['dburl', default_when_missing, None, None]
        assert result.exit_code == 0
        # mock_da.reset_mock()

        # all unknown arguments are parsed as download indices:
        result = runner.invoke(cli, prefix + ['-d', 'dburl', '-c', '-l', '--html', 'abc'])
        lst = list(mock_da.call_args_list[-1][0])
        if is_summary_command:
            # summary command does not have a 4th argument (outfile). This argument is
            # always None for the other two commands because not used anymore. Hacky add:
            lst += [None]
        assert lst == ['dburl', ('-c', '-l', '--html', 'abc',), None, None]
        assert result.exit_code == 0  # --html option not supported
        # mock_da.reset_mock()

        # mock_da.reset_mock()
        # result = runner.invoke(cli, prefix + ['-d', 'dburl', '-g'])
        # assert not mock_da.called
        # assert result.exit_code != 0
        #
        # mock_da.reset_mock()
        # result = runner.invoke(cli, prefix + ['-d', 'dburl', '-did', 'a'])
        # assert not mock_da.called
        # assert result.exit_code != 0


# FINAL NOTE:
# test_classlabel is implemented in test_process_db !!
