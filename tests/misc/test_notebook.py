from stream2segment.process import print_segment_help
from stream2segment.resources import get_resource_abspath

def test_segment_help():
    """This test check that we did not add any new method or attribute to the Segment
    object without considering it in the doc (either make it hidden or visible)
    """
    print_segment_help()


def test_notebook(data):
    import os
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    # cwd = os.getcwd()
    for fle_ in [ 'jupyter.example.ipynb', 'the-segment-object.ipynb']:
        fle = get_resource_abspath('templates', fle_)
        with open(fle) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600)  # , kernel_name='python3')
        cwd = os.path.dirname(fle)
        ep.preprocess(nb, {'metadata': {'path': cwd}})