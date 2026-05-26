import os

def config_dir(*args):
    dir = os.path.join(*args)
    os.makedirs(dir, exist_ok=True)
    return dir

LABWEBVIEW_REPO_DIR                 = os.path.dirname(os.path.abspath(__file__))
LABWEBVIEW_TEMPLATE_DIR             = os.path.join(LABWEBVIEW_REPO_DIR, 'templates') 