import sys

def get_project_root_path_vscode():
    trace_object = getattr(sys, 'gettrace', lambda: None)() #vscode debugging
    if trace_object is not None:
        return "../../../"
    else:
        return "../../../../"
    