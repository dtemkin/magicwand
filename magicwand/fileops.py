import os


def find_reldir(dirname, start=os.getcwd(), direction='back'):
    parts = os.path.split(start)
    if parts[1] == dirname:
        return start
    else:
        return find_reldir(dirname=dirname, start=parts[0])


def filepath(base_dir, *path_args):
    if os.path.isdir(base_dir):
        return os.path.abspath(os.path.join(base_dir, *path_args))
    else:
        base = find_reldir(base_dir)
        return filepath(base_dir=base, *path_args)

