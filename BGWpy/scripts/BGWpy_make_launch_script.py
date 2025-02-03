# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:23:07 2025

@author: Daan Holleman
"""

# %% IMPORT
import argparse
import json
import os
from pathlib import Path


# %% DECLARE COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(
    prog = 'BGWpy-MLS',
    description = 'Reads a JSON file with dependency relations to generate a launch script.',
    epilog = 'Dependencies can be specified when creating tasks and written using workflow objects.'
)
parser.add_argument(
    '-f', '--fname', 
    default = 'dependency_relations.json', 
    help='From which JSON file to read dependencies from'
)
parser.add_argument(
    '-s','--start', 
    action='extend', 
    nargs='+', 
    default = [],
    help='From which tasks to start from'
)
parser.add_argument(
    '-e','--exclude', 
    action='extend', 
    nargs='+', 
    default = [],
    help='Which folders to exclude from adding'
)
parser.add_argument(
    '-c','--comment', 
    action='store_true', 
    help='Comment out in launch script instead of omitting'
)
parser.add_argument(
    '-o','--output', 
    default = 'launch.sh',
    help='Filename of output file'
)
parser.add_argument(
    '-d','--delay',
    default = 0,
    type=int,
    help='Add a delay in between each sbatch command.'
)

config = parser.parse_args()

# %% LOAD JSON

def read_json():
    """
    Read the JSON with the dependency relations.

    Returns
    -------
    relations : list (dict)
        List of dictionaries that contain the dependencies, defers, directory and runscript filename
        of each task.

    """
    with open( config.fname, 'r') as file:
        relations = json.load(file)
    return relations

# %% SORT TASKS

def sort_tasks(relations):
    """
    Sorts a list in O(n log(n)) < O(n^2) / 2
    """
    N = len(relations)
    sorted_relations = list(relations)
    ii = 0
    for attempts in range(N**2):
        ctask = sorted_relations[ii]
        for jj in range(ii+1,N):
            otask = sorted_relations[jj]
            # If the other path is in dependencies, then the order is not correct
            # => swap places
            if otask['path'] in ctask['dependencies']:
                sorted_relations[ii], sorted_relations[jj] = sorted_relations[jj], sorted_relations[ii]
                break
        else:
            ii += 1
            if ii == N:
                break
    else:
        raise Exception('Could not sort list')
    
    return sorted_relations, attempts

def check_valid(relations):
    valid = True
    N = len(relations)
    for ii in range(N-1):
        task = relations[ii]
        for jj in range(ii+1, N):
            other = relations[jj]
            if other['path'] in task['dependencies']:
                valid = False
    return valid

def get_by_path(relations, path):
    for task in relations:
        if path != task['path']: continue
        break
    return task

def get_by_folder(relations, folder):
    for task in relations:
        if folder != os.path.basename(task['dirname']):
            continue
        break
    return task

def follow_defers(relations, task, out = []):
    if task not in out:
        out.append(task)
    for other_path in task['defers']:
        other = get_by_path(relations, other_path)
        out.append(other)
        out = follow_defers(relations, other, out)
    return out

def get_relations():
    relations = read_json()

    if not check_valid(relations):
        relations, attempts = sort_tasks(relations)
    
    return relations

def set_other_exclusions(relations):
    # Early exit if default start should be used
    if len(config.start) == 0:
        return
    
    include = []
    for start_dir in config.start:
        task = get_by_folder(relations, start_dir)
        defers = follow_defers(relations, task)
        include += defers
    additional_exclude = [ os.path.basename(task['dirname']) for task in relations if task not in include ]
    config.exclude += additional_exclude

# %% JOBNAMES

def generate_jobnames(relations):
    jobnames = []
    for task in relations:
        basename = os.path.basename(task['dirname'])
        basename = basename.replace('.','_')
        first_of_run = task['runscript'].split('.')[0]
        jobname = basename + '_' + first_of_run
        jobnames.append(jobname)
    if len(jobnames) == len(set(jobnames)):
        return jobnames
    jobnames = [ 'job{0}'.format(ii) for ii in range(len(relations)) ]
    return jobnames

def find_jobname(relations, jobnames, dependency_path):
    for task, jobname in zip(relations, jobnames):
        if task['path'] == dependency_path:
            return jobname
    raise Exception('Could not find jobname matching with {0}.'.format(dependency_path))


#%% CREATE LAUNCH SCRIPT

def create_header():
    header_lines = [
        '#!/bin/bash',
        'default="--parsable --kill-on-invalid-dep=yes"',
        '',
        '',
    ]
    header = '\n'.join(header_lines)
    return header

def get_dependency_string(relations, jobnames, dependencies):
    dependency_names = []
    for dep_path in dependencies:
        exclude_path = os.path.basename(os.path.dirname(dep_path))
        if exclude_path in config.exclude: continue
        other_job = find_jobname(relations, jobnames, dep_path)
        dependency_names.append('${{{0}}}'.format(other_job))
    
    dependency_string = ''
    if dependency_names:
        seperated_names = ":".join(dependency_names)
        dependency_string = ' --dependency=afterok:' + seperated_names
    
    return dependency_string

def create_body(relations):
    body_lines = []
    jobnames = generate_jobnames(relations)
    list_last_entry = [False] * len(jobnames)
    list_last_entry[-1] = True
    for task, jobname, is_last in zip(relations, jobnames, list_last_entry):
        bodypart = []
        path = task['path']
        dirname = task['dirname']
        runscript = task['runscript']
        dependencies = task['dependencies']
        ppath = Path(path)
        sanitized_dirname = os.path.basename(dirname)
        
        # Check for commenting out
        bool_excluded = dirname in config.exclude or sanitized_dirname in config.exclude
        pre = '# ' if bool_excluded else ''
        if not config.comment and bool_excluded: continue
        
        # Depth to get back from cd
        depth = len(ppath.parents)
        reverse = '../' * (depth-1)
        
        # Enter directory
        bodypart.append('cd {0}'.format(dirname))
        
        # Write sbatch commands
        dependency_string = get_dependency_string(relations, jobnames, dependencies)
        line = '{0}=$(sbatch $default{1} {2})'.format(jobname, dependency_string, runscript)
        bodypart.append(line)
        
        # Leave directory
        bodypart.append('cd {0}'.format(reverse))
        
        # Add delay if needed
        if config.delay > 0 and not is_last:
            bodypart.append('sleep {0}'.format(config.delay))

        # Add comments as necessary
        for line in bodypart:
            body_lines.append(pre + line)
    
    body_lines.append('')
    
    body = '\n'.join(body_lines)
    
    return body

# %% MAIN

def main():
    relations = get_relations()
    set_other_exclusions(relations)
    
    header = create_header()
    body = create_body(relations)
    epilog = '# Created using BGWpy_make_launch_script.py\n'
    
    with open(config.output, 'w') as file:
        file.write(header)
        file.write(body)
        file.write(epilog)

if __name__ == '__main__':
    main()
