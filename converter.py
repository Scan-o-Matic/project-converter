#!/usr/bin/env python
import os
import fnmatch
import json
from scanomatic.io import jsonizer
from scanomatic.models.factories.compile_project_factory import (
    CompileImageAnalysisFactory,
    CompileProjectFactory,
)
from scanomatic.io.paths import Paths
from scanomatic.io.logger import Logger



def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


l = Logger('Converter')
BASE_DIR = '/projects'
p = Paths()

for file_type, pattern, loader, processor in (
    (
        'project compilations',
        p.project_compilation_pattern.format('*'),
        CompileImageAnalysisFactory.serializer.load,
        lambda data: sorted(data, key=lambda e: e.image.index),
    ),
    (
        'project compilation instructions',
        p.project_compilation_instructions_pattern.format('*'),
        CompileProjectFactory.serializer.load_first,
        None,
    ),
):
    l.info('Converting all {} in {}'.format(file_type, BASE_DIR))
    n = 0
    for fname in find_files(BASE_DIR, pattern):
        try:
            with open(fname) as fh:
                if json.load(fh):
                    l.info("Skipping {} because seems already converted".format(fname))
                    continue
        except ValueError:
            pass

        data = loader(fname)
        if processor is not None:
            data = processor(data)
        l.info('Converting {} ({} entries)'.format(fname, len(data) if isinstance(data, list) else 1))
        jsonizer.dump(data, fname)
        n += 1
    l.info('Converted {} {} files'.format(n, file_type))
