#!/usr/bin/env python
from collections import namedtuple
import os
import fnmatch
import json

import numpy as np

from scanomatic.data_processing.phenotypes import PhenotypeDataType
from scanomatic.io import jsonizer
from scanomatic.io.paths import Paths
from scanomatic.io.logger import Logger
from scanomatic.models.factories.compile_project_factory import (
    CompileImageAnalysisFactory,
    CompileProjectFactory,
)
from scanomatic.models.factories.fixture_factories import FixtureFactory
from scanomatic.models.factories.scanning_factory import ScanningModelFactory


def find_files(directory, pattern):
    for root, _, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def get_dump_name(filename, extension):
    if extension is None:
        return filename
    else:
        split_name = os.path.basename(filename).split('.')
        if len(split_name) > 1:
            new_basename = '.'.join(split_name[:-1]) + extension
        else:
            new_basename = split_name[0] + extension
        return os.path.join(os.path.dirname(fname), new_basename)


logger = Logger('Converter')
BASE_DIR = '/projects'
paths = Paths()
TypedParameter = namedtuple("TypedParameter", ["name", "processor"])
PHENOTYPE_PARAMS = (
    TypedParameter("median_kernel_size", int),
    TypedParameter("gaussian_filter_sigma", float),
    TypedParameter("linear_regression_size", int),
    TypedParameter("phenotypes_inclusion", lambda x: PhenotypeDataType[x]),
    TypedParameter("no_growth_monotonicity_threshold", float),
    TypedParameter("no_growth_pop_doublings_threshold", float),
)

for file_type, pattern, loader, processor, new_ext in (
    (
        'project compilations',
        paths.project_compilation_pattern.format('*'),
        CompileImageAnalysisFactory.serializer.load,
        lambda data: sorted(data, key=lambda e: e.image.index),
        None,
    ),
    (
        'project compilation instructions',
        paths.project_compilation_instructions_pattern.format('*'),
        CompileProjectFactory.serializer.load_first,
        None,
        None,
    ),
    (
        'phenotypes extraction parameters',
        paths.phenotypes_extraction_params,
        np.load,
        lambda data: {
            PHENOTYPE_PARAMS[i].name: PHENOTYPE_PARAMS[i].processor(p)
            for i, p in enumerate(data)
        },
        ".json",
    ),
    (
        'local fixture config',
        paths.experiment_local_fixturename,
        FixtureFactory.serializer.load_first,
        None,
        None,
    ),
    (
        'scan project file',
        paths.scan_project_file_pattern.format('*'),
        ScanningModelFactory.serializer.load_first,
        None,
        None,
    ),
):
    logger.info('Converting all {} in {}'.format(file_type, BASE_DIR))
    n = 0
    for fname in find_files(BASE_DIR, pattern):
        try:
            with open(fname) as fh:
                if json.load(fh):
                    logger.info(
                        'Skipping {} because seems already converted'.format(
                            fname,
                        ),
                    )
                    continue
        except ValueError:
            pass

        data = loader(fname)
        if processor is not None:
            data = processor(data)
        logger.info('Converting {} ({} entries)'.format(
            fname,
            len(data) if isinstance(data, list) else int(data is not None),
        ))
        if data is None:
            logger.error('Found nothing to convert in {}'.format(fname))
            continue

        dump_name = get_dump_name(fname, new_ext)
        jsonizer.dump(data, get_dump_name(fname, new_ext))
        if dump_name != fname:
            os.remove(fname)
        n += 1
    logger.info('Converted {} {} files'.format(n, file_type))
