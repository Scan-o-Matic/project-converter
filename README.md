# Scan-o-matic project converted

1. Place all projects you wish to migrate in `/tmp/SoM/`

**NOTE**: There's no way to migrate back from the
python 3 version of projects to the old python 2.7 version
and contents of files will be overwritten.

2. Run `docker-compose up`

The script is designed to ignore already converted files,
so it should be safe to run multiple times on the same
project.
