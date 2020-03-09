#!/usr/bin/env bash

##
# Author: Nikolaus Mayer
##

## Fail if any command fails (use "|| true" if a command is ok to fail)
set -e
## Treat unset variables as error
set -u

DOCKER_CMD='docker run --runtime=nvidia';

${DOCKER_CMD} \
  --rm \
  --volume "${PWD}:/input:ro" \
  --volume "${PWD}:/output:rw" \
  -it flownet2 /bin/bash;

