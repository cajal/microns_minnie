#!/bin/bash
cd /notebooks
jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token=${JUPYTER_PASSWORD:-} --no-browser