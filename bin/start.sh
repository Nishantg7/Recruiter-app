#!/bin/bash
pip install gunicorn
python -m gunicorn app:app 