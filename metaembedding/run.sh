#!/bin/bash
# 가상환경 활성화 후 스크립트 실행
cd "$(dirname "$0")"
source venv/bin/activate
python asdads.py

