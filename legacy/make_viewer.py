#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_viewer.py — results/cases.json 을 index.html 로 굽는다.
=============================================================
index.html 은 데이터를 인라인으로 품은 단일 파일이다(file:// 로 열면 fetch 가 CORS 로 막히므로).
저장소 루트에 index.html 로 두는 이유: `python -m http.server` 로 띄우면 http://localhost:8000/ 이
디렉터리 목록 대신 뷰어를 바로 연다. 탐색기에서 더블클릭해도 같은 파일이다.
매처를 바꾼 뒤에는 export_cases.py → make_viewer.py 순으로 다시 돌리면 된다.

  python3 export_cases.py            # results/cases.json 갱신
  python3 make_viewer.py             # index.html 갱신
  → http://localhost:8000/  (python -m http.server 8000)  또는 index.html 더블클릭

three.js 는 cdnjs 에서 받는다 — 브라우저에 인터넷이 있어야 3D 가 뜬다.
없으면 3D 자리에 안내가 뜨고 오른쪽 목록·수치는 그대로 동작한다.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, 'viewer_template.html')
SRC = os.path.join(HERE, 'results', 'cases.json')
DST = os.path.join(HERE, 'index.html')

data = open(SRC, encoding='utf-8').read()
json.loads(data)                                   # 깨진 JSON 을 굽지 않는다
tpl = open(TPL, encoding='utf-8').read()
if '__DATA__' not in tpl:
    sys.exit('template has no __DATA__ placeholder')
open(DST, 'w', encoding='utf-8').write(tpl.replace('__DATA__', data))
print('index.html  <-  %d bytes of case data' % len(data.encode()))
print('열기:  http://localhost:8000/   (python -m http.server 8000)')
print('   또는 이 폴더의 index.html 더블클릭')
