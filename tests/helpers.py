"""Shared fixtures that are imported rather than injected.

`conftest.py` provides pytest fixtures; this module holds plain constants that
several test modules need at import time, where a fixture would not help.
"""

from __future__ import annotations

TEI_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Deep learning for EEG emotion recognition</title></titleStmt>
      <sourceDesc><biblStruct><analytic>
        <author><persName><forename>Anna</forename><surname>Kowalska</surname></persName></author>
        <idno type="DOI">10.1000/ABC.123</idno>
      </analytic><monogr><title>Journal of Neural Engineering</title>
        <imprint><date when="2021-05-01">2021</date></imprint>
      </monogr></biblStruct></sourceDesc>
    </fileDesc>
    <profileDesc><abstract><p>We study convolutional networks on EEG signals.</p></abstract></profileDesc>
  </teiHeader>
  <text><body>
    <div><head>Introduction</head><p>Emotion recognition from EEG is difficult.</p></div>
    <div><head>Materials and Methods</head><p>We recorded 32 channels at 256 Hz. A convolutional
      network was trained with cross validation on a held out split.</p></div>
    <div><head>Results</head><p>Accuracy reached 0.87 on the held out split.</p></div>
  </body></text>
</TEI>
"""

#: Backwards-compatible alias used by the service tests.
TEI_FOR_SERVICE = TEI_SAMPLE
