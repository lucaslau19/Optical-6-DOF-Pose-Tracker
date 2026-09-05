"""Marker detection and 6-DOF pose estimation.

Phase 1: single-marker pose (`single_marker.py`).
Phase 2: rigid multi-marker bodies -- one solvePnP over every visible corner,
         so the tool keeps a pose when individual markers are occluded.
"""
