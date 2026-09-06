"""Accuracy characterisation (Phase 5).

Four metrics, deliberately separated because they measure different things:

    jitter.py         PRECISION  -- how much a held pose wobbles
    points.py         PRECISION  -- how repeatably a real point can be touched
                      TRUENESS   -- measured vs known distances
                      TRUENESS   -- rigid registration residual
    headline.py       aggregates the latest of each into one citable paragraph

`stats.py` holds the pure maths so it can be tested without a camera, and
`session.py` writes raw samples plus the conditions they were taken under --
a number without its conditions is not a result.
"""
