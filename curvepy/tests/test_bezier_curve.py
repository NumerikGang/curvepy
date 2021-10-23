import pytest

# TODO: Equality (mit Rundung) aller bez_curves (jeweils zwischen 2 damit man am Funktionsnamen erkennen kann
# TODO: welche beiden

# TODO: Equality (ohne Rundung) aller bez_curves auch parallel vs seriell

# TODO: curve auch generell mit ground truth vergleichen (vorgerechnet)  (Approx)

# TODO: Intersect testen

# TODO: collision_check testen (dies ist curveunabhaengig)

# TODO: curve_collision_check testen mit verschiedenen BezCurves (hier einfach mit parametrize)

# TODO: single_forward_difference testen

# TODO: all_forward_differences testen

# TODO: derivative_bezier_curve testen

# TODO: barycentric_combination_bezier testen

# TODO: __call__ testen, explizit mit wylden intervallen

# TODO: Kommutativitaet von Skalarmultiplikation testen (ggf via Hypothesis)

# TODO: BezierCurveApprox Sondertests:
#   - Klappt __add__ mit Approx sowie nicht-Approx?
#   - Gehen bei __add__ (und auch generell und so) die cnt_ts kaputt?
#   - (Nicht als Test schreiben) Klappt eigentlich plot? show_funcs mit nicht Approxes?