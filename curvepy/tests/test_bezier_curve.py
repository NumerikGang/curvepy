import pytest
import numpy as np
import itertools
from curvepy.bezier_curve import *
from curvepy.tests.utility import arrayize
from dataclasses import dataclass


@dataclass
class AllBezierCurves:
    bezier_curve_sympy: BezierCurveSymPy
    bezier_curve_de_caes: BezierCurveDeCaes
    bezier_curve_bernstein: BezierCurveBernstein
    bezier_curve_horner: BezierCurveHorner
    bezier_curve_monomial: BezierCurveMonomial
    bezier_curve_approximation: BezierCurveApproximation

    @staticmethod
    def from_bezier_points(m: np.ndarray, cnt_ts: int = 1000, use_parallel: bool = False,
                           interval: Tuple[int, int] = (0, 1)):
        ...


@pytest.fixture
def use_all_curves():
    ...


# TODO: Equality (mit Rundung) aller bez_curves (jeweils zwischen 2 damit man am Funktionsnamen erkennen kann
# TODO: welche beiden

# TODO: Equality (ohne Rundung) aller bez_curves auch parallel vs seriell

# TODO: curve auch generell mit ground truth vergleichen (vorgerechnet)  (Approx)

FOUR_DISTINCT_SORTED_VALUES = [
    (-114.6046451909072, 68.07019999037828, 97.63733754004107, 181.4022906606396),
    (-165.15907908532319, -137.76317349696706, 4.569499551086864, 10.461931669695446),
    (-95.33903351509477, -3.377791426443224, 55.243150854024094, 185.88124485370747),
    (-162.12249329689334, 56.22860106599177, 171.39441229583935, 181.0643720534207),
    (-41.806181066050925, 27.55277838513686, 35.838278946778246, 79.91887437774545),
    (-169.2250943683963, 25.975556402963974, 46.601444375771365, 192.22729277260788),
    (-131.10770563271063, -78.34996795218112, -37.25603308661664, 79.42180366807719),
    (-106.4993819440181, 24.38032489912493, 97.02674238886834, 103.66324143040731),
    (-140.63523678629196, -126.31564194783653, -88.79548665862771, -71.9993208311482),
    (-121.16629936307142, -41.35199983629252, -24.242928839153734, 170.84800890598638),
    (-175.45420693913502, -162.85983004584523, -81.10319102076669, 170.50978174903577),
    (-16.878369309667335, 49.40761847913677, 73.51163006287982, 179.1502213071604),
    (42.28489094791027, 86.3665585329951, 139.59575491419912, 170.9705656979085),
    (-178.69553705790486, -174.29796158690087, -36.20937278960574, 146.25336284538815),
    (-155.69645298224626, -2.398424142189185, 55.18875614643002, 58.890173673162906),
    (-182.13333753724098, -161.69250245554065, -111.35096523468273, -111.05016273271295),
    (-177.80474295344607, -57.08289785907192, -56.46532360680118, 44.02381298765542),
    (-175.73764416571868, -7.2600612774946, 13.553048739850482, 198.11628534663004),
    (-91.80768559665906, -26.700892452600527, 35.69313638699478, 84.10221468399158),
    (-103.03238750368054, -30.97655505013114, 50.87932947381165, 90.93121981537678),
    (-152.30661956136487, -142.09393128609813, -36.14117162037064, 21.599262269513986),
    (-195.23680596977496, -71.06512889053121, 30.17781637585665, 74.9392404122554),
    (-193.9061431163597, -106.93054475837873, 116.5389278708123, 184.7398844577911),
    (-43.35969627753275, 3.175558826417074, 104.31161669569866, 159.29118588240732),
    (-182.04230540917052, -159.97104572969357, -70.99268263646715, -35.576656242224004),
    (-14.365872730829693, -10.811794098216922, 18.162645294656443, 150.39016414455307),
    (-137.03159624700234, -96.25326936540435, -45.769625179671124, 15.813046384922615),
    (-94.98826622050593, -82.73208019458514, 9.495212479441932, 82.53992083308702),
    (-184.25381863061565, 10.616932886606094, 97.55770304189946, 107.23438163885362),
    (-67.34937184724029, 2.3421457579158584, 13.398842374207163, 128.5036818198554),
    (-192.87427694033292, 115.93817253353001, 128.26605791416353, 178.33791645586615),
    (-182.70434050212089, -120.55938603203043, 40.03600811308695, 165.47459947547213),
    (-124.86738555332985, -121.5547829291344, -85.28222568103807, 41.39625472606133),
    (-131.99832849292102, -123.14767752350573, 121.67735803057332, 196.9506397947154),
    (-130.45211589747038, -77.26533618892293, -48.58435863321114, 23.740795131378974),
    (-150.9801949117121, -135.41485690345576, -104.10565374097298, -97.12824234766143),
    (-140.81112818108807, -1.3662376742552738, 57.4905864518729, 99.04997351835425),
    (121.39900978952602, 153.73277005070452, 180.19218432125137, 185.9006449722977),
    (-69.22185427158502, -13.773811991475782, 57.31892576329568, 138.36892067064286),
    (-69.64395645049373, 81.24484688878414, 102.69633910880879, 192.6809874564098),
    (-192.0183218028243, -1.6128426652080918, 51.06803326342998, 78.64787796655412),
    (-139.77851261149755, -68.10026312061274, -61.01008503662709, -9.564444087830594),
    (-79.5098744843369, -64.4315042891188, -1.1652345334959193, 189.11544053439337),
    (-181.9671014953244, -174.0004871244476, 21.41129241637026, 59.42397658560566),
    (-94.66947189654059, -75.85256460416767, 61.46601986044851, 75.08482434458256),
    (-121.00125733105801, -80.30589286402487, 20.927351426946302, 28.31474263518743),
    (-142.668036828514, -122.05310791938095, -121.27877066107669, 50.71484853790315),
    (-149.02709502706753, -142.32833611548838, -103.46566284352305, 143.9017164258588),
    (-123.44602171596888, 15.219293735911322, 113.94992119550744, 157.76146012185382),
    (-160.87547080705275, -93.07281035612975, -12.104536093860531, 4.419221202642916)
]


@pytest.mark.parametrize('x,y', arrayize([
    ((a, b), (a, b)) for a, b, _, _ in FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_all_points_are_equal(x, y):
    assert AbstractBezierCurve.intersect(x, y)


@pytest.mark.parametrize('x,y', arrayize([
    ((a, b), (b, c)) for a, b, c, _ in FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_at_least_one_equal_point(x, y):
    assert AbstractBezierCurve.intersect(x, y)


@pytest.mark.parametrize('x,y', arrayize([
    ((a, b), (c, d)) for a, b, c, d in FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_disjunct_intervals(x, y):
    assert not AbstractBezierCurve.intersect(x, y)


@pytest.mark.parametrize('x,y', arrayize([
    ((a, d), (b, c)) for a, b, c, d in FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_lies_completely_within_another(x, y):
    assert AbstractBezierCurve.intersect(x, y)


@pytest.mark.parametrize('x,y', arrayize([
    ((a, c), (b, d)) for a, b, c, d in FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_intersects_left_side(x, y):
    assert AbstractBezierCurve.intersect(x, y)


@pytest.mark.parametrize('x,y', arrayize([
    ((b, d), (a, c)) for a, b, c, d in FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_intersects_right_size(x, y):
    assert AbstractBezierCurve.intersect(x, y)


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

@pytest.mark.parametrize('approx_rounds, cnt_bezier_points', itertools.product(range(2,7), range(5,206, 20)))
def test_approx_rounds_to_cnt_ts_to_approx_rounds_equals_id(approx_rounds, cnt_bezier_points):
    assert BezierCurveApproximation.cnt_ts_to_approx_rounds(
        BezierCurveApproximation.approx_rounds_to_cnt_ts(approx_rounds, cnt_bezier_points), cnt_bezier_points
    ) == approx_rounds
