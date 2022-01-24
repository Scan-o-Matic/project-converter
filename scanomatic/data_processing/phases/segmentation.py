import operator
from itertools import izip

import numpy as np
from enum import Enum
from scanomatic.models.phases_models import SegmentationModel


__EMPTY_FILT = np.array([]).astype(bool)


class CurvePhases(Enum):
    """Phases of curves recognized

    Attributes:
        CurvePhases.Multiple: Several types for same position, to be
            considered as an error.
        CurvePhases.Undetermined: Positions yet to be classified
            or not fulfilling any classification
        CurvePhases.Flat: Positions that exhibit no growth or collapse
        CurvePhases.GrowthAcceleration: Positions that are
            characterized by a positive second derivative
            and positive derivative.
        CurvePhases.GrowthRetardation: Positions that are
            characterized by a negative second derivative
            and positive derivative.
        CurvePhases.Impulse: Close to linear segment with growth.
        CurvePhases.Collapse: Close to linear segment with decreasing
            population size.
        CurvePhases.CollapseAcceleration: Positions that are
            characterized by a positive second derivative
            and negative derivative.
        CurvePhases.CollapseRetardation: Positions that are
            characterized by a negative second derivative
            and negative derivative.
        CurvePhases.UndeterminedNonLinear: Positions of curves that
            have only been determined not to be linear.
        CurvePhases.UndeterminedNonFlat: Positions that are not flat
            but whose properties otherwise has yet to be determined

    """
    Multiple = -1
    """:type : CurvePhases"""
    Undetermined = 0
    """:type : CurvePhases"""
    Flat = 1
    """:type : CurvePhases"""
    GrowthAcceleration = 2
    """:type : CurvePhases"""
    GrowthRetardation = 3
    """:type : CurvePhases"""
    Impulse = 4
    """:type : CurvePhases"""
    Collapse = 5
    """:type : CurvePhases"""
    CollapseAcceleration = 6
    """:type : CurvePhases"""
    CollapseRetardation = 7
    """:type : CurvePhases"""
    UndeterminedNonLinear = 8
    """:type : CurvePhases"""
    UndeterminedNonFlat = 9
    """:type : CurvePhases"""


class Thresholds(Enum):
    """Thresholds used by the phase algorithm

    Attributes:
        Thresholds.LinearModelExtension:
            Factor for impulse and collapse slopes to be
            considered equal to max/min point.
        Threshold.PhaseMinimumLength:
            The number of measurements needed for a segment to be
            considered detected.
        Thresholds.FlatlineSlopRequirement:
            Maximum slope for something to be flatline.
        Thresholds.UniformityThreshold:
            The fraction of positions considered that must agree on a
            certain direction of the first or second derivative.
        Thresholds.UniformityTestMinSize:
            The number of measurements included in the
            `UniformityThreshold` test.
        Thresholds.NonFlatLinearMinimumLength:
            Minimum length of collapse or impulse

    """
    LinearModelExtension = 0
    """:type : Thresholds"""
    PhaseMinimumLength = 1
    """:type : Thresholds"""
    FlatlineSlopRequirement = 2
    """:type : Thresholds"""
    UniformityThreshold = 3
    """:type : Thresholds"""
    UniformityTestMinSize = 4
    """:type : Thresholds"""
    SecondDerivativeSigmaAsNotZero = 5
    """:type : Thresholds"""
    NonFlatLinearMinimumLength = 7
    """:type : Thresholds"""
    NonFlatLinearMinimumYield = 8
    """:type : Thresholds"""
    NonFlatLinearMergeLengthMax = 9
    """:type : Thresholds"""
    LinearityPeak = 10
    """:type : Thresholds"""


class PhaseEdge(Enum):
    """Segment edges

    Attributes:
        PhaseEdge.Left: Left edge
        PhaseEdge.Right: Right edge
        PhaseEdge.Intelligent: Most interesting edge
    """
    Left = 0
    """:type : PhaseEdge"""
    Right = 1
    """:type : PhaseEdge"""
    Intelligent = 2
    """:type : PhaseEdge"""


DEFAULT_THRESHOLDS = {
    Thresholds.LinearModelExtension: 0.05,
    Thresholds.PhaseMinimumLength: 3,
    Thresholds.NonFlatLinearMinimumLength: 5,
    Thresholds.FlatlineSlopRequirement: 0.02,
    Thresholds.UniformityThreshold: 0.4,
    Thresholds.UniformityTestMinSize: 7,
    Thresholds.SecondDerivativeSigmaAsNotZero: 0.15,
    Thresholds.NonFlatLinearMinimumYield: 0.1,
    Thresholds.NonFlatLinearMergeLengthMax: 0,
    Thresholds.LinearityPeak: 3}


def is_detected_non_linear(phase_type):

    return phase_type in (CurvePhases.GrowthAcceleration, CurvePhases.GrowthRetardation,
                          CurvePhases.CollapseAcceleration, CurvePhases.CollapseRetardation)


def is_detected_linear(phase_type):

    return phase_type in (CurvePhases.Flat, CurvePhases.Collapse, CurvePhases.Impulse)


def is_undetermined(phase_type):

    return phase_type in (CurvePhases.Undetermined, CurvePhases.UndeterminedNonFlat, CurvePhases.UndeterminedNonLinear,
                          CurvePhases.Multiple)


def segment(segmentation_model, thresholds=None):
    """Iteratively segments a log2_curve into its component CurvePhases

    Args:
        segmentation_model (scanomatic.models.phases_models.SegmentationModel):
            A data model with information
        thresholds:
            The thresholds dictionary to be used.
    """

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # IMPORTANT, should be before having set flat so that there are no edge conditions.
    extensions, _ = get_linear_non_flat_extension_per_position(segmentation_model, thresholds)

    # Mark all flats
    _set_flat_segments(segmentation_model, thresholds)

    yield None

    set_nonflat_linearity_segments(segmentation_model, extensions, thresholds)

    segmentation_model.phases[
        (segmentation_model.phases == CurvePhases.Undetermined.value) |
        (segmentation_model.phases == CurvePhases.UndeterminedNonFlat.value)] = \
        CurvePhases.UndeterminedNonLinear.value

    # Try to classify remaining positions as non linear phases
    for filt in _get_candidate_segment(segmentation_model.phases, test_value=CurvePhases.UndeterminedNonLinear.value):

        phase = _set_nonlinear_phase_type(segmentation_model, thresholds, filt, PhaseEdge.Intelligent)

        yield None

        # If currently considered segment had no phase then it is undetermined
        if phase is CurvePhases.Undetermined:

            segmentation_model.phases[filt] = phase.value
            yield None

    # If there's an offset assume phase carries to edge
    if segmentation_model.offset:
        segmentation_model.phases[:segmentation_model.offset] = \
            segmentation_model.phases[segmentation_model.offset]
        segmentation_model.phases[-segmentation_model.offset:] = \
            segmentation_model.phases[-segmentation_model.offset - 1]
        yield None

    # Bridge neighbouring segments of same type if gap is one
    _fill_undefined_gaps(segmentation_model.phases)


def get_data_needed_for_segmentation(phenotyper_object, plate, pos, thresholds, model=None):
    """Builds a segmentation model

    Args:
        phenotyper_object (scanomatic.data_processing.Phenotyper):
            The projects phenotyer
        plate (int):
            Plate index, zero-based
        pos (Tuple[int]):
            Row and column of position considered
        thresholds (dict):
            Set of thresholds to be used.
        model(scanomatic.models.settings_models.SegmentationModel):
            If a model should be reused, else it is created.
    Returns (scanomatic.models.settings_models.SegmentationModel):
        Data container with information needed for segmentation
    """
    raise NotImplementedError("This has been removed to keep the converter minimal.")


def get_curve_classification_in_steps(phenotyper, plate, position, thresholds=None):
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    model = get_data_needed_for_segmentation(phenotyper, plate, position, thresholds)
    states = [model.phases.copy()]
    for _ in segment(model, thresholds):
        if not (model.phases == states[-1]).all():
            states.append(model.phases.copy())

    return states, model


def _get_flanks(phases, filt):

    n = filt.sum()
    if n == 0:
        return None, None,
    elif n == 1:
        left = right = np.where(filt)[0][0]
    else:
        left, right = np.where(filt)[0][0::n - 1]

    if left > 0 and right < phases.size - 2:
        return phases[left - 1], phases[right + 1]
    elif left > 0:
        return phases[left - 1], None
    elif right < phases.size - 2:
        return None, phases[right + 1]
    else:
        return None, None


def _fill_undefined_gaps(phases):
    """Fills in undefined gaps if same phase on each side

    Maximum gap size is 1

    :param phases: The phase classification array
    """

    undefined, = np.where(phases == CurvePhases.Undetermined.value)
    last_index = phases.size - 1

    # If the log2_curve is just two measurements this makes little sense
    if last_index < 2:
        return

    for loc in undefined:

        if loc == 0:
            if phases[1] != CurvePhases.Undetermined.value:
                phases[loc] = phases[loc + 1]
        elif loc == last_index:
            if phases[loc - 1] != CurvePhases.Undetermined.value:
                phases[loc] = phases[loc - 1]
        elif phases[loc - 1] == phases[loc + 1] and phases[loc + 1] != CurvePhases.Undetermined.value and \
                phases[loc + 1] not in (CurvePhases.Impulse.value, CurvePhases.Collapse.value):
            phases[loc] = phases[loc + 1]


def _get_candidate_segment(complex_segment, test_value=True):
    """While complex_segment contains any test_value the first
    segment of such will be returned as a boolean array

    :param complex_segment: an array
    :param test_value: the value to look for

    :rtype : numpy.ndarray
    """
    raise NotImplementedError("This has been removed to keep the converter minimal.")


def classifier_flat(model):

    return CurvePhases.Flat, _bridge_canditates(model.dydt_signs == 0)


def _set_flat_segments(model, thresholds):

    model.phases[...] = CurvePhases.UndeterminedNonFlat.value
    _, flats = classifier_flat(model)
    for length, left, right in izip(*_get_candidate_lengths_and_edges(flats)):
        if length >= thresholds[Thresholds.PhaseMinimumLength]:
            model.phases[left: right] = CurvePhases.Flat.value


def get_tangent_proximity(model, loc, thresholds):

    # Getting back the sign and values for linear model
    loc_slope = model.dydt[loc]
    loc_value = model.log2_curve[loc]
    loc_time = model.times[loc]

    if np.ma.is_masked(loc_value) or np.ma.is_masked(loc_slope):
        return np.zeros_like(model.times, dtype=bool)

    # Tangent at max
    tangent = (model.times - loc_time) * loc_slope + loc_value

    # Find all candidates
    return (np.abs(model.log2_curve - tangent) <
            np.abs(thresholds[Thresholds.LinearModelExtension] * loc_slope)).filled(False)


def _validate_linear_non_flat_phase(model, elected, phase, thresholds):
    if phase is CurvePhases.Undetermined or elected.sum() < thresholds[Thresholds.NonFlatLinearMinimumLength]:
        # if model.pos == (8, 3):
        #     print("***Failed phase, too short ({3}, {4}) {0} / {1} < {2}".format(
        #         phase, elected.sum(), thresholds[Thresholds.NonFlatLinearMinimumLength], model.plate, model.pos))
        return False

    # Get first and last index of elected stretch
    left, right = np.where(elected)[0][0::elected.sum() - 1]
    if model.offset:

        if (model.log2_curve[model.offset: -model.offset][right] -
                model.log2_curve[model.offset: -model.offset][left]) * \
                (-1 if phase is CurvePhases.Collapse else 1) < \
                thresholds[Thresholds.NonFlatLinearMinimumYield]:

            # print("***Failed phase ({2}, {3}): {0:.2f}".format(
            #    np.abs(model.log2_curve[left] - model.log2_curve[right]), None, model.plate, model.pos))

            return False
    else:

        if (model.log2_curve[right] - model.log2_curve[left]) * (-1 if phase is CurvePhases.Collapse else 1) < \
                thresholds[Thresholds.NonFlatLinearMinimumYield]:

            # print("***Failed phase ({2}, {3}): {0:.2f}".format(
            #    np.abs(model.log2_curve[left] - model.log2_curve[right]), None, model.plate, model.pos))

            return False

    """
    print("*Good phase ({2}, {3}): {0:.2f}, {1:.2f}".format(
        model.log2_curve[left], model.log2_curve[right], model.plate, model.pos))
    """
    return True


def classifier_nonflat_linear(model, thresholds, filt):
    """

    Args:
        model (scanomatic.models.phases_models.SegmentationModel):
            Data container for the log2_curve.
        thresholds (dict):
            Set of thresholds to use
        filt (numpy.ndarray):
            Boolean arrays of what positions to consider

    Returns:

    """
    raise NotImplementedError("This has been removed to keep the converter minimal.")


def _set_nonflat_linear_segment(model, thresholds):
    """

    Args:
        model (scanomatic.models.phases_models.SegmentationModel):
            Data container for the log2_curve
        thresholds (dict):
            Set of thresholds used.

    Returns:

    """
    raise NotImplementedError("This has been removed to keep the converter minimal.")


def _bridge_canditates(candidates, structure=(True, True, True, True, True)):
    raise NotImplementedError("This has been removed to keep the converter minimal.")


def classifier_nonlinear(model, thresholds, filt, test_edge):
    """ Classifies non-linear segments

    Args:
        model (scanomatic.models.phases_models.SegmentationModel):
            Data container for the log2_curve
        thresholds (dict):
            The set of thresholds used
        filt (numpy.ndarray):
            A boolean vector for what positions are considered
        test_edge:
            Which edge to test of the filted segment.

    Returns:

    """
    raise NotImplementedError("This has been removed to keep the converter minimal.")


def classify_non_linear_segment_type(model, thresholds, filt, test_edge):
    """Classifies the primary non-linear phase in a segment

    Args:
        model (scanomatic.models.phases_models.SegmentationModel):
            Data container for the log2_curve
        thresholds (dict):
            Set of thresholds used
        filt (numpy.ndarrya):
            Boolean array defining the segment of interest
        test_edge:
            Which side of the segment is of most concern

    Returns (CurvePhases):
        The type of phase most likely present.
    """
    phase = CurvePhases.Undetermined

    if any((v < thresholds[Thresholds.PhaseMinimumLength] for v in
            (filt.sum(), (~model.dydt[filt].mask).sum(), (~model.d2yd2t[filt].mask).sum()))):

        return CurvePhases.Undetermined

    # Define type at one of the edges
    if test_edge is PhaseEdge.Intelligent:

        # This takes a rough estimate of which side is more interesting
        # based on the location of the steepest slope
        steepest_loc = np.abs(model.dydt[filt]).argmax()
        test_edge = PhaseEdge.Left if steepest_loc / float(filt.sum()) < 0.5 else PhaseEdge.Right

    if test_edge is PhaseEdge.Left:
        for test_length in range(thresholds[Thresholds.PhaseMinimumLength], model.dydt.size, 4):
            d2yd2t_section = model.d2yd2t_signs[filt][:test_length]
            dydt_section = model.dydt_signs[filt][:test_length]
            phase = _classify_non_linear_segment_type(dydt_section, d2yd2t_section, thresholds)
            if phase != CurvePhases.Undetermined:
                break
    elif test_edge is PhaseEdge.Right:
        for test_length in range(thresholds[Thresholds.PhaseMinimumLength], model.dydt.size, 4):
            d2yd2t_section = model.d2yd2t_signs[filt][-test_length:]
            dydt_section = model.dydt_signs[filt][-test_length:]
            phase = _classify_non_linear_segment_type(dydt_section, d2yd2t_section, thresholds)
            if phase != CurvePhases.Undetermined:
                break

    return phase


def _set_nonlinear_phase_type(model, thresholds, filt, test_edge):
    """ Determines type of non-linear phase.

    Function filters the first and second derivatives, only looking
    at a number of measurements near one of the two edges of the
    candidate region. The signs of each (1st and 2nd derivative)
    are used to determine the type of phase.

    Note:
        Both derivatives need a sufficient deviation from 0 to be
        considered to have a sign.

    Args:
        model (scanomatic.models.phases_models.SegmentationModel) :
            Data container for the log2_curve analysed.
        thresholds (dict):
            A collection of thresholds to use
        filt (numpy.ndarray):
            Boolean array of positions considered
        test_edge (PhaseEdge):
            At which edge (left or right) of the filt the
            test should be performed


    Returns: The phase type, any of the following
        CurvePhases.Undetermined (failed detection),
        CurvePhases.GrowthAcceleration,
        CurvePhases.CollapseAcceleration,
        CurvePhases.GrowthRetardation,
        CurvePhases.CollapseRetardation

    """

    phase, candidates = classifier_nonlinear(model, thresholds, filt, test_edge)

    if phase is CurvePhases.Undetermined or candidates.sum() < thresholds[Thresholds.PhaseMinimumLength]:
        return CurvePhases.Undetermined

    if model.offset:
        model.phases[model.offset: -model.offset][candidates] = phase.value
    else:
        model.phases[candidates] = phase.value

    return phase


def _classify_non_linear_segment_type(dydt_section, d2yd2t_section, thresholds):
    """Classifies non linear segment

    Args:
        dydt_section: First derivative signs
        d2yd2t_section: Second derivative signs
        thresholds:

    Returns: CurvePhase

    """

    if d2yd2t_section.size == 0 or d2yd2t_section.sum() == 0 or dydt_section.sum() == 0:
        return CurvePhases.Undetermined

    # Classify as acceleration or retardation
    sign = np.sign(d2yd2t_section.mean())
    if sign == 0:
        return CurvePhases.Undetermined
    op = operator.le if sign < 0 else operator.ge
    value = op(d2yd2t_section, 0).mean() * sign

    if value > thresholds[Thresholds.UniformityThreshold]:
        candidate_phase_types = (CurvePhases.GrowthAcceleration, CurvePhases.CollapseRetardation)
    elif value < -thresholds[Thresholds.UniformityThreshold]:
        candidate_phase_types = (CurvePhases.GrowthRetardation, CurvePhases.CollapseAcceleration)
    else:
        return CurvePhases.Undetermined

    # Classify as acceleration or retardation
    sign = np.sign(dydt_section.mean())
    if sign == 0:
        return CurvePhases.Undetermined
    op = operator.le if sign < 0 else operator.ge
    value = op(dydt_section, 0).mean() * sign

    if value > thresholds[Thresholds.UniformityThreshold]:
        return candidate_phase_types[0]
    elif value < -thresholds[Thresholds.UniformityThreshold]:
        return candidate_phase_types[1]
    else:
        return CurvePhases.Undetermined


def _custom_filt(v, max_gap=3, min_length=3):
    raise NotImplementedError("This has been removed to keep the converter minimal.")


def get_linear_non_flat_extension_per_position(model, thresholds):
    raise NotImplementedError("This has been removed to keep the converter minimal.")


def get_barad_dur_towers(extension_lengths, filt, thresholds):
    raise NotImplementedError("This has been removed to keep the converter minimal.")


def set_nonflat_linearity_segments(model, extension_lengths, thresholds):
    raise NotImplementedError("This has been removed to keep the converter minimal.")
