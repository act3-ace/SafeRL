import saferl.environment
import saferl.aerospace

from saferl.aerospace.models import Dubins2dPlatform
from saferl.aerospace.tasks import DubinsObservationProcessor, RejoinRewardProcessor, RejoinFirstTimeRewardProcessor,\
    TimeRewardProcessor, RejoinDistanceChangeRewardProcessor, FailureRewardProcessor, SuccessRewardProcessor, \
    DubinsInRejoin, DubinsInRejoinPrev, DubinsRejoinTime, DubinsTimeElapsed, DubinsLeadDistance, DubinsFailureStatus, \
    DubinsSuccessStatus
from saferl.environment.models.geometry import RelativeCircle

saferl_lookup = {
    "Dubins2dPlatform": Dubins2dPlatform,
    "RelativeCircle": RelativeCircle,
    "DubinsObservationProcessor": DubinsObservationProcessor,
    "RejoinRewardProcessor": RejoinRewardProcessor,
    "RejoinFirstTimeRewardProcessor": RejoinFirstTimeRewardProcessor,
    "TimeRewardProcessor": TimeRewardProcessor,
    "RejoinDistanceChangeRewardProcessor": RejoinDistanceChangeRewardProcessor,
    "FailureRewardProcessor": FailureRewardProcessor,
    "SuccessRewardProcessor": SuccessRewardProcessor,
    "DubinsInRejoin": DubinsInRejoin,
    "DubinsInRejoinPrev": DubinsInRejoinPrev,
    "DubinsRejoinTime": DubinsRejoinTime,
    "DubinsTimeElapsed": DubinsTimeElapsed,
    "DubinsLeadDistance": DubinsLeadDistance,
    "DubinsFailureStatus": DubinsFailureStatus,
    "DubinsSuccessStatus": DubinsSuccessStatus
}
