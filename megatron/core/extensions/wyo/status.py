from enum import Enum

# 定义一个枚举类
class STATUS(Enum):
    TRACE = 1
    RUN = 2


CURRENT_STATS = STATUS.TRACE

def is_current_status_trace():
    return CURRENT_STATS==STATUS.TRACE

def set_current_status_trace():
    global CURRENT_STATS
    CURRENT_STATS = STATUS.TRACE

def set_current_status_run():
    global CURRENT_STATS
    CURRENT_STATS = STATUS.RUN