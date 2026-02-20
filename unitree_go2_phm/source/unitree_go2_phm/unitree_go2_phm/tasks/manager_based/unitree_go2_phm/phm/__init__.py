"""
PHM (Prognostics and Health Management) Core Package.
Exposes state management, physics interface, and utility functions.
"""

# 1. 핵심 상태 클래스 노출
# MDP나 Env에서 'from .phm import PHMState' 형태로 접근 가능하게 함
from .state import PHMState

# 2. Env와 연결되는 인터페이스 함수 노출
# unitree_go2_phm_env.py의 step() 및 reset() 함수에서 호출되는 핵심 함수들입니다.
from .interface import (
    init_phm_interface,
    update_phm_dynamics,
    reset_phm_interface,
    refresh_phm_sensors,
    clear_step_metrics,
)

# 3. 유틸리티 및 상수 (선택적 노출)
# MDP 관측(Observation) 함수 등에서 자주 사용되는 계산 함수들입니다.
from .utils import (
    compute_battery_voltage,
    compute_component_losses,
    compute_kinematic_accel,
)

# 4. 하위 모듈 명시적 노출 (필요 시)
# 사용자가 'import phm.constants'로 직접 접근할 수 있도록 함
from . import constants
from . import models
from . import buffers
