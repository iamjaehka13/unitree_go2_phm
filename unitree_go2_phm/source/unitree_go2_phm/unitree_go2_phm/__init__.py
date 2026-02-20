# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unitree Go2 PHM extension package for Isaac Lab.
"""

# Register Gym environments via module import side effects.
from . import tasks  # noqa: F401

# Register UI extensions only when Omni UI modules are available.
try:
    from . import ui_extension_example  # noqa: F401
except (ImportError, ModuleNotFoundError):
    # Keep non-UI script imports working in headless/python-only contexts.
    pass
