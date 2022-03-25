#  Copyright 2022 Feedzai
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Helper functions to check installed dependencies.
"""
from typing import Callable, Any, Type


def is_torch_installed() -> bool:
    def func():
        import torch
    return not check_if_raises(func, ImportError)


def is_tensorflow_installed() -> bool:
    def func():
        import tensorflow
    return not check_if_raises(func, ImportError)


def check_if_raises(func: Callable[[], Any], err: Type) -> bool:
    try:
        func()
    except err:
        return True
    return False
