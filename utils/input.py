import cv2
import numpy as np
from PIL import Image
from cv2 import imread
from pathlib import Path, WindowsPath
from functools import partial, wraps
from collections.abc import Generator


def _is_single_input(x):
    '''Test if input is iterable but not a str or numpy array'''
    for type_ in (list, tuple, Generator):
        if isinstance(x, type_):
            return False
    return True


def handle_single_input(preprocess_hook=lambda x: x):
    def decorator(func):
        class decorated_func:
            def __call__(self, *args, ismethod=False, **kwargs):
                input_index = 1 if ismethod else 0
                input_ = args[input_index]
                self.is_single_input = _is_single_input(input_)
                input_ = self.pack_single_iterable(input_)
                input_ = list(map(preprocess_hook, input_))
                args = list(args)
                args[input_index] = input_
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    raise TypeError(
                        'Perhaps you function does not accept an Iterable as input?'
                    ) from e
                result = self.unpack_single_iterable(result)
                return result

            # Descriptor protocol to detect when decorating method Ref: https://docs.python.org/3/howto/descriptor.html
            def __get__(self, instance, _):
                return partial(self.__call__, instance, ismethod=True)

            def pack_single_iterable(self, input_):
                if self.is_single_input:
                    input_ = [input_]
                return input_

            def unpack_single_iterable(self, input_):
                if self.is_single_input:
                    [input_] = input_
                return input_

        return decorated_func()

    return decorator

def _is(type_):
    return lambda x: isinstance(x, type_)

def _is_windows_path(x):
    try:
        return _is(WindowsPath)(Path(x))
    except:
        return False

def imread_windows(path):
    image = bytearray(open(path, 'rb').read())
    image = np.asarray(image, 'uint8')
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image

def imread_buffer(buffer_):
    image = np.frombuffer(buffer_, dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image

def cast_image_to_array(x):
    handlers = {
        _is_windows_path: imread_windows,
        _is(str): imread,
        _is(Path): lambda x: imread(str(x)),
        _is(bytes): imread_buffer,
        _is(np.ndarray): np.array,
        _is(Image.Image): np.array,
    }
    for condition, handler in handlers.items():
        if condition(x):
            return handler(x)
    raise TypeError(f'Unsupported image type {type(input_)}')