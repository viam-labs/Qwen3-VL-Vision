"""Allow torchvision to import on Python builds missing the _lzma extension.

Some pyenv / custom CPythons are built without liblzma. torchvision eagerly
imports datasets utilities that `import lzma`, which then fails with
ModuleNotFoundError: No module named '_lzma'. Transformers wraps that as a
cryptic AutoProcessor import error.

We only need `import lzma` to succeed so processor loading works; .xz dataset
helpers are unused by this module.
"""

from __future__ import annotations

import sys
import types


def ensure_lzma() -> None:
    try:
        import lzma  # noqa: F401

        return
    except ImportError:
        pass

    if "_lzma" not in sys.modules:
        stub = types.ModuleType("_lzma")

        class LZMAError(Exception):
            pass

        class LZMACompressor:
            def __init__(self, *args, **kwargs):
                raise LZMAError("_lzma extension is not available in this Python build")

            def compress(self, data: bytes) -> bytes:
                raise LZMAError("_lzma extension is not available in this Python build")

            def flush(self) -> bytes:
                raise LZMAError("_lzma extension is not available in this Python build")

        class LZMADecompressor:
            def __init__(self, *args, **kwargs):
                raise LZMAError("_lzma extension is not available in this Python build")

            def decompress(self, data: bytes) -> bytes:
                raise LZMAError("_lzma extension is not available in this Python build")

            @property
            def eof(self) -> bool:
                return True

            @property
            def unused_data(self) -> bytes:
                return b""

            @property
            def needs_input(self) -> bool:
                return False

        for name, val in (
            ("CHECK_NONE", 0),
            ("CHECK_CRC32", 1),
            ("CHECK_CRC64", 4),
            ("CHECK_SHA256", 10),
            ("CHECK_ID_MAX", 15),
            ("CHECK_UNKNOWN", 16),
            ("FILTER_LZMA1", 0x4000000000000001),
            ("FILTER_LZMA2", 0x21),
            ("FILTER_DELTA", 3),
            ("FILTER_X86", 4),
            ("FILTER_IA64", 6),
            ("FILTER_ARM", 7),
            ("FILTER_ARMTHUMB", 8),
            ("FILTER_SPARC", 9),
            ("FILTER_POWERPC", 5),
            ("FORMAT_AUTO", 0),
            ("FORMAT_XZ", 1),
            ("FORMAT_ALONE", 2),
            ("FORMAT_RAW", 3),
            ("MF_HC3", 3),
            ("MF_HC4", 4),
            ("MF_BT2", 18),
            ("MF_BT3", 19),
            ("MF_BT4", 20),
            ("MODE_FAST", 1),
            ("MODE_NORMAL", 2),
            ("PRESET_DEFAULT", 6),
            ("PRESET_EXTREME", 1 << 31),
        ):
            setattr(stub, name, val)

        stub.LZMAError = LZMAError
        stub.LZMACompressor = LZMACompressor
        stub.LZMADecompressor = LZMADecompressor
        stub._encode_filter_properties = lambda *args, **kwargs: b""
        stub._decode_filter_properties = lambda *args, **kwargs: {}
        stub.is_check_supported = lambda check_id: False
        sys.modules["_lzma"] = stub

    import lzma  # noqa: F401


ensure_lzma()
