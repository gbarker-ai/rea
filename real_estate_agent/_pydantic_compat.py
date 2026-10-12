"""Compat layer for Pydantic v2 so schemas validate identically in full and bare environments."""

from __future__ import annotations

import re
import typing
from typing import Any, Dict, List, Literal, Optional, get_args, get_origin

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError  # type: ignore
except ImportError:

    class ValidationError(ValueError):  # type: ignore[no-redef]
        def __init__(self, errors: List[Dict[str, Any]]) -> None:
            self._errors = errors
            super().__init__(errors[0]["msg"] if errors else "Validation error")

        def errors(self) -> List[Dict[str, Any]]:
            return self._errors

    def ConfigDict(**kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
        return dict(kwargs)

    class _FieldInfo:
        def __init__(self, default: Any = ..., **kwargs: Any) -> None:
            self.default = default
            self.default_factory = kwargs.get("default_factory")
            self.description = kwargs.get("description", "")
            self.ge = kwargs.get("ge")
            self.le = kwargs.get("le")
            self.min_length = kwargs.get("min_length")
            self.max_length = kwargs.get("max_length")
            self.pattern = kwargs.get("pattern")

    def Field(default: Any = ..., **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return _FieldInfo(default=default, **kwargs)

    class BaseModel:  # type: ignore[no-redef]
        model_config: Dict[str, Any] = {"extra": "forbid"}

        def __init__(self, **data: Any) -> None:
            hints = typing.get_type_hints(self.__class__)
            errors: List[Dict[str, Any]] = []
            if self.model_config.get("extra") == "forbid":
                for k in data:
                    if k not in hints:
                        errors.append({"loc": (k,), "msg": f"Extra field '{k}' is not permitted."})
            for name, ftype in hints.items():
                if name == "model_config":
                    continue
                attr = getattr(self.__class__, name, ...)
                info = attr if isinstance(attr, _FieldInfo) else None
                if name in data:
                    val = data[name]
                elif info is not None and info.default_factory is not None:
                    val = info.default_factory()
                elif info is not None and info.default is not ...:
                    val = info.default
                elif not isinstance(attr, _FieldInfo) and attr is not ...:
                    val = attr
                else:
                    errors.append({"loc": (name,), "msg": f"Field '{name}' is required."})
                    continue
                if val is not None:
                    err = self._check(name, val, ftype, info)
                    if err:
                        errors.append({"loc": (name,), "msg": err})
                        continue
                setattr(self, name, val)
            if errors:
                raise ValidationError(errors)

        @staticmethod
        def _check(name: str, val: Any, ann: Any, info: Optional[_FieldInfo]) -> Optional[str]:
            origin, args = get_origin(ann), get_args(ann)
            if origin is typing.Union:
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    return BaseModel._check(name, val, non_none[0], info)
            if origin is Literal and val not in args:
                return f"Field '{name}' must be one of {args}."
            if ann is int and (not isinstance(val, int) or isinstance(val, bool)):
                return f"Field '{name}' must be an integer."
            if ann is float and not isinstance(val, (int, float)):
                return f"Field '{name}' must be a number."
            if ann is str and not isinstance(val, str):
                return f"Field '{name}' must be a string."
            if info:
                if info.ge is not None and isinstance(val, (int, float)) and val < info.ge:
                    return f"Field '{name}' must be >= {info.ge}."
                if info.le is not None and isinstance(val, (int, float)) and val > info.le:
                    return f"Field '{name}' must be <= {info.le}."
                if info.min_length is not None and isinstance(val, str) and len(val) < info.min_length:
                    return f"Field '{name}' length must be >= {info.min_length}."
                if info.max_length is not None and isinstance(val, str) and len(val) > info.max_length:
                    return f"Field '{name}' length must be <= {info.max_length}."
                if info.pattern is not None and isinstance(val, str) and not re.match(info.pattern, val):
                    return f"Field '{name}' does not match pattern '{info.pattern}'."
            return None

        def model_dump(self) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k in typing.get_type_hints(self.__class__):
                if k != "model_config":
                    v = getattr(self, k, None)
                    out[k] = v.model_dump() if isinstance(v, BaseModel) else (
                        [i.model_dump() if isinstance(i, BaseModel) else i for i in v] if isinstance(v, list) else v
                    )
            return out

        @classmethod
        def model_json_schema(cls) -> Dict[str, Any]:
            props, req = {}, []
            for k, t in typing.get_type_hints(cls).items():
                if k == "model_config":
                    continue
                attr = getattr(cls, k, ...)
                props[k] = {"type": str(t), "description": getattr(attr, "description", "")}
                if isinstance(attr, _FieldInfo) and attr.default is ... and attr.default_factory is None:
                    req.append(k)
            return {
                "title": cls.__name__,
                "type": "object",
                "additionalProperties": cls.model_config.get("extra") != "forbid",
                "properties": props,
                "required": req,
            }

__all__ = ["BaseModel", "ConfigDict", "Field", "ValidationError"]
