"""Compat layer for Pydantic v2 so schemas validate identically in both full and bare environments."""

from __future__ import annotations

import re
import typing
from typing import Any, Dict, List, Literal, Optional, get_args, get_origin

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError  # type: ignore
except ImportError:

    class ValidationError(ValueError):  # type: ignore[no-redef]
        """Compatible Pydantic ValidationError when running in an offline stdlib environment."""

        def __init__(self, errors: List[Dict[str, Any]]) -> None:
            self._errors = errors
            super().__init__(errors[0]["msg"] if errors else "Validation error")

        def errors(self) -> List[Dict[str, Any]]:
            return self._errors

    def ConfigDict(**kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
        return dict(kwargs)

    class _FieldInfo:
        def __init__(
            self,
            default: Any = ...,
            *,
            default_factory: Any = None,
            description: str = "",
            ge: Optional[float] = None,
            le: Optional[float] = None,
            min_length: Optional[int] = None,
            max_length: Optional[int] = None,
            pattern: Optional[str] = None,
        ) -> None:
            self.default = default
            self.default_factory = default_factory
            self.description = description
            self.ge = ge
            self.le = le
            self.min_length = min_length
            self.max_length = max_length
            self.pattern = pattern

    def Field(  # type: ignore[no-redef]
        default: Any = ...,
        *,
        default_factory: Any = None,
        description: str = "",
        ge: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
    ) -> Any:
        return _FieldInfo(
            default=default,
            default_factory=default_factory,
            description=description,
            ge=ge,
            le=le,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
        )

    class BaseModel:  # type: ignore[no-redef]
        """Strict schema validator & JSON Schema generator compatible with Pydantic v2."""

        model_config: Dict[str, Any] = {"extra": "forbid"}

        def __init__(self, **data: Any) -> None:
            hints = typing.get_type_hints(self.__class__)
            errors: List[Dict[str, Any]] = []

            if self.model_config.get("extra") == "forbid":
                for k in data:
                    if k not in hints:
                        errors.append({"loc": (k,), "msg": f"Extra field '{k}' is not permitted."})

            for field_name, field_type in hints.items():
                if field_name == "model_config":
                    continue
                attr_val = getattr(self.__class__, field_name, ...)
                field_info = attr_val if isinstance(attr_val, _FieldInfo) else None

                if field_name in data:
                    val = data[field_name]
                elif field_info is not None:
                    if field_info.default_factory is not None:
                        val = field_info.default_factory()
                    elif field_info.default is not ...:
                        val = field_info.default
                    else:
                        errors.append({"loc": (field_name,), "msg": f"Field '{field_name}' is required."})
                        continue
                elif attr_val is not ...:
                    val = attr_val
                else:
                    errors.append({"loc": (field_name,), "msg": f"Field '{field_name}' is required."})
                    continue

                if val is not None:
                    err = self._validate_field(field_name, val, field_type, field_info)
                    if err:
                        errors.append({"loc": (field_name,), "msg": err})
                        continue
                setattr(self, field_name, val)

            if errors:
                raise ValidationError(errors)

        @staticmethod
        def _validate_field(
            name: str, val: Any, annotation: Any, info: Optional[_FieldInfo]
        ) -> Optional[str]:
            origin = get_origin(annotation)
            args = get_args(annotation)

            if origin is typing.Union:
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    return BaseModel._validate_field(name, val, non_none[0], info)

            if origin is Literal:
                if val not in args:
                    return f"Field '{name}' must be one of {args}, got '{val}'."

            if annotation is int and (not isinstance(val, int) or isinstance(val, bool)):
                return f"Field '{name}' must be an integer."
            if annotation is float and not isinstance(val, (int, float)):
                return f"Field '{name}' must be a number."
            if annotation is str and not isinstance(val, str):
                return f"Field '{name}' must be a string."

            if info is not None:
                if info.ge is not None and isinstance(val, (int, float)) and val < info.ge:
                    return f"Field '{name}' must be >= {info.ge} (got {val})."
                if info.le is not None and isinstance(val, (int, float)) and val > info.le:
                    return f"Field '{name}' must be <= {info.le} (got {val})."
                if info.min_length is not None and isinstance(val, str) and len(val) < info.min_length:
                    return f"Field '{name}' length must be >= {info.min_length}."
                if info.max_length is not None and isinstance(val, str) and len(val) > info.max_length:
                    return f"Field '{name}' length must be <= {info.max_length}."
                if info.pattern is not None and isinstance(val, str):
                    if not re.match(info.pattern, val):
                        return f"Field '{name}' ('{val}') does not match pattern '{info.pattern}'."
            return None

        def model_dump(self) -> Dict[str, Any]:
            hints = typing.get_type_hints(self.__class__)
            out: Dict[str, Any] = {}
            for k in hints:
                if k == "model_config":
                    continue
                v = getattr(self, k, None)
                if isinstance(v, BaseModel):
                    out[k] = v.model_dump()
                elif isinstance(v, list):
                    out[k] = [item.model_dump() if isinstance(item, BaseModel) else item for item in v]
                else:
                    out[k] = v
            return out

        @classmethod
        def model_json_schema(cls) -> Dict[str, Any]:
            hints = typing.get_type_hints(cls)
            props: Dict[str, Any] = {}
            required: List[str] = []
            for k, t in hints.items():
                if k == "model_config":
                    continue
                attr_val = getattr(cls, k, ...)
                desc = attr_val.description if isinstance(attr_val, _FieldInfo) else ""
                props[k] = {"type": str(t), "description": desc}
                if isinstance(attr_val, _FieldInfo) and attr_val.default is ... and attr_val.default_factory is None:
                    required.append(k)
            return {
                "title": cls.__name__,
                "type": "object",
                "additionalProperties": cls.model_config.get("extra") != "forbid" and True or False,
                "properties": props,
                "required": required,
            }


__all__ = ["BaseModel", "ConfigDict", "Field", "ValidationError"]
