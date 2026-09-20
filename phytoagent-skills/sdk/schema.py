"""Offline JSON Schema validation and package-contained schema loading."""

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from referencing import Registry

from sdk.exceptions import ContractError


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    """Reject duplicate keys and non-standard NaN/Infinity JSON values."""
    def reject_constant(value):
        raise ValueError(f"Not a JSON number: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=reject_constant,
        )
        _check_json_value(value)
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContractError(f"Cannot read JSON {path.name}: {exc}") from exc


def package_file(package_dir: Path, relative: str) -> Path:
    """Resolve a portable relative path, refusing traversal and symlinks."""
    if not isinstance(relative, str) or not relative:
        raise ContractError("A non-empty package-relative path is required")
    parts = relative.split("/")
    if "\\" in relative or ":" in relative or any(p in ("", ".", "..") for p in parts):
        raise ContractError(f"Unsafe package path: {relative}")
    if PurePosixPath(relative).is_absolute():
        raise ContractError(f"Absolute package path is forbidden: {relative}")
    root = package_dir.resolve()
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ContractError(f"Linked package resource is forbidden: {relative}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ContractError(f"Missing package file: {relative}")
    return path


def _pointer(document: Any, fragment: str) -> Any:
    if fragment == "":
        return document
    if not fragment.startswith("/"):
        raise ContractError("Use a JSON pointer, for example schema.json#/input")
    node = document
    try:
        for token in fragment[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            node = node[int(token)] if isinstance(node, list) else node[token]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ContractError(f"Unknown JSON pointer: #{fragment}") from exc
    return node


def load_schema(package_dir: Path, reference: str) -> dict:
    filename, separator, fragment = reference.partition("#")
    schema = _pointer(read_json(package_file(package_dir, filename)), fragment if separator else "")
    check_schema(schema)
    return schema


def _schema_nodes(node):
    # Traverse only schema-valued keywords; property names and examples are data.
    yield node
    if not isinstance(node, dict):
        return
    for keyword in ("properties", "patternProperties", "$defs", "dependentSchemas"):
        for child in node.get(keyword, {}).values():
            yield from _schema_nodes(child)
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        for child in node.get(keyword, []):
            yield from _schema_nodes(child)
    for keyword in ("additionalProperties", "unevaluatedProperties", "propertyNames", "items",
                    "unevaluatedItems", "contains", "not", "if", "then", "else"):
        if keyword in node:
            yield from _schema_nodes(node[keyword])


def check_schema(schema: dict) -> None:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ContractError("Skill input and output schemas must describe JSON objects")
    pending, checked = [schema], set()
    while pending:
        current = pending.pop()
        if id(current) in checked:
            continue
        checked.add(id(current))
        if not isinstance(current, (dict, bool)):
            raise ContractError("Schema references must point to an object or boolean schema")
        try:
            Draft202012Validator.check_schema(current)
        except SchemaError as exc:
            raise ContractError(f"Invalid JSON Schema: {exc.message}") from exc
        for node in _schema_nodes(current):
            if not isinstance(node, dict):
                continue
            if node.get("$schema", Draft202012Validator.META_SCHEMA["$id"]) != Draft202012Validator.META_SCHEMA["$id"]:
                raise ContractError("Only JSON Schema draft 2020-12 is supported")
            if "$id" in node or "$dynamicRef" in node:
                raise ContractError("Schema IDs and dynamic references are not supported in SDK v0.1")
            if "$ref" in node:
                ref = node["$ref"]
                if not (ref == "#" or ref.startswith("#/")):
                    raise ContractError("Schema references must be local JSON pointers; network retrieval is disabled")
                # Also inspect targets outside standard schema-valued keywords.
                pending.append(_pointer(schema, ref[1:]))


def _check_json_value(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _check_json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _check_json_value(item)
        return
    raise ContractError("Payload must contain only finite JSON values and string object keys")


def validate_payload(payload: Any, schema: dict, *, label: str) -> None:
    _check_json_value(payload)
    try:
        # A registry without a retriever cannot download remote schema resources.
        validator = Draft202012Validator(schema, format_checker=FormatChecker(), registry=Registry())
        error = next(validator.iter_errors(payload), None)
    except Exception as exc:
        raise ContractError(f"{label}: schema reference could not be resolved") from exc
    if error:
        location = "/".join(str(part) for part in error.absolute_path) or "$"
        raise ContractError(f"{label} at {location}: {error.message}")
