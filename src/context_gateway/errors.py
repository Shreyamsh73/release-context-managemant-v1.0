"""Gateway errors and OpenAI-style error rendering."""

from __future__ import annotations

from typing import Any

_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    404: "not_found_error",
}


class GatewayError(Exception):
    """An error the gateway raises on purpose. Rendered as an OpenAI-shaped JSON body."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        error_type: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.error_type = error_type or _TYPE_BY_STATUS.get(status_code, "server_error")
        self.param = param

    def body(self) -> dict[str, Any]:
        """Return the JSON body for this error."""
        return error_body(self.message, self.error_type, self.code, self.param)


def error_body(
    message: str, error_type: str, code: str | None, param: str | None = None
) -> dict[str, Any]:
    """Build the OpenAI error envelope."""
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


def truncate_message(message: str, limit: int = 500) -> str:
    """Cut long provider messages so they stay small and cannot leak much."""
    if len(message) <= limit:
        return message
    return message[:limit] + "..."
