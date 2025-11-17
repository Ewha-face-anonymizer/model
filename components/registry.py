"""
 나중에 여러 구현을 등록/선택할 수 있게 하는 간단한 레지스트리입니다.
 Lightweight registry utility for wiring interchangeable components.
"""
from typing import Callable, Dict, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    """Maps component names to their constructors using a decorator."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(cls: Callable[..., T]) -> Callable[..., T]:
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        if name not in self._registry:
            raise KeyError(f"Component '{name}' is not registered")
        return self._registry[name]


registry = ComponentRegistry()
