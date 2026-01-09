"""DSPy optimizer registry and wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type

import dspy

from .config import OptimizerConfig


class DSPyOptimizerWrapper(ABC):
    """Base wrapper for DSPy optimizers."""

    name: str = ""

    def __init__(self, metric: Callable, config: OptimizerConfig):
        """
        Initialize optimizer wrapper.

        Parameters
        ----------
        metric : Callable
            DSPy metric function for optimization
        config : OptimizerConfig
            Optimizer configuration
        """
        self.metric = metric
        self.config = config

    @abstractmethod
    def build(self) -> Any:
        """Build the underlying DSPy optimizer."""
        pass

    def compile(
        self,
        student: dspy.Module,
        trainset: List,
        valset: List,
        teacher: dspy.Module | None = None,
    ) -> dspy.Module:
        """
        Run optimization and return compiled module.

        Parameters
        ----------
        student : dspy.Module
            Student module to optimize
        trainset : List
            Training examples
        valset : List
            Validation examples
        teacher : dspy.Module, optional
            Teacher module for bootstrapping

        Returns
        -------
        dspy.Module
            Compiled/optimized module
        """
        optimizer = self.build()
        return optimizer.compile(
            student=student,
            teacher=teacher,
            trainset=trainset,
            valset=valset,
        )


class OptimizerRegistry:
    """Registry for DSPy optimizer wrappers."""

    _registry: Dict[str, Type[DSPyOptimizerWrapper]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register an optimizer wrapper.

        Parameters
        ----------
        name : str
            Name to register the optimizer under
        """

        def decorator(wrapper_cls: Type[DSPyOptimizerWrapper]):
            cls._registry[name] = wrapper_cls
            wrapper_cls.name = name
            return wrapper_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[DSPyOptimizerWrapper]:
        """
        Get optimizer wrapper class by name.

        Parameters
        ----------
        name : str
            Registered optimizer name

        Returns
        -------
        Type[DSPyOptimizerWrapper]
            Optimizer wrapper class

        Raises
        ------
        ValueError
            If optimizer name is not registered
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown optimizer: {name}. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered optimizer names."""
        return list(cls._registry.keys())


@OptimizerRegistry.register("BootstrapFewShot")
class BootstrapFewShotWrapper(DSPyOptimizerWrapper):
    """Wrapper for BootstrapFewShot optimizer."""

    def build(self):
        params = self.config.params
        return dspy.teleprompt.BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=params.get("max_bootstrapped_demos", 4),
            max_labeled_demos=params.get("max_labeled_demos", 4),
            max_rounds=params.get("max_rounds", 1),
        )


@OptimizerRegistry.register("BootstrapFewShotWithRandomSearch")
class BootstrapRandomSearchWrapper(DSPyOptimizerWrapper):
    """Wrapper for BootstrapFewShotWithRandomSearch optimizer."""

    def build(self):
        params = self.config.params
        return dspy.teleprompt.BootstrapFewShotWithRandomSearch(
            metric=self.metric,
            max_bootstrapped_demos=params.get("max_bootstrapped_demos", 2),
            max_labeled_demos=params.get("max_labeled_demos", 2),
            num_candidate_programs=params.get("num_candidate_programs", 5),
            num_threads=params.get("num_threads", 4),
        )


@OptimizerRegistry.register("MIPRO")
class MIPROWrapper(DSPyOptimizerWrapper):
    """Wrapper for MIPROv2 optimizer."""

    def build(self):
        params = self.config.params
        return dspy.teleprompt.MIPROv2(
            metric=self.metric,
            max_bootstrapped_demos=params.get("max_bootstrapped_demos", 3),
            max_labeled_demos=params.get("max_labeled_demos", 3),
            num_candidates=params.get("num_candidates", 10),
            init_temperature=params.get("init_temperature", 1.0),
            verbose=params.get("verbose", False),
            track_stats=params.get("track_stats", True),
        )
