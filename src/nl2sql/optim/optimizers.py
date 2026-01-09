"""DSPy optimizer registry and wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type

import dspy

from .config import OptimizerConfig


class DSPyOptimizerWrapper(ABC):
    """Base wrapper for DSPy optimizers."""

    name: str = ""
    requires_metric: bool = True
    requires_trainset_in_init: bool = False

    def __init__(self, metric: Callable | None, config: OptimizerConfig):
        """
        Initialize optimizer wrapper.

        Parameters
        ----------
        metric : Callable or None
            DSPy metric function for optimization (not all optimizers need it)
        config : OptimizerConfig
            Optimizer configuration
        """
        self.metric = metric
        self.config = config

    @abstractmethod
    def build(self, **kwargs) -> Any:
        """Build the underlying DSPy optimizer."""
        pass

    def compile(
        self,
        student: dspy.Module,
        trainset: List,
        valset: List | None = None,
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
        valset : List, optional
            Validation examples
        teacher : dspy.Module, optional
            Teacher module for bootstrapping

        Returns
        -------
        dspy.Module
            Compiled/optimized module
        """
        optimizer = self.build(trainset=trainset)
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


@OptimizerRegistry.register("LabeledFewShot")
class LabeledFewShotWrapper(DSPyOptimizerWrapper):
    """
    Wrapper for LabeledFewShot optimizer.

    Simply constructs few-shot examples from provided labeled data.
    This is the simplest optimizer - no metric needed, just samples k examples.
    """

    requires_metric = False

    def build(self, **kwargs):
        params = self.config.params
        return dspy.teleprompt.LabeledFewShot(
            k=params.get("k", 16),
        )

    def compile(
        self,
        student: dspy.Module,
        trainset: List,
        valset: List | None = None,
        teacher: dspy.Module | None = None,
    ) -> dspy.Module:
        optimizer = self.build()
        return optimizer.compile(
            student=student,
            trainset=trainset,
            sample=self.config.params.get("sample", True),
        )


@OptimizerRegistry.register("BootstrapFewShot")
class BootstrapFewShotWrapper(DSPyOptimizerWrapper):
    """
    Wrapper for BootstrapFewShot optimizer.

    Uses a teacher module to generate complete demonstrations for every stage
    of your program, using labeled examples.
    """

    def build(self, **kwargs):
        params = self.config.params
        return dspy.teleprompt.BootstrapFewShot(
            metric=self.metric,
            metric_threshold=params.get("metric_threshold"),
            max_bootstrapped_demos=params.get("max_bootstrapped_demos", 4),
            max_labeled_demos=params.get("max_labeled_demos", 16),
            max_rounds=params.get("max_rounds", 1),
            max_errors=params.get("max_errors"),
        )

    def compile(
        self,
        student: dspy.Module,
        trainset: List,
        valset: List | None = None,
        teacher: dspy.Module | None = None,
    ) -> dspy.Module:
        optimizer = self.build()
        return optimizer.compile(
            student=student,
            teacher=teacher,
            trainset=trainset,
        )


@OptimizerRegistry.register("BootstrapFewShotWithRandomSearch")
class BootstrapRandomSearchWrapper(DSPyOptimizerWrapper):
    """
    Wrapper for BootstrapFewShotWithRandomSearch optimizer.

    Applies BootstrapFewShot several times with random search over generated
    demonstrations, and selects the best program based on validation performance.
    """

    def build(self, **kwargs):
        params = self.config.params
        return dspy.teleprompt.BootstrapFewShotWithRandomSearch(
            metric=self.metric,
            max_bootstrapped_demos=params.get("max_bootstrapped_demos", 2),
            max_labeled_demos=params.get("max_labeled_demos", 2),
            num_candidate_programs=params.get("num_candidate_programs", 5),
            num_threads=params.get("num_threads", 4),
        )

    def compile(
        self,
        student: dspy.Module,
        trainset: List,
        valset: List | None = None,
        teacher: dspy.Module | None = None,
    ) -> dspy.Module:
        optimizer = self.build()
        return optimizer.compile(
            student=student,
            teacher=teacher,
            trainset=trainset,
            valset=valset,
        )


@OptimizerRegistry.register("KNNFewShot")
class KNNFewShotWrapper(DSPyOptimizerWrapper):
    """
    Wrapper for KNNFewShot optimizer.

    Uses k-Nearest Neighbors to find the most similar training examples
    for each input at inference time. Requires an embedder for vectorization.

    Note: This optimizer takes trainset in __init__, not compile.
    """

    requires_metric = False
    requires_trainset_in_init = True

    def build(self, **kwargs):
        params = self.config.params
        trainset = kwargs.get("trainset", [])

        # Get or create embedder
        vectorizer = params.get("vectorizer")
        if vectorizer is None:
            # Use default sentence-transformers embedder
            embedding_model = params.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorizer = dspy.Embedder(embedding_model)

        return dspy.teleprompt.KNNFewShot(
            k=params.get("k", 3),
            trainset=trainset,
            vectorizer=vectorizer,
        )

    def compile(
        self,
        student: dspy.Module,
        trainset: List,
        valset: List | None = None,
        teacher: dspy.Module | None = None,
    ) -> dspy.Module:
        # KNNFewShot takes trainset in build (passed to __init__)
        optimizer = self.build(trainset=trainset)
        return optimizer.compile(
            student=student,
            teacher=teacher,
        )


@OptimizerRegistry.register("COPRO")
class COPROWrapper(DSPyOptimizerWrapper):
    """
    Wrapper for COPRO (Coordinate Prompt Optimization) optimizer.

    Generates and refines new instructions for each step using coordinate ascent.
    Good for optimizing instructions without needing many demonstrations.
    """

    def build(self, **kwargs):
        params = self.config.params
        return dspy.teleprompt.COPRO(
            metric=self.metric,
            breadth=params.get("breadth", 10),
            depth=params.get("depth", 3),
            init_temperature=params.get("init_temperature", 1.4),
            track_stats=params.get("track_stats", False),
        )

    def compile(
        self,
        student: dspy.Module,
        trainset: List,
        valset: List | None = None,
        teacher: dspy.Module | None = None,
    ) -> dspy.Module:
        optimizer = self.build()
        params = self.config.params

        # COPRO requires eval_kwargs
        eval_kwargs = params.get("eval_kwargs", {})
        if not eval_kwargs:
            # Provide sensible defaults
            eval_kwargs = {
                "num_threads": params.get("num_threads", 4),
                "display_progress": params.get("display_progress", True),
            }

        return optimizer.compile(
            student=student,
            trainset=trainset,
            eval_kwargs=eval_kwargs,
        )


@OptimizerRegistry.register("MIPRO")
class MIPROWrapper(DSPyOptimizerWrapper):
    """
    Wrapper for MIPROv2 optimizer.

    Generates instructions and few-shot examples using Bayesian Optimization
    to efficiently search the space of possible prompts.
    """

    def build(self, **kwargs):
        params = self.config.params
        return dspy.teleprompt.MIPROv2(
            metric=self.metric,
            auto=params.get("auto", "light"),
            num_candidates=params.get("num_candidates", 10),
            max_bootstrapped_demos=params.get("max_bootstrapped_demos", 4),
            max_labeled_demos=params.get("max_labeled_demos", 4),
            num_threads=params.get("num_threads"),
            init_temperature=params.get("init_temperature", 1.0),
            verbose=params.get("verbose", False),
            track_stats=params.get("track_stats", True),
            seed=params.get("seed", 9),
        )

    def compile(
        self,
        student: dspy.Module,
        trainset: List,
        valset: List | None = None,
        teacher: dspy.Module | None = None,
    ) -> dspy.Module:
        optimizer = self.build()
        params = self.config.params

        compile_kwargs = {
            "student": student,
            "trainset": trainset,
            "teacher": teacher,
        }

        # Add optional parameters if specified
        if valset is not None:
            compile_kwargs["valset"] = valset
        if "num_trials" in params:
            compile_kwargs["num_trials"] = params["num_trials"]
        if "minibatch" in params:
            compile_kwargs["minibatch"] = params["minibatch"]
        if "minibatch_size" in params:
            compile_kwargs["minibatch_size"] = params["minibatch_size"]

        return optimizer.compile(**compile_kwargs)
