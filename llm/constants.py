"""
Constants used throughout the ALeRCE T2S system.
Centralizing constants helps avoid issues with hard-coded values
and makes the codebase more maintainable.
"""

# Classification constants
class DifficultyLevel:
    SIMPLE = "simple"
    MEDIUM = "medium"
    ADVANCED = "advanced"
    
    @classmethod
    def get_valid_levels(cls):
        return [cls.SIMPLE, cls.MEDIUM, cls.ADVANCED]

class SpatialClass:
    SPATIAL = "spatial"
    NOT_SPATIAL = "not_spatial"
    
    @classmethod
    def get_valid_classes(cls):
        return [cls.SPATIAL, cls.NOT_SPATIAL]

# Method types
class GenerationMethod:
    DIRECT = "direct"
    CoT = "cot"
    STEP_BY_STEP = "step-by-step"
    STEP_BY_STEP_COT = "step-by-step-cot"
    @classmethod
    def get_valid_methods(cls):
        return [cls.DIRECT, cls.CoT, cls.STEP_BY_STEP, cls.STEP_BY_STEP_COT]

# Response prefixes
class ResponsePrefix:
    CLASS = "class: "

# SQL Error types
class SQLErrorType:
    TIMEOUT = "timeout"
    UNDEFINED = "undefined"
    OTHER = "other"
    
    @classmethod
    def get_valid_error_types(cls):
        return [cls.TIMEOUT, cls.UNDEFINED, cls.OTHER]
