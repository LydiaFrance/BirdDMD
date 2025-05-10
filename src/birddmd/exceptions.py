"""
Custom exceptions for BirdDMD package.

This module defines custom exception classes for handling various error conditions
that may occur during DMD computation, data validation, and I/O operations.
"""

class BirdDMDError(Exception):
    """Base exception class for all BirdDMD errors."""
    pass

class ValidationError(BirdDMDError):
    """Raised when data validation fails."""
    pass

class ComputationError(BirdDMDError):
    """Raised when DMD computation fails."""
    pass

class DataError(BirdDMDError):
    """Base class for data-related errors."""
    pass

class ShapeError(DataError):
    """Raised when array shapes are invalid."""
    pass

class IOError(BirdDMDError):
    """Base class for I/O related errors."""
    pass

class FileNotFoundError(IOError):
    """Raised when a required file is not found."""
    pass

class DataValidationError(IOError):
    """Raised when data validation fails during I/O operations."""
    pass

class ConfigurationError(BirdDMDError):
    """Raised when there are issues with DMD configuration parameters."""
    pass

class InterpolationError(BirdDMDError):
    """Raised when data interpolation fails."""
    pass

class ReconstructionError(BirdDMDError):
    """Raised when DMD reconstruction fails."""
    pass

class ModeError(BirdDMDError):
    """Raised when there are issues with DMD modes."""
    pass

class EigenvalueError(BirdDMDError):
    """Raised when there are issues with DMD eigenvalues."""
    pass

# Add __all__ to explicitly define what should be imported
__all__ = [
    'BirdDMDError',
    'ValidationError',
    'ComputationError',
    'DataError',
    'ShapeError',
    'IOError',
    'FileNotFoundError',
    'DataValidationError',
    'ConfigurationError',
    'InterpolationError',
    'ReconstructionError',
    'ModeError',
    'EigenvalueError'
] 