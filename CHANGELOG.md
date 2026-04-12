# Changelog

## [1.2.0] - 2026-04-12
### Added
- Added fixed script planning agent implementations.

### Changed
- Enhanced logging callbacks:
    - Added support for trace saving.
    - Added episode timing.
    - Improved error messaging with friendly import errors for missing dependencies.

### Development
- Added unit tests for PDDLEnv, PDDLMaskedEnv, and MinecraftEnv.

## [1.1.1] - 2026-03-12
### Added
- Introduced count_inapplicable parameter in PDDLMaskedEnv for episode length calculation.
    - Relevant when using post-masking.
    - When enabled, inapplicable actions contribute to the episode length; otherwise, they are ignored.

## [1.1.0] - 2026-03-11
### Added
- Configurable action masking strategies: `"pre"` (filters invalid actions before execution) and `"post"` (learns invalid actions from environment feedback)

## [1.0.1] - 2025-12-16
### Changed
- Refactored repository structure

## [1.0.0] - 2025-11-06
### Initial release
- Basic functionality