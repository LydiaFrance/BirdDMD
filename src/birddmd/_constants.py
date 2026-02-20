"""Named constants for the BirdDMD package (private module).

Centralises magic numbers used across modules so they can be
imported by name rather than scattered as literals.
"""

# ── Conjugate-pair detection ────────────────────────────────────────
CONJUGATE_TOLERANCE: float = 1e-10

# ── Frequency classification ────────────────────────────────────────
ZERO_FREQUENCY_THRESHOLD: float = 1e-9

# ── Default DMD parameters ──────────────────────────────────────────
DEFAULT_N_MODES: int = 10
DEFAULT_DELAY: int = 2
DEFAULT_EIG_CONSTRAINTS: set[str] = {"conjugate_pairs"}

# ── Array shape constants ──────────────────────────────────────────
NDIM_2D: int = 2
NDIM_3D: int = 3
N_SPATIAL: int = 3  # x, y, z coordinates per marker

# ── Data filtering defaults ─────────────────────────────────────────
DEFAULT_BIN_SIZE: float = 0.005
DEFAULT_GAP_THRESHOLD: float = 0.15
DEFAULT_TIME_START_MAX: float = 0.15
DEFAULT_MIN_FRAMES: int = 30

# ── Hawk-specific constants ─────────────────────────────────────────
N_BILATERAL_MARKERS: int = 8
N_UNILATERAL_MARKERS: int = 4

MARKER_POSITIONS: list[str] = ["wingtip", "primary", "secondary", "tailtip"]

MARKER_COLOURS: dict[str, str] = {
    "wingtip": "#DF5D99",  # pink
    "primary": "#57B7B0",  # teal
    "secondary": "#6BBB5D",  # green
    "tailtip": "#A86CC5",  # purple
}

DEFAULT_MARKER_NAMES: list[str] = [
    "left_wingtip",
    "left_primary",
    "left_secondary",
    "left_tailtip",
    "right_wingtip",
    "right_primary",
    "right_secondary",
    "right_tailtip",
]

HAWK_META_COLUMNS: list[str] = [
    "BirdID",
    "Year",
    "PerchDistance",
    "Naive",
    "Obstacle",
    "Turn",
    "IMU",
]
