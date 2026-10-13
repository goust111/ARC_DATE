from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


BG = Color.BLACK
MARKER = Color.PINK
FILLER = Color.BLUE

SHAPE_TO_COLOR = {
    "horizontal": Color.RED,
    "vertical": Color.GREEN,
    "l_shape": Color.ORANGE,
    "single": Color.YELLOW,
}

LOGICAL_SIZE_CHOICES = (6, 7, 8)
MAX_ATTEMPTS = 400


def _component_local_mask(shape: tuple[int, int], comp: dict) -> np.ndarray:
    """Return a tight boolean crop for one component only."""
    full = np.zeros(shape, dtype=bool)
    coords = comp["coords"]
    full[coords[:, 0], coords[:, 1]] = True
    return crop_bbox(full, tuple(map(int, comp["bbox"])))


def _classify_marker_shape(mask: np.ndarray) -> str:
    area = int(mask.sum())
    height, width = mask.shape

    if area == 1:
        return "single"
    if height == 1 and area == width and 2 <= width <= 3:
        return "horizontal"
    if width == 1 and area == height and 2 <= height <= 3:
        return "vertical"
    if area == 3 and height == 2 and width == 2:
        return "l_shape"
    raise AssertionError(f"unexpected compressed marker shape: area={area}, shape={mask.shape}")


def _solve_logical_grid(logical: np.ndarray) -> np.ndarray:
    out = logical.copy()
    components = find_components(
        logical,
        bg=BG,
        colors=[MARKER],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    for comp in components:
        local_mask = _component_local_mask(logical.shape, comp)
        kind = _classify_marker_shape(local_mask)
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = SHAPE_TO_COLOR[kind]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downsample the separator lattice and recolor each emergent magenta shape by its compressed geometry.

    Input structure:
    - The input is an even-by-even sparse lattice on a black background.
    - Every odd-indexed row and odd-indexed column is a black separator line.
    - Only even-row/even-column lattice cells contain non-background pixels: ordinary foreground cells and magenta marker cells.

    Rule:
    - Delete all odd rows and odd columns, keeping only the even/even lattice cells as a compressed grid.
    - In that compressed grid, find every 4-connected magenta component.
    - Recolor each magenta component by its compressed shape: horizontal line -> red, vertical line -> green, L-shape -> orange, single cell -> yellow.
    - Leave every non-magenta cell unchanged and return the compressed grid.

    Key insight:
    - The separators hide the marker geometry in the original grid; only after downsampling do the magenta cells become orthogonally adjacent and reveal the color-coding shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0 and g.shape[1] % 2 == 0
    assert np.all(g[1::2, :] == BG), "odd rows must be black separators"
    assert np.all(g[:, 1::2] == BG), "odd columns must be black separators"

    logical = g[::2, ::2].copy()
    return _solve_logical_grid(logical)


def _build_marker_patch(spec: str, rng: np.random.Generator) -> np.ndarray:
    if spec == "single":
        return np.array([[MARKER]], dtype=int)
    if spec == "horizontal2":
        return np.full((1, 2), MARKER, dtype=int)
    if spec == "horizontal3":
        return np.full((1, 3), MARKER, dtype=int)
    if spec == "vertical2":
        return np.full((2, 1), MARKER, dtype=int)
    if spec == "vertical3":
        return np.full((3, 1), MARKER, dtype=int)
    if spec == "l_shape":
        patch = np.array([[MARKER, MARKER], [MARKER, BG]], dtype=int)
        return np.rot90(patch, k=int(rng.integers(0, 4))).copy()
    raise ValueError(f"unknown shape spec: {spec}")


def _marker_area(patch: np.ndarray) -> int:
    return int(np.count_nonzero(patch == MARKER))


def _sample_shape_specs(logical_size: int, rng: np.random.Generator) -> list[str]:
    specs = ["horizontal3", "vertical3", "l_shape", "single"]
    specs.append("horizontal2" if rng.random() < 0.5 else "vertical2")

    extra_pool = [
        "horizontal2",
        "horizontal3",
        "vertical2",
        "vertical3",
        "l_shape",
        "single",
    ]
    extra_count = int(rng.integers(0, 2 if logical_size == 6 else 3))
    for _ in range(extra_count):
        specs.append(extra_pool[int(rng.integers(0, len(extra_pool)))])
    return specs


def _candidate_origins(occupied: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    height, width = occupied.shape
    patch_h, patch_w = patch.shape
    cells = np.argwhere(patch == MARKER)
    origins: list[tuple[int, int]] = []

    for r0 in range(height - patch_h + 1):
        for c0 in range(width - patch_w + 1):
            ok = True
            for dr, dc in cells:
                rr, cc = r0 + int(dr), c0 + int(dc)
                if occupied[rr, cc]:
                    ok = False
                    break
                rr0 = max(0, rr - 1)
                rr1 = min(height, rr + 2)
                cc0 = max(0, cc - 1)
                cc1 = min(width, cc + 2)
                if occupied[rr0:rr1, cc0:cc1].any():
                    ok = False
                    break
            if ok:
                origins.append((r0, c0))
    return origins


def _place_marker_shapes(logical_size: int, rng: np.random.Generator) -> np.ndarray | None:
    logical = np.zeros((logical_size, logical_size), dtype=int)
    occupied = np.zeros_like(logical, dtype=bool)

    patches: list[tuple[float, int, int, np.ndarray]] = []
    for spec in _sample_shape_specs(logical_size, rng):
        patch = _build_marker_patch(spec, rng)
        patches.append((-float(_marker_area(patch)), -(patch.shape[0] + patch.shape[1]), rng.random(), patch))
    patches.sort()

    for _neg_area, _neg_span, _noise, patch in patches:
        origins = _candidate_origins(occupied, patch)
        if not origins:
            return None
        r0, c0 = origins[int(rng.integers(0, len(origins)))]
        paste(logical, patch, r0, c0, transparent=BG)
        coords = np.argwhere(patch == MARKER)
        occupied[r0 + coords[:, 0], c0 + coords[:, 1]] = True

    return logical


def _expand4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr > 0:
            out[rr - 1, cc] = True
        if rr + 1 < mask.shape[0]:
            out[rr + 1, cc] = True
        if cc > 0:
            out[rr, cc - 1] = True
        if cc + 1 < mask.shape[1]:
            out[rr, cc + 1] = True
    return out


def _add_filler_pixels(logical: np.ndarray, rng: np.random.Generator) -> bool:
    marker_mask = logical == MARKER
    candidate_mask = (logical == BG) & ~_expand4(marker_mask)
    candidates = np.argwhere(candidate_mask)
    if candidates.shape[0] < 4:
        candidates = np.argwhere(logical == BG)
    if candidates.shape[0] < 4:
        return False

    max_fill = min(int(candidates.shape[0] - 1), max(4, int(0.28 * logical.size)))
    min_fill = min(max_fill, max(3, logical.shape[0] - 1))
    if max_fill < 3:
        return False

    count = int(rng.integers(min_fill, max_fill + 1))
    chosen = candidates[rng.choice(candidates.shape[0], size=count, replace=False)]
    logical[chosen[:, 0], chosen[:, 1]] = FILLER
    return True


def _render_sparse_lattice(logical: np.ndarray) -> np.ndarray:
    size = logical.shape[0] * 2
    grid = np.zeros((size, size), dtype=int)
    grid[::2, ::2] = logical
    return grid


def _component_kinds(grid: np.ndarray) -> list[tuple[str, int]]:
    comps = find_components(
        grid,
        bg=BG,
        colors=[MARKER],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    kinds: list[tuple[str, int]] = []
    for comp in comps:
        local = _component_local_mask(grid.shape, comp)
        kinds.append((_classify_marker_shape(local), int(local.sum())))
    return kinds


def _validate_instance(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return False
    if inp.shape[0] != inp.shape[1] or inp.shape[0] not in {12, 14, 16}:
        return False
    if np.any(inp[1::2, :] != BG) or np.any(inp[:, 1::2] != BG):
        return False

    logical = inp[::2, ::2].copy()
    if not np.any(logical == FILLER):
        return False
    if not np.any(logical == BG):
        return False

    original_marker_components = find_components(
        inp,
        bg=BG,
        colors=[MARKER],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    if not original_marker_components or any(int(comp["area"]) != 1 for comp in original_marker_components):
        return False

    kinds = _component_kinds(logical)
    kind_names = [kind for kind, _area in kinds]
    if not {"horizontal", "vertical", "l_shape", "single"}.issubset(kind_names):
        return False
    if not any(kind == "horizontal" and area == 3 for kind, area in kinds):
        return False
    if not any(kind == "vertical" and area == 3 for kind, area in kinds):
        return False
    if not any(kind in {"horizontal", "vertical"} and area == 2 for kind, area in kinds):
        return False

    expected = _solve_logical_grid(logical)
    out = main(inp)
    if not np.array_equal(out, expected):
        return False

    if not {Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE}.issubset(set(map(int, out.ravel()))):
        return False
    if not np.array_equal(out[logical != MARKER], logical[logical != MARKER]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    logical_size = int(rng.choice(LOGICAL_SIZE_CHOICES, p=(0.2, 0.45, 0.35)))

    logical = _place_marker_shapes(logical_size, rng)
    if logical is None:
        return None
    if not _add_filler_pixels(logical, rng):
        return None

    inp = _render_sparse_lattice(logical)
    if not _validate_instance(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
