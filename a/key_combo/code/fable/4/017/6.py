from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np

# Shared helpers are loaded from the project utility module so this file can be
# executed standalone via SourceFileLoader in the provided self-check.
_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_spec = importlib.util.spec_from_file_location("arc_utils_for_6", str(_UTILS_PATH))
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to resolve arc_utils.py")
_arc_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_arc_utils)

Color = _arc_utils.Color
bbox = _arc_utils.bbox
colors_present = _arc_utils.colors_present

MOTIF_SHAPES = {
    "plus": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "h": np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    "x": np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    "four_corners": np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
}

MOTIF_LIST = list(MOTIF_SHAPES.items())
_BACKGROUND = int(Color.BLACK)
_OVERLAP_COLOR = int(Color.RED)
_COLOR_POOL = [
    int(Color.BLUE),
    int(Color.GREEN),
    int(Color.YELLOW),
    int(Color.GRAY),
    int(Color.PINK),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
]

_CANVAS_MIN_SIZE = 10
_CANVAS_MAX_SIZE = 30
_ATTEMPT_LIMIT = 200
_GENERATOR_ATTEMPTS = 300
_DIFF_SHAPE_PROB = 0.72
_VARIANT_PROB = 0.45


def _decode_known_contour(motif: np.ndarray) -> np.ndarray | None:
    g = np.asarray(motif, dtype=bool)
    if g.shape != (3, 3):
        return None

    corners = np.array([g[0, 0], g[0, 2], g[2, 0], g[2, 2]], dtype=bool)
    edge_centers = np.array([g[0, 1], g[1, 0], g[1, 2], g[2, 1]], dtype=bool)

    if not bool(corners.any()):
        if not bool(g[1, 1]) or not bool(edge_centers.all()):
            return None
        phase = np.zeros((3, 3), dtype=bool)
        phase[1:, 1:] = True
        return phase

    if not bool(corners.all()):
        return None
    if bool(g[0, 1]) or bool(g[2, 1]):
        return None
    if bool(g[1, 0]) != bool(g[1, 2]):
        return None

    phase = np.zeros((3, 3), dtype=bool)
    phase[1:, 0] = True
    if not bool(g[1, 0]):
        phase[0, 1:] = True
    if not bool(g[1, 1]):
        phase[0, 0] = True
    return phase


def _decode_phase(motif: np.ndarray) -> np.ndarray:
    motif = np.asarray(motif, dtype=bool)
    for rot in range(4):
        r = np.rot90(motif, k=rot)
        for flipped in (False, True):
            cand = np.fliplr(r) if flipped else r
            phase = _decode_known_contour(cand)
            if phase is not None:
                phase = np.rot90(phase, k=(-rot) % 4)
                if flipped:
                    phase = np.fliplr(phase)
                return phase
    if not np.any(motif):
        raise ValueError("glyph must contain at least one colored pixel")
    return motif.copy()


def _glyph_from_grid(g: np.ndarray, color: int) -> tuple[int, int, np.ndarray]:
    mask = (g == int(color))
    b = bbox(mask)
    if b is None:
        raise AssertionError(f"missing glyph for color {color}")
    r0, c0, r1, c1 = b
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        raise AssertionError("each glyph must have a tight 3x3 bbox")
    motif = np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)
    phase = _decode_phase(motif)
    return r0, c0, phase


def _tile_cover(shape: tuple[int, int], phase: np.ndarray, anchor_r: int, anchor_c: int) -> np.ndarray:
    rows = np.arange(int(shape[0]))[:, None]
    cols = np.arange(int(shape[1]))[None, :]
    return phase[(rows - int(anchor_r)) % 3, (cols - int(anchor_c)) % 3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct two hidden period-3 wallpapers from their 3×3 contour glyphs and overlay their visible cells.

    Input structure:
    - The input grid is black with exactly two 3×3 contour glyphs on different non-black colors.
    - The two glyphs do not overlap and appear as the only non-black objects.

    Rule:
    - For each glyph, infer its 3×3 wallpaper phase tile and tile it across the entire grid aligned to the glyph's bbox top-left.
    - Paint a cell by that glyph's color when exactly one tiled wallpaper covers it.
    - Paint a cell red when both tiled wallpapers cover it; leave black otherwise.

    Key insight:
    - The input glyphs are not to be copied directly; each glyph is an implicit sample of a periodic wallpaper anchored by its phase position.
    """
    g = np.asarray(grid, dtype=int)
    colors = colors_present(g, exclude={_BACKGROUND})
    if len(colors) != 2:
        raise AssertionError("expected exactly two non-black colors")
    color_a, color_b = map(int, colors)
    r0_a, c0_a, phase_a = _glyph_from_grid(g, color_a)
    r1_b, c1_b, phase_b = _glyph_from_grid(g, color_b)

    cover_a = _tile_cover(g.shape, phase_a, r0_a, c0_a)
    cover_b = _tile_cover(g.shape, phase_b, r1_b, c1_b)

    out = np.full(g.shape, _BACKGROUND, dtype=int)
    both = cover_a & cover_b
    out[both] = _OVERLAP_COLOR
    out[cover_a & ~both] = color_a
    out[cover_b & ~both] = color_b
    return out


def _transform_glyph(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    g = np.rot90(base, k=int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        g = np.fliplr(g)
    return g


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(_CANVAS_MIN_SIZE, _CANVAS_MAX_SIZE + 1))
    w = int(rng.integers(_CANVAS_MIN_SIZE, _CANVAS_MAX_SIZE + 1))
    grid = np.full((h, w), _BACKGROUND, dtype=int)

    color_a, color_b = map(int, rng.choice(_COLOR_POOL, size=2, replace=False))

    if rng.random() < _DIFF_SHAPE_PROB:
        glyph_names = rng.choice(len(MOTIF_LIST), size=2, replace=False)
    else:
        same_index = int(rng.integers(len(MOTIF_LIST)))
        glyph_names = [same_index, same_index]

    motif_a = _transform_glyph(MOTIF_LIST[int(glyph_names[0])][1], rng)
    motif_b = _transform_glyph(MOTIF_LIST[int(glyph_names[1])][1], rng)

    if rng.random() < _VARIANT_PROB:
        motif_b = np.rot90(motif_b, k=1)
    if rng.random() < 0.3:
        motif_a = np.flipud(motif_a)

    pos_candidates = [(r, c) for r in range(h - 2) for c in range(w - 2)]
    if len(pos_candidates) < 2:
        return None

    idx_a = int(rng.integers(len(pos_candidates)))
    r_a, c_a = pos_candidates[idx_a]

    used = set(pos_candidates)
    for _ in range(_ATTEMPT_LIMIT):
        idx_b = int(rng.integers(len(pos_candidates)))
        r_b, c_b = pos_candidates[idx_b]
        if abs(r_a - r_b) < 3 and abs(c_a - c_b) < 3:
            continue
        if any((r_b == r_a + dr and c_b == c_a + dc) for dr in (-2, -1, 0, 1, 2) for dc in (-2, -1, 0, 1, 2)):
            continue

        try:
            out = grid.copy()
            out[r_a:r_a + 3, c_a:c_a + 3][motif_a] = color_a
            out[r_b:r_b + 3, c_b:c_b + 3][motif_b] = color_b
            if np.any((out[r_a:r_a+3, c_a:c_a+3] != _BACKGROUND) & (out[r_b:r_b+3, c_b:c_b+3] != _BACKGROUND)):
                continue
            _ = main(out)
            return out
        except Exception:
            continue

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GENERATOR_ATTEMPTS):
        out = _generate_candidate(rng)
        if out is not None:
            return out
    raise RuntimeError("failed to generate a valid 6 input")
