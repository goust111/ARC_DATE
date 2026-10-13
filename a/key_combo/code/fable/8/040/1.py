from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, scale_sprite


ORIENTATIONS = ("identity", "flip_lr", "flip_ud", "rot180")
ALL_COLORS = tuple(range(10))
BG_COLOR = 0
MAX_GEN_ATTEMPTS = 600
_SEEN_INPUT_SIGNATURES: set[str] = set()


@dataclass(frozen=True)
class Candidate:
    orientation: str
    scale: int
    y: int
    x: int


def _apply_orientation(sprite: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "identity":
        return sprite.copy()
    if orientation == "flip_lr":
        return np.fliplr(sprite)
    if orientation == "flip_ud":
        return np.flipud(sprite)
    if orientation == "rot180":
        return np.rot90(sprite, 2)
    raise ValueError(f"unknown orientation: {orientation}")


def _top_projection(canvas: np.ndarray, bg: int) -> np.ndarray:
    h, w = canvas.shape
    out = np.full(w, int(bg), dtype=int)
    for c in range(w):
        for r in range(h):
            if int(canvas[r, c]) != int(bg):
                out[c] = int(canvas[r, c])
                break
    return out


def _left_projection(canvas: np.ndarray, bg: int) -> np.ndarray:
    h, w = canvas.shape
    out = np.full(h, int(bg), dtype=int)
    for r in range(h):
        for c in range(w):
            if int(canvas[r, c]) != int(bg):
                out[r] = int(canvas[r, c])
                break
    return out


def _render_inside(
    sprite: np.ndarray,
    interior_shape: tuple[int, int],
    candidate: Candidate,
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = _apply_orientation(sprite, candidate.orientation)
    scaled = scale_sprite(transformed, candidate.scale)
    canvas = np.full(interior_shape, int(bg), dtype=int)
    paste(canvas, scaled, candidate.y, candidate.x, transparent=bg)
    return canvas, scaled


def _enumerate_candidates(
    sprite: np.ndarray,
    interior_shape: tuple[int, int],
    top_view: np.ndarray,
    left_view: np.ndarray,
    bg: int,
) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    frame_h, frame_w = map(int, interior_shape)
    both: list[Candidate] = []
    top_only: list[Candidate] = []
    left_only: list[Candidate] = []

    for orientation in ORIENTATIONS:
        transformed = _apply_orientation(sprite, orientation)
        base_h, base_w = transformed.shape
        max_scale = min(frame_h // base_h, frame_w // base_w)
        for scale in range(1, max_scale + 1):
            scaled = scale_sprite(transformed, scale)
            sh, sw = scaled.shape
            scaled_top = _top_projection(scaled, bg)
            scaled_left = _left_projection(scaled, bg)
            for y in range(frame_h - sh + 1):
                candidate_left = np.full(frame_h, int(bg), dtype=int)
                candidate_left[y:y + sh] = scaled_left
                left_match = np.array_equal(candidate_left, left_view)
                for x in range(frame_w - sw + 1):
                    candidate_top = np.full(frame_w, int(bg), dtype=int)
                    candidate_top[x:x + sw] = scaled_top
                    top_match = np.array_equal(candidate_top, top_view)
                    if not top_match and not left_match:
                        continue
                    cand = Candidate(orientation, int(scale), int(y), int(x))
                    if top_match:
                        top_only.append(cand)
                    if left_match:
                        left_only.append(cand)
                    if top_match and left_match:
                        both.append(cand)
    return both, top_only, left_only


def _is_ring_component(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    color = int(comp["color"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    if int(comp["area"]) != 2 * (h + w) - 4:
        return False
    patch = crop_bbox(grid, comp["bbox"])
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.all(patch[border] == color) and not np.any(patch[~border] == color)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    rings = [comp for comp in components if _is_ring_component(grid, comp)]
    assert len(rings) == 1, f"expected one frame ring, got {len(rings)}"
    comp = rings[0]
    r0, c0, r1, c1 = map(int, comp["bbox"])
    assert r0 >= 1 and c0 >= 1, "frame must leave room for the two silhouettes"
    return r0, c0, r1, c1, int(comp["color"])


def _extract_sprite(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = map(int, frame_bbox)
    clean = np.array(grid, dtype=int, copy=True)
    clean[r0:r1 + 1, c0:c1 + 1] = int(bg)
    clean[r0 - 1, c0 + 1:c1] = int(bg)
    clean[r0 + 1:r1, c0 - 1] = int(bg)
    components = find_components(
        clean,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) == 1, f"expected one external sprite, got {len(components)}"
    return crop_bbox(clean, components[0]["bbox"])


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = int(get_bg(g))
    r0, c0, r1, c1, frame_color = _find_frame(g, bg)
    sprite = _extract_sprite(g, bg, (r0, c0, r1, c1))

    top_view = g[r0 - 1, c0 + 1:c1].copy()
    left_view = g[r0 + 1:r1, c0 - 1].copy()
    interior_shape = (r1 - r0 - 1, c1 - c0 - 1)

    both, top_only, left_only = _enumerate_candidates(sprite, interior_shape, top_view, left_view, bg)
    assert len(both) == 1, f"expected a unique solution, got {len(both)}"

    out = g.copy()
    canvas, _scaled = _render_inside(sprite, interior_shape, both[0], bg)
    paste(out, canvas, r0 + 1, c0 + 1, transparent=bg)

    return out, {
        "bg": bg,
        "frame_color": frame_color,
        "frame_bbox": (r0, c0, r1, c1),
        "sprite": sprite,
        "top_view": top_view,
        "left_view": left_view,
        "both": both,
        "top_only": top_only,
        "left_only": left_only,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a hidden scaled sprite from the top and left silhouettes around a frame, then draw it inside.

    Input structure:
    - One corner of the grid contains a small multicolor sprite.
    - Elsewhere there is a large empty rectangular frame in a single border color.
    - The row just above the frame interior and the column just left of the frame interior
      encode the first non-background color seen from the top and from the left.

    Rule:
    - The missing target is the small sprite after one of: identity, left-right flip,
      up-down flip, or 180-degree rotation, followed by an integer scale-up.
    - Enumerate every orientation, scale, and placement fully inside the frame.
    - Keep the unique candidate whose column-wise top silhouette and row-wise left
      silhouette both match the given edge clues, then paint that sprite into the frame.

    Key insight:
    - Each silhouette alone leaves ambiguity because one axis of translation stays hidden;
      only the intersection of the two orthogonal first-hit views fixes the full placement.
    """
    out, _details = _solve_details(grid)
    return out


def _random_connected_mask(h: int, w: int, n_cells: int) -> np.ndarray:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < n_cells:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    coords = np.argwhere(mask)
    rr0, cc0 = coords.min(axis=0)
    rr1, cc1 = coords.max(axis=0)
    return mask[int(rr0):int(rr1) + 1, int(cc0):int(cc1) + 1]


def _random_sprite(bg: int, frame_color: int) -> np.ndarray:
    usable = [c for c in ALL_COLORS if c not in {bg, frame_color}]
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        n_cells = random.randint(max(4, (h * w) // 2), h * w)
        mask = _random_connected_mask(h, w, n_cells)
        patch = np.full(mask.shape, int(bg), dtype=int)
        color_count = random.randint(2, min(4, int(mask.sum())))
        palette = random.sample(usable, color_count)
        coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
        random.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            if idx < color_count:
                patch[r, c] = palette[idx]
            else:
                patch[r, c] = random.choice(palette)

        transformed = [_apply_orientation(patch, name) for name in ORIENTATIONS]
        if len({arr.tobytes() + str(arr.shape).encode() for arr in transformed}) < 4:
            continue
        if len(set(int(v) for v in patch[patch != bg])) < 2:
            continue
        return patch
    raise RuntimeError("failed to build a nondegenerate sprite")


def _rectangles_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _sprite_corner_positions(h: int, w: int, sh: int, sw: int) -> list[tuple[int, int]]:
    top = random.randint(0, min(2, h - sh))
    left = random.randint(0, min(2, w - sw))
    bottom = h - sh - random.randint(0, min(2, h - sh))
    right = w - sw - random.randint(0, min(2, w - sw))
    return [(top, left), (top, right), (bottom, left), (bottom, right)]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, interior_shape: tuple[int, int], color: int) -> tuple[int, int]:
    h, w = map(int, interior_shape)
    r1 = r0 + h + 1
    c1 = c0 + w + 1
    grid[r0, c0:c1 + 1] = int(color)
    grid[r1, c0:c1 + 1] = int(color)
    grid[r0:r1 + 1, c0] = int(color)
    grid[r0:r1 + 1, c1] = int(color)
    return r1, c1


def _input_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    bg = BG_COLOR
    frame_color = random.choice([c for c in ALL_COLORS if c != bg])
    sprite = _random_sprite(bg, frame_color)
    sprite_h, sprite_w = sprite.shape

    k_choices = [k for k in (1, 2, 3) if sprite_h * k <= 12 and sprite_w * k <= 12]
    if len(k_choices) < 2:
        return None
    true_orientation = random.choice(ORIENTATIONS)
    true_scale = random.choices(k_choices, weights=[1 if k == 1 else 2 for k in k_choices], k=1)[0]
    scaled_truth = scale_sprite(_apply_orientation(sprite, true_orientation), true_scale)
    sh, sw = scaled_truth.shape

    extra_h = random.randint(1, 4)
    extra_w = random.randint(1, 4)
    interior_h = sh + extra_h
    interior_w = sw + extra_w
    true_y = random.randint(0, extra_h)
    true_x = random.randint(0, extra_w)

    frame_foot_h = interior_h + 3
    frame_foot_w = interior_w + 3

    for _ in range(60):
        grid_h = random.randint(max(16, frame_foot_h + 2), 30)
        grid_w = random.randint(max(16, frame_foot_w + 2), 30)
        if grid_h < sprite_h + 1 or grid_w < sprite_w + 1:
            continue

        sprite_positions = _sprite_corner_positions(grid_h, grid_w, sprite_h, sprite_w)
        sr, sc = random.choice(sprite_positions)
        sprite_rect = (sr, sc, sr + sprite_h - 1, sc + sprite_w - 1)

        frame_positions = []
        max_r0 = grid_h - (interior_h + 2)
        max_c0 = grid_w - (interior_w + 2)
        for fr in range(1, max_r0 + 1):
            for fc in range(1, max_c0 + 1):
                footprint = (fr - 1, fc - 1, fr + interior_h + 1, fc + interior_w + 1)
                if _rectangles_separated(sprite_rect, footprint, gap=1):
                    frame_positions.append((fr, fc))
        if not frame_positions:
            continue

        fr, fc = random.choice(frame_positions)
        grid = np.full((grid_h, grid_w), int(bg), dtype=int)
        paste(grid, sprite, sr, sc, transparent=bg)
        r1, c1 = _draw_frame(grid, fr, fc, (interior_h, interior_w), frame_color)

        truth = Candidate(true_orientation, int(true_scale), int(true_y), int(true_x))
        truth_canvas, truth_scaled = _render_inside(sprite, (interior_h, interior_w), truth, bg)
        grid[fr - 1, fc + 1:c1] = _top_projection(truth_canvas, bg)
        grid[fr + 1:r1, fc - 1] = _left_projection(truth_canvas, bg)

        expected = grid.copy()
        paste(expected, truth_scaled, fr + 1 + true_y, fc + 1 + true_x, transparent=bg)

        try:
            solved, details = _solve_details(grid)
        except Exception:
            continue
        if not np.array_equal(solved, expected):
            continue

        both = details["both"]
        top_only = details["top_only"]
        left_only = details["left_only"]
        if len(both) != 1 or both[0] != truth:
            continue
        if len(top_only) < 2 or len(left_only) < 2:
            continue
        if len({cand.y for cand in top_only}) < 2:
            continue
        if len({cand.x for cand in left_only}) < 2:
            continue

        sig = _input_signature(grid)
        if sig in _SEEN_INPUT_SIGNATURES:
            continue
        _SEEN_INPUT_SIGNATURES.add(sig)
        return grid, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _attempt_generate()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid dual-silhouette sprite puzzle")
