from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
FRAME = 5
RED = 2
BLUE = 1
GREEN = 3
DEPTH_COLORS = {1: RED, 2: BLUE, 3: GREEN}
SEED_COLORS = (4, 6, 7, 8, 9)
MAX_SIZE = 30
GEN_RETRIES = 500
PLACEMENT_RETRIES = 300

SEED_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
)


@dataclass
class FrameInfo:
    bbox: tuple[int, int, int, int]
    coords: np.ndarray
    parent: int | None = None
    depth: int = 0


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _rect_frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    local = _mask_from_coords(coords, box)
    expected = np.zeros_like(local, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(local, expected):
        return None
    return box


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _inside_frame_interior(box: tuple[int, int, int, int], coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = box
    return bool(
        np.all(coords[:, 0] > r0)
        and np.all(coords[:, 0] < r1)
        and np.all(coords[:, 1] > c0)
        and np.all(coords[:, 1] < c1)
    )


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _place_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int, require_bg: bool) -> None:
    h, w = mask.shape
    region = grid[top : top + h, left : left + w]
    if require_bg:
        assert np.all(region[mask] == BG), "seed copy must land on empty corridor cells"
    region[mask] = color


def _find_frames(grid: np.ndarray) -> list[FrameInfo]:
    frame_components = _components(grid == FRAME)
    assert frame_components, "expected at least one grey frame"
    frames = []
    for coords in frame_components:
        box = _rect_frame_bbox(coords)
        assert box is not None, "all grey components must be hollow rectangular frames"
        frames.append(FrameInfo(bbox=box, coords=coords))
    return frames


def _annotate_frames(frames: list[FrameInfo]) -> tuple[list[list[int]], list[int]]:
    children: list[list[int]] = [[] for _ in frames]
    for i, frame in enumerate(frames):
        containers = [j for j, other in enumerate(frames) if i != j and _contains(other.bbox, frame.bbox)]
        if containers:
            parent = min(
                containers,
                key=lambda j: (frames[j].bbox[2] - frames[j].bbox[0] + 1) * (frames[j].bbox[3] - frames[j].bbox[1] + 1),
            )
            frame.parent = parent
            children[parent].append(i)
    for siblings in children:
        assert len(siblings) <= 1, "each frame chain should have at most one direct child"

    def depth_of(idx: int) -> int:
        if frames[idx].depth:
            return frames[idx].depth
        parent = frames[idx].parent
        frames[idx].depth = 1 if parent is None else depth_of(parent) + 1
        return frames[idx].depth

    for idx in range(len(frames)):
        depth_of(idx)
        assert 1 <= frames[idx].depth <= 3, "nesting depth must stay within 1..3"

    leaves = [idx for idx, kids in enumerate(children) if not kids]
    assert leaves, "expected at least one innermost frame"
    return children, leaves


def _find_seeds(grid: np.ndarray, frames: list[FrameInfo], leaves: list[int]) -> dict[int, tuple[int, np.ndarray]]:
    seed_mask = (grid != BG) & (grid != FRAME)
    comps = _components(seed_mask)
    assert comps, "expected at least one seed object"

    seeds_by_leaf: dict[int, tuple[int, np.ndarray]] = {}
    for coords in comps:
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1, "each seed must be monochrome"
        color = int(colors[0])
        box = _bbox(coords)
        leaf_candidates = [idx for idx in leaves if _inside_frame_interior(frames[idx].bbox, coords)]
        assert len(leaf_candidates) == 1, "each seed must belong to exactly one innermost frame"
        leaf = leaf_candidates[0]
        assert leaf not in seeds_by_leaf, "each innermost frame must contain exactly one seed"
        seeds_by_leaf[leaf] = (color, _mask_from_coords(coords, box))

    assert set(seeds_by_leaf) == set(leaves), "every innermost frame must contain one seed"
    return seeds_by_leaf


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular frames by depth and copy each innermost seed upward into every enclosing corridor.

    Input structure:
    - bg = black.
    - One or more disjoint groups of one-pixel-thick grey rectangular frames form strict
      nesting chains of depth 1 to 3.
    - Each innermost frame contains exactly one small connected monochrome seed object of a
      non-grey, non-background color.

    Rule:
    - Compute each frame's nesting depth: outermost = 1, one level inside = 2, two levels
      inside = 3.
    - Recolor depth-1 frames red, depth-2 frames blue, and depth-3 frames green.
    - Keep the original seed inside the innermost frame.
    - For every pair of directly nested frames on that seed's ancestor chain, copy the seed
      into the corridor between them with its top row at the outer frame's inner top edge and
      its shape horizontally centered in that outer frame's interior.

    Key insight:
    - The task requires separating frame containment depth from seed geometry, then using the
      innermost seed as a template that propagates one level outward per nesting corridor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    frames = _find_frames(g)
    children, leaves = _annotate_frames(frames)
    seeds_by_leaf = _find_seeds(g, frames, leaves)

    out = g.copy()
    for frame in frames:
        color = DEPTH_COLORS[frame.depth]
        out[frame.coords[:, 0], frame.coords[:, 1]] = color

    for leaf, (seed_color, seed_shape) in seeds_by_leaf.items():
        child = leaf
        while frames[child].parent is not None:
            parent = frames[child].parent
            assert parent is not None
            outer = frames[parent].bbox
            inner = frames[child].bbox
            sh, sw = seed_shape.shape
            top = outer[0] + 1
            interior_w = outer[3] - outer[1] - 1
            left = outer[1] + 1 + (interior_w - sw) // 2
            assert top + sh - 1 < inner[0], "corridor top band must fit the copied seed"
            assert left >= outer[1] + 1 and left + sw - 1 <= outer[3] - 1, "seed copy must stay inside the outer frame"
            _place_mask(out, seed_shape, seed_color, top, left, require_bg=True)
            child = parent

    return out


def _make_group(depth: int, seed_color: int) -> tuple[np.ndarray, np.ndarray]:
    seed_shape = random.choice(SEED_SHAPES).copy()
    sh, sw = seed_shape.shape

    inner_h = sh + random.randint(2, 4)
    inner_w = sw + random.randint(2, 5)
    chain_inner_to_outer = [(0, 0, inner_h + 1, inner_w + 1)]
    current = chain_inner_to_outer[-1]

    for _ in range(depth - 1):
        gap_top = random.randint(sh, sh + 1)
        gap_bottom = random.randint(1, 2)
        gap_left = random.randint(1, 2)
        gap_right = random.randint(1, 2)
        current = (
            current[0] - gap_top - 1,
            current[1] - gap_left - 1,
            current[2] + gap_bottom + 1,
            current[3] + gap_right + 1,
        )
        chain_inner_to_outer.append(current)

    min_r = min(box[0] for box in chain_inner_to_outer)
    min_c = min(box[1] for box in chain_inner_to_outer)
    shifted_inner_to_outer = [
        (r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in chain_inner_to_outer
    ]
    frames = list(reversed(shifted_inner_to_outer))
    leaf = shifted_inner_to_outer[0]

    height = max(box[2] for box in frames) + 1
    width = max(box[3] for box in frames) + 1
    inp = np.full((height, width), BG, dtype=int)
    for box in frames:
        _draw_frame(inp, box, FRAME)

    seed_row = random.randint(leaf[0] + 1, leaf[2] - sh)
    seed_col = random.randint(leaf[1] + 1, leaf[3] - sw)
    _place_mask(inp, seed_shape, seed_color, seed_row, seed_col, require_bg=True)

    out = inp.copy()
    for depth_idx, box in enumerate(frames, start=1):
        _draw_frame(out, box, DEPTH_COLORS[depth_idx])
    for i in range(len(frames) - 1):
        outer = frames[i]
        inner = frames[i + 1]
        top = outer[0] + 1
        left = outer[1] + 1 + (outer[3] - outer[1] - 1 - sw) // 2
        assert top + sh - 1 < inner[0], "generator must leave room for the copied seed"
        _place_mask(out, seed_shape, seed_color, top, left, require_bg=True)

    return inp, out


def _too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _crop_pair_with_margin(inp: np.ndarray, out: np.ndarray, margin_max: int = 3) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(margin_max, int(r0)))
    left = random.randint(0, min(margin_max, int(c0)))
    bottom = random.randint(0, min(margin_max, int(inp.shape[0] - 1 - r1)))
    right = random.randint(0, min(margin_max, int(inp.shape[1] - 1 - c1)))
    row_slice = slice(r0 - top, r1 + bottom + 1)
    col_slice = slice(c0 - left, c1 + right + 1)
    return inp[row_slice, col_slice], out[row_slice, col_slice]


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    n_groups = random.choices((1, 2, 3), weights=(0.35, 0.40, 0.25), k=1)[0]
    depths = [random.randint(1, 3) for _ in range(n_groups)]
    colors = random.sample(SEED_COLORS, k=n_groups)
    patches = [_make_group(depth, color) for depth, color in zip(depths, colors)]

    canvas_in = np.full((MAX_SIZE, MAX_SIZE), BG, dtype=int)
    canvas_out = np.full((MAX_SIZE, MAX_SIZE), BG, dtype=int)
    rects: list[tuple[int, int, int, int]] = []

    order = sorted(range(n_groups), key=lambda idx: patches[idx][0].shape[0] * patches[idx][0].shape[1], reverse=True)
    for idx in order:
        inp_patch, out_patch = patches[idx]
        ph, pw = inp_patch.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, MAX_SIZE - ph)
            left = random.randint(0, MAX_SIZE - pw)
            rect = (top, left, top + ph - 1, left + pw - 1)
            if any(_too_close(rect, other, pad=1) for other in rects):
                continue
            region_in = canvas_in[top : top + ph, left : left + pw]
            region_out = canvas_out[top : top + ph, left : left + pw]
            mask = inp_patch != BG
            region_in[mask] = inp_patch[mask]
            region_out[out_patch != BG] = out_patch[out_patch != BG]
            rects.append(rect)
            placed = True
            break
        if not placed:
            raise ValueError("failed to place all frame groups on the canvas")

    return _crop_pair_with_margin(canvas_in, canvas_out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, out = _generate_case()
            if max(inp.shape) > MAX_SIZE or max(out.shape) > MAX_SIZE:
                continue
            if not np.array_equal(main(inp), out):
                continue
            return inp
        except AssertionError:
            continue
        except ValueError:
            continue
    raise RuntimeError("failed to generate a valid random instance")
