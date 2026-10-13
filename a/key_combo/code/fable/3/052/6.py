from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

ARC_COLORS = tuple(range(10))
FOUR_CORNERS = ("tl", "tr", "bl", "br")
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(grid: np.ndarray, coords: list[Coord], bg: int) -> tuple[np.ndarray, BBox]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0, r1, c1)


def _corner_from_missing(mr: int, mc: int, h: int, w: int) -> str | None:
    if (mr, mc) in {(0, 1), (1, 0)}:
        return "tl"
    if (mr, mc) in {(0, w - 2), (1, w - 1)}:
        return "tr"
    if (mr, mc) in {(h - 2, 0), (h - 1, 1)}:
        return "bl"
    if (mr, mc) in {(h - 2, w - 1), (h - 1, w - 2)}:
        return "br"
    return None


def _detect_frame(grid: np.ndarray, coords: list[Coord], bg: int) -> dict | None:
    patch, bbox = _crop_component(grid, coords, bg)
    h, w = patch.shape
    if h < 4 or w < 4:
        return None

    rel = {(r - bbox[0], c - bbox[1]) for r, c in coords}
    perimeter = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r == 0 or r == h - 1 or c == 0 or c == w - 1
    }
    if rel - perimeter:
        return None
    if len(rel) != len(perimeter) - 1:
        return None
    missing = list(perimeter - rel)
    if len(missing) != 1:
        return None
    corner = _corner_from_missing(missing[0][0], missing[0][1], h, w)
    if corner is None:
        return None
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None

    return {
        "bbox": bbox,
        "patch": patch,
        "shape": patch.shape,
        "corner": corner,
        "color": next(iter(colors)),
    }


def _contains(inner: BBox, outer: BBox) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _paste_non_bg(dst: np.ndarray, src: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = src.shape
    sub = dst[top : top + h, left : left + w]
    mask = src != bg
    assert not np.any((sub != bg) & mask)
    sub[mask] = src[mask]


def _corner_anchor(corner: str, outer_shape: tuple[int, int], inner_shape: tuple[int, int]) -> Coord:
    oh, ow = outer_shape
    ih, iw = inner_shape
    assert ih <= oh - 2 and iw <= ow - 2
    top = 1 if corner[0] == "t" else oh - ih - 1
    left = 1 if corner[1] == "l" else ow - iw - 1
    return top, left


def _compose_aligned(frame_patch: np.ndarray, inner_patch: np.ndarray, corner: str, bg: int) -> np.ndarray:
    frame_h, frame_w = frame_patch.shape
    inner_h, inner_w = inner_patch.shape
    top, left = _corner_anchor(corner, (frame_h, frame_w), (inner_h, inner_w))
    out = frame_patch.copy()
    _paste_non_bg(out, inner_patch, top, left, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the nested inner content into each frame's notch-indicated inner corner from inside out.

    Input structure:
    - The grid has one background color, 2-4 nested one-cell-thick hollow rectangles, and one small sprite inside the innermost frame.
    - Each frame is missing exactly one border cell, and that gap sits orthogonally adjacent to one corner, encoding that corner.
    - Inner frames and the sprite start away from the parent frame's corners.

    Rule:
    - Detect the nested frames and sort them from innermost to outermost by containment.
    - First translate the sprite so its bbox touches the two inner walls of the innermost frame at the encoded corner.
    - Then recursively translate the whole current composite (innermost frame plus everything already moved inside it) into the next frame's encoded corner; keep every frame's notch missing.

    Key insight:
    - Each layer provides only its own target corner, so the final placement of the innermost content is the sum of all inner-to-outer rigid translations, not an independent one-step move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    components = _find_components(g != bg)
    assert components, "expected non-background objects"

    frames: list[dict] = []
    frame_cells: set[Coord] = set()
    for coords in components:
        info = _detect_frame(g, coords, bg)
        if info is not None:
            frames.append(info)
            frame_cells.update(coords)

    assert 2 <= len(frames) <= 4, "expected 2-4 nested frames"
    frames.sort(key=lambda item: (item["shape"][0] * item["shape"][1], item["bbox"][0], item["bbox"][1]))
    for i in range(len(frames) - 1):
        assert _contains(frames[i]["bbox"], frames[i + 1]["bbox"]), "frames must be strictly nested"

    inner_bbox = frames[0]["bbox"]
    sprite_coords = [
        (r, c)
        for r in range(inner_bbox[0] + 1, inner_bbox[2])
        for c in range(inner_bbox[1] + 1, inner_bbox[3])
        if int(g[r, c]) != bg and (r, c) not in frame_cells
    ]
    assert sprite_coords, "expected inner sprite inside the innermost frame"

    outer_bbox = frames[-1]["bbox"]
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == bg:
                continue
            if (r, c) in frame_cells or (r, c) in sprite_coords:
                continue
            assert outer_bbox[0] <= r <= outer_bbox[2] and outer_bbox[1] <= c <= outer_bbox[3], "unexpected foreground outside nested structure"

    current_patch, _ = _crop_component(g, sprite_coords, bg)
    for frame in frames:
        current_patch = _compose_aligned(frame["patch"], current_patch, frame["corner"], bg)

    out = np.full_like(g, bg)
    r0, c0, r1, c1 = frames[-1]["bbox"]
    assert current_patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
    _paste_non_bg(out, current_patch, r0, c0, bg)
    return out


def _grow_connected_mask(height: int, width: int, cells: int, rng: random.Random) -> np.ndarray:
    assert 1 <= cells <= height * width
    mask = np.zeros((height, width), dtype=bool)
    start = (rng.randrange(height), rng.randrange(width))
    mask[start] = True
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while int(mask.sum()) < cells:
        assert frontier
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _make_sprite_patch(color: int, bg: int, rng: random.Random) -> np.ndarray:
    height = rng.randint(2, 4)
    width = rng.randint(2, 4)
    max_cells = min(height * width - 1, 6)
    cells = rng.randint(max(3, min(height * width, 3)), max_cells)
    mask = _grow_connected_mask(height, width, cells, rng)
    coords = [(r, c) for r in range(height) for c in range(width) if mask[r, c]]
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch


def _notch_cell(corner: str, shape: tuple[int, int], rng: random.Random) -> Coord:
    h, w = shape
    if corner == "tl":
        return rng.choice([(0, 1), (1, 0)])
    if corner == "tr":
        return rng.choice([(0, w - 2), (1, w - 1)])
    if corner == "bl":
        return rng.choice([(h - 2, 0), (h - 1, 1)])
    return rng.choice([(h - 2, w - 1), (h - 1, w - 2)])


def _make_frame_patch(shape: tuple[int, int], color: int, corner: str, bg: int, rng: random.Random) -> np.ndarray:
    h, w = shape
    patch = np.full((h, w), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    nr, nc = _notch_cell(corner, shape, rng)
    patch[nr, nc] = bg
    return patch


def _build_input_patch(
    current_patch: np.ndarray,
    shape: tuple[int, int],
    color: int,
    corner: str,
    bg: int,
    rng: random.Random,
) -> np.ndarray:
    frame_patch = _make_frame_patch(shape, color, corner, bg, rng)
    child_h, child_w = current_patch.shape
    frame_h, frame_w = shape

    top_choices = list(range(2, frame_h - child_h - 1))
    left_choices = list(range(2, frame_w - child_w - 1))
    assert top_choices and left_choices
    top = rng.choice(top_choices)
    left = rng.choice(left_choices)

    out = frame_patch.copy()
    _paste_non_bg(out, current_patch, top, left, bg)
    return out


def _sample_layers(sprite_patch: np.ndarray, frame_colors: list[int], bg: int, rng: random.Random) -> tuple[np.ndarray, list[str]]:
    current_patch = sprite_patch
    corners: list[str] = []
    for color in frame_colors:
        add_h = rng.randint(4, 5)
        add_w = rng.randint(4, 5)
        shape = (current_patch.shape[0] + add_h, current_patch.shape[1] + add_w)
        corner = rng.choice(FOUR_CORNERS)
        current_patch = _build_input_patch(current_patch, shape, color, corner, bg, rng)
        corners.append(corner)
    return current_patch, corners


def _sample_canvas(patch: np.ndarray, bg: int, rng: random.Random) -> np.ndarray:
    ph, pw = patch.shape
    extra_h = rng.randint(2, max(2, 30 - ph))
    extra_w = rng.randint(2, max(2, 30 - pw))
    h = ph + extra_h
    w = pw + extra_w
    top = rng.randint(1, h - ph - 1)
    left = rng.randint(1, w - pw - 1)
    grid = np.full((h, w), bg, dtype=int)
    _paste_non_bg(grid, patch, top, left, bg)
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(ARC_COLORS)
    n_layers = rng.randint(2, 4)
    palette = [c for c in ARC_COLORS if c != bg]
    rng.shuffle(palette)
    sprite_color = palette[0]
    frame_colors = palette[1 : n_layers + 1]

    sprite_patch = _make_sprite_patch(sprite_color, bg, rng)
    nested_patch, corners = _sample_layers(sprite_patch, frame_colors, bg, rng)
    if nested_patch.shape[0] > 28 or nested_patch.shape[1] > 28:
        return None

    grid = _sample_canvas(nested_patch, bg, rng)
    if max(grid.shape) > 30:
        return None
    if len(set(corners)) == 1 and n_layers > 2 and rng.random() < 0.5:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape:
        return None
    if _get_bg(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-corner input")
