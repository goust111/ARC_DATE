from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
RING = [2, 7, 4, 3, 8, 1, 6, 9]  # red, orange, yellow, green, azure, blue, magenta, maroon
RING_INDEX = {color: idx for idx, color in enumerate(RING)}
MAX_SIDE = 30
GEN_RETRIES = 400


def _boundary_bg(grid: np.ndarray) -> int:
    border = np.concatenate(
        [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
    )
    values, counts = np.unique(border, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(points: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _perimeter_coords(h: int, w: int, inset: int) -> list[tuple[int, int]]:
    r0 = inset
    c0 = inset
    r1 = h - 1 - inset
    c1 = w - 1 - inset
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        if r1 != r0:
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _parse_frame(
    grid: np.ndarray, points: list[tuple[int, int]], bg: int
) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(points)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape

    layer_colors: list[int] = []
    max_layers = min(3, min(h, w) // 2)
    for inset in range(max_layers + 1):
        coords = _perimeter_coords(h, w, inset)
        values = {int(sub[rr, cc]) for rr, cc in coords}
        if len(values) != 1:
            break
        color = values.pop()
        if color == bg:
            break
        if layer_colors and color == layer_colors[-1]:
            break
        if color not in RING_INDEX:
            break
        if layer_colors:
            expected = RING[
                (RING_INDEX[int(layer_colors[0])] + len(layer_colors)) % len(RING)
            ]
        else:
            expected = color
        if int(color) != int(expected):
            break
        layer_colors.append(color)

    assert len(layer_colors) in (2, 3), "expected a 2- or 3-layer frame"
    inset = len(layer_colors)
    cavity = sub[inset : h - inset, inset : w - inset]
    assert cavity.size > 0, "frame cavity must be nonempty"
    template = cavity == bg
    assert bool(template.any()), "frame cavity must contain a bg template"

    outer = int(layer_colors[0])
    target = RING[(RING_INDEX[outer] + len(layer_colors)) % len(RING)]
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "template": template,
        "target_color": int(target),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer each frame's inner bg cutout onto the sprite whose color is determined by the frame's color-ring step count.

    Input structure:
    - The background is black, and the grid contains several nested rectangular frames plus several separate monochrome sprites.
    - Every frame is one connected multi-color object made of 2 or 3 concentric rectangular border layers with distinct ring colors.
    - Inside the innermost cavity, some cells stay black as a cutout template while the remaining cavity cells are filled by the innermost frame color.

    Rule:
    - Separate non-background components into monochrome sprites and multi-color frames.
    - For each frame, read the outer border color and count its border layers; on the fixed visible color ring, advance from the outer color by that many steps to get the target sprite color.
    - Extract the black cutout mask from the frame's innermost cavity and stamp that mask in grey onto the top-left-aligned region of the matching sprite, only where the sprite already has pixels.

    Key insight:
    - The nested border colors identify which sprite to modify, but the black cavity cutout is the payload that actually gets transferred.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _boundary_bg(g)

    sprites: dict[int, list[dict[str, object]]] = {}
    frames: list[dict[str, object]] = []
    for comp in _components(g != bg):
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) == 1:
            color = int(next(iter(colors)))
            bbox = _bbox(comp)
            sprites.setdefault(color, []).append(
                {
                    "bbox": bbox,
                    "mask": _crop_mask(comp, bbox),
                }
            )
        else:
            frames.append(_parse_frame(g, comp, bg))

    assert frames, "expected at least one nested frame"
    out = g.copy()
    for frame in frames:
        target_color = int(frame["target_color"])
        assert target_color in sprites and len(sprites[target_color]) == 1, "target sprite must be unique"
        sprite = sprites[target_color][0]
        sr0, sc0, _sr1, _sc1 = sprite["bbox"]  # type: ignore[misc]
        sprite_mask = sprite["mask"]  # type: ignore[assignment]
        template = frame["template"]  # type: ignore[assignment]
        hh = min(int(template.shape[0]), int(sprite_mask.shape[0]))
        ww = min(int(template.shape[1]), int(sprite_mask.shape[1]))
        overlap = template[:hh, :ww] & sprite_mask[:hh, :ww]
        patch = out[sr0 : sr0 + hh, sc0 : sc0 + ww]
        patch[overlap] = GREY
    return out


def _grow_connected_mask(
    h: int, w: int, area: int, *, start_on_boundary: bool
) -> np.ndarray | None:
    if not (1 <= area <= h * w):
        return None
    mask = np.zeros((h, w), dtype=bool)
    if start_on_boundary:
        boundary = [(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)]
        boundary += [(r, 0) for r in range(1, h - 1)] + [(r, w - 1) for r in range(1, h - 1)]
        start = random.choice(boundary)
    else:
        start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while int(mask.sum()) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        r, c = cell
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _render_frame(layer_colors: list[int], filler_mask: np.ndarray) -> np.ndarray:
    layers = len(layer_colors)
    cavity_h, cavity_w = filler_mask.shape
    out = np.full((cavity_h + 2 * layers, cavity_w + 2 * layers), BG, dtype=int)
    h, w = out.shape
    for inset, color in enumerate(layer_colors):
        r0, c0 = inset, inset
        r1, c1 = h - 1 - inset, w - 1 - inset
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    cavity = out[layers : h - layers, layers : w - layers]
    cavity[filler_mask] = layer_colors[-1]
    return out


def _make_frame_object(outer_idx: int, layers: int) -> dict[str, object] | None:
    cavity_h = random.randint(4, 7)
    cavity_w = random.randint(4, 7)
    area = cavity_h * cavity_w
    filler_area = random.randint(max(1, area // 3), area - 1)
    filler_mask = _grow_connected_mask(
        cavity_h, cavity_w, filler_area, start_on_boundary=True
    )
    if filler_mask is None:
        return None
    layer_colors = [RING[(outer_idx + step) % len(RING)] for step in range(layers)]
    template = ~filler_mask
    target = RING[(outer_idx + layers) % len(RING)]
    return {
        "grid": _render_frame(layer_colors, filler_mask),
        "template": template,
        "target_color": int(target),
    }


def _make_sprite_object(target_color: int, template: np.ndarray) -> np.ndarray | None:
    th, tw = template.shape
    for _ in range(80):
        sprite_h = max(3, min(10, th + random.choice([-2, -1, 0, 1, 2])))
        sprite_w = max(3, min(10, tw + random.choice([-2, -1, 0, 1, 2])))
        area = sprite_h * sprite_w
        sprite_area = random.randint(max(3, area // 3), max(3, area - 1))
        sprite_mask = _grow_connected_mask(
            sprite_h, sprite_w, sprite_area, start_on_boundary=False
        )
        if sprite_mask is None:
            continue
        hh = min(sprite_h, th)
        ww = min(sprite_w, tw)
        if not np.any(sprite_mask[:hh, :ww] & template[:hh, :ww]):
            continue
        return np.where(sprite_mask, target_color, BG)
    return None


def _sample_frame_specs(n_frames: int) -> list[tuple[int, int]]:
    combos = [(outer_idx, layers) for outer_idx in range(len(RING)) for layers in (2, 3)]
    random.shuffle(combos)
    out: list[tuple[int, int]] = []
    used_targets: set[int] = set()
    for outer_idx, layers in combos:
        target = RING[(outer_idx + layers) % len(RING)]
        if target in used_targets:
            continue
        out.append((outer_idx, layers))
        used_targets.add(int(target))
        if len(out) == n_frames:
            break
    assert len(out) == n_frames
    return out


def _place_objects(objects: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.full((MAX_SIDE, MAX_SIDE), BG, dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)

    order = sorted(range(len(objects)), key=lambda idx: int(np.count_nonzero(objects[idx])), reverse=True)
    for idx in order:
        obj = objects[idx]
        h, w = obj.shape
        if h > MAX_SIDE - 2 or w > MAX_SIDE - 2:
            return None
        placed = False
        for _ in range(300):
            top = random.randint(1, MAX_SIDE - h - 1)
            left = random.randint(1, MAX_SIDE - w - 1)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(MAX_SIDE, top + h + 1)
            c1 = min(MAX_SIDE, left + w + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            patch = canvas[top : top + h, left : left + w]
            patch[obj != BG] = obj[obj != BG]
            occupied[top : top + h, left : left + w] = True
            placed = True
            break
        if not placed:
            return None

    pts = np.argwhere(canvas != BG)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    top_margin = random.randint(1, min(3, int(r0)))
    left_margin = random.randint(1, min(3, int(c0)))
    bottom_margin = random.randint(1, min(3, int(MAX_SIDE - 1 - r1)))
    right_margin = random.randint(1, min(3, int(MAX_SIDE - 1 - c1)))
    return canvas[
        int(r0) - top_margin : int(r1) + bottom_margin + 1,
        int(c0) - left_margin : int(c1) + right_margin + 1,
    ]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_frames = random.choice([2, 3])
        frame_specs = _sample_frame_specs(n_frames)

        frame_objects: list[dict[str, object]] = []
        ok = True
        for outer_idx, layers in frame_specs:
            frame_obj = _make_frame_object(outer_idx, layers)
            if frame_obj is None:
                ok = False
                break
            frame_objects.append(frame_obj)
        if not ok:
            continue

        sprite_objects: list[np.ndarray] = []
        for frame_obj in frame_objects:
            sprite = _make_sprite_object(
                int(frame_obj["target_color"]), frame_obj["template"]  # type: ignore[arg-type]
            )
            if sprite is None:
                ok = False
                break
            sprite_objects.append(sprite)
        if not ok:
            continue

        objects = [frame_obj["grid"] for frame_obj in frame_objects] + sprite_objects
        grid = _place_objects(objects)  # type: ignore[arg-type]
        if grid is None:
            continue
        out = main(grid)
        if np.array_equal(grid, out):
            continue
        if not np.any(out == GREY):
            continue
        assert grid.ndim == 2 and max(grid.shape) <= MAX_SIDE
        assert out.ndim == 2 and max(out.shape) <= MAX_SIDE
        return grid

    raise AssertionError("failed to generate a valid input grid")
