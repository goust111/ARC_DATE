from __future__ import annotations

from collections import defaultdict

import numpy as np


BG = 0
FRAME_COLOR = 5
CONTROL_COLORS = tuple(color for color in range(10) if color not in {BG, FRAME_COLOR})
GEN_RETRIES = 500


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _read_control_segments(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    col = grid[:, 0]
    r = 0
    while r < int(col.shape[0]):
        if int(col[r]) == int(bg):
            r += 1
            continue
        color = int(col[r])
        start = int(r)
        while r < int(col.shape[0]) and int(col[r]) == int(color):
            r += 1
        segments.append((int(color), int(r - start)))
    return segments


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _layer_from_coords(coords: np.ndarray) -> dict[str, object] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    size_r = int(r1 - r0 + 1)
    size_c = int(c1 - c0 + 1)
    if size_r != size_c:
        return None

    local = np.zeros((size_r, size_c), dtype=bool)
    local[coords[:, 0] - int(r0), coords[:, 1] - int(c0)] = True

    if bool(local.all()):
        kind = "solid"
    else:
        if size_r < 3:
            return None
        border = np.zeros_like(local, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not bool(np.array_equal(local, border)):
            return None
        kind = "frame"

    return {
        "coords": coords,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "size": int(size_r),
        "kind": kind,
        "center": (int(r0 + r1), int(c0 + c1)),
    }


def _find_layers(grid: np.ndarray, bg: int, frame_color: int, segment_lengths: list[int]) -> list[dict[str, object]]:
    mask = grid == int(frame_color)
    mask[:, 0] = False

    candidates: list[dict[str, object]] = []
    for coords in _components(mask):
        layer = _layer_from_coords(coords)
        if layer is not None:
            candidates.append(layer)

    if not candidates:
        return []

    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for layer in candidates:
        groups[layer["center"]].append(layer)

    def score(group: list[dict[str, object]]) -> tuple[int, int, int, int]:
        ordered = sorted(group, key=lambda item: -int(item["size"]))
        match_count = sum(
            1
            for idx, seg_len in enumerate(segment_lengths)
            if idx < len(ordered) and int(ordered[idx]["size"]) == int(seg_len)
        )
        gap_count = sum(
            1
            for idx in range(len(ordered) - 1)
            if int(ordered[idx]["size"]) - int(ordered[idx + 1]["size"]) == 4
        )
        return (int(match_count), int(len(ordered)), int(gap_count), int(ordered[0]["size"]))

    best_group = max(groups.values(), key=score)
    return sorted(best_group, key=lambda item: (-int(item["size"]), int(item["bbox"][0]), int(item["bbox"][1])))


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(_mode_color(g))

    segments = _read_control_segments(g, bg)
    segment_colors = [int(color) for color, _length in segments]
    segment_lengths = [int(length) for _color, length in segments]

    body = g[:, 1:]
    body_vals = body[body != int(bg)]
    out = g.copy()
    out[g[:, 0] != int(bg), 0] = int(bg)
    if body_vals.size == 0:
        return out

    values, counts = np.unique(body_vals, return_counts=True)
    frame_color = int(values[int(np.argmax(counts))])
    layers = _find_layers(g, bg, frame_color, segment_lengths)

    for idx, color in enumerate(segment_colors):
        if idx >= len(layers):
            break
        coords = layers[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor concentric square layers from the outside in using the left-edge color-band order.

    Input structure:
    - The background is black.
    - Away from the left edge, grey pixels form several nested square layers with a one-cell black gap between layers.
    - Each layer is either a one-cell-thick hollow square frame or, at the very center, an optional solid grey square.
    - The leftmost column contains contiguous non-black color bands stacked from top to bottom.

    Rule:
    - Read the leftmost-column bands from top to bottom to obtain an ordered color list.
    - Order the concentric grey layers from the outermost square to the innermost square.
    - Recolor layer 1 with the first band color, layer 2 with the second band color, and continue inward in order.
    - If there are fewer bands than layers, leave the remaining inner layers grey.
    - Clear the leftmost color bands back to black in the output.

    Key insight: The left-edge bands encode the outside-to-inside recoloring order, while each band's height matches its target square side length as a visible consistency clue.
    """
    return _transform(grid)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    r1 = int(r0 + size)
    c1 = int(c0 + size)
    grid[int(r0), int(c0):int(c1)] = int(color)
    grid[int(r1 - 1), int(c0):int(c1)] = int(color)
    grid[int(r0):int(r1), int(c0)] = int(color)
    grid[int(r0):int(r1), int(c1 - 1)] = int(color)


def _sample_layout(rng: np.random.Generator) -> tuple[list[int], bool, int]:
    for _ in range(300):
        full_mapping = bool(int(rng.integers(0, 2)))
        if full_mapping:
            layer_count = int(rng.integers(1, 5))
            colored_count = int(layer_count)
        else:
            layer_count = int(rng.integers(2, 8))
            colored_count = int(rng.integers(1, layer_count))

        solid_center = bool(int(rng.integers(0, 2))) if layer_count >= 2 else False
        inner_min, inner_max = (1, 4) if solid_center else (3, 7)
        inner_size = int(rng.integers(inner_min, inner_max + 1))

        sizes = [int(inner_size + 4 * (layer_count - 1 - idx)) for idx in range(layer_count)]
        if sizes[0] > 27:
            continue
        if sum(sizes[:colored_count]) > 30:
            continue
        return sizes, solid_center, colored_count
    raise RuntimeError("failed to sample a valid concentric layout")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sizes, solid_center, colored_count = _sample_layout(rng)
    outer_size = int(sizes[0])
    total_band_height = int(sum(sizes[:colored_count]))

    min_h = max(int(outer_size), int(total_band_height))
    min_w = int(outer_size + 3)
    if min_h > 30 or min_w > 30:
        return None

    height = int(rng.integers(min_h, 31))
    width = int(rng.integers(min_w, 31))
    if width - outer_size < 3:
        return None

    grid = np.full((height, width), BG, dtype=int)

    top = int(rng.integers(0, height - outer_size + 1))
    left = int(rng.integers(3, width - outer_size + 1))
    for idx, size in enumerate(sizes):
        inset = int(2 * idx)
        r0 = int(top + inset)
        c0 = int(left + inset)
        if solid_center and idx == len(sizes) - 1:
            grid[int(r0) : int(r0 + size), int(c0) : int(c0 + size)] = FRAME_COLOR
        else:
            _draw_frame(grid, r0, c0, int(size), FRAME_COLOR)

    colors = [int(c) for c in rng.permutation(CONTROL_COLORS)[:colored_count]]
    start = int(rng.integers(0, height - total_band_height + 1))
    pos = int(start)
    for color, size in zip(colors, sizes[:colored_count]):
        grid[int(pos) : int(pos + size), 0] = int(color)
        pos += int(size)

    segments = _read_control_segments(grid, BG)
    if [length for _color, length in segments] != sizes[:colored_count]:
        return None

    layers = _find_layers(grid, BG, FRAME_COLOR, [length for _color, length in segments])
    if [int(layer["size"]) for layer in layers] != sizes:
        return None

    out = _transform(grid)
    if not bool((out[:, 0] == BG).all()):
        return None

    for idx, color in enumerate(colors):
        coords = layers[idx]["coords"]
        if not bool((out[coords[:, 0], coords[:, 1]] == int(color)).all()):
            return None

    for idx in range(colored_count, len(layers)):
        coords = layers[idx]["coords"]
        if not bool((out[coords[:, 0], coords[:, 1]] == FRAME_COLOR).all()):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
