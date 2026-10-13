from __future__ import annotations

from collections import deque
import random

import numpy as np


BBox = tuple[int, int, int, int]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)

                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "coords": coords,
                }
            )
    return components


def _is_rect_frame(coords: list[tuple[int, int]], bbox: BBox) -> bool:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    expected_area = 2 * (h + w) - 4
    if len(coords) != expected_area:
        return False

    cells = set(coords)
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    return [
        comp
        for comp in _find_components(grid, bg)
        if _is_rect_frame(comp["coords"], comp["bbox"])
    ]


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _inset_bbox(bbox: BBox, delta: int) -> BBox:
    r0, c0, r1, c1 = bbox
    return (int(r0 + delta), int(c0 + delta), int(r1 - delta), int(c1 - delta))


def _ring_is_bg(grid: np.ndarray, bg: int, bbox: BBox, delta: int) -> bool:
    r0, c0, r1, c1 = _inset_bbox(bbox, delta)
    if r0 > r1 or c0 > c1:
        return False
    return bool(
        np.all(grid[r0, c0 : c1 + 1] == bg)
        and np.all(grid[r1, c0 : c1 + 1] == bg)
        and np.all(grid[r0 : r1 + 1, c0] == bg)
        and np.all(grid[r0 : r1 + 1, c1] == bg)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find every nested-frame group and swap the outermost and innermost frame colors.

    Input structure:
    - The background fills the grid.
    - Each object group is made of 2-4 one-pixel-thick rectangular frames.
    - Adjacent frame layers are separated by a one-pixel background gap.

    Rule:
    - For each nested group, preserve the layer order.
    - Recolor the outermost frame with the original innermost color.
    - Recolor the innermost frame with the original outermost color.
    - Keep any middle layers and all background cells unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    frames = _find_frames(g, bg)
    frame_by_bbox = {frame["bbox"]: frame for frame in frames}

    child_of: dict[BBox, BBox] = {}
    parent_of: dict[BBox, BBox] = {}
    for frame in frames:
        bbox = frame["bbox"]
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 7 or c1 - c0 + 1 < 7:
            continue

        inner_bbox = _inset_bbox(bbox, 2)
        if inner_bbox not in frame_by_bbox:
            continue
        if not _ring_is_bg(g, bg, bbox, 1):
            continue

        child_of[bbox] = inner_bbox
        parent_of[inner_bbox] = bbox

    out = g.copy()
    for frame in frames:
        root_bbox = frame["bbox"]
        if root_bbox in parent_of:
            continue

        chain = [frame]
        cursor = root_bbox
        while cursor in child_of:
            cursor = child_of[cursor]
            chain.append(frame_by_bbox[cursor])

        if len(chain) < 2:
            continue

        outer = chain[0]
        inner = chain[-1]
        _draw_frame(out, outer["bbox"], int(inner["color"]))
        _draw_frame(out, inner["bbox"], int(outer["color"]))

    return out


COLOR_IDS = tuple(range(10))
GRID_MIN = 18
GRID_MAX = 30
GROUP_PAD = 2
MAX_ATTEMPTS = 400


def _choose_group_count(rows: int, cols: int) -> int:
    limit = 1
    if min(rows, cols) >= 21:
        limit = 2
    if min(rows, cols) >= 24:
        limit = 3
    if min(rows, cols) >= 28:
        limit = 4
    return random.randint(1, limit)


def _sample_group_spec(rows: int, cols: int, bg: int) -> dict | None:
    layers = random.choice((2, 3, 4))
    min_outer = 3 + 4 * (layers - 1)
    max_outer_h = min(rows - 2, min_outer + 6)
    max_outer_w = min(cols - 2, min_outer + 6)
    if max_outer_h < min_outer or max_outer_w < min_outer:
        return None

    outer_h = random.randint(min_outer, max_outer_h)
    outer_w = random.randint(min_outer, max_outer_w)
    colors = random.sample([c for c in COLOR_IDS if c != bg], layers)
    return {
        "layers": layers,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "colors": colors,
    }


def _place_groups(rows: int, cols: int, specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros((rows, cols), dtype=bool)
    placed: list[dict] = []

    for spec in sorted(specs, key=lambda item: item["outer_h"] * item["outer_w"], reverse=True):
        h = int(spec["outer_h"])
        w = int(spec["outer_w"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(rows - h + 1):
            for c0 in range(cols - w + 1):
                r1 = r0 + h - 1
                c1 = c0 + w - 1
                rr0 = max(0, r0 - GROUP_PAD)
                cc0 = max(0, c0 - GROUP_PAD)
                rr1 = min(rows - 1, r1 + GROUP_PAD)
                cc1 = min(cols - 1, c1 + GROUP_PAD)
                if not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        placed_spec = dict(spec)
        placed_spec["r0"] = int(r0)
        placed_spec["c0"] = int(c0)
        placed.append(placed_spec)

        r1 = r0 + h - 1
        c1 = c0 + w - 1
        rr0 = max(0, r0 - GROUP_PAD)
        cc0 = max(0, c0 - GROUP_PAD)
        rr1 = min(rows - 1, r1 + GROUP_PAD)
        cc1 = min(cols - 1, c1 + GROUP_PAD)
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    return placed


def _render_case(rows: int, cols: int, bg: int, groups: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((rows, cols), int(bg), dtype=int)
    for group in groups:
        r0 = int(group["r0"])
        c0 = int(group["c0"])
        outer_h = int(group["outer_h"])
        outer_w = int(group["outer_w"])
        for layer, color in enumerate(group["colors"]):
            bbox = (
                r0 + 2 * layer,
                c0 + 2 * layer,
                r0 + outer_h - 1 - 2 * layer,
                c0 + outer_w - 1 - 2 * layer,
            )
            _draw_frame(inp, bbox, int(color))

    out = inp.copy()
    for group in groups:
        r0 = int(group["r0"])
        c0 = int(group["c0"])
        outer_h = int(group["outer_h"])
        outer_w = int(group["outer_w"])
        layers = int(group["layers"])
        outer_bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        inner_delta = 2 * (layers - 1)
        inner_bbox = (
            r0 + inner_delta,
            c0 + inner_delta,
            r0 + outer_h - 1 - inner_delta,
            c0 + outer_w - 1 - inner_delta,
        )
        outer_color = int(group["colors"][0])
        inner_color = int(group["colors"][-1])
        _draw_frame(out, outer_bbox, inner_color)
        _draw_frame(out, inner_bbox, outer_color)

    return inp, out


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))]) == int(bg)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rows = random.randint(GRID_MIN, GRID_MAX)
        cols = random.randint(GRID_MIN, GRID_MAX)
        bg = random.choice(COLOR_IDS)
        group_count = _choose_group_count(rows, cols)

        specs: list[dict] = []
        ok = True
        for _group_idx in range(group_count):
            spec = _sample_group_spec(rows, cols, bg)
            if spec is None:
                ok = False
                break
            specs.append(spec)
        if not ok:
            continue

        groups = _place_groups(rows, cols, specs)
        if groups is None:
            continue

        inp, expected = _render_case(rows, cols, bg, groups)
        if not _bg_is_dominant(inp, bg):
            continue
        if np.array_equal(inp, expected):
            continue

        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError("failed to generate a valid nested-frame input")
