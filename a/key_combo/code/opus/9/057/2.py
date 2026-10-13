from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
REFLECT_COLOR = 7
MARKER_PALETTE = (1, 2, 3, 4, 6, 8, 9)
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _find_line_row(grid: np.ndarray) -> int:
    coords = np.argwhere(grid == REFLECT_COLOR)
    assert len(coords) >= 4, "expected a sparse marker line"
    rows = np.unique(coords[:, 0])
    assert len(rows) == 1, "marker line must occupy exactly one row"
    return int(rows[0])


def _find_top_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    markers = [(c, int(grid[0, c])) for c in range(grid.shape[1]) if int(grid[0, c]) != BG]
    assert 2 <= len(markers) <= 3, "expected 2-3 top markers"
    colors = [color for _c, color in markers]
    assert len(set(colors)) == len(colors), "top marker colors must be distinct"
    assert all(color not in (WALL, REFLECT_COLOR) for color in colors)
    return markers


def _find_containers(grid: np.ndarray) -> list[dict[str, int]]:
    containers: list[dict[str, int]] = []
    for coords in _components(grid == WALL):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        template = np.zeros((h, w), dtype=bool)
        template[:, 0] = True
        template[:, -1] = True
        template[-1, :] = True
        if not np.array_equal(mask, template):
            continue
        containers.append(
            {
                "top": int(r0),
                "bottom": int(r1),
                "left": int(c0),
                "right": int(c1),
                "inner_w": int(w - 2),
                "inner_h": int(h - 1),
            }
        )
    assert containers, "expected at least one U container"
    containers.sort(key=lambda box: (box["left"], box["top"]))
    return containers


def _container_for_column(containers: list[dict[str, int]], col: int) -> int | None:
    hits = [idx for idx, box in enumerate(containers) if box["left"] <= col <= box["right"]]
    assert len(hits) <= 1, "marker column must not match multiple containers"
    return hits[0] if hits else None


def _paint_reflection(out: np.ndarray, box: dict[str, int], line_row: int) -> None:
    top = box["top"]
    bottom = box["bottom"]
    left = box["left"]
    right = box["right"]
    h, _w = out.shape
    for r in range(top, bottom + 1):
        rr = 2 * line_row - r
        assert line_row < rr < h, "reflection must stay below the marker line"
        for c in range(left, right + 1):
            value = int(out[r, c])
            if value == BG:
                continue
            color = WALL if value == WALL else REFLECT_COLOR
            existing = int(out[rr, c])
            assert existing in (BG, color), "reflections must not create conflicting overlaps"
            out[rr, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each top marker into a U-container or onto the pre-line floor, then reflect only the filled containers across the sparse marker row.

    Input structure:
    - The grid uses black background, grey U-shaped wall containers in the upper region,
      a sparse orange horizontal marker row in the middle/lower area, and 2-3 distinct
      single-pixel markers on the top row.
    - Each grey container is a simple U shape with two vertical walls and one bottom wall.
      The row immediately above the orange marker row starts empty and acts as the floor
      for missed drops.

    Rule:
    - Erase every top-row marker from its original position.
    - Drop each marker straight down by column: if its column lies within exactly one
      container footprint, fill that container interior with the marker color; otherwise
      fill the entire row directly above the sparse orange marker row with that marker color.
    - After all fills, reflect only the containers that were actually filled across the
      sparse orange marker row. The reflected copy keeps grey walls, but every reflected
      interior cell becomes orange, the marker-row color.
    - Unfilled containers, the orange marker row itself, and any floor puddle remain in place.

    Key insight:
    - The gravity result is the gate for the second step: being filled is what qualifies a
      container for reflection, while missed drops create a floor puddle that does not reflect.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    line_row = _find_line_row(g)
    puddle_row = line_row - 1
    assert puddle_row >= 1, "marker line must not touch the top"
    markers = _find_top_markers(g)
    containers = _find_containers(g)

    out = g.copy()
    filled: dict[int, int] = {}
    miss_color: int | None = None

    for col, color in markers:
        out[0, col] = BG
        idx = _container_for_column(containers, col)
        if idx is None:
            assert miss_color is None, "multiple missed markers would create an ambiguous puddle"
            miss_color = color
            out[puddle_row, :] = color
            continue

        assert idx not in filled, "a container must not receive multiple top markers"
        box = containers[idx]
        out[box["top"] : box["bottom"], box["left"] + 1 : box["right"]] = color
        filled[idx] = color

    for idx in filled:
        _paint_reflection(out, containers[idx], line_row)

    return out


def _random_sparse_positions(width: int) -> list[int]:
    for _ in range(50):
        spacing = random.randint(2, 4)
        offset = random.randint(0, spacing - 1)
        positions = list(range(offset, width, spacing))
        if 4 <= len(positions) <= 7 and positions[-1] >= width - spacing:
            return positions
    return list(range(0, width, 3))


def _random_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    last = -1
    out: list[int] = []
    for cut in cuts + [total + parts - 1]:
        out.append(cut - last - 1)
        last = cut
    return out


def _sample_layout() -> tuple[np.ndarray, dict[str, object]] | None:
    h = random.randint(17, 22)
    w = random.randint(17, 22)
    line_row = random.randint(h // 2, min(h // 2 + 1, h - 6))
    puddle_row = line_row - 1

    marker_count = random.choice((2, 3))
    hit_count = marker_count - 1
    container_count = random.choice((3, 4))
    if container_count < hit_count + 1:
        return None

    hit_slots = set(random.sample(range(container_count), hit_count))
    inner_ws: list[int] = []
    inner_hs: list[int] = []
    widths: list[int] = []
    heights: list[int] = []
    for idx in range(container_count):
        inner_w = 1 if idx in hit_slots else random.randint(2, 4)
        inner_h = random.randint(2, 4)
        inner_ws.append(inner_w)
        inner_hs.append(inner_h)
        widths.append(inner_w + 2)
        heights.append(inner_h + 1)

    base_gaps = [0] + [1] * (container_count - 1) + [0]
    needed = sum(widths) + sum(base_gaps)
    if needed > w:
        return None

    slack = w - needed
    extras = _random_partition(slack, container_count + 1)
    gaps = [base_gaps[i] + extras[i] for i in range(container_count + 1)]

    min_top = max(2, 2 * line_row - (h - 1))
    containers: list[dict[str, int]] = []
    left = gaps[0]
    for idx in range(container_count):
        box_h = heights[idx]
        max_top = line_row - box_h - 2
        if min_top > max_top:
            return None
        top = random.randint(min_top, max_top)
        right = left + widths[idx] - 1
        containers.append(
            {
                "top": top,
                "bottom": top + box_h - 1,
                "left": left,
                "right": right,
                "inner_w": inner_ws[idx],
                "inner_h": inner_hs[idx],
            }
        )
        left = right + 1 + gaps[idx + 1]

    covered_cols = {
        c
        for box in containers
        for c in range(box["left"], box["right"] + 1)
    }
    miss_candidates = [c for c in range(w) if c not in covered_cols]
    if not miss_candidates:
        return None

    hit_indices = sorted(hit_slots)
    hit_cols = [containers[idx]["left"] + 1 for idx in hit_indices]
    if len(set(hit_cols)) != len(hit_cols):
        return None

    miss_col = random.choice(miss_candidates)
    marker_columns = hit_cols + [miss_col]
    if len(set(marker_columns)) != len(marker_columns):
        return None

    colors = random.sample(MARKER_PALETTE, marker_count)
    roles = [("hit", idx) for idx in hit_indices] + [("miss", -1)]
    random.shuffle(roles)
    markers: list[dict[str, int]] = []
    for color, role in zip(colors, roles):
        kind, payload = role
        if kind == "hit":
            box = containers[payload]
            markers.append({"col": box["left"] + 1, "color": color, "target": payload})
        else:
            markers.append({"col": miss_col, "color": color, "target": -1})
    markers.sort(key=lambda item: item["col"])

    grid = np.zeros((h, w), dtype=int)
    for c in _random_sparse_positions(w):
        grid[line_row, c] = REFLECT_COLOR
    for box in containers:
        grid[box["top"] : box["bottom"] + 1, box["left"]] = WALL
        grid[box["top"] : box["bottom"] + 1, box["right"]] = WALL
        grid[box["bottom"], box["left"] : box["right"] + 1] = WALL
    for marker in markers:
        grid[0, marker["col"]] = marker["color"]

    meta: dict[str, object] = {
        "line_row": line_row,
        "puddle_row": puddle_row,
        "containers": containers,
        "markers": markers,
        "hit_indices": set(hit_indices),
        "miss_col": miss_col,
    }
    return grid, meta


def _assert_generated_semantics(inp: np.ndarray, meta: dict[str, object]) -> None:
    out = main(inp)
    line_row = int(meta["line_row"])
    puddle_row = int(meta["puddle_row"])
    containers = list(meta["containers"])
    markers = list(meta["markers"])
    hit_indices = set(meta["hit_indices"])

    assert np.array_equal(out[line_row], inp[line_row]), "marker line must stay unchanged"
    assert np.all(out[0, :] == 0), "top markers must be erased"

    miss_markers = [marker for marker in markers if int(marker["target"]) < 0]
    assert len(miss_markers) == 1
    miss_color = int(miss_markers[0]["color"])
    assert np.all(out[puddle_row, :] == miss_color), "missed marker must create one full-width puddle"

    for idx, box in enumerate(containers):
        top = int(box["top"])
        bottom = int(box["bottom"])
        left = int(box["left"])
        right = int(box["right"])
        src_inner = out[top:bottom, left + 1 : right]
        ref_inner = out[2 * line_row - bottom + 1 : 2 * line_row - top + 1, left + 1 : right]
        ref_left = out[2 * line_row - bottom : 2 * line_row - top + 1, left]
        ref_right = out[2 * line_row - bottom : 2 * line_row - top + 1, right]
        ref_bottom = out[2 * line_row - bottom, left : right + 1]

        if idx in hit_indices:
            colors = {int(marker["color"]) for marker in markers if int(marker["target"]) == idx}
            assert len(colors) == 1
            fill_color = next(iter(colors))
            assert src_inner.size > 0 and np.all(src_inner == fill_color)
            assert np.all(ref_inner == REFLECT_COLOR), "reflected interior must recolor to orange"
            assert np.all(ref_left == WALL) and np.all(ref_right == WALL) and np.all(ref_bottom == WALL)
        else:
            assert src_inner.size > 0 and np.all(src_inner == BG)
            assert np.all(ref_inner == BG), "unfilled containers must not reflect"
            assert np.all(ref_left == BG) and np.all(ref_right == BG) and np.all(ref_bottom == BG)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        sample = _sample_layout()
        if sample is None:
            continue
        grid, meta = sample
        try:
            _assert_generated_semantics(grid, meta)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid conditional-reflection ARC sample")
