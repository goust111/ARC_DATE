from __future__ import annotations

import numpy as np


BG = 0
DRAW_ORDER = (1, 2, 3, 4, 5, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")
SHIFT_BY_DIRECTION = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_GRID_MIN = 14
GEN_GRID_MAX = 24
GEN_OBJECT_MIN = 2
GEN_OBJECT_MAX = 5
GEN_SHAFT_MIN = 1
GEN_SHAFT_MAX = 6
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_rel(direction: str, shaft_len: int) -> set[tuple[int, int]]:
    if direction == "up":
        h = shaft_len + 3
        return {(r, 1) for r in range(h)} | {(1, 0), (1, 2)}
    if direction == "down":
        h = shaft_len + 3
        return {(r, 1) for r in range(h)} | {(h - 2, 0), (h - 2, 2)}
    if direction == "left":
        w = shaft_len + 3
        return {(1, c) for c in range(w)} | {(0, 1), (2, 1)}
    if direction == "right":
        w = shaft_len + 3
        return {(1, c) for c in range(w)} | {(0, w - 2), (2, w - 2)}
    raise ValueError(direction)


def _arrow_bbox_shape(direction: str, shaft_len: int) -> tuple[int, int]:
    if direction in {"up", "down"}:
        return shaft_len + 3, 3
    return 3, shaft_len + 3


def _recognize_arrow(coords: list[tuple[int, int]]) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = {(r - r0, c - c0) for r, c in coords}
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    matches: list[tuple[str, int]] = []
    if w == 3 and h >= 4:
        shaft_len = h - 3
        for direction in ("up", "down"):
            if rel == _arrow_rel(direction, shaft_len):
                matches.append((direction, shaft_len))
    if h == 3 and w >= 4:
        shaft_len = w - 3
        for direction in ("left", "right"):
            if rel == _arrow_rel(direction, shaft_len):
                matches.append((direction, shaft_len))
    if len(matches) != 1:
        raise AssertionError(f"unrecognized arrow component: bbox={(h, w)} coords={sorted(rel)}")
    direction, shaft_len = matches[0]
    dr, dc = SHIFT_BY_DIRECTION[direction]
    return {
        "direction": direction,
        "shaft_len": shaft_len,
        "shift": (dr * shaft_len, dc * shaft_len),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    protected = g != BG

    arrows: list[dict[str, object]] = []
    for comp in _components(g):
        info = _recognize_arrow(comp["coords"])
        info["color"] = int(comp["color"])
        info["coords"] = list(comp["coords"])
        arrows.append(info)

    order_index = {color: i for i, color in enumerate(DRAW_ORDER)}
    arrows.sort(key=lambda item: (order_index.get(int(item["color"]), 999), int(item["color"])))

    for arrow in arrows:
        color = int(arrow["color"])
        shift_r, shift_c = arrow["shift"]
        for r, c in arrow["coords"]:
            rr = r + shift_r
            cc = c + shift_c
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if protected[rr, cc] and int(g[rr, cc]) != color:
                continue
            out[rr, cc] = color

    return out


def _placed_cells(
    top: int,
    left: int,
    rel_cells: set[tuple[int, int]],
    shift: tuple[int, int],
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], bool]:
    original = {(top + dr, left + dc) for dr, dc in rel_cells}
    shifted_all = {(r + shift[0], c + shift[1]) for r, c in original}
    shifted_in = {(r, c) for r, c in shifted_all if 0 <= r < h and 0 <= c < w}
    return original, shifted_in, len(shifted_in) < len(original)


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _render_input(h: int, w: int, placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for item in placed:
        color = int(item["color"])
        for r, c in item["original"]:
            grid[r, c] = color
    return grid


def _place_specs(
    rng: np.random.Generator,
    h: int,
    w: int,
    specs: list[dict[str, object]],
    target_clip: bool,
    target_overlap: bool,
) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    original_union: set[tuple[int, int]] = set()
    original_halo: set[tuple[int, int]] = set()
    shifted_union: set[tuple[int, int]] = set()
    clip_done = False
    overlap_done = False

    order = list(range(len(specs)))
    order.sort(key=lambda i: (-len(specs[i]["rel_cells"]), rng.random()))

    for idx in order:
        spec = specs[idx]
        bbox_h = int(spec["bbox_h"])
        bbox_w = int(spec["bbox_w"])
        rel_cells = spec["rel_cells"]
        shift = spec["shift"]
        candidates: list[dict[str, object]] = []

        for top in range(h - bbox_h + 1):
            for left in range(w - bbox_w + 1):
                original, shifted, is_clip = _placed_cells(top, left, rel_cells, shift, h, w)
                if original & original_halo:
                    continue
                if shifted & original_union:
                    continue
                if original & shifted_union:
                    continue
                overlap_count = len(shifted & shifted_union)
                score = rng.random()
                if target_clip and not clip_done and is_clip:
                    score += 4.0
                if target_overlap and not overlap_done and overlap_count > 0:
                    score += 3.0 + 0.2 * overlap_count
                if not target_overlap and overlap_count > 0:
                    score -= 0.5 * overlap_count
                candidates.append(
                    {
                        "score": score,
                        "top": top,
                        "left": left,
                        "original": original,
                        "shifted": shifted,
                        "clip": is_clip,
                        "overlap_count": overlap_count,
                    }
                )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item["score"], reverse=True)
        pool = candidates[: min(12, len(candidates))]
        chosen = pool[int(rng.integers(len(pool)))]

        entry = {
            "color": int(spec["color"]),
            "direction": spec["direction"],
            "shaft_len": int(spec["shaft_len"]),
            "original": chosen["original"],
            "shifted": chosen["shifted"],
        }
        placed.append(entry)
        original_union |= chosen["original"]
        original_halo |= _halo(chosen["original"], h, w)
        shifted_union |= chosen["shifted"]
        clip_done = clip_done or bool(chosen["clip"])
        overlap_done = overlap_done or int(chosen["overlap_count"]) > 0

    if target_clip and not clip_done:
        return None
    if target_overlap and not overlap_done:
        return None
    return placed


def _make_specs(rng: np.random.Generator, count: int) -> list[dict[str, object]]:
    colors = list(rng.choice(np.array(DRAW_ORDER, dtype=int), size=count, replace=False))
    rng.shuffle(colors)

    direction_pool = list(DIRECTIONS)
    rng.shuffle(direction_pool)
    directions = [direction_pool[i % len(direction_pool)] for i in range(count)]
    for i in range(count):
        if rng.random() < 0.35:
            directions[i] = str(rng.choice(np.array(DIRECTIONS, dtype=object)))

    specs: list[dict[str, object]] = []
    for color, direction in zip(colors, directions):
        shaft_len = int(rng.integers(GEN_SHAFT_MIN, GEN_SHAFT_MAX + 1))
        bbox_h, bbox_w = _arrow_bbox_shape(direction, shaft_len)
        dr, dc = SHIFT_BY_DIRECTION[direction]
        specs.append(
            {
                "color": int(color),
                "direction": direction,
                "shaft_len": shaft_len,
                "bbox_h": bbox_h,
                "bbox_w": bbox_w,
                "rel_cells": _arrow_rel(direction, shaft_len),
                "shift": (dr * shaft_len, dc * shaft_len),
            }
        )
    return specs


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GEN_GRID_MIN, GEN_GRID_MAX + 1))
    w = int(rng.integers(GEN_GRID_MIN, GEN_GRID_MAX + 1))
    count = int(rng.integers(GEN_OBJECT_MIN, GEN_OBJECT_MAX + 1))
    target_clip = bool(rng.random() < 0.45)
    target_overlap = bool(rng.random() < 0.4)

    specs = _make_specs(rng, count)
    placed = _place_specs(rng, h, w, specs, target_clip=target_clip, target_overlap=target_overlap)
    if placed is None:
        return None

    grid = _render_input(h, w, placed)

    comps = _components(grid)
    if len(comps) != count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-translation input")
