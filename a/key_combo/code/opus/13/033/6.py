from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

SPRITE_PATTERNS = [
    np.array([[BLUE, GREEN, BLUE], [GREEN, BLUE, GREEN], [BLUE, BLUE, GREEN]], dtype=int),
    np.array([[GREEN, BLUE, GREEN], [BLUE, GREEN, BLUE], [GREEN, GREEN, BLUE]], dtype=int),
    np.array([[BLUE, BLUE, GREEN], [GREEN, BLUE, GREEN], [BLUE, GREEN, BLUE]], dtype=int),
    np.array([[GREEN, BLUE, BLUE], [BLUE, GREEN, GREEN], [GREEN, BLUE, GREEN]], dtype=int),
]

GEN_HEIGHT_RANGE = (16, 23)
GEN_WIDTH_RANGE = (19, 25)
GEN_CONTAINER_COUNT_CHOICES = (2, 3, 3)
GEN_MARKER_COUNT_RANGE = (2, 4)
GEN_CONTAINER_HEIGHTS = (3, 4, 5)
GEN_CONTAINER_WIDTHS = (3, 5)
GEN_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_sprite(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    mask = (grid == BLUE) | (grid == GREEN)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected one blue/green source sprite"
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert patch.shape == (3, 3), "expected a 3x3 source sprite"
    assert set(map(int, np.unique(patch))) <= {BLUE, GREEN}
    assert BLUE in patch and GREEN in patch
    return (r0, c0, r1, c1), patch


def _container_from_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 3 and w >= 3, "container bbox must be at least 3x3"

    expected = np.full((h, w), bg, dtype=int)
    expected[:, 0] = RED
    expected[:, -1] = RED
    expected[-1, :] = RED
    assert np.array_equal(patch, expected), "red component must be a clean U-shaped container"

    return {
        "top": r0,
        "bottom": r1,
        "left": c0,
        "right": c1,
        "height": h,
        "width": w,
        "center": (c0 + c1) // 2,
    }


def _find_containers(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _components(grid == RED)
    containers = [_container_from_component(grid, comp, bg) for comp in comps]
    containers.sort(key=lambda d: (d["left"], d["top"]))
    return containers


def _stamp_patch(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = out.shape
    ph, pw = patch.shape
    for pr in range(ph):
        rr = top + pr
        if not (0 <= rr < h):
            continue
        for pc in range(pw):
            cc = left + pc
            if 0 <= cc < w:
                out[rr, cc] = int(patch[pr, pc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop top-row yellow markers into red U-containers, filling the caught container and stamping the lower-left sprite above it.

    Input structure:
    - The background is black, with 2-4 yellow marker pixels on the top row.
    - The lower half contains 2-3 red U-shaped containers, each made of two vertical side walls and one bottom wall, with black interior and an upward opening.
    - A single 3x3 blue/green sprite sits in the lower-left area and acts as the stamp template.

    Rule:
    - Remove every top-row yellow marker and erase the original blue/green sprite from the output.
    - For each marker column from left to right, find the unique red container whose horizontal span includes that column.
    - If a container is hit, fill its interior with yellow and stamp the 3x3 sprite above the opening, centered on the container's opening center column, with the sprite bottom row one row above the container top; multiple hits on the same container only act once.
    - If no container is hit, place a yellow trace at that column in the bottom row; red walls remain unchanged, and later stamps overwrite earlier overlapping stamps.

    Key insight:
    - The falling marker selects a container by column, but the resulting stamp is anchored by the container geometry rather than by the marker's final trace position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    marker_cols = [int(c) for c in np.flatnonzero(g[0] == YELLOW)]
    assert marker_cols, "expected at least one top-row marker"

    sprite_bbox, sprite = _find_sprite(g)
    containers = _find_containers(g, bg)
    assert containers, "expected at least one container"

    out = g.copy()
    out[0, marker_cols] = bg
    sr0, sc0, sr1, sc1 = sprite_bbox
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg

    activated: set[int] = set()
    missed_cols: list[int] = []
    for col in sorted(marker_cols):
        hits = [i for i, cont in enumerate(containers) if cont["left"] <= col <= cont["right"]]
        assert len(hits) <= 1, "marker column must identify at most one container"
        if hits:
            activated.add(hits[0])
        else:
            missed_cols.append(col)

    for idx in activated:
        cont = containers[idx]
        out[cont["top"] : cont["bottom"], cont["left"] + 1 : cont["right"]] = YELLOW

    for col in missed_cols:
        out[-1, col] = YELLOW

    for idx in sorted(activated, key=lambda i: (containers[i]["center"], containers[i]["left"])):
        cont = containers[idx]
        stamp_top = cont["top"] - sprite.shape[0]
        stamp_left = cont["center"] - sprite.shape[1] // 2
        _stamp_patch(out, sprite, stamp_top, stamp_left)

    return out


def _draw_container(grid: np.ndarray, top: int, left: int, height: int, width: int) -> dict:
    grid[top : top + height, left] = RED
    grid[top : top + height, left + width - 1] = RED
    grid[top + height - 1, left : left + width] = RED
    return {
        "top": top,
        "bottom": top + height - 1,
        "left": left,
        "right": left + width - 1,
        "height": height,
        "width": width,
        "center": left + width // 2,
    }


def _non_adjacent_available(col: int, chosen: set[int]) -> bool:
    return all(abs(col - other) > 1 for other in chosen)


def _build_expected(inp: np.ndarray, sprite_bbox: tuple[int, int, int, int], sprite: np.ndarray, containers: list[dict], marker_cols: list[int]) -> np.ndarray:
    out = inp.copy()
    out[0, marker_cols] = BG
    sr0, sc0, sr1, sc1 = sprite_bbox
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG

    active: set[int] = set()
    misses: list[int] = []
    for col in sorted(marker_cols):
        hits = [i for i, cont in enumerate(containers) if cont["left"] <= col <= cont["right"]]
        if hits:
            active.add(hits[0])
        else:
            misses.append(col)

    for idx in active:
        cont = containers[idx]
        out[cont["top"] : cont["bottom"], cont["left"] + 1 : cont["right"]] = YELLOW
    for col in misses:
        out[-1, col] = YELLOW
    for idx in sorted(active, key=lambda i: (containers[i]["center"], containers[i]["left"])):
        cont = containers[idx]
        _stamp_patch(out, sprite, cont["top"] - 3, cont["center"] - 1)
    return out


def _wrong_marker_anchored_output(inp: np.ndarray, sprite_bbox: tuple[int, int, int, int], sprite: np.ndarray, containers: list[dict], marker_cols: list[int]) -> np.ndarray:
    out = inp.copy()
    out[0, marker_cols] = BG
    sr0, sc0, sr1, sc1 = sprite_bbox
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG

    stamped_cols: list[tuple[int, int]] = []
    active: set[int] = set()
    for col in sorted(marker_cols):
        hits = [i for i, cont in enumerate(containers) if cont["left"] <= col <= cont["right"]]
        if hits:
            idx = hits[0]
            if idx not in active:
                active.add(idx)
                out[containers[idx]["top"] : containers[idx]["bottom"], containers[idx]["left"] + 1 : containers[idx]["right"]] = YELLOW
                stamped_cols.append((idx, col))
        else:
            out[-1, col] = YELLOW

    for idx, col in stamped_cols:
        _stamp_patch(out, sprite, containers[idx]["top"] - 3, col - 1)
    return out


def _sample_container_layout(h: int, w: int, count: int, sprite_c1: int) -> list[dict] | None:
    start_min = sprite_c1 + 3
    if start_min >= w - 4:
        return None

    widths = [random.choice(GEN_CONTAINER_WIDTHS) for _ in range(count)]
    heights = [random.choice(GEN_CONTAINER_HEIGHTS) for _ in range(count)]
    total_span = sum(widths) + (count - 1)
    usable = w - start_min
    if total_span > usable:
        return None

    slack = usable - total_span
    extras = [0] * (count + 1)
    for _ in range(slack):
        extras[random.randrange(count + 1)] += 1

    top_candidates = list(range(max(2, h // 2 - 1), h - 5))
    if len(top_candidates) < count:
        return None
    tops = random.sample(top_candidates, count)
    random.shuffle(tops)

    containers = []
    cur_left = start_min + extras[0]
    for i in range(count):
        width = widths[i]
        height = heights[i]
        top = tops[i]
        if top + height > h - 1:
            return None
        containers.append({"top": top, "left": cur_left, "height": height, "width": width})
        cur_left += width
        if i < count - 1:
            cur_left += 1 + extras[i + 1]
    return containers


def _pick_markers(containers: list[dict], w: int, reserved_left: int) -> tuple[list[int], list[int], set[int], bool] | None:
    marker_total = random.randint(*GEN_MARKER_COUNT_RANGE)
    max_active = max(1, len(containers) - 1)
    active_count = random.randint(1, min(max_active, marker_total - 1))
    active_ids = set(random.sample(range(len(containers)), active_count))
    hit_total = random.randint(active_count, marker_total - 1)
    miss_total = marker_total - hit_total
    chosen: set[int] = set()
    hits: list[int] = []
    misses: list[int] = []

    active_list = sorted(active_ids)
    forced_boundary_done = False
    off_center_done = False

    for idx in active_list:
        cont = containers[idx]
        options = [cont["left"], cont["right"], cont["center"]]
        options = [c for c in options if cont["left"] <= c <= cont["right"]]
        options = [c for c in options if _non_adjacent_available(c, chosen)]
        if not options:
            return None
        if not forced_boundary_done:
            boundary = [c for c in options if c in (cont["left"], cont["right"])]
            if boundary:
                col = random.choice(boundary)
                forced_boundary_done = True
                off_center_done = col != cont["center"]
            else:
                col = random.choice(options)
                off_center_done = col != cont["center"]
        else:
            col = random.choice(options)
            off_center_done = off_center_done or (col != cont["center"])
        chosen.add(col)
        hits.append(col)

    for _ in range(hit_total - active_count):
        idx = random.choice(active_list)
        cont = containers[idx]
        options = [c for c in range(cont["left"], cont["right"] + 1) if _non_adjacent_available(c, chosen)]
        if not options:
            return None
        col = random.choice(options)
        chosen.add(col)
        hits.append(col)
        off_center_done = off_center_done or (col != cont["center"])

    span_cells = {c for cont in containers for c in range(cont["left"], cont["right"] + 1)}
    gap_candidates = [
        c
        for c in range(reserved_left, w)
        if c not in span_cells and _non_adjacent_available(c, chosen)
    ]
    if len(gap_candidates) < miss_total:
        return None
    random.shuffle(gap_candidates)
    for col in gap_candidates[:miss_total]:
        chosen.add(col)
        misses.append(col)

    if not forced_boundary_done or not off_center_done:
        return None
    marker_cols = sorted(hits + misses)
    if any(marker_cols[i + 1] - marker_cols[i] <= 1 for i in range(len(marker_cols) - 1)):
        return None
    return marker_cols, hits, active_ids, off_center_done


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, sprite_bbox: tuple[int, int, int, int], sprite: np.ndarray, containers: list[dict], marker_cols: list[int], hit_cols: list[int], active_ids: set[int]) -> bool:
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    if not (2 <= len(marker_cols) <= 4):
        return False
    if any(marker_cols[i + 1] - marker_cols[i] <= 1 for i in range(len(marker_cols) - 1)):
        return False
    if len(containers) not in (2, 3):
        return False
    if len({cont["top"] for cont in containers}) != len(containers):
        return False
    if not active_ids or len(active_ids) >= len(containers):
        return False
    if len(hit_cols) == len(marker_cols):
        return False

    for col in marker_cols:
        hit_count = sum(cont["left"] <= col <= cont["right"] for cont in containers)
        if hit_count > 1:
            return False

    wrong = _wrong_marker_anchored_output(inp, sprite_bbox, sprite, containers, marker_cols)
    if np.array_equal(expected, wrong):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    sprite = random.choice(SPRITE_PATTERNS).copy()
    sprite_r0 = h - 4 - random.choice((0, 1))
    sprite_c0 = random.choice((0, 1))
    grid[sprite_r0 : sprite_r0 + 3, sprite_c0 : sprite_c0 + 3] = sprite
    sprite_bbox = (sprite_r0, sprite_c0, sprite_r0 + 2, sprite_c0 + 2)

    count = random.choice(GEN_CONTAINER_COUNT_CHOICES)
    layout = _sample_container_layout(h, w, count, sprite_bbox[3])
    if layout is None:
        return None

    containers = []
    for spec in layout:
        cont = _draw_container(grid, spec["top"], spec["left"], spec["height"], spec["width"])
        containers.append(cont)

    picked = _pick_markers(containers, w, sprite_bbox[3] + 2)
    if picked is None:
        return None
    marker_cols, hit_cols, active_ids, _ = picked
    grid[0, marker_cols] = YELLOW

    expected = _build_expected(grid, sprite_bbox, sprite, containers, marker_cols)
    if not _validate_semantics(grid, expected, sprite_bbox, sprite, containers, marker_cols, hit_cols, active_ids):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ARC input")
