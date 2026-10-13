from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TEMP_CANVAS = 60
MAX_GEN_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4.values():
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    comps = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components_mask(grid == color):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _detect_cups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    cups = []
    for comp in _monochrome_components(g, bg=0):
        if comp["area"] < 5:
            continue
        border_color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        bbox = (r0, c0, r1, c1)
        border_set = {tuple(map(int, rc)) for rc in comp["coords"]}
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        edges = {
            "U": [p for p in border_set if p[0] == r0],
            "D": [p for p in border_set if p[0] == r1],
            "L": [p for p in border_set if p[1] == c0],
            "R": [p for p in border_set if p[1] == c1],
        }
        candidates = [
            (direction, cells[0])
            for direction, cells in edges.items()
            if len(cells) == 1 and cells[0] not in corners
        ]
        passing = []
        for direction, spike in candidates:
            mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[direction])
            color_counts = Counter(
                int(g[r, c])
                for r, c in mouth
                if int(g[r, c]) not in (0, border_color)
            )
            if len(color_counts) != 1:
                continue
            content_color = int(next(iter(color_counts)))
            content_cells = sorted((r, c) for r, c in mouth if int(g[r, c]) == content_color)
            if not content_cells:
                continue
            passing.append(
                {
                    "border_color": border_color,
                    "tail_dir": direction,
                    "spike": tuple(map(int, spike)),
                    "content_color": content_color,
                    "content_cells": content_cells,
                    "bbox": bbox,
                }
            )
        if len(passing) == 1:
            cups.append(passing[0])
    cups.sort(key=lambda cup: (cup["bbox"][0], cup["bbox"][1], cup["border_color"]))
    assert cups, "expected at least one cup-shaped frame"
    return cups


def _apply_ejections(grid: np.ndarray, cups: list[dict]) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    segments = []
    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = 0
    for cup in cups:
        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        cells = []
        for step in range(1, len(cup["content_cells"]) + 1):
            rr = sr + dr * step
            cc = sc + dc * step
            if not _in_bounds(out.shape, rr, cc):
                break
            out[rr, cc] = cup["content_color"]
            cells.append((rr, cc))
        segments.append(
            {
                "color": int(cup["content_color"]),
                "cells": cells,
                "tail_dir": cup["tail_dir"],
                "spike": cup["spike"],
            }
        )
    return out, segments


def _pick_closer_endpoint(
    cells: list[tuple[int, int]],
    other_cells: list[tuple[int, int]],
) -> tuple[int, int]:
    assert cells and other_cells
    ends = [cells[0], cells[-1]]
    scores = [
        min(abs(r - rr) + abs(c - cc) for rr, cc in other_cells)
        for r, c in ends
    ]
    if scores[1] < scores[0]:
        return ends[1]
    if scores[0] < scores[1]:
        return ends[0]
    return ends[-1]


def _extend_toward(
    out: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int],
    color: int,
) -> list[tuple[int, int]]:
    sr, sc = start
    tr, tc = target
    assert sr == tr or sc == tc
    dr = 0 if sr == tr else (1 if tr > sr else -1)
    dc = 0 if sc == tc else (1 if tc > sc else -1)
    added = []
    r, c = sr, sc
    while True:
        nr, nc = r + dr, c + dc
        if (nr, nc) == (tr, tc):
            break
        if not _in_bounds(out.shape, nr, nc) or int(out[nr, nc]) != 0:
            break
        out[nr, nc] = int(color)
        added.append((nr, nc))
        r, c = nr, nc
    return added


def _connect_pair(out: np.ndarray, seg_a: dict, seg_b: dict) -> dict | None:
    if not seg_a["cells"] or not seg_b["cells"]:
        return None
    end_a = _pick_closer_endpoint(seg_a["cells"], seg_b["cells"])
    end_b = _pick_closer_endpoint(seg_b["cells"], seg_a["cells"])
    if end_a[0] != end_b[0] and end_a[1] != end_b[1]:
        return {
            "endpoints": (end_a, end_b),
            "added": [],
            "aligned": False,
        }
    color = int(seg_a["color"])
    added = []
    added.extend(_extend_toward(out, end_a, end_b, color))
    added.extend(_extend_toward(out, end_b, end_a, color))
    return {
        "endpoints": (end_a, end_b),
        "added": added,
        "aligned": True,
    }


def _solve_with_details(grid: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray, list[dict], dict[int, dict | None]]:
    g = np.asarray(grid, dtype=int)
    cups = _detect_cups(g)
    ejected, segments = _apply_ejections(g, cups)
    out = ejected.copy()
    by_color: dict[int, list[dict]] = defaultdict(list)
    for seg in segments:
        by_color[int(seg["color"])].append(seg)
    connections: dict[int, dict | None] = {}
    for color in sorted(by_color):
        if len(by_color[color]) == 2:
            connections[color] = _connect_pair(out, by_color[color][0], by_color[color][1])
        else:
            connections[color] = None
    return out, cups, ejected, segments, connections


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject cup contents along each tail and then connect aligned same-color ejection endpoints until blocked.

    Input structure:
    - The grid has black background with 2-3 hollow cup-shaped border components in distinct frame colors.
    - Each cup is open on one side and has a 1-2 pixel tail spike on the opposite side.
    - Inside each cup are 2-4 content pixels of one non-border color; exactly one content color appears in two cups.

    Rule:
    - Detect every cup's tail direction from its unique spike edge, then erase that cup's interior content pixels.
    - From the spike tip, emit a straight segment in the tail direction whose length equals the erased content count.
    - Keep the cup frame unchanged.
    - After all ejections, if a content color has exactly two emitted segments, choose on each segment the endpoint closer to the other segment.
    - If those chosen endpoints share a row or a column, extend that color from each endpoint toward the other until the next cell is non-black or the other endpoint is reached.

    Key insight:
    - The tail sets the emission direction, the content count sets each segment's endpoint, and only those resulting endpoints determine whether and how the final connection can be drawn before another foreground object blocks it.
    """
    out, _cups, _ejected, _segments, _connections = _solve_with_details(grid)
    return out


def _pick_content_cells(
    interior: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    for _ in range(40):
        sample = random.sample(interior, count)
        rows = {r for r, _ in sample}
        cols = {c for _, c in sample}
        if len(rows) > 1 and len(cols) > 1:
            return sample
    return random.sample(interior, count)


def _rotate_patch_and_cells(
    patch: np.ndarray,
    spike: tuple[int, int],
    content_cells: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(content_cells)
    if direction == "D":
        return (
            np.flipud(patch).copy(),
            (h - 1 - sr, sc),
            [(h - 1 - r, c) for r, c in content_cells],
        )
    if direction == "L":
        return (
            np.rot90(patch, k=1).copy(),
            (w - 1 - sc, sr),
            [(w - 1 - c, r) for r, c in content_cells],
        )
    if direction == "R":
        return (
            np.rot90(patch, k=-1).copy(),
            (sc, h - 1 - sr),
            [(c, h - 1 - r) for r, c in content_cells],
        )
    raise ValueError(direction)


def _build_cup_patch(
    direction: str,
    border_color: int,
    content_color: int,
    content_count: int,
) -> dict:
    inner_width = random.choice([3, 5])
    inner_depth = random.choice([2, 3])
    tail_len = random.choice([1, 2])
    width = inner_width + 2
    height = tail_len + inner_depth + 1
    patch = np.zeros((height, width), dtype=int)
    center = width // 2
    patch[:tail_len, center] = border_color
    patch[tail_len, :] = border_color
    patch[tail_len + 1 :, 0] = border_color
    patch[tail_len + 1 :, -1] = border_color
    interior = [
        (r, c)
        for r in range(tail_len + 1, height)
        for c in range(1, width - 1)
    ]
    content_cells = _pick_content_cells(interior, content_count)
    for r, c in content_cells:
        patch[r, c] = content_color
    patch, spike, content_cells = _rotate_patch_and_cells(
        patch,
        (0, center),
        content_cells,
        direction,
    )
    return {
        "patch": patch,
        "spike_local": tuple(map(int, spike)),
        "content_local": [tuple(map(int, cell)) for cell in content_cells],
        "count": int(content_count),
        "direction": direction,
        "border_color": int(border_color),
        "content_color": int(content_color),
    }


def _place_pair(axis: str, cup_a: dict, cup_b: dict, gap: int) -> tuple[dict, dict, dict]:
    if axis == "H":
        common_row = max(cup_a["spike_local"][0], cup_b["spike_local"][0])
        row_a = common_row - cup_a["spike_local"][0]
        row_b = common_row - cup_b["spike_local"][0]
        col_a = 0
        spike_a_col = col_a + cup_a["spike_local"][1]
        tip_a_col = spike_a_col + cup_a["count"]
        tip_b_col = tip_a_col + gap + 1
        spike_b_col = tip_b_col + cup_b["count"]
        col_b = spike_b_col - cup_b["spike_local"][1]
        meta = {
            "axis": axis,
            "line_value": common_row,
            "tip_a": (common_row, tip_a_col),
            "tip_b": (common_row, tip_b_col),
        }
    else:
        common_col = max(cup_a["spike_local"][1], cup_b["spike_local"][1])
        col_a = common_col - cup_a["spike_local"][1]
        col_b = common_col - cup_b["spike_local"][1]
        row_a = 0
        spike_a_row = row_a + cup_a["spike_local"][0]
        tip_a_row = spike_a_row + cup_a["count"]
        tip_b_row = tip_a_row + gap + 1
        spike_b_row = tip_b_row + cup_b["count"]
        row_b = spike_b_row - cup_b["spike_local"][0]
        meta = {
            "axis": axis,
            "line_value": common_col,
            "tip_a": (tip_a_row, common_col),
            "tip_b": (tip_b_row, common_col),
        }
    cup_a = {**cup_a, "top": int(row_a), "left": int(col_a), "label": "pair_a"}
    cup_b = {**cup_b, "top": int(row_b), "left": int(col_b), "label": "pair_b"}
    return cup_a, cup_b, meta


def _pair_bbox(cups: list[dict]) -> tuple[int, int, int, int]:
    boxes = []
    for cup in cups:
        h, w = cup["patch"].shape
        boxes.append((cup["top"], cup["left"], cup["top"] + h - 1, cup["left"] + w - 1))
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _make_blocker(axis: str, meta: dict, content_color: int, border_color: int) -> dict:
    if axis == "H":
        direction = random.choice(["D", "U"])
        count = random.randint(2, 4)
        blocker = _build_cup_patch(direction, border_color, content_color, count)
        left_tip = meta["tip_a"][1]
        right_tip = meta["tip_b"][1]
        x = random.randint(left_tip + 2, right_tip - 2)
        k = random.randint(1, count)
        if direction == "D":
            spike_row = meta["line_value"] - k
        else:
            spike_row = meta["line_value"] + k
        top = spike_row - blocker["spike_local"][0]
        left = x - blocker["spike_local"][1]
    else:
        direction = random.choice(["R", "L"])
        count = random.randint(2, 4)
        blocker = _build_cup_patch(direction, border_color, content_color, count)
        top_tip = meta["tip_a"][0]
        bottom_tip = meta["tip_b"][0]
        y = random.randint(top_tip + 2, bottom_tip - 2)
        k = random.randint(1, count)
        if direction == "R":
            spike_col = meta["line_value"] - k
        else:
            spike_col = meta["line_value"] + k
        top = y - blocker["spike_local"][0]
        left = spike_col - blocker["spike_local"][1]
    return {**blocker, "top": int(top), "left": int(left), "label": "blocker"}


def _make_decoy(axis: str, pair_cups: list[dict], content_color: int, border_color: int) -> dict:
    bbox = _pair_bbox(pair_cups)
    if axis == "H":
        direction = "D"
        decoy = _build_cup_patch(direction, border_color, content_color, random.randint(2, 4))
        top = bbox[2] + 2
        left = random.randint(bbox[1], bbox[3] - decoy["patch"].shape[1] + 1)
    else:
        direction = "R"
        decoy = _build_cup_patch(direction, border_color, content_color, random.randint(2, 4))
        top = random.randint(bbox[0], bbox[2] - decoy["patch"].shape[0] + 1)
        left = bbox[3] + 2
    return {**decoy, "top": int(top), "left": int(left), "label": "decoy"}


def _shift_cups(cups: list[dict], dr: int, dc: int) -> list[dict]:
    return [{**cup, "top": cup["top"] + dr, "left": cup["left"] + dc} for cup in cups]


def _render_cups(cups: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    for cup in cups:
        patch = cup["patch"]
        top = int(cup["top"])
        left = int(cup["left"])
        h, w = patch.shape
        if top < 0 or left < 0 or top + h > shape[0] or left + w > shape[1]:
            return None
        view = grid[top : top + h, left : left + w]
        patch_mask = patch != 0
        if np.any(view[patch_mask] != 0):
            return None
        view[patch_mask] = patch[patch_mask]
    return grid


def _nonzero_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    return _bbox_coords(coords)


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    r0, c0, r1, c1 = _nonzero_bbox((inp != 0) | (out != 0))
    pads = [1, 2, 3]
    random.shuffle(pads)
    for pad in pads:
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(inp.shape[0] - 1, r1 + pad)
        cc1 = min(inp.shape[1] - 1, c1 + pad)
        cropped_inp = inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        cropped_out = out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        if max(cropped_inp.shape) <= 30 and max(cropped_out.shape) <= 30:
            return cropped_inp, cropped_out
    return None


def _global_content_cells(cup: dict) -> list[tuple[int, int]]:
    return [(cup["top"] + r, cup["left"] + c) for r, c in cup["content_local"]]


def _validate_generated(
    inp: np.ndarray,
    out: np.ndarray,
    mode: str,
    pair_color: int,
    pair_cups: list[dict],
) -> bool:
    try:
        solved, cups, ejected, segments, connections = _solve_with_details(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False
    if np.array_equal(inp, out):
        return False
    cup_color_counts = Counter(int(cup["content_color"]) for cup in cups)
    repeated = sorted(count for count in cup_color_counts.values() if count > 1)
    if repeated != [2]:
        return False
    pair_segments = [seg for seg in segments if int(seg["color"]) == int(pair_color)]
    if len(pair_segments) != 2:
        return False
    planned_counts = sorted(cup["count"] for cup in pair_cups)
    actual_counts = sorted(len(seg["cells"]) for seg in pair_segments)
    if actual_counts != planned_counts:
        return False
    connection = connections.get(int(pair_color))
    if connection is None or not connection["aligned"]:
        return False
    if len(connection["added"]) == 0:
        return False
    end_a, end_b = connection["endpoints"]
    total_gap = abs(end_a[0] - end_b[0]) + abs(end_a[1] - end_b[1]) - 1
    if total_gap < 1:
        return False
    if mode == "blocked":
        if len(connection["added"]) >= total_gap:
            return False
        if end_a[0] == end_b[0]:
            line = [tuple([end_a[0], c]) for c in range(min(end_a[1], end_b[1]) + 1, max(end_a[1], end_b[1]))]
        else:
            line = [tuple([r, end_a[1]]) for r in range(min(end_a[0], end_b[0]) + 1, max(end_a[0], end_b[0]))]
        if not any(int(ejected[r, c]) != 0 for r, c in line):
            return False
    else:
        if len(connection["added"]) != total_gap:
            return False
    if np.array_equal(ejected, out):
        return False

    alt = inp.copy()
    ar, ac = _global_content_cells(pair_cups[0])[0]
    alt[ar, ac] = 0
    try:
        alt_out = main(alt)
    except Exception:
        return False
    if np.array_equal(alt_out, out):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    axis = random.choice(["H", "V"])
    mode = random.choices(
        ["blocked", "decoy", "pair_only"],
        weights=[45, 25, 30],
        k=1,
    )[0]
    colors = random.sample(PALETTE, 5 if mode != "pair_only" else 3)
    pair_color = colors[0]
    border_a = colors[1]
    border_b = colors[2]
    extra_content = colors[3] if mode != "pair_only" else None
    extra_border = colors[4] if mode != "pair_only" else None

    gap = random.randint(4, 7) if mode == "blocked" else random.randint(2, 6)
    if axis == "H":
        cup_a = _build_cup_patch("R", border_a, pair_color, random.randint(3, 4))
        cup_b = _build_cup_patch("L", border_b, pair_color, random.randint(3, 4))
    else:
        cup_a = _build_cup_patch("D", border_a, pair_color, random.randint(3, 4))
        cup_b = _build_cup_patch("U", border_b, pair_color, random.randint(3, 4))
    cup_a, cup_b, meta = _place_pair(axis, cup_a, cup_b, gap)
    cups = [cup_a, cup_b]
    if mode == "blocked":
        cups.append(_make_blocker(axis, meta, int(extra_content), int(extra_border)))
    elif mode == "decoy":
        cups.append(_make_decoy(axis, cups, int(extra_content), int(extra_border)))

    min_row = min(cup["top"] for cup in cups)
    min_col = min(cup["left"] for cup in cups)
    cups = _shift_cups(cups, 20 - min_row, 20 - min_col)
    inp = _render_cups(cups, (TEMP_CANVAS, TEMP_CANVAS))
    if inp is None:
        return None
    out = main(inp)
    cropped = _crop_pair(inp, out)
    if cropped is None:
        return None
    cropped_inp, cropped_out = cropped

    solved_again = main(cropped_inp)
    if not np.array_equal(solved_again, cropped_out):
        return None

    pair_cups = [cup for cup in cups if cup["label"] in {"pair_a", "pair_b"}]
    crop_r0, crop_c0, _crop_r1, _crop_c1 = _nonzero_bbox((inp != 0) | (out != 0))
    for pad in range(0, 4):
        if crop_r0 - pad >= 0 and crop_c0 - pad >= 0:
            offset_r = crop_r0 - pad
            offset_c = crop_c0 - pad
            break
    else:
        offset_r = crop_r0
        offset_c = crop_c0
    pair_cups = _shift_cups(pair_cups, -offset_r, -offset_c)

    if not _validate_generated(cropped_inp, cropped_out, mode, pair_color, pair_cups):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
