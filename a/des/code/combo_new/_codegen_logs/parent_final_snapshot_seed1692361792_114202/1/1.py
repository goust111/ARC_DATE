import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == bg or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r1 = max(rows)
            c1 = max(cols)
            r0b = min(rows)
            c0b = min(cols)
            mask = np.zeros((r1 - r0b + 1, c1 - c0b + 1), dtype=bool)
            for r, c in coords:
                mask[r - r0b, c - c0b] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0b, c0b, r1, c1),
                    "area": len(coords),
                    "mask": mask,
                }
            )
    return comps


def _extract_frame_gaps(component: dict) -> list[dict] | None:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 6 or w < 6:
        return None

    occ = set(component["coords"])
    if any((r not in (r0, r1) and c not in (c0, c1)) for r, c in occ):
        return None
    if any(corner not in occ for corner in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]):
        return None

    def scan_side(side: str) -> dict | None:
        if side == "top":
            positions = [(r0, c) for c in range(c0 + 1, c1)]
        elif side == "bottom":
            positions = [(r1, c) for c in range(c0 + 1, c1)]
        elif side == "left":
            positions = [(r, c0) for r in range(r0 + 1, r1)]
        else:
            positions = [(r, c1) for r in range(r0 + 1, r1)]

        missing_idx = [idx for idx, pos in enumerate(positions) if pos not in occ]
        if not missing_idx:
            return None
        if missing_idx != list(range(missing_idx[0], missing_idx[-1] + 1)):
            return None
        length = len(missing_idx)
        if length not in (2, 3):
            return None
        start = missing_idx[0] + 1
        cells = [positions[idx] for idx in missing_idx]
        return {"side": side, "length": length, "start": start, "cells": cells}

    gaps = []
    for side in ("top", "bottom", "left", "right"):
        gap = scan_side(side)
        if gap is None:
            return None
        gaps.append(gap)
    return gaps


def _side_to_base(side: str, start: int, shape: tuple[int, int], frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
    h, w = shape
    if side == "top":
        return 0, start, h, 0
    if side == "bottom":
        return frame_h - h, start, -h, 0
    if side == "left":
        return start, 0, 0, w
    return start, frame_w - w, 0, -w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each outside key into the uniquely matching frame gap and repeat it inward by the count of same-colored interior pixels.

    Input structure:
    - Background is black; one large grey 1-pixel rectangular frame has exactly one 2- or 3-cell gap on each of its four sides.
    - Inside the frame are isolated single-pixel colored markers; each marker color appears 1 to 4 times.
    - Outside the frame are monochrome 4-connected "key" objects whose colors match the interior marker colors.

    Rule:
    - Identify the grey frame and its four side gaps.
    - For each outside key, match it to the unique gap whose direction-compatible gap length equals the key bbox size: top/bottom use bbox width, left/right use bbox height.
    - Count how many interior pixels share the key color, place one translated copy so its matching bbox edge fills the gap, then repeat the same translated key inward that many times with no spacing between copies.

    Key insight:
    - Geometry chooses which side each key belongs to, while color equality with the interior markers independently determines how deep that key chain extends.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    comps = _find_monochrome_components(g, bg)

    color_groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value != bg:
                color_groups.setdefault(value, []).append((r, c))

    frame_data = []
    for color, coords in color_groups.items():
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        comp = {
            "color": color,
            "coords": coords,
            "bbox": (min(rows), min(cols), max(rows), max(cols)),
            "area": len(coords),
        }
        gaps = _extract_frame_gaps(comp)
        if gaps is not None:
            frame_data.append((comp, gaps))
    if not frame_data:
        return g.copy()

    frame_comp, gaps = max(frame_data, key=lambda item: item[0]["area"])
    frame_color = int(frame_comp["color"])
    fr0, fc0, fr1, fc1 = frame_comp["bbox"]
    gap_by_side = {gap["side"]: gap for gap in gaps}

    inside_counts: Counter[int] = Counter()
    for r in range(fr0 + 1, fr1):
        for c in range(fc0 + 1, fc1):
            value = int(g[r, c])
            if value != bg and value != frame_color:
                inside_counts[value] += 1

    out = g.copy()
    used_sides = set()

    for comp in comps:
        if int(comp["color"]) == frame_color:
            continue

        color = int(comp["color"])
        if color not in inside_counts:
            continue

        coords = comp["coords"]
        if all(fr0 < r < fr1 and fc0 < c < fc1 for r, c in coords):
            continue

        ch, cw = comp["mask"].shape
        candidate_sides = []
        for side, gap in gap_by_side.items():
            if side in ("top", "bottom") and cw == gap["length"]:
                candidate_sides.append(side)
            if side in ("left", "right") and ch == gap["length"]:
                candidate_sides.append(side)
        if len(candidate_sides) != 1:
            continue

        side = candidate_sides[0]
        if side in used_sides:
            continue
        used_sides.add(side)

        gap = gap_by_side[side]
        base_r, base_c, step_r, step_c = _side_to_base(side, gap["start"], comp["mask"].shape, fr1 - fr0 + 1, fc1 - fc0 + 1)
        rel_pixels = np.argwhere(comp["mask"])
        repeat_count = int(inside_counts[color])

        for k in range(repeat_count):
            top = fr0 + base_r + k * step_r
            left = fc0 + base_c + k * step_c
            for rr, cc in rel_pixels:
                gr = top + int(rr)
                gc = left + int(cc)
                if 0 <= gr < h and 0 <= gc < w:
                    out[gr, gc] = color

    return out


def _make_horizontal_mask(gap_len: int, side: str, deep: bool, rng: random.Random) -> np.ndarray:
    if not deep:
        mask = np.ones((1, gap_len), dtype=bool)
    else:
        mask = np.zeros((4, gap_len), dtype=bool)
        mask[0, :] = True
        stem_col = rng.randrange(gap_len)
        mask[:, stem_col] = True
        if gap_len > 1:
            if stem_col == 0:
                extra_col = 1
            elif stem_col == gap_len - 1:
                extra_col = gap_len - 2
            else:
                extra_col = rng.choice([stem_col - 1, stem_col + 1])
            mask[2:, extra_col] = True
    if side == "bottom":
        mask = np.flipud(mask)
    return mask


def _make_vertical_mask(gap_len: int, side: str, deep: bool, rng: random.Random) -> np.ndarray:
    if not deep:
        mask = np.ones((gap_len, 1), dtype=bool)
    else:
        mask = np.zeros((gap_len, 4), dtype=bool)
        mask[:, 0] = True
        stem_row = rng.randrange(gap_len)
        mask[stem_row, :] = True
        if gap_len > 1:
            if stem_row == 0:
                extra_row = 1
            elif stem_row == gap_len - 1:
                extra_row = gap_len - 2
            else:
                extra_row = rng.choice([stem_row - 1, stem_row + 1])
            mask[extra_row, 2:] = True
    if side == "right":
        mask = np.fliplr(mask)
    return mask


def _intervals_disjoint(start_a: int, len_a: int, start_b: int, len_b: int) -> bool:
    end_a = start_a + len_a - 1
    end_b = start_b + len_b - 1
    return end_a < start_b or end_b < start_a


def _simulate_chains(frame_h: int, frame_w: int, specs: dict, gap_starts: dict) -> tuple[dict, set] | None:
    gap_cells = {}
    for side, spec in specs.items():
        length = int(spec["gap_len"])
        start = int(gap_starts[side])
        if side == "top":
            gap_cells[side] = {(0, c) for c in range(start, start + length)}
        elif side == "bottom":
            gap_cells[side] = {(frame_h - 1, c) for c in range(start, start + length)}
        elif side == "left":
            gap_cells[side] = {(r, 0) for r in range(start, start + length)}
        else:
            gap_cells[side] = {(r, frame_w - 1) for r in range(start, start + length)}

    chain_by_side = {}
    all_cells = set()
    for side in ("top", "bottom", "left", "right"):
        spec = specs[side]
        mask = spec["mask"]
        count = int(spec["count"])
        base_r, base_c, step_r, step_c = _side_to_base(side, gap_starts[side], mask.shape, frame_h, frame_w)
        rel = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        cells = set()
        for k in range(count):
            top = base_r + k * step_r
            left = base_c + k * step_c
            for rr, cc in rel:
                r = top + rr
                c = left + cc
                if not (0 <= r < frame_h and 0 <= c < frame_w):
                    return None
                if (r in (0, frame_h - 1) or c in (0, frame_w - 1)) and (r, c) not in gap_cells[side]:
                    return None
                if (r, c) in all_cells:
                    return None
                cells.add((r, c))
                all_cells.add((r, c))
        chain_by_side[side] = cells
    return chain_by_side, all_cells


def _sample_gap_starts(frame_h: int, frame_w: int, specs: dict, rng: random.Random) -> tuple[dict, dict, set] | None:
    top_candidates = list(range(1, frame_w - specs["top"]["gap_len"]))
    bottom_candidates = list(range(1, frame_w - specs["bottom"]["gap_len"]))
    left_candidates = list(range(1, frame_h - specs["left"]["gap_len"]))
    right_candidates = list(range(1, frame_h - specs["right"]["gap_len"]))
    rng.shuffle(top_candidates)
    rng.shuffle(bottom_candidates)
    rng.shuffle(left_candidates)
    rng.shuffle(right_candidates)

    for top_start in top_candidates:
        for bottom_start in bottom_candidates:
            if not _intervals_disjoint(top_start, specs["top"]["gap_len"], bottom_start, specs["bottom"]["gap_len"]):
                continue
            for left_start in left_candidates:
                for right_start in right_candidates:
                    if not _intervals_disjoint(left_start, specs["left"]["gap_len"], right_start, specs["right"]["gap_len"]):
                        continue
                    gap_starts = {
                        "top": top_start,
                        "bottom": bottom_start,
                        "left": left_start,
                        "right": right_start,
                    }
                    simulated = _simulate_chains(frame_h, frame_w, specs, gap_starts)
                    if simulated is not None:
                        chain_by_side, all_cells = simulated
                        return gap_starts, chain_by_side, all_cells
    return None


def _sample_markers(frame_h: int, frame_w: int, specs: dict, chain_cells: set, rng: random.Random) -> dict | None:
    chain_neighbors = set(chain_cells)
    for r, c in list(chain_cells):
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 < rr < frame_h - 1 and 0 < cc < frame_w - 1:
                chain_neighbors.add((rr, cc))

    pool = [
        (r, c)
        for r in range(1, frame_h - 1)
        for c in range(1, frame_w - 1)
        if (r, c) not in chain_neighbors
    ]
    chosen = []
    result = {}
    side_order = sorted(specs, key=lambda side: -specs[side]["count"])
    for side in side_order:
        color = int(specs[side]["color"])
        result[color] = []
        for _ in range(int(specs[side]["count"])):
            candidates = [
                cell
                for cell in pool
                if cell not in chosen
                and all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in chosen)
            ]
            if not candidates:
                return None
            cell = rng.choice(candidates)
            chosen.append(cell)
            result[color].append(cell)
    return result


def _draw_frame(grid: np.ndarray, fr0: int, fc0: int, frame_h: int, frame_w: int, gap_starts: dict, specs: dict) -> None:
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME

    for c in range(fc0 + gap_starts["top"], fc0 + gap_starts["top"] + specs["top"]["gap_len"]):
        grid[fr0, c] = BG
    for c in range(fc0 + gap_starts["bottom"], fc0 + gap_starts["bottom"] + specs["bottom"]["gap_len"]):
        grid[fr1, c] = BG
    for r in range(fr0 + gap_starts["left"], fr0 + gap_starts["left"] + specs["left"]["gap_len"]):
        grid[r, fc0] = BG
    for r in range(fr0 + gap_starts["right"], fr0 + gap_starts["right"] + specs["right"]["gap_len"]):
        grid[r, fc1] = BG


def _can_place_key(grid: np.ndarray, mask: np.ndarray, top: int, left: int, frame_bbox: tuple[int, int, int, int]) -> bool:
    h, w = mask.shape
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False

    fr0, fc0, fr1, fc1 = frame_bbox
    if not (top + h - 1 < fr0 or top > fr1 or left + w - 1 < fc0 or left > fc1):
        return False

    coords = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
    coord_set = set(coords)
    for r, c in coords:
        if int(grid[r, c]) != BG:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < gh and 0 <= cc < gw and (rr, cc) not in coord_set and int(grid[rr, cc]) != BG:
                    return False
    return True


def _place_keys(grid: np.ndarray, specs: dict, frame_bbox: tuple[int, int, int, int], rng: random.Random) -> bool:
    items = sorted(specs.values(), key=lambda spec: -int(np.sum(spec["mask"])))
    gh, gw = grid.shape
    for spec in items:
        mask = spec["mask"]
        color = int(spec["color"])
        h, w = mask.shape
        positions = [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]
        rng.shuffle(positions)
        placed = False
        for top, left in positions:
            if not _can_place_key(grid, mask, top, left, frame_bbox):
                continue
            for rr, cc in np.argwhere(mask):
                grid[top + int(rr), left + int(cc)] = color
            placed = True
            break
        if not placed:
            return False
    return True


def _build_expected_output(inp: np.ndarray, specs: dict, chain_by_side: dict, fr0: int, fc0: int) -> np.ndarray:
    out = np.array(inp, copy=True)
    for side, cells in chain_by_side.items():
        color = int(specs[side]["color"])
        for r, c in cells:
            out[fr0 + r, fc0 + c] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    colors = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)

    h_lens = [2, 3]
    v_lens = [2, 3]
    rng.shuffle(h_lens)
    rng.shuffle(v_lens)

    deep_flags = {side: rng.choice([False, True]) for side in ("top", "bottom", "left", "right")}
    if not any(deep_flags.values()):
        deep_flags[rng.choice(["top", "bottom", "left", "right"])] = True

    counts = {side: rng.randint(1, 4) for side in ("top", "bottom", "left", "right")}
    if max(counts.values()) == 1:
        counts[rng.choice(["top", "bottom", "left", "right"])] = rng.randint(2, 4)

    specs = {
        "top": {
            "side": "top",
            "gap_len": h_lens[0],
            "color": colors[0],
            "count": counts["top"],
            "mask": _make_horizontal_mask(h_lens[0], "top", deep_flags["top"], rng),
        },
        "bottom": {
            "side": "bottom",
            "gap_len": h_lens[1],
            "color": colors[1],
            "count": counts["bottom"],
            "mask": _make_horizontal_mask(h_lens[1], "bottom", deep_flags["bottom"], rng),
        },
        "left": {
            "side": "left",
            "gap_len": v_lens[0],
            "color": colors[2],
            "count": counts["left"],
            "mask": _make_vertical_mask(v_lens[0], "left", deep_flags["left"], rng),
        },
        "right": {
            "side": "right",
            "gap_len": v_lens[1],
            "color": colors[3],
            "count": counts["right"],
            "mask": _make_vertical_mask(v_lens[1], "right", deep_flags["right"], rng),
        },
    }

    min_frame_h = max(
        specs["top"]["count"] * specs["top"]["mask"].shape[0],
        specs["bottom"]["count"] * specs["bottom"]["mask"].shape[0],
        8,
    ) + 1
    min_frame_w = max(
        specs["left"]["count"] * specs["left"]["mask"].shape[1],
        specs["right"]["count"] * specs["right"]["mask"].shape[1],
        8,
    ) + 1

    max_frame_h = 22
    max_frame_w = 22
    if min_frame_h > max_frame_h or min_frame_w > max_frame_w:
        return None

    frame_h = rng.randint(min_frame_h, max_frame_h)
    frame_w = rng.randint(min_frame_w, max_frame_w)

    gap_sample = _sample_gap_starts(frame_h, frame_w, specs, rng)
    if gap_sample is None:
        return None
    gap_starts, chain_by_side, chain_cells = gap_sample

    markers = _sample_markers(frame_h, frame_w, specs, chain_cells, rng)
    if markers is None:
        return None

    min_grid_h = max(frame_h + 7, 24)
    min_grid_w = max(frame_w + 7, 24)
    if min_grid_h > 30 or min_grid_w > 30:
        return None

    gh = rng.randint(min_grid_h, 30)
    gw = rng.randint(min_grid_w, 30)
    fr0 = rng.randint(2, gh - frame_h - 2)
    fc0 = rng.randint(2, gw - frame_w - 2)
    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    inp = np.full((gh, gw), BG, dtype=int)
    _draw_frame(inp, fr0, fc0, frame_h, frame_w, gap_starts, specs)

    for color, cells in markers.items():
        for r, c in cells:
            inp[fr0 + r, fc0 + c] = int(color)

    if not _place_keys(inp, specs, frame_bbox, rng):
        return None

    expected = _build_expected_output(inp, specs, chain_by_side, fr0, fc0)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 500 attempts")
