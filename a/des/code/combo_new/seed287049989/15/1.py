import random
from collections import Counter

import numpy as np


BG = 0
FRAME_COLOR = 5
FRAME_OUTER = 6
FRAME_INNER = 4
GRID_MIN = 12
GRID_MAX = 22
GEN_ATTEMPTS = 800
SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _gap_runs(line: np.ndarray, color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, val in enumerate(line):
        if int(val) != int(color):
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(line) - start))
    return runs


def _find_frame(grid: np.ndarray, bg: int) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    colors = [int(c) for c in np.unique(grid) if int(c) != int(bg)]
    for color in colors:
        cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))]
        if not cells:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 6:
            continue
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in cells):
            continue
        top = grid[r0, c0 + 1 : c1]
        bottom = grid[r1, c0 + 1 : c1]
        left = grid[r0 + 1 : r1, c0]
        right = grid[r0 + 1 : r1, c1]
        side_lines = {"top": top, "bottom": bottom, "left": left, "right": right}
        gaps: dict[str, dict[str, int]] = {}
        ok = True
        for side, line in side_lines.items():
            if not np.all((line == int(color)) | (line == int(bg))):
                ok = False
                break
            runs = _gap_runs(line, color)
            if len(runs) != 1:
                ok = False
                break
            start, width = runs[0]
            if not (2 <= int(width) <= 4):
                ok = False
                break
            if side in {"top", "bottom"}:
                gaps[side] = {"start": c0 + 1 + int(start), "width": int(width)}
            else:
                gaps[side] = {"start": r0 + 1 + int(start), "width": int(width)}
        if not ok:
            continue
        candidates.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "gaps": gaps,
                "area": (r1 - r0 + 1) * (c1 - c0 + 1),
            }
        )
    if not candidates:
        raise AssertionError("frame not found")
    candidates.sort(key=lambda item: int(item["area"]), reverse=True)
    return candidates[0]


def _find_seed(grid: np.ndarray, frame: dict[str, object], bg: int) -> dict[str, object]:
    r0, c0, r1, c1 = frame["bbox"]
    fc = int(frame["color"])
    inner = np.zeros_like(grid, dtype=bool)
    inner[r0 + 1 : r1, c0 + 1 : c1] = True
    mask = inner & (grid != int(bg)) & (grid != fc)
    comps = _components(mask)
    for comp in comps:
        br0, bc0, br1, bc1 = _bbox(comp)
        color = int(grid[comp[0]])
        if br0 == br1 and 2 <= bc1 - bc0 + 1 <= 3:
            return {"cells": comp, "color": color, "orient": "H"}
        if bc0 == bc1 and 2 <= br1 - br0 + 1 <= 3:
            return {"cells": comp, "color": color, "orient": "V"}
    raise AssertionError("seed strip not found")


def _find_fragments(
    grid: np.ndarray, frame: dict[str, object], seed: dict[str, object], bg: int
) -> list[dict[str, object]]:
    fc = int(frame["color"])
    mask = (grid != int(bg)) & (grid != fc)
    for r, c in seed["cells"]:
        mask[r, c] = False
    frags: list[dict[str, object]] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        color = int(grid[comp[0]])
        if np.any(patch != int(color)):
            raise AssertionError("fragment must be a solid rectangle")
        ph, pw = patch.shape
        frags.append(
            {
                "cells": comp,
                "bbox": (r0, c0, r1, c1),
                "patch": patch.copy(),
                "color": color,
                "span": max(int(ph), int(pw)),
            }
        )
    if len(frags) != 4:
        raise AssertionError("expected four outside fragments")
    return frags


def _active_sides(orient: str) -> tuple[str, str]:
    return ("left", "right") if orient == "H" else ("top", "bottom")


def _target_origin(
    side: str, frame: dict[str, object], gap: dict[str, int], patch: np.ndarray
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    ph, pw = map(int, patch.shape)
    if side == "top":
        return r0, int(gap["start"])
    if side == "bottom":
        return r1 - ph + 1, int(gap["start"])
    if side == "left":
        return int(gap["start"]), c0
    if side == "right":
        return int(gap["start"]), c1 - pw + 1
    raise ValueError(side)


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the gap-matching outer fragments from the seed strip's orientation, dock them into the frame, and fill the enclosed seed-side cavity.

    Input structure:
    - Background is black.
    - One grey hollow rectangular frame sits in the grid; each of its four sides contains exactly one contiguous gap.
    - Inside the frame is a short non-grey seed strip, either horizontal or vertical.
    - Four monochrome rectangular fragments lie outside the frame, one per side-width class.

    Rule:
    - Read the seed strip orientation: a horizontal seed activates the left and right gaps, while a vertical seed activates the top and bottom gaps.
    - Among the four outside fragments, find the ones whose span exactly matches the active gap widths, erase all outside fragments, and translate only the active ones into their matching gaps.
    - After docking, flood-fill every black 4-connected component that touches the seed strip and is fully enclosed from the outer grid, using the seed strip color.

    Key insight:
    - The seed strip first chooses which pair of gaps matters, then those matched fragments change the frame topology, and only the resulting enclosed seed-adjacent background gets filled.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame = _find_frame(g, bg)
    seed = _find_seed(g, frame, bg)
    fragments = _find_fragments(g, frame, seed, bg)

    out = g.copy()
    for frag in fragments:
        for r, c in frag["cells"]:
            out[r, c] = int(bg)

    active = _active_sides(str(seed["orient"]))
    remaining = fragments[:]
    placements: dict[str, dict[str, object]] = {}
    for side in active:
        need = int(frame["gaps"][side]["width"])
        matches = [frag for frag in remaining if int(frag["span"]) == need]
        if len(matches) != 1:
            raise AssertionError("active fragment match must be unique")
        placements[side] = matches[0]
        remaining.remove(matches[0])

    for side, frag in placements.items():
        patch = np.asarray(frag["patch"], dtype=int)
        target_r, target_c = _target_origin(side, frame, frame["gaps"][side], patch)
        ph, pw = patch.shape
        out[target_r : target_r + ph, target_c : target_c + pw] = patch

    seed_cells = [tuple(map(int, cell)) for cell in seed["cells"]]
    fill_color = int(seed["color"])
    h, w = out.shape
    for comp in _components(out == int(bg)):
        if _touches_border(comp, h, w):
            continue
        comp_set = set(comp)
        if not any(any((nr, nc) in comp_set for nr, nc in _neighbors4(r, c)) for r, c in seed_cells):
            continue
        for r, c in comp:
            out[r, c] = fill_color
    return out


def _draw_frame_patch(gap_specs: dict[str, tuple[int, int]], seed_spec: dict[str, int], seed_color: int) -> np.ndarray:
    patch = np.zeros((FRAME_OUTER, FRAME_OUTER), dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR

    top_start, top_w = gap_specs["top"]
    bottom_start, bottom_w = gap_specs["bottom"]
    left_start, left_w = gap_specs["left"]
    right_start, right_w = gap_specs["right"]
    patch[0, top_start : top_start + top_w] = BG
    patch[-1, bottom_start : bottom_start + bottom_w] = BG
    patch[left_start : left_start + left_w, 0] = BG
    patch[right_start : right_start + right_w, -1] = BG

    if seed_spec["orient"] == "H":
        row = int(seed_spec["row"])
        col = int(seed_spec["col"])
        length = int(seed_spec["length"])
        patch[row, col : col + length] = int(seed_color)
    else:
        row = int(seed_spec["row"])
        col = int(seed_spec["col"])
        length = int(seed_spec["length"])
        patch[row : row + length, col] = int(seed_color)
    return patch


def _fragment_patch(side: str, span: int, thickness: int, color: int) -> np.ndarray:
    if side in {"left", "right"}:
        return np.full((int(span), int(thickness)), int(color), dtype=int)
    return np.full((int(thickness), int(span)), int(color), dtype=int)


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, int(r0) - 1)
    cc0 = max(0, int(c0) - 1)
    rr1 = min(blocked.shape[0], int(r0) + int(h) + 1)
    cc1 = min(blocked.shape[1], int(c0) + int(w) + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_patches(
    rng: random.Random, grid: np.ndarray, frame_bbox: tuple[int, int, int, int], patches: dict[str, np.ndarray]
) -> bool:
    blocked = np.zeros_like(grid, dtype=bool)
    r0, c0, r1, c1 = frame_bbox
    _mark_blocked(blocked, r0, c0, r1 - r0 + 1, c1 - c0 + 1)

    items = list(patches.items())
    items.sort(key=lambda item: int(item[1].shape[0] * item[1].shape[1]), reverse=True)
    for _side, patch in items:
        ph, pw = map(int, patch.shape)
        positions: list[tuple[int, int]] = []
        for rr in range(grid.shape[0] - ph + 1):
            for cc in range(grid.shape[1] - pw + 1):
                if bool(blocked[rr : rr + ph, cc : cc + pw].any()):
                    continue
                positions.append((rr, cc))
        if not positions:
            return False
        rr, cc = rng.choice(positions)
        grid[rr : rr + ph, cc : cc + pw] = patch
        _mark_blocked(blocked, rr, cc, ph, pw)
    return True


def _sample_local_specs(rng: random.Random) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
    orient = rng.choice(["H", "V"])
    if orient == "H":
        active_left = rng.choice([2, 4])
        active_right = 6 - active_left
        seed_len = rng.choice([2, 3])
        seed_row = rng.choice([1, 4])
        seed_col = rng.randint(1, FRAME_INNER - seed_len + 1)
        gaps = {
            "top": (rng.choice([1, 2]), 3),
            "bottom": (rng.choice([1, 2]), 3),
            "left": (rng.randint(1, FRAME_INNER - active_left + 1), active_left),
            "right": (rng.randint(1, FRAME_INNER - active_right + 1), active_right),
        }
        seed = {"orient": "H", "row": seed_row, "col": seed_col, "length": seed_len}
        return gaps, seed

    active_top = rng.choice([2, 4])
    active_bottom = 6 - active_top
    seed_len = rng.choice([2, 3])
    seed_col = rng.choice([1, 4])
    seed_row = rng.randint(1, FRAME_INNER - seed_len + 1)
    gaps = {
        "top": (rng.randint(1, FRAME_INNER - active_top + 1), active_top),
        "bottom": (rng.randint(1, FRAME_INNER - active_bottom + 1), active_bottom),
        "left": (rng.choice([1, 2]), 3),
        "right": (rng.choice([1, 2]), 3),
    }
    seed = {"orient": "V", "row": seed_row, "col": seed_col, "length": seed_len}
    return gaps, seed


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    if height < FRAME_OUTER + 4 or width < FRAME_OUTER + 4:
        return None

    grid = np.zeros((height, width), dtype=int)
    frame_r = rng.randint(2, height - FRAME_OUTER - 2)
    frame_c = rng.randint(2, width - FRAME_OUTER - 2)
    frame_bbox = (frame_r, frame_c, frame_r + FRAME_OUTER - 1, frame_c + FRAME_OUTER - 1)

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 5)
    seed_color = int(colors[0])
    frag_colors = dict(zip(SIDES, map(int, colors[1:])))

    gaps, seed_spec = _sample_local_specs(rng)
    frame_patch = _draw_frame_patch(gaps, seed_spec, seed_color)
    grid[frame_r : frame_r + FRAME_OUTER, frame_c : frame_c + FRAME_OUTER] = frame_patch

    active = _active_sides(str(seed_spec["orient"]))
    patches: dict[str, np.ndarray] = {}
    for side in SIDES:
        thickness = 2 if side in active else rng.choice([1, 2])
        patches[side] = _fragment_patch(side, int(gaps[side][1]), thickness, frag_colors[side])

    if not _place_patches(rng, grid, frame_bbox, patches):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == seed_color)) <= int(np.count_nonzero(grid == seed_color)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to sample a valid puzzle")
