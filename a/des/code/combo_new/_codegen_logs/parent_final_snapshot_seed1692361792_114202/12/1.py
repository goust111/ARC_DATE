from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FRAME_COLOR = 5
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
CORNER_OFFSETS = [
    (1, 1),   # top-left
    (1, -2),  # top-right
    (-2, -2), # bottom-right
    (-2, 1),  # bottom-left
]
CORNER_TO_ROT_K = {
    0: 0,  # top-left -> no rotation
    1: 3,  # top-right -> clockwise 90 = ccw 270
    2: 2,  # bottom-right -> 180
    3: 1,  # bottom-left -> counterclockwise 90
}
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_bbox_if_valid(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the bbox if the component is exactly a 1-pixel rectangular border."""
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 4 or w < 4:
        return None
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    cells = {tuple(map(int, xy)) for xy in coords}
    if cells != expected:
        return None
    return r0, c0, r1, c1


def _crop_component(coords: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for rr, cc in coords:
        patch[int(rr) - r0, int(cc) - c0] = int(color)
    return patch, (r0, c0, r1, c1)


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return 0


def _bbox_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    return _interval_gap(box_a[0], box_a[2], box_b[0], box_b[2]) + _interval_gap(box_a[1], box_a[3], box_b[1], box_b[3])


def _bbox_center_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    ar = box_a[0] + box_a[2]
    ac = box_a[1] + box_a[3]
    br = box_b[0] + box_b[2]
    bc = box_b[1] + box_b[3]
    return abs(ar - br) + abs(ac - bc)


def _detect_frames(g: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in map(int, np.unique(g)):
        if color == BG:
            continue
        for comp in _components(g == color):
            box = _frame_bbox_if_valid(comp)
            if box is None:
                continue
            r0, c0, r1, c1 = box
            corner_cells = [
                (r0 + 1, c0 + 1),
                (r0 + 1, c1 - 1),
                (r1 - 1, c1 - 1),
                (r1 - 1, c0 + 1),
            ]
            marks = [(idx, pos, int(g[pos])) for idx, pos in enumerate(corner_cells) if int(g[pos]) not in (BG, color)]
            if len(marks) != 1:
                continue
            corner_idx, marker_pos, marker_color = marks[0]
            frames.append(
                {
                    "frame_color": int(color),
                    "bbox": box,
                    "corner_idx": int(corner_idx),
                    "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
                    "marker_color": int(marker_color),
                    "rot_k": int(CORNER_TO_ROT_K[int(corner_idx)]),
                }
            )
    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1], f["bbox"][2], f["bbox"][3]))
    return frames


def _assign_components_to_frames(frames: list[dict], comp_map: dict[int, list[dict]]) -> dict[int, dict]:
    assignments: dict[int, dict] = {}
    by_color: dict[int, list[tuple[int, dict]]] = {}
    for idx, frame in enumerate(frames):
        by_color.setdefault(int(frame["marker_color"]), []).append((idx, frame))

    for color, frame_group in by_color.items():
        pieces = comp_map.get(int(color), [])
        assert len(pieces) == len(frame_group) and pieces, "every frame color must have exactly one matching sprite component"
        best_perm: tuple[int, ...] | None = None
        best_score: tuple[int, int] | None = None
        for perm in itertools.permutations(range(len(pieces)), len(frame_group)):
            score_gap = 0
            score_center = 0
            for (frame_idx, frame), piece_idx in zip(frame_group, perm):
                score_gap += _bbox_gap(frame["bbox"], pieces[piece_idx]["bbox"])
                score_center += _bbox_center_distance(frame["bbox"], pieces[piece_idx]["bbox"])
            score = (int(score_gap), int(score_center))
            if best_score is None or score < best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None
        for (frame_idx, _frame), piece_idx in zip(frame_group, best_perm):
            assignments[int(frame_idx)] = pieces[int(piece_idx)]
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each outside sprite according to a colored inner-corner marker and center it inside its frame.

    Input structure:
    - The background is black.
    - One or more 1-pixel-thick hollow rectangular frames appear as monochrome border components.
    - Inside each frame, exactly one inner-corner cell is recolored with a non-frame, non-black color.
    - Outside each frame, a nearby 4-connected monochrome sprite of the same color appears on the black background.

    Rule:
    - Detect each hollow frame and read which of its four inner corners is colored.
    - Decode that corner as a rotation instruction: top-left = 0 degrees, top-right = clockwise 90, bottom-right = 180, bottom-left = counterclockwise 90.
    - Crop the matching outside sprite to its bounding box, rotate it by the decoded amount, erase the original outside sprite, and paste the rotated sprite centered inside the frame interior.
    - Keep the frame borders and the colored corner-marker cells unchanged.

    Key insight:
    - The lone colored corner is not part of the picture to copy; it is a positional code that tells how the same-colored sprite must be rotated before moving into the frame.
    """
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    assert frames, "expected at least one rectangular frame"

    marker_mask = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        mr, mc = frame["marker_pos"]
        marker_mask[mr, mc] = True

    sprite_components: dict[int, list[dict]] = {}
    needed_colors = sorted({int(frame["marker_color"]) for frame in frames})
    for color in needed_colors:
        comps: list[dict] = []
        for comp in _components((g == int(color)) & (~marker_mask)):
            patch, box = _crop_component(comp, int(color))
            comps.append({"coords": comp, "patch": patch, "bbox": box, "color": int(color)})
        sprite_components[int(color)] = comps

    assigned = _assign_components_to_frames(frames, sprite_components)
    out = g.copy()

    for piece in assigned.values():
        for rr, cc in piece["coords"]:
            out[int(rr), int(cc)] = BG

    for idx, frame in enumerate(frames):
        piece = assigned[int(idx)]
        rotated = np.rot90(piece["patch"], int(frame["rot_k"]))
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = int(r1 - r0 - 1)
        inner_w = int(c1 - c0 - 1)
        ph, pw = map(int, rotated.shape)
        top = int(r0 + 1 + (inner_h - ph) // 2)
        left = int(c0 + 1 + (inner_w - pw) // 2)
        mask = rotated != BG
        window = out[top : top + ph, left : left + pw]
        window[mask] = rotated[mask]
        mr, mc = frame["marker_pos"]
        out[mr, mc] = int(frame["marker_color"])

    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask != 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _rotation_symmetric(mask: np.ndarray) -> bool:
    base = _normalize_mask(mask)
    for k in (1, 2, 3):
        if np.array_equal(base, _normalize_mask(np.rot90(base, int(k)))):
            return True
    return False


def _random_sprite_mask() -> np.ndarray:
    for _ in range(200):
        target_area = random.randint(3, 6)
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < target_area:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in DIRS4:
                nbr = (cell[0] + dr, cell[1] + dc)
                if nbr not in cells:
                    frontier.add(nbr)
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        min_r, min_c = min(rows), min(cols)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        h = max(r for r, _c in norm) + 1
        w = max(c for _r, c in norm) + 1
        mask = np.zeros((h, w), dtype=int)
        for r, c in norm:
            mask[int(r), int(c)] = 1
        if _rotation_symmetric(mask):
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate sprite")


def _candidate_cells_clear(shape: tuple[int, int], occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = map(int, shape)
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    for r, c in cell_set:
        if not (0 <= int(r) < h and 0 <= int(c) < w):
            return False
        for nr in range(int(r) - 1, int(r) + 2):
            for nc in range(int(c) - 1, int(c) + 2):
                if 0 <= nr < h and 0 <= nc < w and bool(occupied[nr, nc]):
                    return False
    return True


def _place_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    marker_idx: int,
    source_patch: np.ndarray,
    rotated_patch: np.ndarray,
) -> dict | None:
    h, w = map(int, grid.shape)
    out_h, out_w = map(int, rotated_patch.shape)
    inner_h = int(out_h + 2 * random.randint(1, 3) + 2 * random.randint(0, 1))
    inner_w = int(out_w + 2 * random.randint(1, 3) + 2 * random.randint(0, 1))
    frame_h = int(inner_h + 2)
    frame_w = int(inner_w + 2)
    src_h, src_w = map(int, source_patch.shape)

    sides = ["top", "right", "bottom", "left"]
    for _ in range(300):
        side = random.choice(sides)
        gap = random.randint(1, 3)

        if side == "top":
            min_fr = gap + src_h
            max_fr = h - frame_h
            min_fc = 0
            max_fc = w - frame_w
        elif side == "bottom":
            min_fr = 0
            max_fr = h - frame_h - gap - src_h
            min_fc = 0
            max_fc = w - frame_w
        elif side == "left":
            min_fr = 0
            max_fr = h - frame_h
            min_fc = gap + src_w
            max_fc = w - frame_w
        else:
            min_fr = 0
            max_fr = h - frame_h
            min_fc = 0
            max_fc = w - frame_w - gap - src_w

        if min_fr > max_fr or min_fc > max_fc:
            continue

        fr0 = random.randint(int(min_fr), int(max_fr))
        fc0 = random.randint(int(min_fc), int(max_fc))

        if side == "top":
            sr0 = int(fr0 - gap - src_h)
            sc_lo = max(0, int(fc0 - src_w + 1))
            sc_hi = min(int(w - src_w), int(fc0 + frame_w - 1))
            if sc_lo > sc_hi:
                continue
            sc0 = random.randint(sc_lo, sc_hi)
        elif side == "bottom":
            sr0 = int(fr0 + frame_h + gap)
            sc_lo = max(0, int(fc0 - src_w + 1))
            sc_hi = min(int(w - src_w), int(fc0 + frame_w - 1))
            if sc_lo > sc_hi:
                continue
            sc0 = random.randint(sc_lo, sc_hi)
        elif side == "left":
            sc0 = int(fc0 - gap - src_w)
            sr_lo = max(0, int(fr0 - src_h + 1))
            sr_hi = min(int(h - src_h), int(fr0 + frame_h - 1))
            if sr_lo > sr_hi:
                continue
            sr0 = random.randint(sr_lo, sr_hi)
        else:
            sc0 = int(fc0 + frame_w + gap)
            sr_lo = max(0, int(fr0 - src_h + 1))
            sr_hi = min(int(h - src_h), int(fr0 + frame_h - 1))
            if sr_lo > sr_hi:
                continue
            sr0 = random.randint(sr_lo, sr_hi)

        border_cells = [
            (r, c)
            for r in range(fr0, fr0 + frame_h)
            for c in range(fc0, fc0 + frame_w)
            if r in (fr0, fr0 + frame_h - 1) or c in (fc0, fc0 + frame_w - 1)
        ]
        mr_off_r, mr_off_c = CORNER_OFFSETS[int(marker_idx)]
        marker_pos = (
            int(fr0 + (1 if mr_off_r > 0 else frame_h - 2)),
            int(fc0 + (1 if mr_off_c > 0 else frame_w - 2)),
        )
        sprite_cells = [
            (int(sr0 + r), int(sc0 + c))
            for r, c in np.argwhere(source_patch != 0)
        ]
        candidate_cells = border_cells + [marker_pos] + sprite_cells
        if not _candidate_cells_clear(grid.shape, occupied, candidate_cells):
            continue

        for r, c in border_cells:
            grid[int(r), int(c)] = FRAME_COLOR
            occupied[int(r), int(c)] = True
        grid[int(marker_pos[0]), int(marker_pos[1])] = int(color)
        occupied[int(marker_pos[0]), int(marker_pos[1])] = True
        for r, c in sprite_cells:
            grid[int(r), int(c)] = int(color)
            occupied[int(r), int(c)] = True

        return {
            "frame_bbox": (int(fr0), int(fc0), int(fr0 + frame_h - 1), int(fc0 + frame_w - 1)),
            "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
            "marker_color": int(color),
            "rot_k": int(CORNER_TO_ROT_K[int(marker_idx)]),
            "source_cells": sprite_cells,
            "source_patch": source_patch.copy(),
            "rotated_patch": rotated_patch.copy(),
        }
    return None


def _expected_output(inp: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for pair in pairs:
        for r, c in pair["source_cells"]:
            out[int(r), int(c)] = BG
    for pair in pairs:
        r0, c0, r1, c1 = pair["frame_bbox"]
        rotated = pair["rotated_patch"]
        inner_h = int(r1 - r0 - 1)
        inner_w = int(c1 - c0 - 1)
        ph, pw = map(int, rotated.shape)
        top = int(r0 + 1 + (inner_h - ph) // 2)
        left = int(c0 + 1 + (inner_w - pw) // 2)
        mask = rotated != 0
        out[top : top + ph, left : left + pw][mask] = pair["marker_color"]
        mr, mc = pair["marker_pos"]
        out[int(mr), int(mc)] = int(pair["marker_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice([1, 2, 2, 3])
    min_side = 12 + 4 * (n_pairs - 1)
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = random.sample(SPRITE_COLORS, int(n_pairs))
    marker_order = random.sample([0, 1, 2, 3], int(n_pairs))
    pairs: list[dict] = []

    for color, marker_idx in zip(colors, marker_order):
        source_patch = _random_sprite_mask() * int(color)
        source_patch = np.rot90(source_patch, random.randrange(4))
        rotated_patch = np.rot90(source_patch, int(CORNER_TO_ROT_K[int(marker_idx)]))
        pair = _place_pair(grid, occupied, int(color), int(marker_idx), source_patch, rotated_patch)
        if pair is None:
            return None
        pairs.append(pair)

    expected = _expected_output(grid, pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
