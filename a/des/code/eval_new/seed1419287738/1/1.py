from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
BASE_CANVAS = 30
FRAME_PAD = 3


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _flood_compartment(g: np.ndarray, bbox: tuple[int, int, int, int], start: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    sr, sc = start
    assert r0 < sr < r1 and c0 < sc < c1, "door must look into frame interior"
    assert int(g[sr, sc]) != GREY, "door-adjacent interior cell cannot be a wall"
    q = deque([(sr, sc)])
    seen = {start}
    pts = [start]
    while q:
        cr, cc = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = cr + dr, cc + dc
            if not (r0 < nr < r1 and c0 < nc < c1):
                continue
            if int(g[nr, nc]) == GREY or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            pts.append((nr, nc))
            q.append((nr, nc))
    return pts


def _door_info(r: int, c: int, bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if r == r0:
        return (-1, 0), (r + 1, c)
    if r == r1:
        return (1, 0), (r - 1, c)
    if c == c0:
        return (0, -1), (r, c + 1)
    if c == c1:
        return (0, 1), (r, c - 1)
    raise AssertionError("door must sit on the frame perimeter")


def _release_delta(
    content: list[tuple[int, int]], bbox: tuple[int, int, int, int], release_dir: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    rows = [r for r, _ in content]
    cols = [c for _, c in content]
    dr, dc = release_dir
    if dr < 0:
        return (r0 - 1) - max(rows), 0
    if dr > 0:
        return (r1 + 1) - min(rows), 0
    if dc < 0:
        return 0, (c0 - 1) - max(cols)
    if dc > 0:
        return 0, (c1 + 1) - min(cols)
    raise AssertionError("release direction must be cardinal")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock colored border doors with matching internal key pixels and eject the adjacent compartment contents outside the grey frame.

    Input structure:
    - Black background surrounds one or more grey rectangular hollow frames.
    - Each frame border may contain one or more one-cell colored doors embedded in the perimeter.
    - Grey internal walls partition each frame interior into orthogonal compartments.
    - Inside a frame, some door colors reappear as isolated one-cell keys; compartments may also contain colored content patterns.

    Rule:
    - For each frame, detect every colored perimeter door and every internal key whose color matches a door.
    - A matched door opens; its border side determines the release direction, and the compartment directly behind that door is selected.
    - Move every non-key colored pixel in that compartment rigidly outward along the door direction until the released pattern sits just outside the frame.
    - Clear the opened door cell, clear the matching key pixel, and empty the released compartment inside the frame; unmatched doors and their compartments stay unchanged.

    Key insight:
    - The transformation chains three cues: color matches keys to doors, door position encodes movement direction, and the door-facing compartment determines which interior pattern gets expelled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    occupied_targets: set[tuple[int, int]] = set()
    actions: list[dict[str, object]] = []
    key_cells_to_clear: set[tuple[int, int]] = set()
    door_cells_to_clear: set[tuple[int, int]] = set()

    grey_components = _components(g == GREY)
    assert grey_components, "expected at least one grey frame"

    for grey_comp in grey_components:
        bbox = _bbox(grey_comp)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

        doors: list[dict[str, object]] = []
        perimeter = []
        for c in range(c0, c1 + 1):
            perimeter.append((r0, c))
            perimeter.append((r1, c))
        for r in range(r0 + 1, r1):
            perimeter.append((r, c0))
            perimeter.append((r, c1))

        for r, c in perimeter:
            val = int(g[r, c])
            assert val != BG, "frame perimeter cannot contain open black gaps in the input"
            if val != GREY:
                release_dir, inside = _door_info(r, c, bbox)
                doors.append(
                    {
                        "pos": (r, c),
                        "color": val,
                        "release_dir": release_dir,
                        "inside": inside,
                    }
                )

        assert doors, "each frame must contain at least one colored door"
        door_colors = {int(d["color"]) for d in doors}

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_mask = (inner != BG) & (inner != GREY)
        key_positions: dict[int, list[tuple[int, int]]] = {}
        for comp in _components(inner_mask):
            if int(comp.shape[0]) != 1:
                continue
            rr, cc = map(int, comp[0])
            ar, ac = r0 + 1 + rr, c0 + 1 + cc
            col = int(g[ar, ac])
            if col in door_colors:
                key_positions.setdefault(col, []).append((ar, ac))

        unlocked_colors = {int(d["color"]) for d in doors if int(d["color"]) in key_positions}
        for color in unlocked_colors:
            for pos in key_positions[color]:
                key_cells_to_clear.add(pos)

        for door in doors:
            color = int(door["color"])
            if color not in unlocked_colors:
                continue
            door_pos = tuple(door["pos"])
            release_dir = tuple(door["release_dir"])
            inside = tuple(door["inside"])
            compartment = _flood_compartment(g, bbox, inside)
            key_cells_in_frame = {pos for cells in key_positions.values() for pos in cells}
            content = sorted(
                (r, c)
                for r, c in compartment
                if int(g[r, c]) not in (BG, GREY) and (r, c) not in key_cells_in_frame
            )
            targets: list[tuple[int, int]] = []
            if content:
                dr, dc = _release_delta(content, bbox, release_dir)
                for sr, sc in content:
                    tr, tc = sr + dr, sc + dc
                    assert 0 <= tr < h and 0 <= tc < w, "released content must stay within the grid"
                    assert int(g[tr, tc]) == BG, "released content must move into empty background"
                    assert (tr, tc) not in occupied_targets, "released patterns may not overlap"
                    occupied_targets.add((tr, tc))
                    targets.append((tr, tc))
            door_cells_to_clear.add(door_pos)
            actions.append(
                {
                    "door_pos": door_pos,
                    "compartment": compartment,
                    "content": content,
                    "targets": targets,
                }
            )

    for r, c in key_cells_to_clear:
        out[r, c] = BG
    for r, c in door_cells_to_clear:
        out[r, c] = BG

    for action in actions:
        for r, c in action["compartment"]:
            if int(g[r, c]) not in (BG, GREY):
                out[r, c] = BG

    for action in actions:
        for (sr, sc), (tr, tc) in zip(action["content"], action["targets"]):
            out[tr, tc] = int(g[sr, sc])

    return out


PATTERN_MASKS = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
]


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for k in range(4):
        cur = np.rot90(mask, k)
        key = cur.tobytes()
        if key not in seen:
            seen.add(key)
            variants.append(cur.copy())
    return variants


def _frame_geometry(scheme: str) -> tuple[int, int, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    if scheme == "v":
        h, w = random.choice([7, 8, 9]), 9
        wall_col = w // 2
        walls = [(1, wall_col, h - 2, wall_col)]
        comps = [(1, 1, h - 2, wall_col - 1), (1, wall_col + 1, h - 2, w - 2)]
        return h, w, walls, comps
    if scheme == "h":
        h, w = 9, random.choice([7, 8, 9])
        wall_row = h // 2
        walls = [(wall_row, 1, wall_row, w - 2)]
        comps = [(1, 1, wall_row - 1, w - 2), (wall_row + 1, 1, h - 2, w - 2)]
        return h, w, walls, comps
    if scheme == "vh":
        h = w = 9
        wall_row = h // 2
        wall_col = w // 2
        walls = [(1, wall_col, h - 2, wall_col), (wall_row, 1, wall_row, w - 2)]
        comps = [
            (1, 1, wall_row - 1, wall_col - 1),
            (1, wall_col + 1, wall_row - 1, w - 2),
            (wall_row + 1, 1, h - 2, wall_col - 1),
            (wall_row + 1, wall_col + 1, h - 2, w - 2),
        ]
        return h, w, walls, comps
    raise AssertionError("unknown frame scheme")


def _compartment_sides(
    box: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]
) -> dict[str, list[int]]:
    r0, c0, r1, c1 = box
    fr0, fc0, fr1, fc1 = frame_bbox
    sides: dict[str, list[int]] = {}
    if r0 == fr0 + 1:
        sides["top"] = list(range(c0, c1 + 1))
    if r1 == fr1 - 1:
        sides["bottom"] = list(range(c0, c1 + 1))
    if c0 == fc0 + 1:
        sides["left"] = list(range(r0, r1 + 1))
    if c1 == fc1 - 1:
        sides["right"] = list(range(r0, r1 + 1))
    return sides


def _place_pattern(
    sub: np.ndarray,
    comp_box: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
    disallow: set[tuple[int, int]],
    color: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = comp_box
    ch = r1 - r0 + 1
    cw = c1 - c0 + 1
    masks = []
    for base in PATTERN_MASKS:
        for var in _mask_variants(base):
            mh, mw = map(int, var.shape)
            if mh <= ch and mw <= cw:
                masks.append(var)
    random.shuffle(masks)
    for mask in masks:
        mh, mw = map(int, mask.shape)
        offsets = [(rr, cc) for rr in range(r0, r1 - mh + 2) for cc in range(c0, c1 - mw + 2)]
        random.shuffle(offsets)
        for rr, cc in offsets:
            cells = [(rr + mr, cc + mc) for mr in range(mh) for mc in range(mw) if mask[mr, mc]]
            if any(cell in occupied or cell in disallow for cell in cells):
                continue
            for cr, cc2 in cells:
                sub[cr, cc2] = color
                occupied.add((cr, cc2))
            return cells
    return []


def _isolated_key_spots(
    sub: np.ndarray, comp_box: tuple[int, int, int, int], occupied: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = comp_box
    spots = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in occupied or int(sub[r, c]) != BG:
                continue
            ok = True
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                    continue
                if (nr, nc) in occupied:
                    ok = False
                    break
            if ok:
                spots.append((r, c))
    return spots


def _make_frame_subgrid() -> tuple[np.ndarray, bool] | None:
    scheme = random.choice(["v", "h", "vh"])
    outer_h, outer_w, walls, compartments = _frame_geometry(scheme)
    sub = np.zeros((outer_h + 2 * FRAME_PAD, outer_w + 2 * FRAME_PAD), dtype=int)
    fr0, fc0 = FRAME_PAD, FRAME_PAD
    fr1, fc1 = fr0 + outer_h - 1, fc0 + outer_w - 1

    sub[fr0, fc0 : fc1 + 1] = GREY
    sub[fr1, fc0 : fc1 + 1] = GREY
    sub[fr0 : fr1 + 1, fc0] = GREY
    sub[fr0 : fr1 + 1, fc1] = GREY

    abs_compartments = []
    for wr0, wc0, wr1, wc1 in walls:
        sub[fr0 + wr0 : fr0 + wr1 + 1, fc0 + wc0 : fc0 + wc1 + 1] = GREY
    for cr0, cc0, cr1, cc1 in compartments:
        abs_compartments.append((fr0 + cr0, fc0 + cc0, fr0 + cr1, fc0 + cc1))

    max_doors = min(len(abs_compartments), 4)
    n_doors = random.randint(1, max_doors)
    chosen_comps = random.sample(range(len(abs_compartments)), n_doors)
    door_colors = random.sample(COLOR_POOL, n_doors)

    used_sides: set[str] = set()
    doors: list[dict[str, object]] = []
    for comp_idx, color in zip(chosen_comps, door_colors):
        sides = _compartment_sides(abs_compartments[comp_idx], (fr0, fc0, fr1, fc1))
        usable = [side for side in sides if side not in used_sides]
        if not usable:
            return None
        side = random.choice(usable)
        used_sides.add(side)
        along = random.choice(sides[side])
        if side == "top":
            dr, dc = fr0, along
        elif side == "bottom":
            dr, dc = fr1, along
        elif side == "left":
            dr, dc = along, fc0
        else:
            dr, dc = along, fc1
        sub[dr, dc] = color
        doors.append({"comp_idx": comp_idx, "side": side, "pos": (dr, dc), "color": color})

    unlock_count = random.randint(1, len(doors))
    unlocked_indices = set(random.sample(range(len(doors)), unlock_count))
    occupied: set[tuple[int, int]] = set()
    content_by_comp: dict[int, list[tuple[int, int]]] = {}

    for idx, comp_box in enumerate(abs_compartments):
        has_door = any(int(door["comp_idx"]) == idx for door in doors)
        wants_content = random.random() < (0.75 if has_door else 0.45)
        if not wants_content:
            content_by_comp[idx] = []
            continue
        content_color = random.choice(COLOR_POOL)
        if content_color in door_colors:
            candidates = [c for c in COLOR_POOL if c not in door_colors]
            content_color = random.choice(candidates)
        cells = _place_pattern(sub, comp_box, occupied, set(), content_color)
        content_by_comp[idx] = cells

    if not any(content_by_comp.get(int(door["comp_idx"]), []) for i, door in enumerate(doors) if i in unlocked_indices):
        candidate_door = random.choice([doors[i] for i in unlocked_indices])
        comp_idx = int(candidate_door["comp_idx"])
        if content_by_comp.get(comp_idx):
            pass
        else:
            candidates = [c for c in COLOR_POOL if c not in door_colors]
            color = random.choice(candidates)
            cells = _place_pattern(sub, abs_compartments[comp_idx], occupied, set(), color)
            if not cells:
                return None
            content_by_comp[comp_idx] = cells

    placed_keys: set[tuple[int, int]] = set()
    for door_idx in random.sample(list(unlocked_indices), len(unlocked_indices)):
        door = doors[door_idx]
        preferred = [i for i in range(len(abs_compartments)) if i != int(door["comp_idx"])]
        random.shuffle(preferred)
        order = preferred + [int(door["comp_idx"])]
        placed = False
        for comp_idx in order:
            spots = _isolated_key_spots(sub, abs_compartments[comp_idx], occupied | placed_keys)
            if not spots:
                continue
            kr, kc = random.choice(spots)
            sub[kr, kc] = int(door["color"])
            placed_keys.add((kr, kc))
            placed = True
            break
        if not placed:
            return None

    try:
        produced = main(sub)
    except Exception:
        return None
    if np.array_equal(produced, sub):
        return None
    return sub, bool(len(doors) > len(unlocked_indices))


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice([1, 2])
    frames: list[np.ndarray] = []
    for _ in range(frame_count):
        made = _make_frame_subgrid()
        if made is None:
            return None
        sub, _has_locked = made
        frames.append(sub)

    canvas = np.zeros((BASE_CANVAS, BASE_CANVAS), dtype=int)
    rects: list[tuple[int, int, int, int]] = []
    for sub in frames:
        sh, sw = map(int, sub.shape)
        placed = False
        for _ in range(200):
            top = random.randint(0, BASE_CANVAS - sh)
            left = random.randint(0, BASE_CANVAS - sw)
            rect = (top, left, top + sh - 1, left + sw - 1)
            if any(_rectangles_overlap(rect, other) for other in rects):
                continue
            block = canvas[top : top + sh, left : left + sw]
            assert not np.any((block != BG) & (sub != BG))
            block[sub != BG] = sub[sub != BG]
            rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    min_r = min(r0 for r0, _, _, _ in rects)
    min_c = min(c0 for _, c0, _, _ in rects)
    max_r = max(r1 for _, _, r1, _ in rects)
    max_c = max(c1 for _, _, _, c1 in rects)
    inp = canvas[min_r : max_r + 1, min_c : max_c + 1]
    if inp.size == 0 or max(inp.shape) > 30:
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if max(out.shape) > 30 or np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
