import random
from typing import Iterable

import numpy as np


BG = 0
SEP = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUADRANTS = ("tl", "tr", "br", "bl")
GEN_RETRIES = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_slot_lattice(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int], tuple[int, int, int, int]]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        row_lines = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == color)]
        col_lines = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == color)]
        if len(row_lines) < 2 or len(col_lines) < 2:
            continue
        if row_lines[0] != r0 or row_lines[-1] != r1 or col_lines[0] != c0 or col_lines[-1] != c1:
            continue
        row_steps = {b - a for a, b in zip(row_lines, row_lines[1:])}
        col_steps = {b - a for a, b in zip(col_lines, col_lines[1:])}
        if len(row_steps) != 1 or len(col_steps) != 1:
            continue
        row_step = next(iter(row_steps))
        col_step = next(iter(col_steps))
        if row_step <= 1 or col_step <= 1:
            continue
        candidates.append((len(coords), color, row_lines, col_lines, (r0, c0, r1, c1)))
    assert candidates, "slot lattice not found"
    _, color, row_lines, col_lines, bbox = max(candidates)
    return color, row_lines, col_lines, bbox


def _parse_slots(
    grid: np.ndarray, bg: int, sep: int, row_lines: list[int], col_lines: list[int]
) -> dict[int, tuple[int, int, int, int]]:
    slots: dict[int, tuple[int, int, int, int]] = {}
    slot_hs = []
    slot_ws = []
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            r0 = row_lines[i] + 1
            r1 = row_lines[i + 1] - 1
            c0 = col_lines[j] + 1
            c1 = col_lines[j + 1] - 1
            slot_hs.append(r1 - r0 + 1)
            slot_ws.append(c1 - c0 + 1)
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            marks = np.argwhere((sub != bg) & (sub != sep))
            assert len(marks) == 1, "each slot must contain one label"
            label_r, label_c = map(int, marks[0])
            label = int(sub[label_r, label_c])
            assert label not in slots, "slot labels must be unique"
            slots[label] = (r0, c0, r1 - r0 + 1, c1 - c0 + 1)
    assert len(set(slot_hs)) == 1 and len(set(slot_ws)) == 1, "slots must be equal-sized"
    return slots


def _rotation_k(h: int, w: int, seed_r: int, seed_c: int) -> int:
    top = 2 * seed_r < h
    left = 2 * seed_c < w
    if top and left:
        return 0
    if top and not left:
        return 3
    if not top and not left:
        return 2
    return 1


def _extract_templates(grid: np.ndarray, bg: int, sep: int) -> dict[int, tuple[np.ndarray, int]]:
    parts: dict[int, tuple[np.ndarray, int]] = {}
    comps = _components((grid != bg) & (grid != sep))
    for comp in comps:
        if len(comp) <= 1:
            continue
        colors, counts = np.unique(grid[comp[:, 0], comp[:, 1]], return_counts=True)
        if len(colors) != 2 or np.sum(counts == 1) != 1:
            continue
        seed_color = int(colors[counts == 1][0])
        body_color = int(colors[counts != 1][0])
        r0, c0, r1, c1 = _bbox(comp)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        seed_pos = np.argwhere(crop == seed_color)
        assert len(seed_pos) == 1, "seed must be unique within template"
        sr, sc = map(int, seed_pos[0])
        body = crop == body_color
        k = _rotation_k(body.shape[0], body.shape[1], sr, sc)
        parts[seed_color] = (np.rot90(body, k), body_color)
    assert parts, "no templates found"
    return parts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep, row_lines, col_lines, _ = _find_slot_lattice(g, bg)
    slots = _parse_slots(g, bg, sep, row_lines, col_lines)
    templates = _extract_templates(g, bg, sep)

    assert set(slots) == set(templates), "template seeds and slot labels must match exactly"

    out = np.full_like(g, bg)
    out[g == sep] = sep
    for seed_color, (mask, body_color) in templates.items():
        slot_r, slot_c, slot_h, slot_w = slots[seed_color]
        mh, mw = mask.shape
        assert mh <= slot_h and mw <= slot_w, "rotated template must fit in its slot"
        top = slot_r + (slot_h - mh) // 2
        left = slot_c + (slot_w - mw) // 2
        patch = out[top : top + mh, left : left + mw]
        patch[mask] = body_color
    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = [tuple(map(int, coords[0]))]
    seen[stack[0]] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == len(coords)


def _rotation_signatures(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {(np.rot90(mask, k).shape, np.rot90(mask, k).tobytes()) for k in range(4)}


def _quadrant_candidates(h: int, w: int, quadrant: str) -> list[tuple[int, int]]:
    top_rows = range(1, h // 2)
    bottom_rows = range(h // 2, h - 1)
    left_cols = range(1, w // 2)
    right_cols = range(w // 2, w - 1)
    if quadrant == "tl":
        rows, cols = top_rows, left_cols
    elif quadrant == "tr":
        rows, cols = top_rows, right_cols
    elif quadrant == "br":
        rows, cols = bottom_rows, right_cols
    else:
        rows, cols = bottom_rows, left_cols
    return [(r, c) for r in rows for c in cols]


def _choose_from_nearest(mask: np.ndarray, target: tuple[int, int]) -> tuple[int, int]:
    coords = np.argwhere(mask)
    dists = np.abs(coords[:, 0] - target[0]) + np.abs(coords[:, 1] - target[1])
    best = coords[dists == dists.min()]
    pick = best[random.randrange(len(best))]
    return int(pick[0]), int(pick[1])


def _connect_target(mask: np.ndarray, target: tuple[int, int]) -> None:
    r, c = _choose_from_nearest(mask, target)
    tr, tc = target
    while (r, c) != (tr, tc):
        choices = []
        if r != tr:
            choices.append(("r", 1 if tr > r else -1))
        if c != tc:
            choices.append(("c", 1 if tc > c else -1))
        axis, step = random.choice(choices)
        if axis == "r":
            r += step
        else:
            c += step
        mask[r, c] = True


def _grow_connected(mask: np.ndarray, target_area: int) -> None:
    h, w = mask.shape
    while int(mask.sum()) < target_area:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return
        r, c = random.choice(frontier)
        mask[r, c] = True


def _touches_all_sides(mask: np.ndarray) -> bool:
    return (
        np.any(mask[0, :])
        and np.any(mask[-1, :])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    )


def _make_object(h: int, w: int, quadrant: str) -> tuple[np.ndarray, tuple[int, int]]:
    candidates = _quadrant_candidates(h, w, quadrant)
    for _ in range(300):
        seed = random.choice(candidates)
        mask = np.zeros((h, w), dtype=bool)
        sr, sc = seed
        mask[sr - 1 : sr + 2, sc - 1 : sc + 2] = True

        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(anchors)
        for anchor in anchors:
            _connect_target(mask, anchor)

        base = int(mask.sum())
        target_area = random.randint(base, min(h * w, base + random.randint(2, max(3, h + w - 2))))
        _grow_connected(mask, target_area)

        body = mask.copy()
        body[sr, sc] = False
        if not _connected(body):
            continue
        if not _touches_all_sides(mask):
            continue
        if len(_rotation_signatures(body)) < 4:
            continue
        return body, seed
    raise RuntimeError("failed to make a valid object")


def _non_touching_place(
    board: np.ndarray, occ: np.ndarray, patch: np.ndarray, colors: tuple[int, int]
) -> bool:
    h, w = board.shape
    ph, pw = patch.shape
    tries = 250
    seed_color, body_color = colors
    for _ in range(tries):
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(h, top + ph + 1)
        c1 = min(w, left + pw + 1)
        if np.any(occ[r0:r1, c0:c1]):
            continue
        occ[top : top + ph, left : left + pw] |= patch != BG
        sub = board[top : top + ph, left : left + pw]
        sub[patch == 1] = body_color
        sub[patch == 2] = seed_color
        return True
    return False


def _slot_positions(slot_rows: int, slot_cols: int, slot_h: int, slot_w: int) -> list[tuple[int, int]]:
    positions = []
    for i in range(slot_rows):
        for j in range(slot_cols):
            positions.append((1 + i * (slot_h + 1), 1 + j * (slot_w + 1)))
    return positions


def _build_template_patch(body: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    patch = np.zeros(body.shape, dtype=int)
    patch[body] = 1
    patch[seed] = 2
    return patch


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(GEN_RETRIES):
        n_objects = random.choice([4, 6])
        if n_objects == 4:
            slot_rows, slot_cols = random.choice([(2, 2), (1, 4), (4, 1)])
        else:
            slot_rows, slot_cols = random.choice([(2, 3), (3, 2)])

        slot_h = random.randint(6, 8)
        slot_w = random.randint(6, 8)
        panel_h = 1 + slot_rows * (slot_h + 1)
        panel_w = 1 + slot_cols * (slot_w + 1)

        extra_quadrants = [random.choice(QUADRANTS) for _ in range(n_objects - 4)]
        quadrants = list(QUADRANTS) + extra_quadrants
        random.shuffle(quadrants)

        seed_colors = random.sample(colors, n_objects)
        body_palette = [c for c in colors if c not in seed_colors]
        body_colors = [random.choice(body_palette) for _ in range(n_objects)]

        objects = []
        max_h = 0
        total_w = 0
        try:
            for quadrant, seed_color, body_color in zip(quadrants, seed_colors, body_colors):
                h_choices = [4] + ([6] if slot_h >= 7 else [])
                w_choices = [4] + ([6] if slot_w >= 7 else [])
                h = random.choice(h_choices)
                w = random.choice(w_choices)
                body, seed = _make_object(h, w, quadrant)
                patch = _build_template_patch(body, seed)
                objects.append((patch, seed_color, body_color))
                max_h = max(max_h, h)
                total_w += w + 2
        except RuntimeError:
            continue

        left_h = panel_h
        left_w = max(
            10,
            min(20, total_w // 2 + random.randint(0, 3)),
            max_h + 4,
        )
        gap = random.randint(1, 3)
        grid = np.full((panel_h, left_w + gap + panel_w), BG, dtype=int)

        panel_c0 = left_w + gap
        for r in range(0, panel_h, slot_h + 1):
            grid[r, panel_c0 : panel_c0 + panel_w] = SEP
        for c in range(0, panel_w, slot_w + 1):
            grid[:panel_h, panel_c0 + c] = SEP

        occ = np.zeros((left_h, left_w), dtype=bool)
        order = list(range(n_objects))
        random.shuffle(order)
        ok = True
        for idx in order:
            patch, seed_color, body_color = objects[idx]
            if not _non_touching_place(grid[:, :left_w], occ, patch, (seed_color, body_color)):
                ok = False
                break
        if not ok:
            continue

        slots = _slot_positions(slot_rows, slot_cols, slot_h, slot_w)
        random.shuffle(slots)
        for (patch, seed_color, _body_color), (r0, c0) in zip(objects, slots):
            grid[r0, panel_c0 + c0] = seed_color

        # Reject any accidental extra multi-color component on the right.
        if len(_extract_templates(grid, BG, SEP)) != n_objects:
            continue

        return grid
    raise RuntimeError("failed to generate valid input")
