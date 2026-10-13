from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 600


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, r0, c0


def _rotate_masks(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        cur = np.rot90(mask, k)
        key = (cur.shape, cur.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(cur)
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((cur.shape, cur.tobytes()) for cur in _rotate_masks(mask))


def _match_under_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    return any(a.shape == cur.shape and np.array_equal(a, cur) for cur in _rotate_masks(b))


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BG) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _slot_pixels_from_cells(cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = set(cells)
    pix: set[tuple[int, int]] = set()
    for r, c in cells:
        pix.add((2 * r + 1, 2 * c + 1))
        if (r + 1, c) in cells:
            pix.add((2 * r + 2, 2 * c + 1))
        if (r, c + 1) in cells:
            pix.add((2 * r + 1, 2 * c + 2))
    return pix


def _paint_points(arr: np.ndarray, pts: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in pts:
        arr[r, c] = int(color)


def _candidate_feasible(
    candidates: list[list[int]],
    constraints: list[tuple[int, int, int]],
    assigned: list[int | None],
    used: set[int],
) -> bool:
    for i, cand in enumerate(candidates):
        if assigned[i] is None and not any(col not in used for col in cand):
            return False

    for a, b, col in constraints:
        va, vb = assigned[a], assigned[b]
        if va == col or vb == col:
            continue
        if col in used:
            return False
        can_a = va is None and col in candidates[a]
        can_b = vb is None and col in candidates[b]
        if not (can_a or can_b):
            return False
    return True


def _find_solutions(
    candidates: list[list[int]], constraints: list[tuple[int, int, int]], max_solutions: int = 2
) -> list[list[int]]:
    n = len(candidates)
    degrees = [0] * n
    for a, b, _ in constraints:
        degrees[a] += 1
        degrees[b] += 1
    order = sorted(range(n), key=lambda i: (len(candidates[i]), -degrees[i], i))
    assigned: list[int | None] = [None] * n
    used: set[int] = set()
    solutions: list[list[int]] = []

    def dfs(pos: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if pos == n:
            solutions.append([int(x) for x in assigned])
            return
        slot = order[pos]
        avail = sorted({int(col) for col in candidates[slot] if col not in used})
        for col in avail:
            assigned[slot] = int(col)
            used.add(int(col))
            if _candidate_feasible(candidates, constraints, assigned, used):
                dfs(pos + 1)
            used.remove(int(col))
            assigned[slot] = None

    if _candidate_feasible(candidates, constraints, assigned, used):
        dfs(0)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    comps = _components(g != bg)
    assert comps, "input must contain a frame"
    frame_coords = max(comps, key=lambda pts: int(len(pts)))
    frame_mask = np.zeros_like(g, dtype=bool)
    frame_mask[frame_coords[:, 0], frame_coords[:, 1]] = True

    frame_vals = [int(g[r, c]) for r, c in frame_coords]
    grey = int(Counter(frame_vals).most_common(1)[0][0])
    fr0, fc0, fr1, fc1 = _bbox_coords(frame_coords)
    sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]

    slots = []
    slot_id_map = np.full(sub.shape, -1, dtype=int)
    for comp in _components(sub == bg):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == sub.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == sub.shape[1] - 1):
            continue
        mask, r0, c0 = _tight_mask(comp)
        idx = len(slots)
        slot_id_map[comp[:, 0], comp[:, 1]] = idx
        slots.append(
            {
                "coords_local": comp,
                "coords_global": comp + np.array([[fr0, fc0]], dtype=int),
                "mask": mask,
                "bbox_local": (int(r0), int(c0)),
            }
        )
    slots.sort(key=lambda s: _bbox_coords(s["coords_local"])[0:2])

    # Rebuild id map after sorting.
    slot_id_map[:] = -1
    for idx, slot in enumerate(slots):
        coords = slot["coords_local"]
        slot_id_map[coords[:, 0], coords[:, 1]] = idx

    pieces = []
    for comp in comps:
        if comp is frame_coords:
            continue
        vals = {int(g[r, c]) for r, c in comp}
        if len(vals) != 1:
            continue
        color = int(next(iter(vals)))
        mask, _r0, _c0 = _tight_mask(comp)
        pieces.append({"color": color, "mask": mask})

    assert len(pieces) == len(slots), "piece count must equal slot count"

    constraints: list[tuple[int, int, int]] = []
    for r in range(sub.shape[0]):
        for c in range(sub.shape[1]):
            col = int(sub[r, c])
            if col in (bg, grey):
                continue
            neigh = set()
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1]:
                    sid = int(slot_id_map[nr, nc])
                    if sid >= 0:
                        neigh.add(sid)
            if len(neigh) == 2:
                a, b = sorted(neigh)
                constraints.append((int(a), int(b), col))

    candidates: list[list[int]] = []
    for slot in slots:
        cand = [int(piece["color"]) for piece in pieces if _match_under_rotation(slot["mask"], piece["mask"])]
        cand = sorted(set(cand))
        assert cand, "every slot must have at least one matching piece"
        candidates.append(cand)

    solutions = _find_solutions(candidates, constraints, max_solutions=2)
    assert len(solutions) == 1, "slot assignment must be unique"
    assignment = solutions[0]

    out = np.full_like(g, int(bg))
    out[frame_mask] = int(grey)
    for idx, slot in enumerate(slots):
        color = int(assignment[idx])
        coords = slot["coords_global"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _neighbors_of_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                out.add((nr, nc))
    return out


def _grow_slot_layout() -> tuple[list[set[tuple[int, int]]], int, int] | None:
    gh = random.randint(4, 6)
    gw = random.randint(4, 6)
    n_slots = random.choice([4, 5])
    targets = [random.choice([3, 4]) for _ in range(n_slots)]

    seeds: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()
    first = (random.randrange(gh), random.randrange(gw))
    seeds.append(first)
    occupied.add(first)
    for _ in range(1, n_slots):
        frontier = sorted(_neighbors_of_cells(occupied, gh, gw) - occupied)
        if not frontier:
            return None
        cell = random.choice(frontier)
        seeds.append(cell)
        occupied.add(cell)

    slots = [{seed} for seed in seeds]
    occupied = set(seeds)
    total_need = sum(targets) - n_slots
    for _ in range(total_need):
        unfinished = [i for i in range(n_slots) if len(slots[i]) < targets[i]]
        random.shuffle(unfinished)
        placed = False
        for i in unfinished:
            cand = sorted(_neighbors_of_cells(slots[i], gh, gw) - occupied)
            if not cand:
                continue
            cell = random.choice(cand)
            slots[i].add(cell)
            occupied.add(cell)
            placed = True
            break
        if not placed:
            return None

    return slots, gh, gw


def _render_layout(
    slots: list[set[tuple[int, int]]]
) -> tuple[np.ndarray, list[dict], dict[tuple[int, int], list[tuple[int, int]]]] | None:
    union = sorted(set().union(*slots))
    if not union:
        return None
    min_r = min(r for r, _ in union)
    max_r = max(r for r, _ in union)
    min_c = min(c for _, c in union)
    max_c = max(c for _, c in union)
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    shifted_slots: list[set[tuple[int, int]]] = []
    cell_owner: dict[tuple[int, int], int] = {}
    for idx, cells in enumerate(slots):
        shifted = {(r - min_r, c - min_c) for r, c in cells}
        shifted_slots.append(shifted)
        for cell in shifted:
            cell_owner[cell] = idx

    patch = np.full((2 * h + 1, 2 * w + 1), FRAME, dtype=int)
    slot_infos: list[dict] = []
    for idx, cells in enumerate(shifted_slots):
        pix = _slot_pixels_from_cells(cells)
        _paint_points(patch, pix, BG)
        coords = np.array(sorted(pix), dtype=int)
        mask, _r0, _c0 = _tight_mask(coords)
        slot_infos.append({"cells": cells, "pix": pix, "coords": coords, "mask": mask, "index": idx})

    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for (r, c), owner in cell_owner.items():
        for dr, dc in ((1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            other = cell_owner.get((nr, nc))
            if other is None or other == owner:
                continue
            a, b = sorted((owner, other))
            if dr == 1:
                wall = (2 * r + 2, 2 * c + 1)
            else:
                wall = (2 * r + 1, 2 * c + 2)
            edges[(a, b)].append(wall)

    if not edges:
        return None
    return patch, slot_infos, edges


def _build_shape_candidates(colors: list[int], slot_infos: list[dict]) -> list[list[int]]:
    candidates: list[list[int]] = []
    for slot in slot_infos:
        mask = slot["mask"]
        cand = [int(colors[j]) for j, other in enumerate(slot_infos) if _match_under_rotation(mask, other["mask"])]
        candidates.append(sorted(set(cand)))
    return candidates


def _choose_markers(
    slot_infos: list[dict], colors: list[int], edges: dict[tuple[int, int], list[tuple[int, int]]]
) -> list[tuple[int, int, int]]:
    candidates = _build_shape_candidates(colors, slot_infos)
    if len(_find_solutions(candidates, [], max_solutions=2)) <= 1:
        return []

    options = []
    for edge in edges:
        a, b = edge
        options.append((edge, int(colors[a])))
        options.append((edge, int(colors[b])))
    random.shuffle(options)

    chosen: list[tuple[int, int, int]] = []
    used_edges: set[tuple[int, int]] = set()
    cur_count = len(_find_solutions(candidates, chosen, max_solutions=200))
    while cur_count > 1:
        best_count = cur_count
        best_options: list[tuple[tuple[int, int], int, int]] = []
        for edge, col in options:
            if edge in used_edges:
                continue
            a, b = edge
            trial = chosen + [(int(a), int(b), int(col))]
            new_count = len(_find_solutions(candidates, trial, max_solutions=200))
            if new_count == 0 or new_count >= cur_count:
                continue
            if new_count < best_count:
                best_count = new_count
                best_options = [(edge, int(col), int(new_count))]
            elif new_count == best_count:
                best_options.append((edge, int(col), int(new_count)))
        if not best_options:
            return []
        edge, col, cur_count = random.choice(best_options)
        a, b = edge
        chosen.append((int(a), int(b), int(col)))
        used_edges.add(edge)
    return chosen


def _frame_position(canvas_h: int, canvas_w: int, frame_h: int, frame_w: int) -> tuple[int, int]:
    base_r = (canvas_h - frame_h) // 2
    base_c = (canvas_w - frame_w) // 2
    jitter_r = random.randint(-1, 1) if canvas_h - frame_h >= 4 else 0
    jitter_c = random.randint(-1, 1) if canvas_w - frame_w >= 4 else 0
    r0 = min(max(1, base_r + jitter_r), canvas_h - frame_h - 1)
    c0 = min(max(1, base_c + jitter_c), canvas_w - frame_w - 1)
    return int(r0), int(c0)


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> None:
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(blocked.shape[0], r0 + h + padding)
    cc1 = min(blocked.shape[1], c0 + w + padding)
    blocked[rr0:rr1, cc0:cc1] = True


def _random_place_patch(blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = blocked.shape
    ph, pw = patch.shape
    places: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                continue
            places.append((r0, c0))
    if not places:
        return None
    return random.choice(places)


def _make_piece(mask: np.ndarray, color: int) -> np.ndarray:
    rot = np.rot90(mask, random.randint(0, 3))
    patch = np.full(rot.shape, BG, dtype=int)
    patch[rot] = int(color)
    return patch


def _gen_attempt() -> np.ndarray | None:
    grown = _grow_slot_layout()
    if grown is None:
        return None
    slots, _gh, _gw = grown

    rendered = _render_layout(slots)
    if rendered is None:
        return None
    frame_patch, slot_infos, edges = rendered

    keys = [_canonical_key(info["mask"]) for info in slot_infos]
    counts = Counter(keys)
    if max(counts.values()) < 2:
        return None

    palette = [c for c in range(1, 10) if c != FRAME]
    if len(palette) < len(slot_infos):
        return None
    colors = random.sample(palette, len(slot_infos))

    markers = _choose_markers(slot_infos, colors, edges)
    if not markers:
        return None

    shape_only = _find_solutions(_build_shape_candidates(colors, slot_infos), [], max_solutions=2)
    if len(shape_only) <= 1:
        return None

    constrained = _find_solutions(_build_shape_candidates(colors, slot_infos), markers, max_solutions=2)
    if len(constrained) != 1:
        return None
    expected_assignment = [int(col) for col in colors]
    if constrained[0] != expected_assignment:
        return None

    input_frame = frame_patch.copy()
    output_frame = frame_patch.copy()
    for idx, info in enumerate(slot_infos):
        _paint_points(output_frame, info["pix"], colors[idx])
    for a, b, col in markers:
        pos = random.choice(edges[tuple(sorted((a, b)))])
        input_frame[pos] = int(col)

    frame_h, frame_w = frame_patch.shape
    piece_patches = [_make_piece(info["mask"], colors[idx]) for idx, info in enumerate(slot_infos)]
    piece_area = sum(int((patch != BG).sum()) for patch in piece_patches)
    canvas_h = frame_h + random.randint(8, 13)
    canvas_w = frame_w + random.randint(8, 13)
    while canvas_h * canvas_w - frame_h * frame_w < piece_area * 7:
        if random.random() < 0.5:
            canvas_h += 1
        else:
            canvas_w += 1

    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    out = np.full((canvas_h, canvas_w), BG, dtype=int)
    fr0, fc0 = _frame_position(canvas_h, canvas_w, frame_h, frame_w)
    _paste(inp, input_frame, fr0, fc0, transparent=BG)
    _paste(out, output_frame, fr0, fc0, transparent=BG)

    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)
    _mark_blocked(blocked, fr0, fc0, frame_h, frame_w, padding=1)

    order = list(range(len(piece_patches)))
    random.shuffle(order)
    for idx in order:
        patch = piece_patches[idx]
        pos = _random_place_patch(blocked, patch)
        if pos is None:
            return None
        r0, c0 = pos
        _paste(inp, patch, r0, c0, transparent=BG)
        _mark_blocked(blocked, r0, c0, patch.shape[0], patch.shape[1], padding=1)

    if _most_common_color(inp) != BG:
        return None
    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
