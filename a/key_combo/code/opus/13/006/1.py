from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
ARROW_COLOR = 5
BRIDGE_COLOR = 8
RECT_COLORS = [1, 2, 3, 4, 6, 7, 9]

Rect = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> Rect:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _shift_rect(rect: Rect, dr: int, dc: int) -> Rect:
    r0, c0, r1, c1 = rect
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _transpose_rect(rect: Rect) -> Rect:
    r0, c0, r1, c1 = rect
    return c0, r0, c1, r1


def _paint_rect(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _rectangles_by_color(grid: np.ndarray) -> dict[int, list[Rect]]:
    rectangles: dict[int, list[Rect]] = defaultdict(list)
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, ARROW_COLOR)):
        for coords in _components(grid == color):
            rect = _bbox(coords)
            r0, c0, r1, c1 = rect
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert coords.shape[0] == h * w, "each foreground component must be a solid rectangle"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
            rectangles[color].append(rect)
    return dict(rectangles)


def _candidate_gap(grid: np.ndarray, a: Rect, b: Rect) -> dict | None:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    candidates: list[dict] = []

    shared_r0 = max(ar0, br0)
    shared_r1 = min(ar1, br1)
    shared_c0 = max(ac0, bc0)
    shared_c1 = min(ac1, bc1)
    assert shared_r1 < shared_r0 or shared_c1 < shared_c0, "rectangles may not overlap"

    if shared_r1 - shared_r0 + 1 >= 3 and (ac1 < bc0 or bc1 < ac0):
        if ac1 < bc0:
            gap_c0, gap_c1 = ac1 + 1, bc0 - 1
        else:
            gap_c0, gap_c1 = bc1 + 1, ac0 - 1
        if gap_c0 <= gap_c1 and np.all(grid[shared_r0 : shared_r1 + 1, gap_c0 : gap_c1 + 1] == BG):
            candidates.append(
                {
                    "orientation": "H",
                    "length": gap_c1 - gap_c0 + 1,
                    "raw": (shared_r0, gap_c0, shared_r1, gap_c1),
                    "paint": (shared_r0 + 1, gap_c0, shared_r1 - 1, gap_c1),
                }
            )

    if shared_c1 - shared_c0 + 1 >= 3 and (ar1 < br0 or br1 < ar0):
        if ar1 < br0:
            gap_r0, gap_r1 = ar1 + 1, br0 - 1
        else:
            gap_r0, gap_r1 = br1 + 1, ar0 - 1
        if gap_r0 <= gap_r1 and np.all(grid[gap_r0 : gap_r1 + 1, shared_c0 : shared_c1 + 1] == BG):
            candidates.append(
                {
                    "orientation": "V",
                    "length": gap_r1 - gap_r0 + 1,
                    "raw": (gap_r0, shared_c0, gap_r1, shared_c1),
                    "paint": (gap_r0, shared_c0 + 1, gap_r1, shared_c1 - 1),
                }
            )

    assert len(candidates) <= 1, "a rectangle pair should face in at most one direction"
    return candidates[0] if candidates else None


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for coords in _components(grid == ARROW_COLOR):
        rect = _bbox(coords)
        r0, c0, r1, c1 = rect
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "grey object must be a 2x2 arrow sprite"
        assert coords.shape[0] == 3, "grey arrow sprite must have exactly one missing corner"
        sprite = np.zeros((2, 2), dtype=bool)
        for rr, cc in coords:
            sprite[int(rr) - r0, int(cc) - c0] = True
        missing = [(rr, cc) for rr in range(2) for cc in range(2) if not sprite[rr, cc]]
        assert len(missing) == 1
        direction = {
            (0, 1): "right",
            (0, 0): "up",
            (1, 1): "down",
            (1, 0): "left",
        }[missing[0]]
        arrows.append({"bbox": rect, "direction": direction})
    return arrows


def _ray_hits_bbox(arrow: dict, box: Rect) -> bool:
    r0, c0, _r1, _c1 = arrow["bbox"]
    br0, bc0, br1, bc1 = box
    direction = arrow["direction"]
    if direction == "right":
        return br0 <= r0 + 1 and br1 >= r0 and bc1 >= c0 + 2
    if direction == "left":
        return br0 <= r0 + 1 and br1 >= r0 and bc0 <= c0 - 1
    if direction == "down":
        return bc0 <= c0 + 1 and bc1 >= c0 and br1 >= r0 + 2
    return bc0 <= c0 + 1 and bc1 >= c0 and br0 <= r0 - 1


def _collect_candidates(grid: np.ndarray) -> tuple[dict[int, list[Rect]], dict[int, list[dict]]]:
    rectangles_by_color = _rectangles_by_color(grid)
    candidates_by_color: dict[int, list[dict]] = defaultdict(list)
    for color, rects in rectangles_by_color.items():
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                gap = _candidate_gap(grid, rects[i], rects[j])
                if gap is not None:
                    candidates_by_color[color].append(gap)
    return rectangles_by_color, dict(candidates_by_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter same-color rectangle gaps with invisible arrow rays, then bridge only the shortest activated gap per color.

    Input structure:
    - The grid has black background, 3-cell grey arrow sprites shaped as 2x2 blocks with one missing corner, and several same-color solid rectangles.
    - Rectangles of one color may form one or more horizontal or vertical empty gaps whose shared span is at least three cells.
    - Grey appears only in the arrow sprites; teal is reserved for the bridge added in the output.

    Rule:
    - Infer each arrow direction from its missing corner: top-right means right, top-left means up, bottom-right means down, and bottom-left means left.
    - Treat each arrow as emitting an invisible straight ray to the grid edge; a rectangle gap is activated if any ray passes through at least one raw gap cell.
    - For each color independently, choose the shortest activated gap, paint that gap teal using the shared span eroded by one cell on both ends, and leave colors with no activated gaps unchanged.
    - Finally erase every grey arrow pixel to black while preserving all rectangles and untouched background.

    Key insight:
    - The arrows do not draw visible output; they only decide which rectangle gaps are eligible before the shortest-gap rule is applied inside each color group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    arrows = _find_arrows(g)
    _rectangles, candidates_by_color = _collect_candidates(g)

    out = g.copy()
    out[out == ARROW_COLOR] = BG

    for color, candidates in candidates_by_color.items():
        activated = [cand for cand in candidates if any(_ray_hits_bbox(arrow, cand["raw"]) for arrow in arrows)]
        if not activated:
            continue
        chosen = min(activated, key=lambda cand: (cand["length"], cand["raw"], cand["paint"]))
        r0, c0, r1, c1 = chosen["paint"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BRIDGE_COLOR

    return out


def _transpose_module(module: dict) -> dict:
    out = {
        "color": module["color"],
        "special": module["special"],
        "rects": [_transpose_rect(rect) for rect in module["rects"]],
        "candidates": [],
    }
    for cand in module["candidates"]:
        out["candidates"].append(
            {
                "name": cand["name"],
                "orientation": "V" if cand["orientation"] == "H" else "H",
                "length": cand["length"],
                "raw": _transpose_rect(cand["raw"]),
                "paint": _transpose_rect(cand["paint"]),
            }
        )
    return out


def _make_special_module(color: int) -> dict | None:
    top_h = random.randint(3, 4)
    mid_h = random.randint(3, 4)
    bot_h = random.randint(3, 4)
    sep1 = random.randint(1, 2)
    sep2 = random.randint(1, 2)
    hub_h = top_h + sep1 + mid_h + sep2 + bot_h
    hub_w = random.randint(3, 5)
    left_space = random.randint(6, 8)
    right_gap = random.randint(1, 4)
    mid_w = random.randint(3, 5)

    for _ in range(60):
        top_w = random.randint(3, 5)
        bot_w = random.randint(3, 5)
        top_c0 = random.randint(0, left_space - top_w - 1)
        bot_c0 = random.randint(0, left_space - bot_w - 1)
        if _overlap_len(top_c0, top_c0 + top_w - 1, bot_c0, bot_c0 + bot_w - 1) >= 3:
            continue

        hub_c0 = left_space
        hub_c1 = hub_c0 + hub_w - 1
        mid_c0 = hub_c1 + 1 + right_gap
        top_r0 = 0
        mid_r0 = top_h + sep1
        bot_r0 = hub_h - bot_h

        rect_top = (top_r0, top_c0, top_r0 + top_h - 1, top_c0 + top_w - 1)
        rect_hub = (0, hub_c0, hub_h - 1, hub_c1)
        rect_mid = (mid_r0, mid_c0, mid_r0 + mid_h - 1, mid_c0 + mid_w - 1)
        rect_bot = (bot_r0, bot_c0, bot_r0 + bot_h - 1, bot_c0 + bot_w - 1)

        top_gap = hub_c0 - rect_top[3] - 1
        mid_gap = rect_mid[1] - hub_c1 - 1
        bot_gap = hub_c0 - rect_bot[3] - 1
        if min(top_gap, mid_gap, bot_gap) < 1:
            continue

        module = {
            "color": color,
            "special": True,
            "rects": [rect_top, rect_hub, rect_mid, rect_bot],
            "candidates": [
                {
                    "name": "top",
                    "orientation": "H",
                    "length": top_gap,
                    "raw": (rect_top[0], rect_top[3] + 1, rect_top[2], hub_c0 - 1),
                    "paint": (rect_top[0] + 1, rect_top[3] + 1, rect_top[2] - 1, hub_c0 - 1),
                },
                {
                    "name": "mid",
                    "orientation": "H",
                    "length": mid_gap,
                    "raw": (rect_mid[0], hub_c1 + 1, rect_mid[2], rect_mid[1] - 1),
                    "paint": (rect_mid[0] + 1, hub_c1 + 1, rect_mid[2] - 1, rect_mid[1] - 1),
                },
                {
                    "name": "bot",
                    "orientation": "H",
                    "length": bot_gap,
                    "raw": (rect_bot[0], rect_bot[3] + 1, rect_bot[2], hub_c0 - 1),
                    "paint": (rect_bot[0] + 1, rect_bot[3] + 1, rect_bot[2] - 1, hub_c0 - 1),
                },
            ],
        }
        if len({cand["length"] for cand in module["candidates"]}) >= 2:
            if random.random() < 0.5:
                return module
            return _transpose_module(module)
    return None


def _make_pair_module(color: int) -> dict:
    h1 = random.randint(3, 5)
    h2 = random.randint(3, 5)
    w1 = random.randint(3, 5)
    w2 = random.randint(3, 5)
    gap = random.randint(1, 4)
    total_h = max(h1, h2) + 2
    r1 = random.randint(0, total_h - h1)
    r2 = random.randint(0, total_h - h2)
    while _overlap_len(r1, r1 + h1 - 1, r2, r2 + h2 - 1) < 3:
        r1 = random.randint(0, total_h - h1)
        r2 = random.randint(0, total_h - h2)

    rect_a = (r1, 0, r1 + h1 - 1, w1 - 1)
    rect_b = (r2, w1 + gap, r2 + h2 - 1, w1 + gap + w2 - 1)
    shared_r0 = max(rect_a[0], rect_b[0])
    shared_r1 = min(rect_a[2], rect_b[2])
    module = {
        "color": color,
        "special": False,
        "rects": [rect_a, rect_b],
        "candidates": [
            {
                "name": "pair",
                "orientation": "H",
                "length": gap,
                "raw": (shared_r0, rect_a[3] + 1, shared_r1, rect_b[1] - 1),
                "paint": (shared_r0 + 1, rect_a[3] + 1, shared_r1 - 1, rect_b[1] - 1),
            }
        ],
    }
    if random.random() < 0.5:
        return module
    return _transpose_module(module)


def _module_bbox(module: dict) -> Rect:
    r0 = min(rect[0] for rect in module["rects"])
    c0 = min(rect[1] for rect in module["rects"])
    r1 = max(rect[2] for rect in module["rects"])
    c1 = max(rect[3] for rect in module["rects"])
    return r0, c0, r1, c1


def _module_size(module: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = _module_bbox(module)
    return r1 - r0 + 1, c1 - c0 + 1


def _bboxes_far_enough(a: Rect, b: Rect, pad: int) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _place_modules(modules: list[dict], height: int, width: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(modules)), key=lambda idx: _module_size(modules[idx])[0] * _module_size(modules[idx])[1], reverse=True)
    placed: dict[int, tuple[int, int]] = {}
    boxes: dict[int, Rect] = {}

    def search(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        h, w = _module_size(modules[idx])
        positions = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        random.shuffle(positions)
        for r, c in positions:
            box = (r, c, r + h - 1, c + w - 1)
            if any(not _bboxes_far_enough(box, other, pad=2) for other in boxes.values()):
                continue
            placed[idx] = (r, c)
            boxes[idx] = box
            if search(k + 1):
                return True
            del placed[idx]
            del boxes[idx]
        return False

    if not search(0):
        return None
    return [placed[idx] for idx in range(len(modules))]


def _shift_module(module: dict, dr: int, dc: int) -> dict:
    out = {"color": module["color"], "special": module["special"], "rects": [], "candidates": []}
    out["rects"] = [_shift_rect(rect, dr, dc) for rect in module["rects"]]
    for cand in module["candidates"]:
        out["candidates"].append(
            {
                "name": cand["name"],
                "orientation": cand["orientation"],
                "length": cand["length"],
                "raw": _shift_rect(cand["raw"], dr, dc),
                "paint": _shift_rect(cand["paint"], dr, dc),
            }
        )
    return out


def _arrow_sprite(direction: str) -> np.ndarray:
    sprite = np.full((2, 2), ARROW_COLOR, dtype=int)
    missing = {
        "right": (0, 1),
        "up": (0, 0),
        "down": (1, 1),
        "left": (1, 0),
    }[direction]
    sprite[missing] = BG
    return sprite


def _clear_with_padding(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not np.any(occupied[rr0:rr1, cc0:cc1])


def _arrow_hits_candidate(arrow: dict, cand: dict) -> bool:
    return _ray_hits_bbox({"bbox": arrow["bbox"], "direction": arrow["direction"]}, cand["raw"])


def _arrow_line_id(direction: str, bbox: Rect) -> tuple[str, int]:
    r0, c0, _r1, _c1 = bbox
    return ("H", r0) if direction in ("left", "right") else ("V", c0)


def _enumerate_arrow_options(
    occupied: np.ndarray,
    target_id: tuple[int, int],
    target: dict,
    all_candidates: dict[tuple[int, int], dict],
    forbidden: set[tuple[int, int]],
    used_lines: set[tuple[str, int]],
) -> list[dict]:
    h, w = occupied.shape
    raw_r0, raw_c0, raw_r1, raw_c1 = target["raw"]
    options: list[dict] = []

    def consider(direction: str, r0: int, c0: int) -> None:
        bbox = (r0, c0, r0 + 1, c0 + 1)
        if r0 < 0 or c0 < 0 or r0 + 1 >= h or c0 + 1 >= w:
            return
        line_id = _arrow_line_id(direction, bbox)
        if line_id in used_lines:
            return
        if not _clear_with_padding(occupied, r0, c0, 2, 2, pad=1):
            return
        arrow = {"bbox": bbox, "direction": direction}
        hit_set = {cid for cid, cand in all_candidates.items() if _arrow_hits_candidate(arrow, cand)}
        if target_id not in hit_set:
            return
        if hit_set & forbidden:
            return
        options.append(
            {
                "direction": direction,
                "bbox": bbox,
                "hit_set": hit_set,
                "line_id": line_id,
            }
        )

    if target["orientation"] == "H":
        rows = list(range(raw_r0, raw_r1))
        random.shuffle(rows)
        for r0 in rows:
            for c0 in range(raw_c0 - 2, -1, -1):
                consider("right", r0, c0)
            for c0 in range(raw_c1 + 1, w - 1):
                consider("left", r0, c0)
    else:
        cols = list(range(raw_c0, raw_c1))
        random.shuffle(cols)
        for c0 in cols:
            for r0 in range(raw_r0 - 2, -1, -1):
                consider("down", r0, c0)
            for r0 in range(raw_r1 + 1, h - 1):
                consider("up", r0, c0)

    options.sort(key=lambda opt: (len(opt["hit_set"]), opt["bbox"]))
    return options


def _place_target_arrows(
    grid: np.ndarray,
    target_ids: list[tuple[int, int]],
    all_candidates: dict[tuple[int, int], dict],
) -> list[dict] | None:
    occupied = grid != BG
    forbidden = set(all_candidates) - set(target_ids)
    placed: list[dict] = []
    used_lines: set[tuple[str, int]] = set()

    ordered_targets = sorted(target_ids, key=lambda cid: (all_candidates[cid]["orientation"], all_candidates[cid]["raw"]))
    for target_id in ordered_targets:
        options = _enumerate_arrow_options(occupied, target_id, all_candidates[target_id], all_candidates, forbidden, used_lines)
        options = [opt for opt in options if opt["hit_set"] <= set(target_ids)]
        if not options:
            return None
        best_size = min(len(opt["hit_set"]) for opt in options)
        best = [opt for opt in options if len(opt["hit_set"]) == best_size]
        choice = random.choice(best)
        sprite = _arrow_sprite(choice["direction"])
        r0, c0, _r1, _c1 = choice["bbox"]
        mask = sprite == ARROW_COLOR
        grid[r0 : r0 + 2, c0 : c0 + 2][mask] = ARROW_COLOR
        occupied[r0 : r0 + 2, c0 : c0 + 2] |= mask
        placed.append(choice)
        used_lines.add(choice["line_id"])

    union_hits = set().union(*(arrow["hit_set"] for arrow in placed))
    if union_hits != set(target_ids):
        return None
    return placed


def _scene_metadata(modules: list[dict]) -> dict[tuple[int, int], dict]:
    all_candidates: dict[tuple[int, int], dict] = {}
    for mi, module in enumerate(modules):
        for ci, cand in enumerate(module["candidates"]):
            entry = dict(cand)
            entry["color"] = module["color"]
            entry["module_index"] = mi
            entry["special"] = module["special"]
            all_candidates[(mi, ci)] = entry
    return all_candidates


def _activated_candidate_ids(grid: np.ndarray) -> set[tuple[int, int, int]]:
    arrows = _find_arrows(grid)
    activated: set[tuple[int, int, int]] = set()
    for color, candidates in _collect_candidates(grid)[1].items():
        for ci, cand in enumerate(candidates):
            if any(_ray_hits_bbox(arrow, cand["raw"]) for arrow in arrows):
                activated.add((color, ci, cand["length"]))
    return activated


def _attempt_generate() -> np.ndarray | None:
    n_groups = random.choice([3, 4])
    colors = random.sample(RECT_COLORS, n_groups)
    special = _make_special_module(colors[0])
    if special is None:
        return None
    modules = [special] + [_make_pair_module(color) for color in colors[1:]]

    height = random.randint(18, 24)
    width = random.randint(20, 26)
    placements = _place_modules(modules, height, width)
    if placements is None:
        return None

    shifted_modules = [_shift_module(module, pos[0], pos[1]) for module, pos in zip(modules, placements)]
    grid = np.zeros((height, width), dtype=int)
    for module in shifted_modules:
        for rect in module["rects"]:
            _paint_rect(grid, rect, module["color"])

    all_candidates = _scene_metadata(shifted_modules)
    special_ids = [cid for cid, cand in all_candidates.items() if cand["special"]]
    distinct_pairs = [
        pair
        for pair in ((special_ids[0], special_ids[1]), (special_ids[0], special_ids[2]), (special_ids[1], special_ids[2]))
        if all_candidates[pair[0]]["length"] != all_candidates[pair[1]]["length"]
    ]
    if not distinct_pairs:
        return None
    target_ids = list(random.choice(distinct_pairs))

    simple_ids = [cid for cid, cand in all_candidates.items() if not cand["special"]]
    if simple_ids and random.random() < 0.55:
        random.shuffle(simple_ids)
        for cid in simple_ids:
            trial_grid = grid.copy()
            arrows = _place_target_arrows(trial_grid, target_ids + [cid], all_candidates)
            if arrows is not None:
                grid = trial_grid
                target_ids = target_ids + [cid]
                break
        else:
            arrows = _place_target_arrows(grid, target_ids, all_candidates)
            if arrows is None:
                return None
    else:
        arrows = _place_target_arrows(grid, target_ids, all_candidates)
        if arrows is None:
            return None

    if len(arrows) not in (2, 3):
        return None
    if len({_arrow_line_id(arrow["direction"], arrow["bbox"]) for arrow in arrows}) != len(arrows):
        return None
    if not all(arrow["hit_set"] for arrow in arrows):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    actual_rectangles, actual_candidates = _collect_candidates(grid)
    if len(actual_rectangles) != n_groups:
        return None
    if len(actual_rectangles[colors[0]]) < 4:
        return None

    actual_activated_by_color: dict[int, list[dict]] = defaultdict(list)
    arrows_found = _find_arrows(grid)
    for color, candidates in actual_candidates.items():
        for cand in candidates:
            if any(_ray_hits_bbox(arrow, cand["raw"]) for arrow in arrows_found):
                actual_activated_by_color[color].append(cand)

    special_activated = actual_activated_by_color[colors[0]]
    if len(actual_candidates[colors[0]]) < 3 or len(special_activated) != 2:
        return None

    chosen_special = min(special_activated, key=lambda cand: (cand["length"], cand["raw"], cand["paint"]))
    expected = grid.copy()
    expected[expected == ARROW_COLOR] = BG
    for color, candidates in actual_candidates.items():
        activated = actual_activated_by_color.get(color, [])
        if not activated:
            continue
        chosen = min(activated, key=lambda cand: (cand["length"], cand["raw"], cand["paint"]))
        r0, c0, r1, c1 = chosen["paint"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = BRIDGE_COLOR

    if not np.array_equal(out, expected):
        return None

    sr0, sc0, sr1, sc1 = chosen_special["paint"]
    if not np.all(out[sr0 : sr1 + 1, sc0 : sc1 + 1] == BRIDGE_COLOR):
        return None
    if np.count_nonzero(grid == ARROW_COLOR) != 3 * len(arrows):
        return None
    if np.count_nonzero(out == BRIDGE_COLOR) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
