# Combo variant of edb79dae (index 11)
# combo_spec: D10/1+D12/1
# Name: 迭代放置+最近匹配

from collections import deque
import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# - Concrete behavior:
#   The frame still contains solid NxN placeholder blocks on a regular lattice, the outside still
#   contains one sprite template and one two-color legend per placeholder color, and legends still
#   define the recoloring. The fusion change is that outside sprites are no longer used in parallel.
#   They are processed one by one in visible reading order, and each sprite chooses only one target
#   block of its own color. "Nearest" is not a fixed geometric lookup anymore: it is recomputed after
#   every placement on the graph formed by the filled placeholder slots. A slot already used by an
#   earlier sprite becomes blocked and cannot be traversed by later searches.
# - Small trace:
#   Consider a 3x4 frame grid whose top row contains three connected blocks that act like entrances
#   from the sprites placed above the frame. Suppose blue is the first top sprite and green is the
#   second. Blue has two blue targets and chooses the nearer one through the current slot graph. That
#   chosen blue slot is then marked occupied. When green is processed, its static nearest green target
#   might have been reachable through the blue slot, but that path is now severed, so green switches
#   to its other same-color target. In the source task every blue and green placeholder would be
#   filled; in D12/1 only one per color survives but choices are still independent; in this combo the
#   later green choice depends on the earlier blue placement.
# - Complexity calibration:
#   To keep the combo visible, the generator forces repeated placeholder colors, keeps the filled-slot
#   lattice connected, and rejects samples unless at least one later color chooses a different target
#   under dynamic blocking than it would under static nearest matching. The outside sprites are placed
#   in a clean left-to-right row above the frame so humans can see the processing order directly.
# - Hidden assumptions from the original main():
#   1. Legend parsing is still valid because the glyph shape is unchanged.
#   2. Sprite extraction is still valid because there is still exactly one non-legend sprite per
#      placeholder color and it still shares the placeholder bbox size.
#   3. The old final loop assumed "every placeholder of color c gets stamped". That fails here; we
#      must group by color, sort sprites by visible order, keep mutable occupancy state, and stamp
#      only one target per color.
#   4. The original task did not need a lattice model. This combo adds one: all placeholder blocks
#      must lie on one regular NxN-with-1-gap grid so we can infer top-edge entry columns and slot
#      adjacency. The generator keeps the original global-N construction, so this assumption holds.
# - Placement legality:
#   Generation is output-first. It first builds a connected set of filled placeholder slots, then
#   searches for a color assignment and top-sprite ordering whose dynamic selection is well-defined
#   and non-degenerate. The deterministic canvas layout keeps the top sprite row, the frame, and the
#   right-side legends separated by background gaps, so nothing overlaps or leaves the canvas.
# - Visual observability:
#   No hidden color-id ordering is used. The order cue is spatial: outside sprites are read left to
#   right. The distance cue is structural: a later color searches through adjacent filled placeholder
#   slots, and earlier chosen slots visibly become occupied blockers in the output. Samples are
#   accepted only when this stateful blocking changes at least one later choice.
# - Visual cleanliness:
#   Sprites form one top row, the frame is a single central object, and legends are stacked on the
#   right. This avoids the clutter that would come from mixing both order markers and legends in the
#   same outside region. Inside the frame, the regular lattice and low empty-slot count keep the path
#   structure readable.
# - Color mapping inference:
#   Recoloring still comes entirely from the legend glyphs. The new reasoning burden is separate and
#   visible: choose the only reachable/nearest same-color block at that time in the placement order.
# - New solver assumptions reviewed:
#   The solver now assumes the filled placeholder-slot graph is connected, each placeholder color has
#   at least two candidate slots, each sprite lies outside the frame, and dynamic target selection has
#   a deterministic best choice. The generator explicitly enforces each of these conditions.


FRAME_COL = 5


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    frame_col, bbox = candidates[0]
    return int(frame_col), bbox


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _bbox_center2(b: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = b
    return r0 + r1, c0 + c1


def _side_of_sprite(
    sprite_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]
) -> str:
    sr0, sc0, sr1, sc1 = sprite_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if sr1 < fr0:
        return "top"
    if sr0 > fr1:
        return "bottom"
    if sc1 < fc0:
        return "left"
    if sc0 > fc1:
        return "right"
    gaps = {
        "top": abs(sr1 - fr0),
        "bottom": abs(sr0 - fr1),
        "left": abs(sc1 - fc0),
        "right": abs(sc0 - fc1),
    }
    return min(gaps.items(), key=lambda item: (item[1], item[0]))[0]


def _nearest_rank(center2_values: list[int], target2: int) -> int:
    scored = [(abs(v - target2), idx) for idx, v in enumerate(center2_values)]
    scored.sort()
    return scored[0][1]


def _slot_neighbors(slot: tuple[int, int], filled_slots: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = slot
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (r + dr, c + dc)
        if nxt in filled_slots:
            out.append(nxt)
    return out


def _slot_distance(
    start: tuple[int, int],
    goal: tuple[int, int],
    filled_slots: set[tuple[int, int]],
    blocked_slots: set[tuple[int, int]],
) -> int | None:
    allowed = filled_slots - blocked_slots
    if start not in allowed or goal not in allowed:
        return None
    q = deque([(start, 0)])
    seen = {start}
    while q:
        cur, dist = q.popleft()
        if cur == goal:
            return dist
        for nxt in _slot_neighbors(cur, allowed):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, dist + 1))
    return None


def _boundary_slots(side: str, slots: set[tuple[int, int]], rows: int, cols: int) -> list[tuple[int, int]]:
    if side == "top":
        return sorted(slot for slot in slots if slot[0] == 0)
    if side == "bottom":
        return sorted(slot for slot in slots if slot[0] == rows - 1)
    if side == "left":
        return sorted(slot for slot in slots if slot[1] == 0)
    if side == "right":
        return sorted(slot for slot in slots if slot[1] == cols - 1)
    raise ValueError(f"unknown side: {side}")


def _boundary_rank_delta(side: str, boundary_slot: tuple[int, int], entry_rank: int) -> int:
    if side in ("top", "bottom"):
        return abs(boundary_slot[1] - entry_rank)
    return abs(boundary_slot[0] - entry_rank)


def _build_placeholder_lattice(
    placeholders_by_color: dict[int, list[dict[str, object]]]
) -> tuple[list[dict[str, object]], list[int], list[int], int, int, int, int]:
    entries = [entry for items in placeholders_by_color.values() for entry in items]
    heights = {int(entry["rel_bbox"][2]) - int(entry["rel_bbox"][0]) + 1 for entry in entries}
    widths = {int(entry["rel_bbox"][3]) - int(entry["rel_bbox"][1]) + 1 for entry in entries}
    assert len(heights) == 1 and len(widths) == 1, "combo expects one global placeholder size"
    slot_h = next(iter(heights))
    slot_w = next(iter(widths))
    row_tops = sorted({int(entry["rel_bbox"][0]) for entry in entries})
    col_lefts = sorted({int(entry["rel_bbox"][1]) for entry in entries})
    rows = len(row_tops)
    cols = len(col_lefts)
    row_index = {r: i for i, r in enumerate(row_tops)}
    col_index = {c: i for i, c in enumerate(col_lefts)}
    for entry in entries:
        r0, c0, r1, c1 = entry["rel_bbox"]
        entry["slot"] = (row_index[int(r0)], col_index[int(c0)])
        entry["center2"] = (int(r0) + int(r1), int(c0) + int(c1))
    row_centers2 = [2 * r + slot_h - 1 for r in row_tops]
    col_centers2 = [2 * c + slot_w - 1 for c in col_lefts]
    return entries, row_centers2, col_centers2, rows, cols, slot_h, slot_w


def _entry_rank_for_sprite(
    side: str,
    sprite_bbox: tuple[int, int, int, int],
    row_centers2: list[int],
    col_centers2: list[int],
) -> int:
    sr2, sc2 = _bbox_center2(sprite_bbox)
    if side in ("top", "bottom"):
        return _nearest_rank(col_centers2, sc2)
    return _nearest_rank(row_centers2, sr2)


def _choose_target(
    color_entries: list[dict[str, object]],
    all_entries: list[dict[str, object]],
    side: str,
    entry_rank: int,
    rows: int,
    cols: int,
    occupied_slots: set[tuple[int, int]],
) -> tuple[dict[str, object], int, list[tuple[int, tuple[int, int]]]]:
    filled_slots = {tuple(entry["slot"]) for entry in all_entries}
    start_slots = _boundary_slots(side, filled_slots - occupied_slots, rows, cols)
    assert start_slots, f"no free boundary slots available for side {side}"
    scored: list[tuple[int, int, int, dict[str, object]]] = []
    score_pairs: list[tuple[int, tuple[int, int]]] = []
    for entry in color_entries:
        slot = tuple(entry["slot"])
        best = None
        for start in start_slots:
            dist = _slot_distance(start, slot, filled_slots, occupied_slots)
            if dist is None:
                continue
            total = _boundary_rank_delta(side, start, entry_rank) + dist
            pair = (total, start[0], start[1])
            if best is None or pair < best:
                best = pair
        assert best is not None, f"no reachable target for color {entry['color']}"
        rr0, cc0, _, _ = entry["rel_bbox"]
        scored.append((best[0], int(rr0), int(cc0), entry))
        score_pairs.append((best[0], slot))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    score_pairs.sort()
    if len(score_pairs) >= 2:
        assert score_pairs[0][0] < score_pairs[1][0], f"ambiguous nearest target for color {color_entries[0]['color']}"
    return scored[0][3], scored[0][0], score_pairs


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_col, frame_bbox = _find_frame(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"

    placeholders_by_color: dict[int, list[dict[str, object]]] = {col: [] for col in sorted(placeholder_colors)}
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            placeholders_by_color[col].append(
                {
                    "color": int(col),
                    "rel_bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "abs_bbox": (fr0 + int(r0), fc0 + int(c0), fr0 + int(r1), fc0 + int(c1)),
                }
            )
        assert len(placeholders_by_color[col]) >= 2, f"combo expects repeated placeholders for color {col}"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": len(pts_tuples),
                }
            )
            cid += 1

    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors:
                continue
            if len(sub_colors) != 2:
                continue
            cset = set(sub_colors)
            if ci["col"] not in cset or cj["col"] not in cset:
                continue

            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]

            assert src in placeholder_colors, f"legend source color {src} is not a placeholder"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])
    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )

    entries, row_centers2, col_centers2, rows, cols, slot_h, slot_w = _build_placeholder_lattice(placeholders_by_color)
    filled_slots = {tuple(entry["slot"]) for entry in entries}
    assert len(filled_slots) == len(entries), "placeholder slots must be unique"
    start = next(iter(filled_slots))
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nxt in _slot_neighbors(cur, filled_slots):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert seen == filled_slots, "combo expects a connected filled-slot graph"

    sprite_data: dict[int, dict[str, object]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1) == slot_h and (c["bbox"][3] - c["bbox"][1] + 1) == slot_w
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(c["pts"])
        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_data[src_col] = {
            "coords": coords,
            "shape": (slot_h, slot_w),
            "bbox": (sr0, sc0, sr1, sc1),
        }

    return {
        "grid": g,
        "bg": int(bg),
        "frame_col": int(frame_col),
        "frame_bbox": frame_bbox,
        "crop": crop,
        "placeholders_by_color": placeholders_by_color,
        "entries": entries,
        "row_centers2": row_centers2,
        "col_centers2": col_centers2,
        "rows": rows,
        "cols": cols,
        "slot_h": slot_h,
        "slot_w": slot_w,
        "sprite_data": sprite_data,
        "color_map": color_map,
    }


def _compute_selection(scene: dict[str, object], *, dynamic_blocking: bool) -> dict[str, object]:
    placeholders_by_color = scene["placeholders_by_color"]
    entries = scene["entries"]
    rows = int(scene["rows"])
    cols = int(scene["cols"])
    frame_bbox = scene["frame_bbox"]
    row_centers2 = scene["row_centers2"]
    col_centers2 = scene["col_centers2"]
    sprite_data = scene["sprite_data"]

    sprite_order = sorted(
        ((src_col, data) for src_col, data in sprite_data.items()),
        key=lambda item: (item[1]["bbox"][0], item[1]["bbox"][1], item[0]),
    )
    occupied_slots: set[tuple[int, int]] = set()
    chosen_by_color: dict[int, dict[str, object]] = {}
    costs_by_color: dict[int, int] = {}
    ranked_scores: dict[int, list[tuple[int, tuple[int, int]]]] = {}

    for src_col, data in sprite_order:
        side = _side_of_sprite(data["bbox"], frame_bbox)
        entry_rank = _entry_rank_for_sprite(side, data["bbox"], row_centers2, col_centers2)
        chosen, cost, score_pairs = _choose_target(
            placeholders_by_color[src_col],
            entries,
            side,
            entry_rank,
            rows,
            cols,
            occupied_slots if dynamic_blocking else set(),
        )
        chosen_by_color[src_col] = chosen
        costs_by_color[src_col] = cost
        ranked_scores[src_col] = score_pairs
        if dynamic_blocking:
            occupied_slots.add(tuple(chosen["slot"]))

    return {
        "order": [src_col for src_col, _ in sprite_order],
        "chosen_by_color": chosen_by_color,
        "costs_by_color": costs_by_color,
        "ranked_scores": ranked_scores,
    }


def _combo_metrics(grid: np.ndarray) -> dict[str, object]:
    scene = _parse_scene(grid)
    dynamic = _compute_selection(scene, dynamic_blocking=True)
    static = _compute_selection(scene, dynamic_blocking=False)
    changed_colors = [
        col
        for col in dynamic["order"][1:]
        if tuple(dynamic["chosen_by_color"][col]["slot"]) != tuple(static["chosen_by_color"][col]["slot"])
    ]
    return {
        "order": dynamic["order"],
        "changed_colors": changed_colors,
        "changed_count": len(changed_colors),
        "dynamic_slots": {col: tuple(info["slot"]) for col, info in dynamic["chosen_by_color"].items()},
        "static_slots": {col: tuple(info["slot"]) for col, info in static["chosen_by_color"].items()},
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place one recolored sprite at a time, letting earlier placements block later searches.

    Input structure:
    - A gray rectangular frame encloses repeated solid placeholder blocks of several colors on a
      regular lattice.
    - Outside the frame there is one sprite template per placeholder color plus one two-color legend
      per placeholder color; in generated samples the sprites form a readable row above the frame.

    Rule:
    - Parse the frame, placeholders, outside sprites, and legend recoloring pairs.
    - Process the outside sprites in reading order. For each sprite color, consider only the inside
      blocks of that color and choose the single nearest target by shortest travel on the current
      filled-slot graph, entering from the frame side nearest that sprite.
    - After a target is chosen, mark that slot as occupied so later colors must route around it, then
      stamp the sprite once at that slot and recolor it according to the legend.

    Key insight:
    - The same outside cues now serve two roles: their order says which sprite acts first, and their
      position says where that sprite enters the frame. Because each chosen slot blocks later routing,
      "nearest" must be recomputed after every placement instead of once per color.
    """
    scene = _parse_scene(grid)
    bg = int(scene["bg"])
    color_map = scene["color_map"]
    sprite_data = scene["sprite_data"]
    crop = np.array(scene["crop"], copy=True)

    selection = _compute_selection(scene, dynamic_blocking=True)

    out = crop.copy()
    out[1:-1, 1:-1] = bg
    for src_col in selection["order"]:
        chosen = selection["chosen_by_color"][src_col]
        tgt_col = int(color_map[src_col])
        coords = sprite_data[src_col]["coords"]
        sh, sw = sprite_data[src_col]["shape"]
        pr0, pc0, pr1, pc1 = chosen["rel_bbox"]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = tgt_col
    return out


# --------------- generator (combo) ---------------

# The generator keeps the original global geometry family but narrows some degrees of freedom so the
# stateful blocking effect is reliably visible from a few examples.
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 4)
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
GEN_RETRIES = 500
LAYOUT_RETRIES = 300
SPRITE_GAP = 1
FRAME_GAP = 2
LEGEND_GAP = 1
SIDE_GAP = 2


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    target = random.randint(lo, hi)

    mask = np.zeros((n, n), dtype=bool)
    c = random.randrange(n)
    for r in range(n):
        c = max(0, min(n - 1, c + random.choice([-1, 0, 1])))
        mask[r, c] = True
    mask[random.randrange(n), :] = True

    if random.random() < 0.5:
        mask = mask.T.copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    while int(mask.sum()) < target:
        dilated = np.zeros_like(mask)
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        frontier = list(zip(*np.where(dilated & ~mask)))
        r, c = random.choice(frontier)
        mask[r, c] = True
    return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1


def _choose_structure() -> tuple[int, int, int, int] | None:
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                slots = rows * cols
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + 4
                base_fw = cw + 4
                max_k = min(NUM_COLORS_RANGE[1], cols, slots // 2)
                min_k = NUM_COLORS_RANGE[0]
                if max_k < min_k:
                    continue
                for total in range(TOTAL_BLOCKS_RANGE[0], TOTAL_BLOCKS_RANGE[1] + 1):
                    empty = slots - total
                    if not (0 <= empty <= MAX_EMPTY_SLOTS):
                        continue
                    if base_fh > MAX_FRAME_DIM or base_fw > MAX_FRAME_DIM:
                        continue
                    options.append((n, rows, cols, total))
    return random.choice(options) if options else None


def _is_connected(slots: set[tuple[int, int]]) -> bool:
    start = next(iter(slots))
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nxt in _slot_neighbors(cur, slots):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen == slots


def _choose_filled_slots(rows: int, cols: int, total: int) -> list[tuple[int, int]] | None:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, total):
        filled = set(combo)
        if len({r for r, _ in filled}) != rows:
            continue
        if len({c for _, c in filled}) != cols:
            continue
        if not _is_connected(filled):
            continue
        valid.append(tuple(sorted(filled)))
    return list(random.choice(valid)) if valid else None


def _repeat_counts(total: int, k: int) -> list[int]:
    counts = [2] * k
    for _ in range(total - 2 * k):
        counts[random.randrange(k)] += 1
    random.shuffle(counts)
    return counts


def _abstract_choose_target(
    color_slots: list[tuple[int, int]],
    filled_slots: set[tuple[int, int]],
    side: str,
    entry_rank: int,
    rows: int,
    cols: int,
    occupied_slots: set[tuple[int, int]],
) -> tuple[tuple[int, int], int, list[tuple[int, tuple[int, int]]]] | None:
    starts = _boundary_slots(side, filled_slots - occupied_slots, rows, cols)
    if not starts:
        return None
    scores = []
    for slot in sorted(color_slots):
        best = None
        for start in starts:
            dist = _slot_distance(start, slot, filled_slots, occupied_slots)
            if dist is None:
                continue
            total = _boundary_rank_delta(side, start, entry_rank) + dist
            pair = (total, start[0], start[1])
            if best is None or pair < best:
                best = pair
        if best is None:
            return None
        scores.append((best[0], slot))
    scores.sort()
    if len(scores) >= 2 and scores[0][0] == scores[1][0]:
        return None
    return scores[0][1], scores[0][0], scores


def _abstract_run(
    rows: int,
    cols: int,
    filled_slots: set[tuple[int, int]],
    slots_by_color: dict[int, list[tuple[int, int]]],
    order: list[int],
    entry_rank_by_color: dict[int, int],
    *,
    dynamic_blocking: bool,
) -> dict[int, tuple[int, int]] | None:
    occupied: set[tuple[int, int]] = set()
    chosen: dict[int, tuple[int, int]] = {}
    for color in order:
        result = _abstract_choose_target(
            slots_by_color[color],
            filled_slots,
            "top",
            entry_rank_by_color[color],
            rows,
            cols,
            occupied if dynamic_blocking else set(),
        )
        if result is None:
            return None
        slot, _cost, _scores = result
        chosen[color] = slot
        if dynamic_blocking:
            occupied.add(slot)
    return chosen


def _search_layout(
    rows: int,
    cols: int,
    total: int,
    usable: list[int],
) -> dict[str, object] | None:
    filled_slots_list = _choose_filled_slots(rows, cols, total)
    if filled_slots_list is None:
        return None
    filled_slots = set(filled_slots_list)

    max_k = min(NUM_COLORS_RANGE[1], cols, total // 2)
    if max_k < NUM_COLORS_RANGE[0]:
        return None

    for _ in range(LAYOUT_RETRIES):
        k = random.randint(NUM_COLORS_RANGE[0], max_k)
        src_cols = random.sample(usable, k)
        counts = _repeat_counts(total, k)
        bag = []
        for col, cnt in zip(src_cols, counts):
            bag.extend([col] * cnt)
        random.shuffle(bag)
        cell_colors = {slot: col for slot, col in zip(filled_slots_list, bag)}
        slots_by_color: dict[int, list[tuple[int, int]]] = {col: [] for col in src_cols}
        for slot, col in cell_colors.items():
            slots_by_color[col].append(slot)
        if any(len(v) < 2 for v in slots_by_color.values()):
            continue

        entry_cols = sorted(random.sample(range(cols), k))
        order = random.sample(src_cols, k)
        entry_rank_by_color = {color: entry_cols[i] for i, color in enumerate(order)}

        dynamic = _abstract_run(
            rows,
            cols,
            filled_slots,
            slots_by_color,
            order,
            entry_rank_by_color,
            dynamic_blocking=True,
        )
        static = _abstract_run(
            rows,
            cols,
            filled_slots,
            slots_by_color,
            order,
            entry_rank_by_color,
            dynamic_blocking=False,
        )
        if dynamic is None or static is None:
            continue
        changed = [color for color in order[1:] if dynamic[color] != static[color]]
        if not changed:
            continue
        return {
            "src_cols": src_cols,
            "cell_colors": cell_colors,
            "slots_by_color": slots_by_color,
            "order": order,
            "entry_rank_by_color": entry_rank_by_color,
            "chosen_dynamic": dynamic,
        }
    return None


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    layout = _search_layout(rows, cols, total, usable)
    if layout is None:
        return None

    src_cols = layout["src_cols"]
    cell_colors = layout["cell_colors"]
    chosen_dynamic = layout["chosen_dynamic"]
    order = layout["order"]
    entry_rank_by_color = layout["entry_rank_by_color"]

    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    frame_h = random.randint(min_fh, min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK)))
    frame_w = random.randint(min_fw, min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK)))
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    slot_box_rel: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        slot_box_rel[(rr, cc)] = (r0, c0, r0 + n - 1, c0 + n - 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)

    for src, slot in chosen_dynamic.items():
        r0, c0, _, _ = slot_box_rel[slot]
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    sprite_row_w = len(order) * n + (len(order) - 1) * SPRITE_GAP
    legend_col_h = len(src_cols) * 3 + (len(src_cols) - 1) * LEGEND_GAP
    min_canvas_h = max(n + FRAME_GAP + frame_h, legend_col_h)
    min_canvas_w = max(sprite_row_w, frame_w + SIDE_GAP + 4)
    if min_canvas_h > CANVAS_H_RANGE[1] - 1 or min_canvas_w > CANVAS_W_RANGE[1] - 1:
        return None

    h = random.randint(max(CANVAS_H_RANGE[0], min_canvas_h), CANVAS_H_RANGE[1] - 1)
    w = random.randint(max(CANVAS_W_RANGE[0], min_canvas_w), CANVAS_W_RANGE[1] - 1)
    grid = np.full((h, w), bg, dtype=int)

    frame_top = n + FRAME_GAP + random.randint(0, h - (n + FRAME_GAP + frame_h))
    frame_left = random.randint(0, w - frame_w - SIDE_GAP - 4)
    legend_left = frame_left + frame_w + SIDE_GAP
    if legend_left + 4 > w:
        return None

    sprite_lefts = []
    sprite_top = max(0, frame_top - n - FRAME_GAP)
    if sprite_top + n > frame_top:
        return None
    for color in order:
        entry_col = entry_rank_by_color[color]
        center_c = frame_left + left_margin + entry_col * (n + 1) + (n - 1) / 2
        sprite_left = int(round(center_c - (n - 1) / 2))
        sprite_lefts.append(sprite_left)
    if sorted(sprite_lefts) != sprite_lefts:
        return None
    if any(sprite_lefts[i + 1] < sprite_lefts[i] + n + SPRITE_GAP for i in range(len(sprite_lefts) - 1)):
        return None
    if sprite_lefts[0] < 0 or sprite_lefts[-1] + n > w:
        return None

    paste(grid, input_frame, frame_top, frame_left)

    for color, sprite_left in zip(order, sprite_lefts):
        paste(grid, _colored_sprite(masks[color], color, bg), sprite_top, sprite_left, transparent=bg)

    legend_top = random.randint(0, h - legend_col_h)
    for i, src in enumerate(sorted(src_cols)):
        paste(grid, _legend(src, color_to_tgt[src]), legend_top + i * (3 + LEGEND_GAP), legend_left)

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
        metrics = _combo_metrics(grid)
        if metrics["changed_count"] < 1:
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid combo edb79dae input")
