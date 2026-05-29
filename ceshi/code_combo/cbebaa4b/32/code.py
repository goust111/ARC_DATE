# Combo variant of cbebaa4b (index 32)
# combo_spec: D3/1+D7/0
# Name: 无固定锚点 + 由外向内组装

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(_ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Source task: a yellow center piece is visibly special, stays fixed, and the
#      solver may attach any compatible red-connector sprite next.
#    - D3/1 removes the yellow privilege. In this combo, the stationary seed is the
#      unique largest sprite by filled area, so the first reasoning step becomes
#      "which piece is the start?" rather than "where is yellow?".
#    - D7/0 must be made visible at the pixel level; otherwise it would collapse into
#      a hidden DFS tie-breaker. The combo therefore turns the puzzle into a one-sided
#      contraction chain:
#        seed (largest leaf) -> segment 1 -> segment 2 -> ... -> segment N
#      All non-seed sprites lie on one visible side of the seed in the input, and
#      their order is encoded by projected distance along that side.
#    - Small trace:
#        input: a large teal seed is fixed near the left edge. Four smaller bars sit
#        to its right at clearly different distances. The farthest bar is maroon, then
#        orange, then blue, then green nearest the seed.
#        output: those same bars collapse into a connected bar-chain extending right
#        from the teal seed, in the order maroon -> orange -> blue -> green.
#      Relative to the source task, both the start-piece cue and the assembly-order cue
#      have changed.
#
# b) Complexity calibration
#    - A fully general no-anchor tree plus outer-to-inner ordering would be hard to
#      visualize and easy to accidentally degenerate. This combo uses a controlled
#      chain geometry so the distance ordering is repeated several times in one sample.
#    - The pieces are still distinct colored sprites with red 2-cell connectors, so
#      the task remains a translation-only connector puzzle rather than becoming a pure
#      sorting puzzle.
#    - The seed is made substantially larger than the others, not just barely larger,
#      to keep the D3 cue human-visible and avoid accidental ties.
#
# c) Hidden assumptions audit of the original main()
#    - "exactly one yellow anchor exists": broken; yellow is removed completely.
#    - "the anchor is the fixed piece": replaced by "the unique max-area piece is the
#      fixed seed".
#    - "any remaining compatible piece may attach next": broken; in this combo only
#      the distance-ranked next segment is legal, because several intermediate chain
#      pieces are intentionally connector-compatible and would otherwise be swappable.
#    - "generic backtracking is enough to define the task": no longer true as a task
#      definition. Here, distance order is part of the visible rule, not an internal
#      optimization. The solver therefore reads the input-side ordering explicitly.
#
# d) Placement legality
#    - The generator is still output-first: it builds the solved chain first, validates
#      red-over-red overlaps, then scatters the non-seed pieces.
#    - Input scattering keeps every non-seed piece on one side of the seed, strictly
#      ordered by projected distance, non-overlapping, in-bounds, and separated by a
#      padding box so component extraction remains stable.
#    - Output legality is simple and robust because every join is a 2-cell connector
#      overlap on the chain axis; no non-red cells are allowed to collide.
#
# e) Visual observability
#    - Humans can see two repeated cues from 2-4 examples:
#        1. there is no yellow; the largest piece is the only plausible fixed seed;
#        2. all other pieces sit on one side of it and are ordered from far to near.
#    - The output then shows those same pieces compressed into one connected chain,
#      preserving the far-to-near identity order.
#    - No hidden metadata such as color id arithmetic, generation index, or invisible
#      object labels is used.
#
# f) Visual cleanliness
#    - Every sprite is a single connected bar-like component with at most two connector
#      pairs, so the scene stays legible.
#    - The input uses one directional cone rather than random all-around scatter, which
#      makes the "contract from outside inward" idea readable instead of cluttered.
#
# g) Change inferability and degeneracy control
#    - Colors are not remapped; they only help track piece identity across input/output.
#    - Several intermediate pieces share the same connector signature. Without the
#      D7 ordering cue, many permutations would be geometrically valid. This is
#      intentional: the input distance order does real disambiguation work.
#    - Post-implementation metric: compare against the D3-only solver. A high rate of
#      D3-only failure or mismatched output confirms that the outer-to-inner ranking is
#      genuinely active rather than decorative.
#
# h) New solver assumptions and generator guarantees
#    - main() assumes exactly one strict max-area component exists. The generator uses a
#      longer seed bar and checks the rendered areas.
#    - main() assumes all non-seed components lie in one dominant direction from the
#      seed. The scatter stage enforces a directional cone.
#    - main() assumes projected distances are strictly ordered. The scatter stage places
#      every piece in a separate projection band and validates the resulting order.
#    - main() assumes all non-final moved pieces have two connector sides aligned with
#      the chain direction, and the final moved piece has only the incoming connector.
#      The generator builds exactly that topology.


GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)

PIECE_COUNT_RANGE = (4, 6)  # non-seed pieces
THICKNESS_CHOICES = [4, 5]
ROOT_LEN_CHOICES = [7]
SEG_LEN_CHOICES = [4]
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
ROOT_AREA_MARGIN = 3

SCATTER_PADDING = 2
SCATTER_ATTEMPTS = 300
GEN_RETRY_LIMIT = 500

DIRS = ("right", "left", "down", "up")
OPPOSITE = {"right": "left", "left": "right", "down": "up", "up": "down"}
AXIS = {"right": "h", "left": "h", "down": "v", "up": "v"}
FORWARD_SIDE = {"right": "right", "left": "left", "down": "bottom", "up": "top"}
BACK_SIDE = {"right": "left", "left": "right", "down": "top", "up": "bottom"}
SIDE_TO_DIR = {"right": "right", "left": "left", "bottom": "down", "top": "up"}


def _sorted_group(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _find_red_groups(arr: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    groups = []
    for pts in components_mask(arr == 2, connectivity=4):
        groups.append(_sorted_group([(int(r), int(c)) for r, c in pts]))
    groups.sort()
    return groups


def _connector_sides(arr: np.ndarray) -> set[str]:
    h, w = arr.shape
    out = set()
    if np.any(arr[0, :] == 2):
        out.add("top")
    if np.any(arr[h - 1, :] == 2):
        out.add("bottom")
    if np.any(arr[:, 0] == 2):
        out.add("left")
    if np.any(arr[:, w - 1] == 2):
        out.add("right")
    return out


def _side_red_group(arr: np.ndarray, side: str) -> tuple[tuple[int, int], ...]:
    h, w = arr.shape
    if side == "top":
        pts = [(0, c) for c in range(w) if arr[0, c] == 2]
    elif side == "bottom":
        pts = [(h - 1, c) for c in range(w) if arr[h - 1, c] == 2]
    elif side == "left":
        pts = [(r, 0) for r in range(h) if arr[r, 0] == 2]
    else:
        pts = [(r, w - 1) for r in range(h) if arr[r, w - 1] == 2]
    return _sorted_group(pts)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _projection(center: tuple[float, float], seed_center: tuple[float, float], direction: str) -> float:
    dr = center[0] - seed_center[0]
    dc = center[1] - seed_center[1]
    if direction == "right":
        return float(dc)
    if direction == "left":
        return float(-dc)
    if direction == "down":
        return float(dr)
    return float(-dr)


def _cross_offset(center: tuple[float, float], seed_center: tuple[float, float], direction: str) -> float:
    dr = center[0] - seed_center[0]
    dc = center[1] - seed_center[1]
    return float(abs(dr) if direction in ("right", "left") else abs(dc))


def _infer_direction(seed_center: tuple[float, float], pieces: list[dict]) -> str:
    dr_sum = 0.0
    dc_sum = 0.0
    for p in pieces:
        pr, pc = _bbox_center(p["bbox"])
        dr_sum += pr - seed_center[0]
        dc_sum += pc - seed_center[1]
    if abs(dc_sum) >= abs(dr_sum):
        return "right" if dc_sum > 0 else "left"
    return "down" if dr_sum > 0 else "up"


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = [(int(r), int(c)) for r, c in np.argwhere(arr == 2)]
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": reds,
                "red_groups": _find_red_groups(arr),
                "connector_sides": _connector_sides(arr),
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
            }
        )
    return pieces


def _try_place(canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None

    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != bg
    if not np.any(mask):
        return None

    conflict = mask & (sub != bg) & ~(((arr == 2) & (sub == 2)) | (arr == sub))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _match_group_translations(
    local_group: tuple[tuple[int, int], ...], target_group: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
    target = set(target_group)
    out: set[tuple[int, int]] = set()
    for lr, lc in local_group:
        for tr, tc in target_group:
            top = int(tr - lr)
            left = int(tc - lc)
            shifted = {(int(r + top), int(c + left)) for r, c in local_group}
            if shifted == target:
                out.add((top, left))
    return sorted(out)


def _abs_group(group: tuple[tuple[int, int], ...], top: int, left: int) -> tuple[tuple[int, int], ...]:
    return _sorted_group([(r + top, c + left) for r, c in group])


def _select_seed_piece(pieces: list[dict]) -> tuple[dict, list[dict]]:
    max_area = max(p["area"] for p in pieces)
    seeds = [p for p in pieces if p["area"] == max_area]
    assert len(seeds) == 1, "input must have exactly one largest piece"
    seed = seeds[0]
    rest = [p for p in pieces if p is not seed]
    return seed, rest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered connector sprites by inferring the largest seed and then contracting the remaining pieces from the outside inward.

    Input structure:
    - Background is black (0). There is no yellow anchor anywhere in the puzzle.
    - Exactly one ordinary colored sprite has the largest filled area; it is the stationary seed.
    - Every other sprite lies on one visible side of the seed and carries one or two 2-cell red connector groups aligned with a single chain direction.

    Rule:
    - Keep the unique largest sprite fixed in place.
    - Infer the contraction direction from where the other sprites sit relative to the seed, then rank those sprites from farthest to nearest along that direction.
    - Attach them in that ranked order so each piece's incoming red group overlaps the current open red group of the growing chain; all red cells must finish paired exactly once.

    Key insight:
    - The task is no longer "find yellow and expand outward". First identify the seed by size, then read the outer-to-inner ordering from the input distances, because several middle segments are connector-compatible and distance order is what disambiguates their final positions.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert len(pieces) >= 2, "input must contain multiple pieces"

    seed, rest = _select_seed_piece(pieces)
    seed_center = _bbox_center(seed["bbox"])
    assert len(seed["connector_sides"]) == 1, "seed must have exactly one visible connector side"
    direction = SIDE_TO_DIR[next(iter(seed["connector_sides"]))]
    forward_side = FORWARD_SIDE[direction]
    incoming = BACK_SIDE[direction]

    ranked = sorted(
        rest,
        key=lambda p: (
            -_projection(_bbox_center(p["bbox"]), seed_center, direction),
            p["bbox"],
        ),
    )

    projs = [_projection(_bbox_center(p["bbox"]), seed_center, direction) for p in ranked]
    assert min(projs) > 0, "all non-seed pieces must lie in front of the seed"
    assert all(projs[i] > projs[i + 1] for i in range(len(projs) - 1)), "distance order must be strict"

    assert seed["connector_sides"] == {forward_side}, "seed must expose exactly one forward connector"
    for i, p in enumerate(ranked):
        expected = {incoming} if i == len(ranked) - 1 else {incoming, forward_side}
        assert p["connector_sides"] == expected, "piece connector pattern does not match the contraction chain"
        for grp in p["red_groups"]:
            assert len(grp) == 2, "each connector group must contain exactly two red cells"

    canvas = np.full((h, w), bg, dtype=int)
    sr0, sc0, _, _ = seed["bbox"]
    cur = _try_place(canvas, seed["arr"], sr0, sc0, bg)
    assert cur is not None, "seed must fit at its input position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector count must be even"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in seed["reds"]:
        key = (int(sr0 + rr), int(sc0 + cc))
        red_counts[key] = red_counts.get(key, 0) + 1

    active_tail = _abs_group(_side_red_group(seed["arr"], forward_side), sr0, sc0)
    assert len(active_tail) == 2, "seed must have a 2-cell starting connector"

    for i, p in enumerate(ranked):
        local_in = _side_red_group(p["arr"], incoming)
        assert len(local_in) == 2, "each moved piece needs a 2-cell incoming connector"
        moves = _match_group_translations(local_in, active_tail)
        assert moves, "ranked piece must align to the active connector"

        placed = None
        for top, left in moves:
            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            next_rc = dict(red_counts)
            for rr, cc in p["reds"]:
                key = (int(top + rr), int(left + cc))
                next_rc[key] = next_rc.get(key, 0) + 1
            if any(v > 2 for v in next_rc.values()):
                continue

            unpaired = _sorted_group([k for k, v in next_rc.items() if v == 1])
            if i < len(ranked) - 1:
                next_tail = _abs_group(_side_red_group(p["arr"], forward_side), top, left)
                if unpaired != next_tail:
                    continue
            else:
                if unpaired:
                    continue
                next_tail = ()

            placed = (nxt, next_rc, next_tail)
            break

        assert placed is not None, "distance-ranked placement must be valid"
        cur, red_counts, active_tail = placed

    assert len(red_counts) == target_unique_red, "assembled red coordinates mismatch the expected pairing count"
    assert all(v == 2 for v in red_counts.values()), "every red connector must be paired exactly once"
    return cur


def _render_piece(axis: str, length: int, thickness: int, color: int, ports: set[str]) -> np.ndarray:
    if axis == "h":
        arr = np.zeros((thickness, length), dtype=int)
        arr[:, 1 : length - 1] = color
        k = thickness // 2 - 1
        if "left" in ports:
            arr[k : k + 2, 0] = 2
        else:
            arr[k : k + 2, 0] = color
        if "right" in ports:
            arr[k : k + 2, length - 1] = 2
        else:
            arr[k : k + 2, length - 1] = color
    else:
        arr = np.zeros((length, thickness), dtype=int)
        arr[1 : length - 1, :] = color
        k = thickness // 2 - 1
        if "top" in ports:
            arr[0, k : k + 2] = 2
        else:
            arr[0, k : k + 2] = color
        if "bottom" in ports:
            arr[length - 1, k : k + 2] = 2
        else:
            arr[length - 1, k : k + 2] = color

    assert len(components_mask(arr != 0, connectivity=4)) == 1, "piece must stay connected"
    return arr


def _node_bbox(node: dict, pos_key: str = "top") -> tuple[int, int, int, int]:
    top = node[pos_key]
    left = node["left" if pos_key == "top" else "left"]
    h, w = node["arr"].shape
    return (top, left, top + h - 1, left + w - 1)


def _union_bbox(nodes: list[dict], pos_key: str = "top") -> tuple[int, int, int, int]:
    boxes = []
    for node in nodes:
        top = node[pos_key]
        left = node["left"]
        h, w = node["arr"].shape
        boxes.append((top, left, top + h - 1, left + w - 1))
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _build_nodes(direction: str) -> list[dict] | None:
    n_seg = random.randint(*PIECE_COUNT_RANGE)
    axis = AXIS[direction]
    thickness = random.choice(THICKNESS_CHOICES)
    root_len = random.choice(ROOT_LEN_CHOICES)
    seg_lens = [random.choice(SEG_LEN_CHOICES) for _ in range(n_seg)]
    colors = random.sample(PIECE_COLORS, n_seg + 1)

    nodes = []
    root = {
        "role": "seed",
        "arr": _render_piece(axis, root_len, thickness, colors[0], {FORWARD_SIDE[direction]}),
        "top": 0,
        "left": 0,
    }
    nodes.append(root)

    for i, seg_len in enumerate(seg_lens):
        ports = {BACK_SIDE[direction]} if i == n_seg - 1 else {BACK_SIDE[direction], FORWARD_SIDE[direction]}
        nodes.append(
            {
                "role": "terminal" if i == n_seg - 1 else "mid",
                "arr": _render_piece(axis, seg_len, thickness, colors[i + 1], ports),
                "top": 0,
                "left": 0,
            }
        )

    areas = [int(np.sum(node["arr"] != 0)) for node in nodes]
    if areas[0] != max(areas) or areas.count(areas[0]) != 1:
        return None
    if areas[0] - max(areas[1:]) < ROOT_AREA_MARGIN:
        return None
    return nodes


def _set_local_chain_positions(nodes: list[dict], direction: str) -> None:
    nodes[0]["top"] = 0
    nodes[0]["left"] = 0
    prev = nodes[0]
    for node in nodes[1:]:
        ph, pw = prev["arr"].shape
        nh, nw = node["arr"].shape
        if direction == "right":
            node["top"] = prev["top"]
            node["left"] = prev["left"] + pw - 1
        elif direction == "left":
            node["top"] = prev["top"]
            node["left"] = prev["left"] - nw + 1
        elif direction == "down":
            node["top"] = prev["top"] + ph - 1
            node["left"] = prev["left"]
        else:
            node["top"] = prev["top"] - nh + 1
            node["left"] = prev["left"]
        prev = node


def _place_assembled(nodes: list[dict], h: int, w: int, direction: str) -> np.ndarray | None:
    _set_local_chain_positions(nodes, direction)
    r0, c0, r1, c1 = _union_bbox(nodes)
    union_h = r1 - r0 + 1
    union_w = c1 - c0 + 1
    if union_h > h or union_w > w:
        return None

    if direction == "right":
        union_left = random.randint(0, max(0, min(2, w - union_w)))
        top_min = 4
        top_max = h - union_h - 5
        if top_min > top_max:
            return None
        union_top = random.randint(top_min, top_max)
    elif direction == "left":
        left_min = max(0, w - union_w - 2)
        left_max = w - union_w
        union_left = random.randint(left_min, left_max)
        top_min = 4
        top_max = h - union_h - 5
        if top_min > top_max:
            return None
        union_top = random.randint(top_min, top_max)
    elif direction == "down":
        union_top = random.randint(0, max(0, min(2, h - union_h)))
        left_min = 4
        left_max = w - union_w - 5
        if left_min > left_max:
            return None
        union_left = random.randint(left_min, left_max)
    else:
        top_min = max(0, h - union_h - 2)
        top_max = h - union_h
        union_top = random.randint(top_min, top_max)
        left_min = 4
        left_max = w - union_w - 5
        if left_min > left_max:
            return None
        union_left = random.randint(left_min, left_max)

    shift_r = union_top - r0
    shift_c = union_left - c0
    out = np.zeros((h, w), dtype=int)
    red_counts: dict[tuple[int, int], int] = {}

    for node in nodes:
        node["top"] += shift_r
        node["left"] += shift_c
        nxt = _try_place(out, node["arr"], node["top"], node["left"], 0)
        if nxt is None:
            return None
        out = nxt
        for rr, cc in np.argwhere(node["arr"] == 2):
            key = (int(node["top"] + rr), int(node["left"] + cc))
            red_counts[key] = red_counts.get(key, 0) + 1

    total_ports = sum(int(np.sum(node["arr"] == 2)) for node in nodes)
    if len(red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    if np.any(out == 4):
        return None
    return out


def _rect_clear(occ: np.ndarray, top: int, left: int, ph: int, pw: int, padding: int) -> bool:
    h, w = occ.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + ph + padding)
    c1 = min(w, left + pw + padding)
    return not np.any(occ[r0:r1, c0:c1])


def _min_projection(seed_node: dict, node: dict, direction: str) -> float:
    sh, sw = seed_node["arr"].shape
    nh, nw = node["arr"].shape
    seed_half = (sw - 1) / 2.0 if direction in ("right", "left") else (sh - 1) / 2.0
    node_half = (nw - 1) / 2.0 if direction in ("right", "left") else (nh - 1) / 2.0
    return seed_half + node_half + SCATTER_PADDING + 1.0


def _max_projection(seed_center: tuple[float, float], node: dict, h: int, w: int, direction: str) -> float:
    ph, pw = node["arr"].shape
    half_r = (ph - 1) / 2.0
    half_c = (pw - 1) / 2.0
    if direction == "right":
        return (w - 1 - half_c) - seed_center[1]
    if direction == "left":
        return seed_center[1] - half_c
    if direction == "down":
        return (h - 1 - half_r) - seed_center[0]
    return seed_center[0] - half_r


def _sample_scatter_pos(
    occ: np.ndarray,
    node: dict,
    seed_center: tuple[float, float],
    direction: str,
    target_proj: float,
) -> tuple[int, int] | None:
    h, w = occ.shape
    ph, pw = node["arr"].shape
    for _ in range(SCATTER_ATTEMPTS):
        band = max(2.0, target_proj / 2.5)
        if direction == "right":
            center_r = seed_center[0] + random.uniform(-band, band)
            center_c = seed_center[1] + target_proj + random.uniform(-0.35, 0.35)
        elif direction == "left":
            center_r = seed_center[0] + random.uniform(-band, band)
            center_c = seed_center[1] - target_proj + random.uniform(-0.35, 0.35)
        elif direction == "down":
            center_r = seed_center[0] + target_proj + random.uniform(-0.35, 0.35)
            center_c = seed_center[1] + random.uniform(-band, band)
        else:
            center_r = seed_center[0] - target_proj + random.uniform(-0.35, 0.35)
            center_c = seed_center[1] + random.uniform(-band, band)

        top = int(round(center_r - (ph - 1) / 2.0))
        left = int(round(center_c - (pw - 1) / 2.0))
        if not _rect_clear(occ, top, left, ph, pw, SCATTER_PADDING):
            continue

        bbox = (top, left, top + ph - 1, left + pw - 1)
        center = _bbox_center(bbox)
        proj = _projection(center, seed_center, direction)
        cross = _cross_offset(center, seed_center, direction)
        if proj <= 0 or abs(proj - target_proj) > 0.9:
            continue
        if cross > max(2.0, proj / 2.0):
            continue
        return top, left
    return None


def _scatter(nodes: list[dict], h: int, w: int, direction: str) -> np.ndarray | None:
    inp = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    seed = nodes[0]
    seed_top, seed_left = seed["top"], seed["left"]
    sh, sw = seed["arr"].shape
    inp = _try_place(inp, seed["arr"], seed_top, seed_left, 0)
    assert inp is not None
    occ[seed_top : seed_top + sh, seed_left : seed_left + sw] |= seed["arr"] != 0

    seed_center = _bbox_center((seed_top, seed_left, seed_top + sh - 1, seed_left + sw - 1))
    segs = nodes[1:]

    min_proj = max(_min_projection(seed, node, direction) for node in segs)
    max_proj = min(_max_projection(seed_center, node, h, w, direction) for node in segs)
    if max_proj - min_proj < 1.45 * (len(segs) - 1):
        return None

    targets = np.linspace(max_proj - 0.35, min_proj + 0.35, len(segs))
    for node, target_proj in zip(segs, targets):
        pos = _sample_scatter_pos(occ, node, seed_center, direction, float(target_proj))
        if pos is None:
            return None
        top, left = pos
        nxt = _try_place(inp, node["arr"], top, left, 0)
        if nxt is None:
            return None
        inp = nxt
        ph, pw = node["arr"].shape
        occ[top : top + ph, left : left + pw] |= node["arr"] != 0
        node["scatter_top"] = top
        node["scatter_left"] = left

    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    direction = random.choice(DIRS)
    nodes = _build_nodes(direction)
    if nodes is None:
        return None

    out = _place_assembled(nodes, h, w, direction)
    if out is None:
        return None

    inp = _scatter(nodes, h, w, direction)
    if inp is None:
        return None

    pieces = _extract_pieces(inp, 0)
    seed, rest = _select_seed_piece(pieces)
    seed_center = _bbox_center(seed["bbox"])
    inferred = SIDE_TO_DIR[next(iter(seed["connector_sides"]))]
    if inferred != direction:
        return None
    ranked = sorted(rest, key=lambda p: -_projection(_bbox_center(p["bbox"]), seed_center, inferred))
    projs = [_projection(_bbox_center(p["bbox"]), seed_center, inferred) for p in ranked]
    if not all(projs[i] > projs[i + 1] for i in range(len(projs) - 1)):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
