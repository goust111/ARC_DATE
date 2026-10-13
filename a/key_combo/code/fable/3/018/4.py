from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, STRUCT8, bbox_coords, colors_present, components_mask, get_bg, paste


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

BG = Color.BLACK
BORDER_PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
TOTAL_CUPS_CHOICES = (3, 4, 5)
GEN_RETRIES = 500


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood inside a cup bbox, seeded from the open side opposite the tail."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    seen = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in seen or (nr, nc) in border_set:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return seen


def _detect_cups(grid: np.ndarray, bg: int) -> list[dict]:
    cups: list[dict] = []
    for border_color in colors_present(grid, exclude={bg, Color.YELLOW}):
        for border_coords in components_mask(grid == int(border_color), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue
            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            border_set = {(int(r), int(c)) for r, c in border_coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            edges = (
                ("U", [p for p in border_set if p[0] == r0]),
                ("D", [p for p in border_set if p[0] == r1]),
                ("L", [p for p in border_set if p[1] == c0]),
                ("R", [p for p in border_set if p[1] == c1]),
            )
            candidates = [
                (direction, cells[0])
                for direction, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            passing: list[tuple[str, tuple[int, int], int, list[tuple[int, int]]]] = []
            for direction, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, (r0, c0, r1, c1), OPPOSITE[direction])
                counts = Counter()
                for r, c in mouth:
                    val = int(grid[r, c])
                    if val not in (bg, int(border_color), int(Color.YELLOW)):
                        counts[val] += 1
                if not counts:
                    continue
                content_color = int(counts.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(grid[p[0], p[1]]) == content_color]
                if not content_cells:
                    continue
                passing.append((direction, spike, content_color, content_cells))
            if len(passing) != 1:
                continue
            direction, spike, content_color, content_cells = passing[0]
            cups.append(
                {
                    "border_color": int(border_color),
                    "bbox": (r0, c0, r1, c1),
                    "border_set": border_set,
                    "direction": direction,
                    "vector": DIR4[direction],
                    "spike": (int(spike[0]), int(spike[1])),
                    "content_color": int(content_color),
                    "content_cells": [(int(r), int(c)) for r, c in content_cells],
                }
            )
    cups.sort(key=lambda cup: (cup["bbox"][0], cup["bbox"][1], cup["border_color"]))
    return cups


def _find_ignition(grid: np.ndarray, cups: list[dict]) -> tuple[int, tuple[int, int]]:
    yellow_coords = np.argwhere(grid == Color.YELLOW)
    assert len(yellow_coords) == 1, "expected exactly one yellow ignition marker"
    ignition = tuple(int(v) for v in yellow_coords[0])
    matches = []
    for idx, cup in enumerate(cups):
        dr, dc = cup["vector"]
        if ignition == (cup["spike"][0] + dr, cup["spike"][1] + dc):
            matches.append(idx)
    assert len(matches) == 1, "yellow marker must sit just outside exactly one tail spike"
    return int(matches[0]), ignition


def _fire_chain(grid: np.ndarray, cups: list[dict], ignition_idx: int, ignition_pos: tuple[int, int], bg: int) -> tuple[np.ndarray, list[int], list[int | None]]:
    out = np.array(grid, dtype=int)
    out[ignition_pos] = int(bg)
    border_owner: dict[tuple[int, int], int] = {}
    for idx, cup in enumerate(cups):
        for cell in cup["border_set"]:
            border_owner[cell] = idx

    fired_order: list[int] = []
    hits: list[int | None] = []
    current = int(ignition_idx)
    seen: set[int] = set()
    H, W = out.shape

    while current not in seen:
        seen.add(current)
        fired_order.append(current)
        cup = cups[current]
        for r, c in cup["content_cells"]:
            out[r, c] = int(bg)

        dr, dc = cup["vector"]
        rr = int(cup["spike"][0] + dr)
        cc = int(cup["spike"][1] + dc)
        next_idx: int | None = None
        for _ in range(len(cup["content_cells"])):
            if not (0 <= rr < H and 0 <= cc < W):
                break
            owner = border_owner.get((rr, cc))
            if owner is not None and owner != current:
                next_idx = int(owner)
                break
            out[rr, cc] = int(cup["content_color"])
            rr += dr
            cc += dc
        hits.append(next_idx)
        if next_idx is None or next_idx in seen:
            break
        current = next_idx
    return out, fired_order, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire only the cups reached by the ignition-hit chain, erasing their contents and ejecting them along each tail.

    Input structure:
    - A black background contains several hollow cup frames, each in a unique border color.
    - Every cup has a convergent tail spike on one side and a monochrome set of content pixels in the opposite mouth interior.
    - Exactly one yellow ignition pixel sits immediately outside one cup's tail spike.

    Rule:
    - Detect the ignition cup from the yellow marker and let only that cup fire first.
    - When a cup fires, erase all of its mouth content and paint one straight tail-direction pixel per erased content pixel starting just outside the spike.
    - If the ray reaches another cup's border before running out, stop one cell before the border, discard the remaining pixels, and let the hit cup fire next; cups never hit by the chain keep their contents, and the yellow marker is removed.

    Key insight:
    - This is a causal chain, not a simultaneous rule: whether a cup empties depends entirely on whether an earlier ray reaches its frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    cups = _detect_cups(g, bg)
    ignition_idx, ignition_pos = _find_ignition(g, cups)
    out, _fired_order, _hits = _fire_chain(g, cups, ignition_idx, ignition_pos, bg)
    return out


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a cup with tail UP before rotation."""
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1 : L + hL + 1, 0] = color
    if hR:
        patch[R + 1 : R + hR + 1, w - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, open_side: str) -> list[tuple[int, int]]:
    h, w = patch.shape
    border_mask = patch == int(border_color)
    if open_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif open_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif open_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if not border_mask[r, w - 1]]
    seen = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and not border_mask[nr, nc]:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return sorted((int(r), int(c)) for r, c in seen)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for r, c in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for r, c in mouth]
    raise ValueError(direction)


def _sample_cup_spec(rng: np.random.Generator, border_color: int, content_color: int, min_capacity: int) -> dict:
    while True:
        direction = str(rng.choice(np.array(list(DIR4))))
        L = int(rng.integers(1, 4))
        R = int(rng.integers(1, 4))
        hL = int(rng.integers(0, 4))
        hR = int(rng.integers(0, 4))
        if hL + hR == 0:
            if rng.random() < 0.5:
                hL = 1
            else:
                hR = 1
        patch_canon, spike_canon = _build_canonical_cup(L, R, hL, hR, border_color)
        mouth_canon = _mouth_interior_coords(patch_canon, border_color, open_side="D")
        patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
        if len(mouth) < min_capacity:
            continue
        border_local = [(int(r), int(c)) for r, c in np.argwhere(patch != BG)]
        return {
            "patch": patch,
            "patch_mask": patch != BG,
            "border_local": border_local,
            "mouth_local": [(int(r), int(c)) for r, c in mouth],
            "spike_local": (int(spike[0]), int(spike[1])),
            "direction": direction,
            "vector": DIR4[direction],
            "border_color": int(border_color),
            "content_color": int(content_color),
        }


def _exposed_impact_cells(border_local: list[tuple[int, int]], incoming_direction: str) -> list[tuple[int, int]]:
    if incoming_direction == "R":
        best: dict[int, int] = {}
        for r, c in border_local:
            best[r] = c if r not in best else min(best[r], c)
        return [(int(r), int(c)) for r, c in best.items()]
    if incoming_direction == "L":
        best = {}
        for r, c in border_local:
            best[r] = c if r not in best else max(best[r], c)
        return [(int(r), int(c)) for r, c in best.items()]
    if incoming_direction == "D":
        best = {}
        for r, c in border_local:
            best[c] = r if c not in best else min(best[c], r)
        return [(int(r), int(c)) for c, r in best.items()]
    best = {}
    for r, c in border_local:
        best[c] = r if c not in best else max(best[c], r)
    return [(int(r), int(c)) for c, r in best.items()]


def _offset_coords(coords: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in coords]


def _ray_cells(start: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = shape
    rr, cc = (int(start[0]), int(start[1]))
    dr, dc = DIR4[direction]
    cells = []
    while 0 <= rr < H and 0 <= cc < W:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _shuffled_positions(rng: np.random.Generator, max_r: int, max_c: int) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(max_r + 1) for c in range(max_c + 1)]
    order = rng.permutation(len(coords))
    return [coords[int(i)] for i in order]


def _can_place_patch(occ_cup: np.ndarray, reserved_clear: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch_mask.shape
    H, W = occ_cup.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    occ_block = ndimage.binary_dilation(occ_cup, structure=STRUCT8)
    if bool(occ_block[r0 : r0 + h, c0 : c0 + w][patch_mask].any()):
        return False
    if bool(reserved_clear[r0 : r0 + h, c0 : c0 + w][patch_mask].any()):
        return False
    return True


def _commit_cup(cup: dict, r0: int, c0: int, occ_cup: np.ndarray) -> None:
    cup["r0"] = int(r0)
    cup["c0"] = int(c0)
    h, w = cup["patch"].shape
    cup["bbox"] = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
    cup["spike"] = (int(r0 + cup["spike_local"][0]), int(c0 + cup["spike_local"][1]))
    cup["border_set"] = set(_offset_coords(cup["border_local"], r0, c0))
    cup["mouth_abs"] = _offset_coords(cup["mouth_local"], r0, c0)
    dr, dc = cup["vector"]
    cup["emit_start"] = (int(cup["spike"][0] + dr), int(cup["spike"][1] + dc))
    for rr, cc in cup["border_set"]:
        occ_cup[rr, cc] = True


def _try_place_initial_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    cup: dict,
    occ_cup: np.ndarray,
    reserved_clear: np.ndarray,
    needs_successor: bool,
) -> tuple[int, int] | None:
    h, w = cup["patch"].shape
    required_forward = 2 if needs_successor else 1
    for r0, c0 in _shuffled_positions(rng, H - h, W - w):
        if not _can_place_patch(occ_cup, reserved_clear, cup["patch_mask"], r0, c0):
            continue
        dr, dc = cup["vector"]
        start = (int(r0 + cup["spike_local"][0] + dr), int(c0 + cup["spike_local"][1] + dc))
        if not (0 <= start[0] < H and 0 <= start[1] < W):
            continue
        if occ_cup[start] or reserved_clear[start]:
            continue
        if len(_ray_cells(start, cup["direction"], (H, W))) < required_forward:
            continue
        return int(r0), int(c0)
    return None


def _choose_hit_count(rng: np.random.Generator, capacity: int, safe_len: int) -> int:
    minimum = int(safe_len + 1)
    max_extra = int(min(2, capacity - minimum))
    extra = 0
    if max_extra > 0:
        if rng.random() < 0.65:
            extra = int(rng.integers(1, max_extra + 1))
        else:
            extra = 0
    return int(minimum + extra)


def _try_place_successor(
    rng: np.random.Generator,
    H: int,
    W: int,
    prev: dict,
    cup: dict,
    occ_cup: np.ndarray,
    reserved_clear: np.ndarray,
) -> tuple[int, int, int, list[tuple[int, int]]] | None:
    ray = _ray_cells(prev["emit_start"], prev["direction"], (H, W))
    max_safe = min(len(ray) - 1, len(prev["mouth_local"]) - 1)
    if max_safe < 1:
        return None

    safe_options = [int(v) for v in range(1, max_safe + 1)]
    safe_options = [safe_options[int(i)] for i in rng.permutation(len(safe_options))]
    impact_cells = _exposed_impact_cells(cup["border_local"], prev["direction"])

    for safe_len in safe_options:
        corridor = ray[:safe_len]
        if any(occ_cup[r, c] or reserved_clear[r, c] for r, c in corridor):
            continue
        target = ray[safe_len]
        shuffled_impacts = [impact_cells[int(i)] for i in rng.permutation(len(impact_cells))]
        for local_r, local_c in shuffled_impacts:
            r0 = int(target[0] - local_r)
            c0 = int(target[1] - local_c)
            if not _can_place_patch(occ_cup, reserved_clear, cup["patch_mask"], r0, c0):
                continue
            overlaps_corridor = False
            for rr, cc in corridor:
                pr = rr - r0
                pc = cc - c0
                if 0 <= pr < cup["patch_mask"].shape[0] and 0 <= pc < cup["patch_mask"].shape[1] and cup["patch_mask"][pr, pc]:
                    overlaps_corridor = True
                    break
            if overlaps_corridor:
                continue
            return r0, c0, int(safe_len), corridor
    return None


def _assign_terminal_count(
    rng: np.random.Generator,
    current_idx: int,
    cup: dict,
    all_cups: list[dict],
    reserved_clear: np.ndarray,
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    border_owner: dict[tuple[int, int], int] = {}
    for idx, other in enumerate(all_cups):
        if "border_set" not in other:
            continue
        for cell in other["border_set"]:
            border_owner[cell] = idx

    ray = _ray_cells(cup["emit_start"], cup["direction"], shape)
    if not ray:
        return None
    first_hit = None
    for idx, cell in enumerate(ray):
        owner = border_owner.get(cell)
        if owner is not None and owner != current_idx:
            first_hit = idx
            break
    max_free = len(ray) if first_hit is None else int(first_hit)
    if max_free < 1:
        return None

    modes = ["free"]
    if first_hit is None and len(cup["mouth_local"]) > len(ray):
        modes.append("edge")
    mode = str(rng.choice(np.array(modes)))

    if mode == "free":
        max_count = min(max_free, len(cup["mouth_local"]))
        if max_count < 1:
            return None
        cup["content_count"] = int(rng.integers(1, max_count + 1))
        painted = ray[: cup["content_count"]]
    else:
        min_count = len(ray) + 1
        if len(cup["mouth_local"]) < min_count:
            return None
        cup["content_count"] = int(rng.integers(min_count, len(cup["mouth_local"]) + 1))
        painted = ray[:]

    if any(reserved_clear[r, c] for r, c in painted):
        return None
    cup["terminal_mode"] = mode
    return painted


def _place_bystander(
    rng: np.random.Generator,
    H: int,
    W: int,
    cup: dict,
    occ_cup: np.ndarray,
    reserved_clear: np.ndarray,
) -> bool:
    placement = _try_place_initial_cup(rng, H, W, cup, occ_cup, reserved_clear, needs_successor=False)
    if placement is None:
        return False
    _commit_cup(cup, placement[0], placement[1], occ_cup)
    cup["content_count"] = int(rng.integers(1, len(cup["mouth_local"]) + 1))
    return True


def _assign_content_pixels(rng: np.random.Generator, cups: list[dict]) -> None:
    for cup in cups:
        idxs = rng.choice(len(cup["mouth_abs"]), size=cup["content_count"], replace=False)
        cup["content_cells"] = [cup["mouth_abs"][int(i)] for i in idxs]


def _render_input(cups: list[dict], shape: tuple[int, int], ignition_pos: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for cup in cups:
        paste(grid, cup["patch"], cup["r0"], cup["c0"], transparent=BG)
    for cup in cups:
        for r, c in cup["content_cells"]:
            grid[r, c] = int(cup["content_color"])
    grid[ignition_pos] = int(Color.YELLOW)
    return grid


def _validate_instance(inp: np.ndarray, expected: np.ndarray, cups: list[dict], chain_len: int, ignition_pos: tuple[int, int]) -> bool:
    if int(np.count_nonzero(inp == Color.YELLOW)) != 1:
        return False

    planned_chain_colors = [cups[i]["border_color"] for i in range(chain_len)]
    oracle_out, oracle_fired, oracle_hits = _fire_chain(inp, cups, 0, ignition_pos, BG)
    if not np.array_equal(oracle_out, expected):
        return False
    if oracle_fired != list(range(chain_len)):
        return False
    if oracle_hits[: chain_len - 1] != [i + 1 for i in range(chain_len - 1)]:
        return False
    if oracle_hits[chain_len - 1] is not None:
        return False

    for idx, cup in enumerate(cups):
        for r, c in cup["content_cells"]:
            expected_val = BG if idx < chain_len else cup["content_color"]
            if int(expected[r, c]) != int(expected_val):
                return False

    detected = _detect_cups(inp, BG)
    if len(detected) != len(cups):
        return False
    by_border = {cup["border_color"]: cup for cup in detected}
    if set(by_border) != {cup["border_color"] for cup in cups}:
        return False
    for cup in cups:
        detected_cup = by_border[cup["border_color"]]
        if detected_cup["direction"] != cup["direction"]:
            return False
        if detected_cup["content_color"] != cup["content_color"]:
            return False
        if len(detected_cup["content_cells"]) != cup["content_count"]:
            return False

    detected_ign_idx, detected_ignition = _find_ignition(inp, detected)
    if detected_ignition != ignition_pos:
        return False
    _det_out, fired_detected, hits_detected = _fire_chain(inp, detected, detected_ign_idx, detected_ignition, BG)
    fired_colors = [detected[i]["border_color"] for i in fired_detected]
    if fired_colors != planned_chain_colors:
        return False
    if hits_detected[: chain_len - 1] != [fired_detected[i + 1] for i in range(chain_len - 1)]:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    total_cups = int(rng.choice(np.array(TOTAL_CUPS_CHOICES)))
    chain_len = int(rng.integers(1, total_cups + 1))

    border_colors = [BORDER_PALETTE[int(i)] for i in rng.permutation(len(BORDER_PALETTE))[:total_cups]]
    cups: list[dict] = []
    for idx in range(total_cups):
        border_color = int(border_colors[idx])
        color_choices = [c for c in BORDER_PALETTE if c != border_color]
        content_color = int(color_choices[int(rng.integers(len(color_choices)))])
        min_capacity = 2 if idx < chain_len - 1 else 1
        cups.append(_sample_cup_spec(rng, border_color, content_color, min_capacity=min_capacity))

    occ_cup = np.zeros((H, W), dtype=bool)
    reserved_clear = np.zeros((H, W), dtype=bool)

    first_placement = _try_place_initial_cup(rng, H, W, cups[0], occ_cup, reserved_clear, needs_successor=chain_len > 1)
    if first_placement is None:
        return None
    _commit_cup(cups[0], first_placement[0], first_placement[1], occ_cup)
    ignition_pos = cups[0]["emit_start"]

    for idx in range(1, chain_len):
        placed = _try_place_successor(rng, H, W, cups[idx - 1], cups[idx], occ_cup, reserved_clear)
        if placed is None:
            return None
        r0, c0, safe_len, corridor = placed
        _commit_cup(cups[idx], r0, c0, occ_cup)
        cups[idx - 1]["content_count"] = _choose_hit_count(rng, len(cups[idx - 1]["mouth_local"]), safe_len)
        cups[idx - 1]["planned_hit"] = idx
        for r, c in corridor:
            reserved_clear[r, c] = True

    painted_last = _assign_terminal_count(rng, chain_len - 1, cups[chain_len - 1], cups[:chain_len], reserved_clear, (H, W))
    if painted_last is None:
        return None
    for r, c in painted_last:
        reserved_clear[r, c] = True
    cups[chain_len - 1]["planned_hit"] = None

    for idx in range(chain_len, total_cups):
        if not _place_bystander(rng, H, W, cups[idx], occ_cup, reserved_clear):
            return None
        reserved_clear[cups[idx]["emit_start"]] = True
        cups[idx]["planned_hit"] = None

    _assign_content_pixels(rng, cups)
    inp = _render_input(cups, (H, W), ignition_pos)
    expected, _fired_order, _hits = _fire_chain(inp, cups, 0, ignition_pos, BG)

    if not _validate_instance(inp, expected, cups, chain_len, ignition_pos):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
