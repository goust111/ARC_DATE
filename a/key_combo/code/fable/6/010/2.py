from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


DIR4 = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

ARM_LEN_RANGE = (1, 2)
WALL_EXT_RANGE = (0, 2)
CHAIN_LEN_CHOICES = (2, 3)
GAP_RANGE = (2, 4)
MAX_CONTENT = 4
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: Iterable[int] | None = None) -> list[int]:
    vals = set(map(int, grid.ravel()))
    if exclude is not None:
        vals -= set(map(int, exclude))
    return sorted(vals)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
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


def _canonical_coord(local: tuple[int, int], shape: tuple[int, int], direction: str) -> tuple[int, int]:
    r, c = local
    h, w = shape
    if direction == "R":
        return r, c
    if direction == "D":
        return w - 1 - c, r
    if direction == "L":
        return h - 1 - r, w - 1 - c
    return c, h - 1 - r


def _fill_order(
    cells: Iterable[tuple[int, int]],
    direction: str,
    *,
    from_mouth: bool,
    bbox: tuple[int, int, int, int] | None = None,
    shape: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    cells = list(cells)
    if not cells:
        return []
    if bbox is not None:
        r0, c0, r1, c1 = bbox
        use_shape = (r1 - r0 + 1, c1 - c0 + 1)
        local_map = {cell: (cell[0] - r0, cell[1] - c0) for cell in cells}
    else:
        assert shape is not None
        use_shape = shape
        local_map = {cell: cell for cell in cells}

    def key(cell: tuple[int, int]) -> tuple[int, int]:
        rr, cc = _canonical_coord(local_map[cell], use_shape, direction)
        return (cc, rr) if from_mouth else (-cc, rr)

    return sorted(cells, key=key)


def _entry_cells(interior: Iterable[tuple[int, int]], bbox: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if direction == "U":
        return sorted([p for p in interior if p[0] == r1])
    if direction == "D":
        return sorted([p for p in interior if p[0] == r0])
    if direction == "L":
        return sorted([p for p in interior if p[1] == c1])
    return sorted([p for p in interior if p[1] == c0])


def _extract_cups(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    cups: list[dict] = []
    for border_color in colors_present(g, exclude={bg}):
        for comp in components_mask(g == border_color, connectivity=8):
            if comp.shape[0] < 4:
                continue
            r0, c0, r1, c1 = bbox_coords(comp)
            bbox = (r0, c0, r1, c1)
            border_set = {(int(r), int(c)) for r, c in comp}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            edges = {
                "U": [p for p in border_set if p[0] == r0],
                "D": [p for p in border_set if p[0] == r1],
                "L": [p for p in border_set if p[1] == c0],
                "R": [p for p in border_set if p[1] == c1],
            }
            candidates = [
                (d, cells[0])
                for d, cells in edges.items()
                if len(cells) == 1 and cells[0] not in corners
            ]
            passing = []
            for direction, spike in candidates:
                interior = _flood_from_bbox_side(border_set, bbox, OPPOSITE[direction])
                counts = Counter()
                for r, c in interior:
                    v = int(g[r, c])
                    if v != bg and v != border_color:
                        counts[v] += 1
                if not counts:
                    continue
                content_color = int(counts.most_common(1)[0][0])
                content_cells = sorted([p for p in interior if int(g[p[0], p[1]]) == content_color])
                passing.append((direction, spike, interior, content_color, content_cells))
            if len(passing) != 1:
                continue
            direction, spike, interior, content_color, content_cells = passing[0]
            cups.append(
                {
                    "border_color": int(border_color),
                    "bbox": bbox,
                    "direction": direction,
                    "spike": tuple(map(int, spike)),
                    "interior": sorted(interior),
                    "entry_cells": _entry_cells(interior, bbox, direction),
                    "stack_slots": _fill_order(interior, direction, from_mouth=False, bbox=bbox),
                    "content_slots": _fill_order(interior, direction, from_mouth=True, bbox=bbox),
                    "content_color": content_color,
                    "content_cells": content_cells,
                }
            )
    cups.sort(key=lambda cup: (cup["bbox"][0], cup["bbox"][1], cup["border_color"]))
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject every cup's content along its tail, but redirect rays that enter another cup's opening into stacked payloads.

    Input structure:
    - Black background with several hollow cup frames, each in a unique border color.
    - Every cup has one converging tail spike on one side and an open mouth on the opposite side.
    - Each cup contains several same-colored content pixels inside its interior near the mouth side.

    Rule:
    - Identify every cup's tail direction from the unique spike side, then erase that cup's original contents.
    - Emit all of that cup's content from the spike tip along the tail direction.
    - If the ray first enters another cup through that target cup's mouth opening, redirect the whole payload into that cup and stack it neatly from the target's bottom side inward.
    - Otherwise keep the payload as a straight line outward from the spike, clipped at the grid edge.
    - A cup still emits its own original payload even if it also receives another cup's payload.

    Key insight:
    The cups form a directed transfer graph: first recover every cup's orientation and interior, then decide for each ray whether it stays free or becomes a stacked payload inside the next cup it enters through the mouth.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    cups = _extract_cups(g)

    entry_to_cup: dict[tuple[int, int], int] = {}
    for idx, cup in enumerate(cups):
        for cell in cup["entry_cells"]:
            entry_to_cup[cell] = idx

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

    incoming: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(len(cups))}
    for src_idx, cup in enumerate(cups):
        count = len(cup["content_cells"])
        if count == 0:
            continue
        dr, dc = DIR4[cup["direction"]]
        sr, sc = cup["spike"]
        receiver = None
        hit_t = None
        t = 1
        while True:
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < h and 0 <= cc < w):
                break
            idx = entry_to_cup.get((rr, cc))
            if idx is not None and idx != src_idx:
                receiver = idx
                hit_t = t
                break
            t += 1
        if receiver is None:
            for step in range(1, count + 1):
                rr = sr + dr * step
                cc = sc + dc * step
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                out[rr, cc] = cup["content_color"]
        else:
            incoming[receiver].append((int(hit_t), src_idx, cup["content_color"]))

    for dst_idx, payloads in incoming.items():
        if not payloads:
            continue
        slots = cups[dst_idx]["stack_slots"]
        pos = 0
        for _hit_t, src_idx, color in sorted(payloads):
            count = len(cups[src_idx]["content_cells"])
            for _ in range(count):
                if pos >= len(slots):
                    break
                r, c = slots[pos]
                out[r, c] = color
                pos += 1

    return out


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1:L + hL + 1, 0] = color
    if hR:
        patch[R + 1:R + hR + 1, w - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
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
    return sorted(seen)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    cells: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(cells)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for (r, c) in cells]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for (r, c) in cells]
    return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for (r, c) in cells]


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _sample_right_cup(rng: np.random.Generator, border_color: int) -> dict | None:
    for _ in range(80):
        L = int(rng.integers(ARM_LEN_RANGE[0], ARM_LEN_RANGE[1] + 1))
        R = int(rng.integers(ARM_LEN_RANGE[0], ARM_LEN_RANGE[1] + 1))
        hL = int(rng.integers(WALL_EXT_RANGE[0], WALL_EXT_RANGE[1] + 1))
        hR = int(rng.integers(WALL_EXT_RANGE[0], WALL_EXT_RANGE[1] + 1))
        if hL + hR == 0:
            continue
        patch_u, spike_u = _build_canonical_cup(L, R, hL, hR, border_color)
        if int(np.count_nonzero(patch_u)) < 5:
            continue
        mouth_u = _mouth_interior_coords(patch_u, border_color, "D")
        if not mouth_u:
            continue
        patch, spike, interior = _rotate_to_direction(patch_u, spike_u, mouth_u, "R")
        h, w = patch.shape
        bbox = (0, 0, h - 1, w - 1)
        entry = _entry_cells(interior, bbox, "R")
        if not entry:
            continue
        return {
            "border_color": border_color,
            "patch": patch,
            "shape": patch.shape,
            "spike_rel": spike,
            "interior_rel": sorted(interior),
            "entry_rel": entry,
            "stack_rel": _fill_order(interior, "R", from_mouth=False, shape=patch.shape),
            "content_rel": _fill_order(interior, "R", from_mouth=True, shape=patch.shape),
        }
    return None


def _expected_output_from_meta(h: int, w: int, cups: list[dict]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for cup in cups:
        r0, c0 = cup["top_left"]
        patch = cup["patch"]
        mask = patch != 0
        out[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]][mask] = patch[mask]

    for idx in range(1, len(cups)):
        prev = cups[idx - 1]
        cur = cups[idx]
        slots = cur["stack_abs"]
        for pos in range(prev["count"]):
            r, c = slots[pos]
            out[r, c] = prev["content_color"]

    last = cups[-1]
    sr, sc = last["spike_abs"]
    for step in range(1, last["count"] + 1):
        out[sr, sc + step] = last["content_color"]
    return out


def _build_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    chain_len = int(rng.choice(CHAIN_LEN_CHOICES, p=[0.4, 0.6]))
    palette = rng.choice(np.arange(1, 10), size=2 * chain_len, replace=False).tolist()
    cups_rel: list[dict] = []
    for i in range(chain_len):
        border_color = int(palette[2 * i])
        content_color = int(palette[2 * i + 1])
        cup = _sample_right_cup(rng, border_color)
        if cup is None:
            return None
        cup["content_color"] = content_color
        cups_rel.append(cup)

    tops: list[tuple[int, int]] = [(0, 0)]
    bboxes: list[tuple[int, int, int, int]] = []
    h0, w0 = cups_rel[0]["shape"]
    bboxes.append((0, 0, h0 - 1, w0 - 1))
    prev_spike_abs = cups_rel[0]["spike_rel"]

    for idx in range(1, chain_len):
        placed = False
        h_i, w_i = cups_rel[idx]["shape"]
        for _ in range(120):
            entry_r, entry_c = cups_rel[idx]["entry_rel"][int(rng.integers(0, len(cups_rel[idx]["entry_rel"])))]
            gap = int(rng.integers(GAP_RANGE[0], GAP_RANGE[1] + 1))
            r0 = prev_spike_abs[0] - entry_r
            c0 = prev_spike_abs[1] + gap - entry_c
            bb = (r0, c0, r0 + h_i - 1, c0 + w_i - 1)
            if any(_rects_too_close(bb, old, margin=1) for old in bboxes):
                continue
            tops.append((r0, c0))
            bboxes.append(bb)
            prev_spike_abs = (r0 + cups_rel[idx]["spike_rel"][0], c0 + cups_rel[idx]["spike_rel"][1])
            placed = True
            break
        if not placed:
            return None

    for idx, cup in enumerate(cups_rel[:-1]):
        max_fit = min(MAX_CONTENT, len(cup["content_rel"]), len(cups_rel[idx + 1]["stack_rel"]))
        if max_fit < 1:
            return None
        cup["count"] = int(rng.integers(1, max_fit + 1))
    last_max = min(MAX_CONTENT, len(cups_rel[-1]["content_rel"]))
    if last_max < 1:
        return None
    cups_rel[-1]["count"] = int(rng.integers(1, last_max + 1))

    counts = [cup["count"] for cup in cups_rel]
    if max(counts) == 1:
        return None
    if len(set(counts)) == 1:
        return None

    min_r = min(bb[0] for bb in bboxes)
    min_c = min(bb[1] for bb in bboxes)
    shifted_tops = [(r - min_r, c - min_c) for r, c in tops]
    shifted_bboxes = []
    cups: list[dict] = []
    for cup, (r0, c0) in zip(cups_rel, shifted_tops):
        h_i, w_i = cup["shape"]
        shifted_bboxes.append((r0, c0, r0 + h_i - 1, c0 + w_i - 1))
        cups.append(dict(cup, top_left=(r0, c0)))

    last = cups[-1]
    sr_last = last["top_left"][0] + last["spike_rel"][0]
    sc_last = last["top_left"][1] + last["spike_rel"][1]
    base_h = max(bb[2] for bb in shifted_bboxes) + 1
    base_w = max(max(bb[3] for bb in shifted_bboxes), sc_last + last["count"]) + 1
    if base_h > 26 or base_w > 26:
        return None

    top_pad = int(rng.integers(0, 3))
    left_pad = int(rng.integers(0, 3))
    bottom_pad = int(rng.integers(0, 3))
    right_pad = int(rng.integers(0, 3))
    h = base_h + top_pad + bottom_pad
    w = base_w + left_pad + right_pad
    if h > 30 or w > 30:
        return None

    inp = np.zeros((h, w), dtype=int)
    for cup in cups:
        r0 = cup["top_left"][0] + top_pad
        c0 = cup["top_left"][1] + left_pad
        cup["top_left"] = (r0, c0)
        cup["spike_abs"] = (r0 + cup["spike_rel"][0], c0 + cup["spike_rel"][1])
        cup["stack_abs"] = [(r0 + r, c0 + c) for r, c in cup["stack_rel"]]
        cup["content_abs"] = [(r0 + r, c0 + c) for r, c in cup["content_rel"]]
        patch = cup["patch"]
        mask = patch != 0
        inp[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]][mask] = patch[mask]
        for pos in range(cup["count"]):
            rr, cc = cup["content_abs"][pos]
            inp[rr, cc] = cup["content_color"]

    out = _expected_output_from_meta(h, w, cups)
    total_in = sum(cup["count"] for cup in cups)
    total_out = int(np.count_nonzero((out != 0) & ~np.isin(out, [cup["border_color"] for cup in cups])))
    if total_in != total_out:
        return None

    k = int(rng.integers(0, 4))
    inp = np.rot90(inp, k=k).copy()
    out = np.rot90(out, k=k).copy()
    return inp, out, cups


def _validate_generated_case(inp: np.ndarray, expected: np.ndarray, cup_count: int) -> bool:
    try:
        got = main(inp)
    except Exception:
        return False
    if not np.array_equal(got, expected):
        return False
    cups = _extract_cups(inp)
    if len(cups) != cup_count:
        return False
    if np.array_equal(inp, expected):
        return False
    input_payload = sum(len(cup["content_cells"]) for cup in cups)
    border_colors = {cup["border_color"] for cup in cups}
    output_payload = 0
    inside_payload = 0
    for cup in cups:
        border = cup["border_color"]
        for r, c in cup["interior"]:
            if expected[r, c] not in (0, border):
                inside_payload += 1
    for r in range(expected.shape[0]):
        for c in range(expected.shape[1]):
            v = int(expected[r, c])
            if v != 0 and v not in border_colors:
                output_payload += 1
    if input_payload != output_payload:
        return False
    if inside_payload == 0 or inside_payload == output_payload:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        built = _build_attempt(rng)
        if built is None:
            continue
        inp, expected, cups = built
        if _validate_generated_case(inp, expected, len(cups)):
            return inp
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
