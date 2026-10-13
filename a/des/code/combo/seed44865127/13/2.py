from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    total = int(prefix[r1, c1])
    if r0:
        total -= int(prefix[r0 - 1, c1])
    if c0:
        total -= int(prefix[r1, c0 - 1])
    if r0 and c0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _side_info(line: np.ndarray, bg: int) -> tuple[int, list[int]] | None:
    vals = [int(v) for v in line.tolist()]
    non_bg = [v for v in vals if v != bg]
    if len(non_bg) < 2 or len(set(non_bg)) != 1:
        return None
    gaps = [i for i, v in enumerate(vals) if v == bg]
    if any(i == 0 or i == len(vals) - 1 for i in gaps):
        return None
    if any(b == a + 1 for a, b in zip(gaps, gaps[1:])):
        return None
    return int(non_bg[0]), gaps


def _lock_candidate(
    grid: np.ndarray,
    bg: int,
    prefix: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> dict | None:
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None
    if int(grid[r0, c0]) == bg or int(grid[r0, c1]) == bg or int(grid[r1, c0]) == bg or int(grid[r1, c1]) == bg:
        return None
    if _rect_sum(prefix, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
        return None

    top = _side_info(grid[r0, c0 + 1 : c1], bg)
    bottom = _side_info(grid[r1, c0 + 1 : c1], bg)
    left = _side_info(grid[r0 + 1 : r1, c0], bg)
    right = _side_info(grid[r0 + 1 : r1, c1], bg)
    if top is None or bottom is None or left is None or right is None:
        return None

    top_color, top_gaps = top
    bottom_color, bottom_gaps = bottom
    left_color, left_gaps = left
    right_color, right_gaps = right
    if len({top_color, bottom_color, left_color, right_color}) < 2:
        return None

    gap_entries: list[tuple[int, int, int]] = []
    for idx in top_gaps:
        gap_entries.append((r0, c0 + 1 + idx, top_color))
    for idx in left_gaps:
        gap_entries.append((r0 + 1 + idx, c0, left_color))
    for idx in right_gaps:
        gap_entries.append((r0 + 1 + idx, c1, right_color))
    for idx in bottom_gaps:
        gap_entries.append((r1, c0 + 1 + idx, bottom_color))
    if not gap_entries:
        return None

    gap_entries.sort(key=lambda item: (item[0], item[1]))
    sequence = tuple(color for _r, _c, color in gap_entries)
    return {
        "bbox": (r0, c0, r1, c1),
        "sequence": sequence,
    }


def _find_locks(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    non_bg = (g != bg).astype(int)
    prefix = non_bg.cumsum(axis=0).cumsum(axis=1)

    candidates: list[dict] = []
    rows = [int(r) for r in np.flatnonzero(non_bg.any(axis=1))]
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            if r1 - r0 < 4:
                continue
            common_cols = np.flatnonzero((g[r0] != bg) & (g[r1] != bg))
            if len(common_cols) < 2:
                continue
            cols = [int(c) for c in common_cols]
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    if c1 - c0 < 4:
                        continue
                    info = _lock_candidate(g, bg, prefix, r0, c0, r1, c1)
                    if info is not None:
                        candidates.append(info)

    candidates.sort(
        key=lambda item: (
            (item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1),
            item["bbox"][0],
            item["bbox"][1],
        )
    )

    selected: list[dict] = []
    for cand in candidates:
        if any(_boxes_overlap(cand["bbox"], other["bbox"]) for other in selected):
            continue
        selected.append(cand)

    selected.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return selected


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_key(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict | None:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if np.any(patch == bg):
        return None

    h, w = patch.shape
    variants: list[dict] = []

    if h >= 2 and w >= 3:
        body = patch[:, :-1]
        body_color = int(body[0, 0])
        teeth = patch[:, -1]
        if np.all(body == body_color) and np.all(teeth != body_color):
            variants.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "body_color": body_color,
                    "sequence": tuple(int(v) for v in teeth.tolist()),
                }
            )
        body = patch[:, 1:]
        body_color = int(body[0, 0])
        teeth = patch[:, 0]
        if np.all(body == body_color) and np.all(teeth != body_color):
            variants.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "body_color": body_color,
                    "sequence": tuple(int(v) for v in teeth.tolist()),
                }
            )

    if h >= 3 and w >= 2:
        body = patch[1:, :]
        body_color = int(body[0, 0])
        teeth = patch[0, :]
        if np.all(body == body_color) and np.all(teeth != body_color):
            variants.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "body_color": body_color,
                    "sequence": tuple(int(v) for v in teeth.tolist()),
                }
            )
        body = patch[:-1, :]
        body_color = int(body[0, 0])
        teeth = patch[-1, :]
        if np.all(body == body_color) and np.all(teeth != body_color):
            variants.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "body_color": body_color,
                    "sequence": tuple(int(v) for v in teeth.tolist()),
                }
            )

    uniq: list[dict] = []
    seen = set()
    for item in variants:
        key = (item["bbox"], item["body_color"], item["sequence"])
        if key in seen or len(item["sequence"]) < 2:
            continue
        seen.add(key)
        uniq.append(item)
    if len(uniq) != 1:
        return None
    return uniq[0]


def _find_keys(grid: np.ndarray, bg: int, locks: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    lock_mask = np.zeros_like(g, dtype=bool)
    for lock in locks:
        r0, c0, r1, c1 = lock["bbox"]
        lock_mask[r0, c0 : c1 + 1] = True
        lock_mask[r1, c0 : c1 + 1] = True
        lock_mask[r0 : r1 + 1, c0] = True
        lock_mask[r0 : r1 + 1, c1] = True

    mask = (g != bg) & (~lock_mask)
    keys: list[dict] = []
    for comp in _components(mask):
        key = _parse_key(g, comp, bg)
        if key is not None:
            keys.append(key)
    keys.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match color-code keys to hollow locks and fill each unlocked interior with the key body color.

    Input structure:
    - Background is the dominant color.
    - Locks are hollow one-cell-thick rectangles with side colors and single-cell gaps on the frame.
    - Keys are solid monochrome bodies with a one-cell tooth strip on one side.

    Rule:
    - Read each lock's password from the colors associated with its gaps in row-major order.
    - Read each key's code from the tooth-strip colors in natural strip order.
    - When a key code matches a lock code exactly, fill the lock interior with the key body color.
    - Remove used keys; leave unmatched locks and keys unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    locks = _find_locks(g, bg)
    if not locks:
        return out
    keys = _find_keys(g, bg, locks)

    by_sequence: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for key in keys:
        by_sequence[key["sequence"]].append(key)

    used_boxes: set[tuple[int, int, int, int]] = set()
    for lock in locks:
        matches = by_sequence.get(lock["sequence"], [])
        key = None
        while matches:
            candidate = matches.pop(0)
            if candidate["bbox"] not in used_boxes:
                key = candidate
                break
        if key is None:
            continue
        used_boxes.add(key["bbox"])

        r0, c0, r1, c1 = lock["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(key["body_color"])

        kr0, kc0, kr1, kc1 = key["bbox"]
        out[kr0 : kr1 + 1, kc0 : kc1 + 1] = bg

    return out


def _safe_gap_positions(length: int) -> list[int]:
    return list(range(1, length - 1))


def _sample_gaps(h: int, w: int) -> dict[str, list[int]] | None:
    top_len = w - 2
    side_len = h - 2
    candidates = (
        [("top", idx) for idx in _safe_gap_positions(top_len)]
        + [("bottom", idx) for idx in _safe_gap_positions(top_len)]
        + [("left", idx) for idx in _safe_gap_positions(side_len)]
        + [("right", idx) for idx in _safe_gap_positions(side_len)]
    )
    if len(candidates) < 2:
        return None

    goal = random.randint(2, min(5, len(candidates)))
    random.shuffle(candidates)
    picked: dict[str, list[int]] = {"top": [], "bottom": [], "left": [], "right": []}
    for side, idx in candidates:
        if len(picked["top"]) + len(picked["bottom"]) + len(picked["left"]) + len(picked["right"]) >= goal:
            break
        if any(abs(idx - prev) <= 1 for prev in picked[side]):
            continue
        picked[side].append(idx)

    total = sum(len(v) for v in picked.values())
    if total < 2:
        return None
    for side in picked:
        picked[side].sort()
    return picked


def _lock_patch(bg: int) -> dict | None:
    for _ in range(80):
        h = random.randint(5, 9)
        w = random.randint(5, 10)
        gaps = _sample_gaps(h, w)
        if gaps is None:
            continue

        colors = [c for c in range(10) if c != bg]
        palette = random.sample(colors, random.randint(2, 4))
        side_colors = {
            "top": random.choice(palette),
            "bottom": random.choice(palette),
            "left": random.choice(palette),
            "right": random.choice(palette),
        }
        if len(set(side_colors.values())) < 2:
            continue

        patch = np.full((h, w), bg, dtype=int)
        patch[0, :] = side_colors["top"]
        patch[-1, :] = side_colors["bottom"]
        patch[1:-1, 0] = side_colors["left"]
        patch[1:-1, -1] = side_colors["right"]

        for idx in gaps["top"]:
            patch[0, 1 + idx] = bg
        for idx in gaps["bottom"]:
            patch[-1, 1 + idx] = bg
        for idx in gaps["left"]:
            patch[1 + idx, 0] = bg
        for idx in gaps["right"]:
            patch[1 + idx, -1] = bg

        seq_entries: list[tuple[int, int, int]] = []
        for idx in gaps["top"]:
            seq_entries.append((0, 1 + idx, side_colors["top"]))
        for idx in gaps["left"]:
            seq_entries.append((1 + idx, 0, side_colors["left"]))
        for idx in gaps["right"]:
            seq_entries.append((1 + idx, w - 1, side_colors["right"]))
        for idx in gaps["bottom"]:
            seq_entries.append((h - 1, 1 + idx, side_colors["bottom"]))
        seq_entries.sort(key=lambda item: (item[0], item[1]))
        sequence = tuple(color for _r, _c, color in seq_entries)

        return {
            "patch": patch,
            "sequence": sequence,
        }
    return None


def _key_patch(bg: int, sequence: tuple[int, ...]) -> dict:
    orient = random.choice(("right", "left", "top", "bottom"))
    colors = [c for c in range(10) if c != bg and c not in set(sequence)]
    body_color = random.choice(colors)

    if orient in ("right", "left"):
        body_h = len(sequence)
        body_w = random.randint(2, 4)
        patch = np.full((body_h, body_w + 1), body_color, dtype=int)
        tooth_col = 0 if orient == "left" else body_w
        patch[:, tooth_col] = np.array(sequence, dtype=int)
    else:
        body_h = random.randint(2, 4)
        body_w = len(sequence)
        patch = np.full((body_h + 1, body_w), body_color, dtype=int)
        tooth_row = 0 if orient == "top" else body_h
        patch[tooth_row, :] = np.array(sequence, dtype=int)

    return {
        "patch": patch,
        "sequence": sequence,
        "body_color": body_color,
    }


def _random_sequence(bg: int, forbidden: set[tuple[int, ...]]) -> tuple[int, ...]:
    colors = [c for c in range(10) if c != bg]
    for _ in range(120):
        length = random.randint(2, 5)
        seq = tuple(random.choice(colors) for _ in range(length))
        if seq not in forbidden and len(set(seq)) >= 1:
            return seq
    raise RuntimeError("failed to sample a distinct key sequence")


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    occupied[top : top + h, left : left + w] = True


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _paint_expected(expected: np.ndarray, bg: int, locks: list[dict], keys: list[dict]) -> None:
    matched_keys = {key["match_lock"]: key for key in keys if key.get("match_lock") is not None}
    for lock_idx, key in matched_keys.items():
        lock = locks[lock_idx]
        r0, c0 = lock["top"], lock["left"]
        h, w = lock["patch"].shape
        expected[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = int(key["body_color"])
        kh, kw = key["patch"].shape
        expected[key["top"] : key["top"] + kh, key["left"] : key["left"] + kw] = bg


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.randrange(10)
    scenario = random.choice(("perfect", "extra_keys", "extra_locks", "mixed"))
    if scenario == "perfect":
        n_locks = random.randint(1, 2)
        n_match = n_locks
        n_extra_keys = 0
    elif scenario == "extra_keys":
        n_locks = random.randint(1, 2)
        n_match = n_locks
        n_extra_keys = random.randint(1, 2)
    elif scenario == "extra_locks":
        n_locks = random.randint(2, 3)
        n_match = random.randint(1, n_locks - 1)
        n_extra_keys = 0
    else:
        n_locks = random.randint(2, 3)
        n_match = random.randint(1, n_locks - 1)
        n_extra_keys = random.randint(1, 2)

    locks: list[dict] = []
    used_sequences: set[tuple[int, ...]] = set()
    for _ in range(n_locks):
        for _attempt in range(100):
            lock = _lock_patch(bg)
            if lock is None or lock["sequence"] in used_sequences:
                continue
            used_sequences.add(lock["sequence"])
            locks.append(lock)
            break
        else:
            return None

    matched_indices = sorted(random.sample(range(n_locks), n_match))
    keys: list[dict] = []
    for lock_idx in matched_indices:
        key = _key_patch(bg, locks[lock_idx]["sequence"])
        key["match_lock"] = lock_idx
        keys.append(key)

    for _ in range(n_extra_keys):
        seq = _random_sequence(bg, used_sequences)
        used_sequences.add(seq)
        key = _key_patch(bg, seq)
        key["match_lock"] = None
        keys.append(key)

    objects = [("lock", idx, locks[idx]["patch"].shape) for idx in range(len(locks))]
    objects += [("key", idx, keys[idx]["patch"].shape) for idx in range(len(keys))]
    objects.sort(key=lambda item: item[2][0] * item[2][1], reverse=True)

    h = random.randint(22, 32)
    w = random.randint(22, 34)
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for kind, idx, shape in objects:
        oh, ow = shape
        placed = False
        for _ in range(200):
            top = random.randint(0, h - oh)
            left = random.randint(0, w - ow)
            if not _can_place(occupied, top, left, oh, ow, pad=1):
                continue
            _mark_place(occupied, top, left, oh, ow)
            if kind == "lock":
                locks[idx]["top"] = top
                locks[idx]["left"] = left
                _paste(grid, locks[idx]["patch"], top, left)
            else:
                keys[idx]["top"] = top
                keys[idx]["left"] = left
                _paste(grid, keys[idx]["patch"], top, left)
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    _paint_expected(expected, bg, locks, keys)

    if _get_bg(grid) != bg:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _build_case()
        if case is not None:
            grid, _expected = case
            return grid
    raise RuntimeError("failed to generate a valid lock-key ARC input")
