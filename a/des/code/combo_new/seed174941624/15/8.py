from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 8
FRAME = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 400
PLACE_RETRIES = 160


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
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_bg(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]))
    return int(Counter(int(v) for v in border).most_common(1)[0][0])


def _coords_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _detect_key(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict[str, object] | None:
    if not coords:
        return None
    color = int(grid[coords[0]])
    if color == bg:
        return None
    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = _coords_mask(coords, bbox)

    def build(target_side: str, bits: np.ndarray) -> dict[str, object] | None:
        pattern = tuple(int(v) for v in bits)
        if len(pattern) < 4 or len(pattern) > 6:
            return None
        if not any(pattern) or all(pattern):
            return None
        return {
            "color": color,
            "coords": [(int(r), int(c)) for r, c in coords],
            "bbox": bbox,
            "target_side": target_side,
            "pattern": pattern,
        }

    if h == 2 and 4 <= w <= 6:
        top = mask[0, :]
        bottom = mask[1, :]
        if np.all(top) and np.any(bottom) and np.any(~bottom):
            return build("top", bottom.astype(int))
        if np.all(bottom) and np.any(top) and np.any(~top):
            return build("bottom", top.astype(int))
        return None

    if w == 2 and 4 <= h <= 6:
        left = mask[:, 0]
        right = mask[:, 1]
        if np.all(left) and np.any(right) and np.any(~right):
            return build("left", right.astype(int))
        if np.all(right) and np.any(left) and np.any(~left):
            return build("right", left.astype(int))
        return None

    return None


def _detect_container(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    bg: int,
    allowed_patterns: dict[str, set[tuple[int, ...]]],
) -> dict[str, object] | None:
    if not coords:
        return None
    color = int(grid[coords[0]])
    if color == bg:
        return None

    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    coord_set = set(coords)
    for r, c in coord_set:
        if not (r == r0 or r == r1 or c == c0 or c == c1):
            return None

    if r1 - r0 >= 2 and c1 - c0 >= 2:
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size and np.any(interior != bg):
            return None

    lines = {
        "top": np.array([(r0, c) in coord_set for c in range(c0, c1 + 1)], dtype=bool),
        "bottom": np.array([(r1, c) in coord_set for c in range(c0, c1 + 1)], dtype=bool),
        "left": np.array([(r, c0) in coord_set for r in range(r0, r1 + 1)], dtype=bool),
        "right": np.array([(r, c1) in coord_set for r in range(r0, r1 + 1)], dtype=bool),
    }
    non_full = [side for side, line in lines.items() if np.any(~line)]
    if len(non_full) != 1:
        return None

    side = non_full[0]
    line = lines[side]
    n = len(line)
    patterns = allowed_patterns.get(side, set())
    if not patterns:
        return None

    candidates: list[tuple[int, tuple[int, ...]]] = []
    for length in range(4, 7):
        if n < length + 2:
            continue
        for start in range(1, n - length):
            segment = line[start : start + length]
            if not (np.any(segment) and np.any(~segment)):
                continue
            if not np.all(line[:start]) or not np.all(line[start + length :]):
                continue
            pattern = tuple(1 if not v else 0 for v in segment)
            if pattern in patterns:
                candidates.append((int(start), pattern))

    if len(candidates) != 1:
        bounded = [cand for cand in candidates if cand[1][0] == 1 and cand[1][-1] == 1]
        if len(bounded) == 1:
            candidates = bounded
        else:
            return None

    start, pattern = candidates[0]
    return {
        "color": color,
        "bbox": bbox,
        "side": side,
        "start": start,
        "pattern": pattern,
    }


def _draw_match(out: np.ndarray, key: dict[str, object], container: dict[str, object]) -> None:
    color = int(key["color"])
    r0, c0, r1, c1 = container["bbox"]
    start = int(container["start"])
    pattern = tuple(int(v) for v in container["pattern"])
    length = len(pattern)

    if container["side"] == "top":
        cols = [c0 + start + i for i in range(length)]
        out[r0 - 1, cols] = color
        for bit, col in zip(pattern, cols):
            if not bit:
                continue
            out[r0, col] = color
            out[r0 + 1 : r1, col] = color
        return

    if container["side"] == "bottom":
        cols = [c0 + start + i for i in range(length)]
        out[r1 + 1, cols] = color
        for bit, col in zip(pattern, cols):
            if not bit:
                continue
            out[r1, col] = color
            out[r0 + 1 : r1, col] = color
        return

    if container["side"] == "left":
        rows = [r0 + start + i for i in range(length)]
        out[rows, c0 - 1] = color
        for bit, row in zip(pattern, rows):
            if not bit:
                continue
            out[row, c0] = color
            out[row, c0 + 1 : c1] = color
        return

    rows = [r0 + start + i for i in range(length)]
    out[rows, c1 + 1] = color
    for bit, row in zip(pattern, rows):
        if not bit:
            continue
        out[row, c1] = color
        out[row, c0 + 1 : c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each toothed key into the matching jagged frame notch and extend its teeth through the container interior.

    Input structure:
    - The background is azure and fills the grid border.
    - Several grey one-pixel rectangular containers appear, each with exactly one side containing a 4-6 cell mixed missing/present notch segment.
    - Separate colored key fragments are single-color 4-connected components shaped as a straight 4-6 cell bar with one-cell teeth on one side.

    Rule:
    - Read each key's tooth pattern from the cells that protrude from its straight bar; the tooth side fixes which container edge it can enter.
    - For each grey rectangle, locate the unique notch segment on its damaged side whose missing/present pattern matches one key's teeth.
    - Erase the key from its original location, translate it onto that edge so the protruding teeth fill the missing border cells, and keep the intact grey border cells unchanged.
    - Every protruding tooth then continues inward in the insertion direction, coloring the full interior column or row until the opposite border.

    Key insight:
    - The colored fragments are not paintbrushes or labels: their side and binary tooth pattern are geometric keys that identify a matching notch, and only the toothed positions grow across the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _border_bg(g)

    comps: list[list[tuple[int, int]]] = []
    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        comps.extend(_components(g == color))

    keys: list[dict[str, object]] = []
    leftovers: list[dict[str, object]] = []
    for comp in comps:
        key = _detect_key(g, comp, bg)
        if key is not None:
            keys.append(key)
        else:
            leftovers.append(
                {
                    "color": int(g[comp[0]]),
                    "coords": [(int(r), int(c)) for r, c in comp],
                    "bbox": _bbox(comp),
                }
            )

    allowed_patterns: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for key in keys:
        allowed_patterns[str(key["target_side"])].add(tuple(int(v) for v in key["pattern"]))

    containers: list[dict[str, object]] = []
    leftovers_by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for item in leftovers:
        leftovers_by_color[int(item["color"])].append(item)

    for color_items in leftovers_by_color.values():
        order = sorted(range(len(color_items)), key=lambda i: len(color_items[i]["coords"]), reverse=True)
        used: set[int] = set()
        for idx in order:
            if idx in used:
                continue
            base = color_items[idx]
            br0, bc0, br1, bc1 = (int(v) for v in base["bbox"])
            merged = list(base["coords"])
            member_ids = [idx]
            for j, item in enumerate(color_items):
                if j == idx or j in used:
                    continue
                ir0, ic0, ir1, ic1 = (int(v) for v in item["bbox"])
                if not (br0 <= ir0 <= ir1 <= br1 and bc0 <= ic0 <= ic1 <= bc1):
                    continue
                if all(r in {br0, br1} or c in {bc0, bc1} for r, c in item["coords"]):
                    merged.extend(item["coords"])
                    member_ids.append(j)
            container = _detect_container(g, merged, bg, allowed_patterns)
            if container is not None:
                containers.append(container)
                used.update(member_ids)

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = bg

    keys_by_sig: dict[tuple[str, tuple[int, ...]], list[dict[str, object]]] = defaultdict(list)
    containers_by_sig: dict[tuple[str, tuple[int, ...]], list[dict[str, object]]] = defaultdict(list)
    for key in keys:
        sig = (str(key["target_side"]), tuple(int(v) for v in key["pattern"]))
        keys_by_sig[sig].append(key)
    for container in containers:
        sig = (str(container["side"]), tuple(int(v) for v in container["pattern"]))
        containers_by_sig[sig].append(container)

    for sig, matched_keys in keys_by_sig.items():
        matched_containers = containers_by_sig.get(sig, [])
        assert len(matched_keys) == len(matched_containers) and matched_keys, "keys and containers must match uniquely"
        matched_keys.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))
        matched_containers.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))
        for key, container in zip(matched_keys, matched_containers):
            _draw_match(out, key, container)

    return out


def _random_pattern(rng: random.Random, length: int) -> tuple[int, ...]:
    while True:
        pattern = [1]
        pattern.extend(rng.randint(0, 1) for _ in range(length - 2))
        pattern.append(1)
        if any(v == 0 for v in pattern):
            return tuple(pattern)


def _reserve_box(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(occ.shape[0], r + h + pad)
    cc1 = min(occ.shape[1], c + w + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _find_free_box(
    occ: np.ndarray,
    h: int,
    w: int,
    rng: random.Random,
    *,
    pad: int,
    margin: int,
) -> tuple[int, int] | None:
    height, width = occ.shape
    if height < h + 2 * margin or width < w + 2 * margin:
        return None

    positions = [
        (r, c)
        for r in range(margin, height - h - margin + 1)
        for c in range(margin, width - w - margin + 1)
    ]
    rng.shuffle(positions)
    for r, c in positions[:PLACE_RETRIES]:
        rr0 = max(0, r - pad)
        cc0 = max(0, c - pad)
        rr1 = min(height, r + h + pad)
        cc1 = min(width, c + w + pad)
        if not np.any(occ[rr0:rr1, cc0:cc1]):
            return int(r), int(c)
    return None


def _paint_container(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    side: str,
    start: int,
    pattern: tuple[int, ...],
) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    if side in {"top", "bottom"}:
        row = r0 if side == "top" else r1
        for i, bit in enumerate(pattern):
            if bit:
                grid[row, c0 + start + i] = BG
        return

    col = c0 if side == "left" else c1
    for i, bit in enumerate(pattern):
        if bit:
            grid[r0 + start + i, col] = BG


def _key_patch(side: str, pattern: tuple[int, ...], color: int) -> np.ndarray:
    length = len(pattern)
    if side == "top":
        patch = np.full((2, length), BG, dtype=int)
        patch[0, :] = color
        for i, bit in enumerate(pattern):
            if bit:
                patch[1, i] = color
        return patch
    if side == "bottom":
        patch = np.full((2, length), BG, dtype=int)
        patch[1, :] = color
        for i, bit in enumerate(pattern):
            if bit:
                patch[0, i] = color
        return patch
    if side == "left":
        patch = np.full((length, 2), BG, dtype=int)
        patch[:, 0] = color
        for i, bit in enumerate(pattern):
            if bit:
                patch[i, 1] = color
        return patch
    patch = np.full((length, 2), BG, dtype=int)
    patch[:, 1] = color
    for i, bit in enumerate(pattern):
        if bit:
            patch[i, 0] = color
    return patch


def _place_patch(grid: np.ndarray, occ: np.ndarray, patch: np.ndarray, rng: random.Random) -> bool:
    h, w = patch.shape
    pos = _find_free_box(occ, h, w, rng, pad=1, margin=1)
    if pos is None:
        return False
    r0, c0 = pos
    mask = patch != BG
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
    _reserve_box(occ, r0, c0, h, w, pad=1)
    return True


def _spec_size(
    side: str,
    length: int,
    pair_count: int,
    height: int,
    width: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    cap = {1: 10, 2: 9, 3: 8}[pair_count]
    h_max = min(cap, height - 2)
    w_max = min(cap, width - 2)
    if side in {"top", "bottom"}:
        h_min, w_min = 6, max(length + 2, 6)
    else:
        h_min, w_min = max(length + 2, 6), 6
    if h_min > h_max or w_min > w_max:
        return None
    return rng.randint(h_min, h_max), rng.randint(w_min, w_max)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(1, 3)
    if pair_count == 1:
        height = rng.randint(12, 20)
        width = rng.randint(12, 20)
    elif pair_count == 2:
        height = rng.randint(18, 26)
        width = rng.randint(18, 26)
    else:
        height = rng.randint(22, 30)
        width = rng.randint(22, 30)

    grid = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    used_signatures: set[tuple[str, tuple[int, ...]]] = set()
    colors = list(KEY_COLORS)
    rng.shuffle(colors)
    specs: list[dict[str, object]] = []

    for i in range(pair_count):
        for _ in range(PLACE_RETRIES):
            side = rng.choice(("top", "bottom", "left", "right"))
            length = rng.randint(4, 6)
            pattern = _random_pattern(rng, length)
            sig = (side, pattern)
            if sig in used_signatures:
                continue
            size = _spec_size(side, length, pair_count, height, width, rng)
            if size is None:
                continue
            h, w = size
            if side in {"top", "bottom"}:
                start = rng.randint(1, w - length - 1)
            else:
                start = rng.randint(1, h - length - 1)
            specs.append(
                {
                    "side": side,
                    "pattern": pattern,
                    "h": h,
                    "w": w,
                    "start": start,
                    "color": colors[i],
                }
            )
            used_signatures.add(sig)
            break
        else:
            return None

    specs.sort(key=lambda spec: max(int(spec["h"]), int(spec["w"])), reverse=True)

    for spec in specs:
        h = int(spec["h"])
        w = int(spec["w"])
        pos = _find_free_box(occ, h, w, rng, pad=1, margin=1)
        if pos is None:
            return None
        r0, c0 = pos
        spec["r0"] = r0
        spec["c0"] = c0
        _paint_container(
            grid,
            r0,
            c0,
            h,
            w,
            str(spec["side"]),
            int(spec["start"]),
            tuple(int(v) for v in spec["pattern"]),
        )
        _reserve_box(occ, r0, c0, h, w, pad=1)

    rng.shuffle(specs)
    for spec in specs:
        patch = _key_patch(str(spec["side"]), tuple(int(v) for v in spec["pattern"]), int(spec["color"]))
        if not _place_patch(grid, occ, patch, rng):
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
