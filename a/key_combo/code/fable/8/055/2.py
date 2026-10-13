import random
from collections import Counter

import numpy as np


BG = 0
SAND = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _cup_outline_info(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None

    pts = {tuple(map(int, p)) for p in coords}
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r1, c))

    if pts != expected:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "mouth_cols": tuple(range(c0 + 1, c1)),
        "inner_w": c1 - c0 - 1,
        "inner_h": r1 - r0 - 1,
    }


def _find_cups(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    bg = _get_bg(grid)
    cups: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, SAND}):
        for comp in _components_of_color(grid, color):
            info = _cup_outline_info(comp)
            if info is None:
                continue
            r0, c0, r1, c1 = info["bbox"]
            markers = []
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    value = int(grid[r, c])
                    if value != bg and value != int(color):
                        markers.append((r, c, value))
            assert len(markers) == 1, "each cup must contain exactly one marker pixel"
            mr, mc, marker_color = markers[0]
            cup = {
                "wall_color": int(color),
                "marker_color": int(marker_color),
                "marker_pos": (int(mr), int(mc)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mouth_cols": tuple(int(c) for c in info["mouth_cols"]),
                "inner_w": int(info["inner_w"]),
                "inner_h": int(info["inner_h"]),
                "capacity": int(info["inner_w"] * info["inner_h"]),
            }
            cups.append(cup)

    cups.sort(key=lambda cup: cup["bbox"][1])
    assert cups, "expected at least one cup"

    claimed_cols = {}
    for idx, cup in enumerate(cups):
        for c in cup["mouth_cols"]:
            assert 0 <= c < w
            assert c not in claimed_cols, "cup mouth intervals must be disjoint"
            claimed_cols[c] = idx
    return cups


def _stack_positions(cup: dict, height: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = cup["bbox"]
    cols = range(c0 + 1, c1)
    inside = [(r, c) for r in range(r1 - 1, r0, -1) for c in cols]
    overflow = [(r, c) for r in range(r0 - 1, -1, -1) for c in cols]
    return inside + overflow


def _render_output(shape: tuple[int, int], cups: list[dict], sand_coords: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)

    claimed_cols = {}
    for idx, cup in enumerate(cups):
        r0, c0, r1, c1 = cup["bbox"]
        for r in range(r0, r1 + 1):
            out[r, c0] = cup["wall_color"]
            out[r, c1] = cup["wall_color"]
        out[r1, c0 : c1 + 1] = cup["wall_color"]
        for c in cup["mouth_cols"]:
            claimed_cols[c] = idx

    per_cup = [0] * len(cups)
    missed = Counter()
    for _r, c in sand_coords:
        if c in claimed_cols:
            per_cup[claimed_cols[c]] += 1
        else:
            missed[int(c)] += 1

    for idx, cup in enumerate(cups):
        total = per_cup[idx] + 1
        positions = _stack_positions(cup, h)
        assert total <= len(positions), "overflow exceeds the canvas top"
        for r, c in positions[:total]:
            assert out[r, c] == BG
            out[r, c] = cup["marker_color"]

    for c, count in missed.items():
        assert 0 <= c < w
        for offset in range(count):
            r = h - 1 - offset
            assert out[r, c] == BG
            out[r, c] = SAND

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each grey sand pixel by column, recolor the sand that falls into a cup, and let excess spill above the rim.

    Input structure:
    - The grid has a black background, one or more upward-open cup outlines, and isolated grey sand pixels above them.
    - Each cup is a monochrome U-shaped outline with a single non-background marker pixel somewhere inside.
    - Different cups have disjoint mouth column intervals, so every sand column is claimed by at most one cup.

    Rule:
    - Clear all original sand pixels and all marker pixels, but keep every cup wall unchanged.
    - A grey sand pixel belongs to the unique cup whose mouth interval contains its column; otherwise it misses every cup.
    - For each cup, add its claimed sand count to its marker pixel (the marker counts as one in-cup grain), recolor that total to the marker color, and stack it bottom-up inside the cup, filling each row left-to-right.
    - If a cup fills completely, continue stacking the same colored grains in the same width directly above the rim, row by row upward.
    - Missed sand stays grey and stacks at the bottom of its own column.

    Key insight:
    - Cup ownership is decided only by the mouth column interval; the marker provides the color and also contributes one conserved grain to the final stack.
    """
    g = np.asarray(grid, dtype=int)
    cups = _find_cups(g)

    sand_coords = [tuple(map(int, p)) for p in np.argwhere(g == SAND)]
    return _render_output(g.shape, cups, sand_coords)


def _split_nonnegative(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    prev = -1
    out = []
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _draw_cup(grid: np.ndarray, cup: dict) -> None:
    r0, c0, r1, c1 = cup["bbox"]
    color = cup["wall_color"]
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color
    grid[r1, c0:c1 + 1] = color
    mr, mc = cup["marker_pos"]
    grid[mr, mc] = cup["marker_color"]


def _claim_columns(mouth_cols: tuple[int, ...], count: int) -> list[int]:
    if count <= 0:
        return []
    cols = list(mouth_cols)
    out: list[int] = []
    if len(cols) >= 2 and count >= 2:
        edges = [cols[0], cols[-1]]
        random.shuffle(edges)
        take = min(2, count)
        out.extend(edges[:take])
    while len(out) < count:
        pool = cols.copy()
        random.shuffle(pool)
        out.extend(pool)
    return out[:count]


def _place_sands(shape: tuple[int, int], band_h: int, jobs: list[int]) -> list[tuple[int, int]] | None:
    h, _w = shape
    assert band_h < h
    target = list(jobs)
    for _ in range(120):
        random.shuffle(target)
        target.sort(key=lambda c: (target.count(c), random.random()), reverse=True)
        placed: list[tuple[int, int]] = []
        used = set()
        ok = True
        for col in target:
            rows = list(range(band_h))
            random.shuffle(rows)
            rows.sort(reverse=True)
            pick = None
            for row in rows:
                if (row, col) in used:
                    continue
                blocked = False
                for rr, cc in placed:
                    if max(abs(rr - row), abs(cc - col)) <= 1:
                        blocked = True
                        break
                if not blocked:
                    pick = (row, col)
                    break
            if pick is None:
                ok = False
                break
            placed.append(pick)
            used.add(pick)
        if ok:
            return placed
    return None


def _make_cups(shape: tuple[int, int], n_cups: int, band_h: int) -> list[dict] | None:
    h, w = shape
    cups = []
    dims = []
    max_inner_w = 4 if n_cups == 3 else 5
    for _ in range(n_cups):
        inner_w = random.randint(2, max_inner_w)
        inner_h = random.randint(2, 3)
        dims.append((inner_w + 2, inner_h + 2, inner_w, inner_h))

    total_w = sum(bw for bw, _bh, _iw, _ih in dims)
    min_gap = 2
    border = 1
    needed = total_w + min_gap * (n_cups - 1) + 2 * border
    if needed > w:
        return None

    slack = w - needed
    extras = _split_nonnegative(slack, n_cups + 1)
    c = border + extras[0]
    boxes = []
    for i, (bw, bh, _iw, _ih) in enumerate(dims):
        boxes.append((c, bw, bh))
        c += bw + min_gap + extras[i + 1]

    colors = random.sample(PALETTE, 2 * n_cups)
    for idx, (c0, bw, bh) in enumerate(boxes):
        r0_min = band_h + 1
        r0_max = h - bh - 3
        if r0_min > r0_max:
            return None
        r0 = random.randint(r0_min, r0_max)
        c1 = c0 + bw - 1
        r1 = r0 + bh - 1
        marker_pos = (
            random.randint(r0 + 1, r1 - 1),
            random.randint(c0 + 1, c1 - 1),
        )
        cups.append(
            {
                "bbox": (r0, c0, r1, c1),
                "wall_color": colors[2 * idx],
                "marker_color": colors[2 * idx + 1],
                "marker_pos": marker_pos,
                "mouth_cols": tuple(range(c0 + 1, c1)),
                "inner_w": bw - 2,
                "inner_h": bh - 2,
                "capacity": (bw - 2) * (bh - 2),
            }
        )
    return cups


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    h = random.randint(20, 29)
    w = random.randint(20, 29)
    n_cups = random.choices([1, 2, 3], weights=[1, 5, 4])[0]
    band_h = random.randint(8, 11)

    cups = _make_cups((h, w), n_cups, band_h)
    if cups is None:
        return None

    overflow_candidates = []
    for idx, cup in enumerate(cups):
        soft_max = cup["inner_w"] * ((band_h + 1) // 2)
        if soft_max >= cup["capacity"]:
            overflow_candidates.append(idx)
        cup["soft_max"] = soft_max

    need_overflow = bool(overflow_candidates) and random.random() < 0.7
    need_missed = random.random() < 0.7
    need_zero = n_cups >= 2 and random.random() < 0.5

    overflow_idx = random.choice(overflow_candidates) if need_overflow else None
    zero_choices = [i for i in range(n_cups) if i != overflow_idx]
    zero_idx = random.choice(zero_choices) if need_zero and zero_choices else None

    total_claimed = 0
    for idx, cup in enumerate(cups):
        cap = cup["capacity"]
        soft_max = min(cup["soft_max"], cap + cup["inner_w"])
        if soft_max < 1:
            return None
        if idx == overflow_idx:
            if soft_max < cap:
                return None
            claimed = random.randint(cap, soft_max)
        elif idx == zero_idx:
            claimed = 0
        else:
            high = min(soft_max, max(1, cap - 1))
            if random.random() < 0.25 and cap - 1 <= soft_max:
                claimed = cap - 1
            else:
                claimed = random.randint(1, high)
        cup["claimed"] = claimed
        total_claimed += claimed

    blocked = set()
    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        blocked.update(range(c0, c1 + 1))
    free_cols = [c for c in range(w) if c not in blocked]
    missed_counts = {}
    if need_missed:
        if not free_cols:
            return None
        n_missed_cols = random.randint(1, min(2, len(free_cols)))
        for col in random.sample(free_cols, n_missed_cols):
            missed_counts[col] = random.randint(1, 3)

    jobs = []
    for cup in cups:
        jobs.extend(_claim_columns(cup["mouth_cols"], cup["claimed"]))
    for col, count in missed_counts.items():
        jobs.extend([col] * count)

    if not jobs:
        return None

    max_sands = ((band_h + 1) // 2) * ((w + 1) // 2)
    if len(jobs) > max_sands:
        return None

    sand_positions = _place_sands((h, w), band_h, jobs)
    if sand_positions is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for cup in cups:
        _draw_cup(grid, cup)
    for r, c in sand_positions:
        if grid[r, c] != BG:
            return None
        grid[r, c] = SAND

    meta = {
        "cups": cups,
        "missed_counts": dict(missed_counts),
        "sand_positions": sand_positions,
        "need_overflow": need_overflow,
        "need_missed": need_missed,
        "need_zero": need_zero,
    }
    return grid, meta


def _validate_instance(grid: np.ndarray, meta: dict) -> bool:
    parsed = _find_cups(grid)
    if len(parsed) != len(meta["cups"]):
        return False

    for want, got in zip(sorted(meta["cups"], key=lambda cup: cup["bbox"][1]), parsed):
        if want["bbox"] != got["bbox"]:
            return False
        if want["marker_color"] != got["marker_color"]:
            return False
        if want["mouth_cols"] != got["mouth_cols"]:
            return False

    out = main(grid)
    expected = _render_output(grid.shape, meta["cups"], meta["sand_positions"])
    if not np.array_equal(out, expected):
        return False

    total_input_sand = len(meta["sand_positions"])
    total_output_sand = int(np.count_nonzero(out == SAND))
    recolored = sum(int(np.count_nonzero(out == cup["marker_color"])) for cup in meta["cups"])
    if total_output_sand + recolored != total_input_sand + len(meta["cups"]):
        return False

    if meta["need_overflow"] and not any(cup["claimed"] + 1 > cup["capacity"] for cup in meta["cups"]):
        return False
    if meta["need_missed"] and not meta["missed_counts"]:
        return False
    if meta["need_zero"] and not any(cup["claimed"] == 0 for cup in meta["cups"]):
        return False
    if not any(cup["claimed"] > 0 for cup in meta["cups"]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _gen_attempt()
        if built is None:
            continue
        grid, meta = built
        if _validate_instance(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid funnel-and-cup input")
