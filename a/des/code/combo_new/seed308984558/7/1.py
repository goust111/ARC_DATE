import math
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("T", "B", "L", "R")
FRACTIONS = ((1, 2), (1, 3), (2, 3), (1, 4), (3, 4))
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250
SIDE_ORDER = {"T": 0, "B": 1, "L": 2, "R": 3}


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(flat).argmax())


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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_candidate(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    pts = {tuple(map(int, xy)) for xy in coords}
    boundary: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        boundary.add((r0, c))
        boundary.add((r1, c))
    for r in range(r0 + 1, r1):
        boundary.add((r, c0))
        boundary.add((r, c1))

    if not pts <= boundary:
        return None

    missing = sorted(boundary - pts)
    if not missing:
        return None

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if any(cell in corners for cell in missing):
        return None

    side_positions: dict[str, list[int]] = {side: [] for side in SIDES}
    pattern: list[tuple[str, int, int]] = []
    for r, c in missing:
        if r == r0:
            side, pos, den = "T", c - c0, w - 1
        elif r == r1:
            side, pos, den = "B", c - c0, w - 1
        elif c == c0:
            side, pos, den = "L", r - r0, h - 1
        else:
            side, pos, den = "R", r - r0, h - 1
        if pos <= 0 or pos >= den:
            return None
        side_positions[side].append(pos)
        g = math.gcd(pos, den)
        pattern.append((side, pos // g, den // g))

    for positions in side_positions.values():
        positions.sort()
        if any(b - a == 1 for a, b in zip(positions, positions[1:])):
            return None

    pattern.sort(key=lambda item: (SIDE_ORDER[item[0]], item[1], item[2]))
    return {"bbox": (r0, c0, r1, c1), "pattern": tuple(pattern), "coords": coords}


def _filled_rectangle(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if coords.shape[0] != h * w:
        return None
    return {"bbox": (r0, c0, r1, c1), "coords": coords}


def _touches_bbox(frame_bbox: tuple[int, int, int, int], block_bbox: tuple[int, int, int, int]) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    br0, bc0, br1, bc1 = block_bbox
    if bc0 == fc1 + 1 and not (br1 < fr0 or br0 > fr1):
        return True
    if bc1 + 1 == fc0 and not (br1 < fr0 or br0 > fr1):
        return True
    if br0 == fr1 + 1 and not (bc1 < fc0 or bc0 > fc1):
        return True
    if br1 + 1 == fr0 and not (bc1 < fc0 or bc0 > fc1):
        return True
    return False


def _detect_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _background_color(grid)
    candidate_by_color: dict[int, list[dict]] = {}
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        base_mask = grid == color
        merged_mask = _dilate4(base_mask)
        cands: list[dict] = []
        for comp in _components(merged_mask):
            region = np.zeros_like(base_mask, dtype=bool)
            region[comp[:, 0], comp[:, 1]] = True
            actual = np.argwhere(region & base_mask)
            cand = _frame_candidate(actual)
            if cand is not None:
                cand["color"] = color
                cands.append(cand)
        if cands:
            candidate_by_color[color] = cands

    assert candidate_by_color, "expected at least one frame color"
    ranked = sorted(candidate_by_color.items(), key=lambda item: len(item[1]), reverse=True)
    frame_color, frames = ranked[0]
    if len(ranked) > 1:
        assert len(frames) > len(ranked[1][1]), "frame color must be uniquely identifiable"
    return int(frame_color), frames


def _legend_blocks(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    blocks: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color in (bg, frame_color):
            continue
        for comp in _components(grid == color):
            block = _filled_rectangle(comp)
            if block is not None:
                block["color"] = color
                blocks.append(block)
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each hollow frame's border-gap pattern to a legend entry and fill the matched frame interior with that legend color.

    Input structure:
    - The background is black or another dominant blank color.
    - Several large monochrome hollow rectangles appear in the grid; each border has one or more single-cell gaps at specific relative positions.
    - Separate legend entries use the same border color to draw smaller hollow rectangles with matching gap patterns, and each small frame touches a solid colored block that gives the fill color.

    Rule:
    - Detect the shared border color and extract every rectangular frame candidate.
    - For each candidate, convert its missing border cells into a normalized side-and-position pattern so large and small frames can be compared across sizes.
    - Treat frame candidates that touch a solid color block as legend entries, build a pattern-to-color mapping from them, and fill the interior background pixels of every non-legend large frame with its matched legend color.

    Key insight:
    - The absolute frame size is irrelevant; the relative gap positions along the frame sides form the lookup key that connects each large frame to exactly one legend color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frame_color, frames = _detect_frames(g)
    blocks = _legend_blocks(g, bg, frame_color)
    assert blocks, "expected legend color blocks"

    legends: dict[tuple[tuple[str, int, int], ...], int] = {}
    targets: list[dict] = []
    for frame in frames:
        matches = [block for block in blocks if _touches_bbox(frame["bbox"], block["bbox"])]
        if matches:
            assert len(matches) == 1, "legend frame must touch exactly one color block"
            pattern = frame["pattern"]
            color = int(matches[0]["color"])
            assert pattern not in legends, "legend patterns must be unique"
            legends[pattern] = color
        else:
            targets.append(frame)

    assert legends and targets, "expected both legend frames and large frames"

    out = g.copy()
    for frame in targets:
        pattern = frame["pattern"]
        assert pattern in legends, "every large frame must have a matching legend pattern"
        fill = legends[pattern]
        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == bg] = fill
    return out


def _pattern_signature(pattern: tuple[tuple[str, int, int], ...]) -> tuple[tuple[str, int, int], ...]:
    return tuple(sorted(pattern, key=lambda item: (SIDE_ORDER[item[0]], item[1], item[2])))


def _random_pattern(used: set[tuple[tuple[str, int, int], ...]]) -> tuple[tuple[str, int, int], ...] | None:
    for _ in range(200):
        gap_count = random.choice((1, 2))
        sides = random.sample(SIDES, gap_count)
        pattern = tuple((side, *random.choice(FRACTIONS)) for side in sides)
        key = _pattern_signature(pattern)
        if key not in used:
            return key
    return None


def _pattern_lcms(pattern: tuple[tuple[str, int, int], ...]) -> tuple[int, int]:
    lcm_h = 1
    lcm_v = 1
    for side, _num, den in pattern:
        if side in ("T", "B"):
            lcm_h = math.lcm(lcm_h, den)
        else:
            lcm_v = math.lcm(lcm_v, den)
    return lcm_h, lcm_v


def _size_options(low: int, high: int, required_lcm: int) -> list[int]:
    return [size for size in range(low, high + 1) if (size - 1) % required_lcm == 0]


def _choose_sizes(pattern: tuple[tuple[str, int, int], ...]) -> tuple[int, int, int, int] | None:
    lcm_h, lcm_v = _pattern_lcms(pattern)
    small_ws = _size_options(4, 7, lcm_h)
    small_hs = _size_options(4, 7, lcm_v)
    big_ws = _size_options(6, 10, lcm_h)
    big_hs = _size_options(6, 10, lcm_v)
    if not small_ws or not small_hs or not big_ws or not big_hs:
        return None

    combos: list[tuple[int, int, int, int]] = []
    for sw in small_ws:
        for sh in small_hs:
            for bw in big_ws:
                for bh in big_hs:
                    if bw > sw or bh > sh:
                        combos.append((sh, sw, bh, bw))
    if not combos:
        return None
    return random.choice(combos)


def _draw_frame(h: int, w: int, pattern: tuple[tuple[str, int, int], ...], color: int = FRAME_COLOR) -> np.ndarray:
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for side, num, den in pattern:
        if side in ("T", "B"):
            assert (w - 1) % den == 0
            c = ((w - 1) // den) * num
            r = 0 if side == "T" else h - 1
        else:
            assert (h - 1) % den == 0
            r = ((h - 1) // den) * num
            c = 0 if side == "L" else w - 1
        patch[r, c] = BG
    return patch


def _make_legend_entry(pattern: tuple[tuple[str, int, int], ...], sh: int, sw: int, fill_color: int) -> np.ndarray:
    frame = _draw_frame(sh, sw, pattern)
    block_w = random.choice((1, 2))
    block_h = random.randint(2, sh)
    row0 = random.randint(0, sh - block_h)
    entry = np.full((sh, sw + block_w), BG, dtype=int)
    entry[:, :sw] = frame
    entry[row0 : row0 + block_h, sw : sw + block_w] = fill_color
    return entry


def _reserve(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = occ.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not np.any(occ[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    target = grid[top : top + h, left : left + w]
    target[mask] = patch[mask]


def _legend_layout(entries: list[np.ndarray], side: str, grid_h: int, grid_w: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    gaps = [3 for _ in range(max(0, len(entries) - 1))]
    shapes = [entry.shape for entry in entries]
    if side in ("L", "R"):
        total_h = sum(h for h, _ in shapes) + sum(gaps)
        strip_w = max(w for _, w in shapes)
        y0 = random.randint(1, grid_h - total_h - 1)
        x0 = 1 if side == "L" else grid_w - strip_w - 1
        positions = []
        cur_y = y0
        for idx, entry in enumerate(entries):
            eh, ew = entry.shape
            dx = random.randint(0, strip_w - ew)
            positions.append((cur_y, x0 + dx))
            cur_y += eh + (gaps[idx] if idx < len(gaps) else 0)
        return positions, (total_h, strip_w)

    total_w = sum(w for _, w in shapes) + sum(gaps)
    strip_h = max(h for h, _ in shapes)
    x0 = random.randint(1, grid_w - total_w - 1)
    y0 = 1 if side == "T" else grid_h - strip_h - 1
    positions = []
    cur_x = x0
    for idx, entry in enumerate(entries):
        eh, ew = entry.shape
        dy = random.randint(0, strip_h - eh)
        positions.append((y0 + dy, cur_x))
        cur_x += ew + (gaps[idx] if idx < len(gaps) else 0)
    return positions, (strip_h, total_w)


def _usable_region(side: str, grid_h: int, grid_w: int, strip_size: tuple[int, int]) -> tuple[int, int, int, int]:
    strip_h, strip_w = strip_size
    if side == "L":
        return 1, strip_w + 4, grid_h - 2, grid_w - 2
    if side == "R":
        return 1, 1, grid_h - 2, grid_w - strip_w - 5
    if side == "T":
        return strip_h + 4, 1, grid_h - 2, grid_w - 2
    return 1, 1, grid_h - strip_h - 5, grid_w - 2


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    legend_side = random.choice(SIDES)
    used_patterns: set[tuple[tuple[str, int, int], ...]] = set()
    specs = []
    for _ in range(n_frames):
        pattern = _random_pattern(used_patterns)
        if pattern is None:
            return None
        sizes = _choose_sizes(pattern)
        if sizes is None:
            return None
        sh, sw, bh, bw = sizes
        used_patterns.add(pattern)
        specs.append({"pattern": pattern, "small": (sh, sw), "big": (bh, bw)})

    fill_colors = random.sample(FILL_COLORS, n_frames)
    random.shuffle(specs)
    for spec, fill in zip(specs, fill_colors):
        spec["fill"] = fill
        spec["legend"] = _make_legend_entry(spec["pattern"], spec["small"][0], spec["small"][1], fill)
        spec["big_patch"] = _draw_frame(spec["big"][0], spec["big"][1], spec["pattern"])

    entries = [spec["legend"] for spec in specs]
    max_big_h = max(spec["big"][0] for spec in specs)
    max_big_w = max(spec["big"][1] for spec in specs)

    if legend_side in ("L", "R"):
        total_h = sum(entry.shape[0] for entry in entries) + (len(entries) - 1) * 3
        strip_w = max(entry.shape[1] for entry in entries)
        min_h = max(total_h + 2, max_big_h + 6)
        min_w = strip_w + max_big_w + 8
    else:
        total_w = sum(entry.shape[1] for entry in entries) + (len(entries) - 1) * 3
        strip_h = max(entry.shape[0] for entry in entries)
        min_h = strip_h + max_big_h + 8
        min_w = max(total_w + 2, max_big_w + 6)

    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    entry_positions, strip_size = _legend_layout(entries, legend_side, grid_h, grid_w)
    strip_h, strip_w = strip_size
    if legend_side in ("L", "R"):
        full_strip_r0 = 0
        full_strip_r1 = grid_h - 1
        full_strip_c0 = 0 if legend_side == "L" else grid_w - strip_w - 2
        full_strip_c1 = strip_w + 1 if legend_side == "L" else grid_w - 1
        _reserve(occ, full_strip_r0, full_strip_c0, full_strip_r1, full_strip_c1, pad=0)
    else:
        full_strip_r0 = 0 if legend_side == "T" else grid_h - strip_h - 2
        full_strip_r1 = strip_h + 1 if legend_side == "T" else grid_h - 1
        full_strip_c0 = 0
        full_strip_c1 = grid_w - 1
        _reserve(occ, full_strip_r0, full_strip_c0, full_strip_r1, full_strip_c1, pad=0)

    for entry, (top, left) in zip(entries, entry_positions):
        _paste(grid, entry, top, left)

    region_r0, region_c0, region_r1, region_c1 = _usable_region(legend_side, grid_h, grid_w, strip_size)
    if region_r1 - region_r0 + 1 < max_big_h or region_c1 - region_c0 + 1 < max_big_w:
        return None

    region_area = (region_r1 - region_r0 + 1) * (region_c1 - region_c0 + 1)
    padded_need = sum((spec["big"][0] + 2) * (spec["big"][1] + 2) for spec in specs)
    if padded_need > int(region_area * 0.65):
        return None

    for spec in sorted(specs, key=lambda item: item["big"][0] * item["big"][1], reverse=True):
        bh, bw = spec["big"]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(region_r0, region_r1 - bh + 1)
            left = random.randint(region_c0, region_c1 - bw + 1)
            if not _can_place(occ, top, left, top + bh - 1, left + bw - 1, pad=2):
                continue
            _paste(grid, spec["big_patch"], top, left)
            _reserve(occ, top, left, top + bh - 1, left + bw - 1, pad=2)
            placed = True
            break
        if not placed:
            return None

    # Sanity-check that the generated input is non-degenerate for the solver.
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
