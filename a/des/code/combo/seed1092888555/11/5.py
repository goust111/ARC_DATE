import random
from collections import Counter, defaultdict

import numpy as np


BG = 3
ARC_COLORS = tuple(range(10))
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, str, dict[int, int], np.ndarray]] = []

    def add_candidate(side: str, entries: list[tuple[int, int, int]]) -> None:
        if len(entries) != 5:
            return
        widths = {gap for _, gap, _ in entries}
        if widths != {1, 2, 3, 4, 5}:
            return
        mapping = {gap: color for _, gap, color in entries}
        mask = np.zeros((h, w), dtype=bool)
        for pos, gap, _color in entries:
            if side == "left":
                mask[pos, gap] = True
            elif side == "right":
                mask[pos, w - 1 - gap] = True
            elif side == "top":
                mask[gap, pos] = True
            else:
                mask[h - 1 - gap, pos] = True
        # Prefer candidates whose marker cells are all isolated singletons.
        score = 0
        for r, c in zip(*np.where(mask), strict=False):
            same = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == int(grid[r, c]):
                    same += 1
            if same == 0:
                score += 1
        candidates.append((score, side, mapping, mask))

    left_entries = []
    for r in range(h):
        cols = np.flatnonzero(grid[r] != bg)
        if len(cols) == 1 and 1 <= int(cols[0]) <= 5:
            left_entries.append((r, int(cols[0]), int(grid[r, int(cols[0])])))
    add_candidate("left", left_entries)

    right_entries = []
    for r in range(h):
        cols = np.flatnonzero(grid[r] != bg)
        if len(cols) == 1:
            gap = w - 1 - int(cols[0])
            if 1 <= gap <= 5:
                right_entries.append((r, gap, int(grid[r, int(cols[0])])))
    add_candidate("right", right_entries)

    top_entries = []
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] != bg)
        if len(rows) == 1 and 1 <= int(rows[0]) <= 5:
            top_entries.append((c, int(rows[0]), int(grid[int(rows[0]), c])))
    add_candidate("top", top_entries)

    bottom_entries = []
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] != bg)
        if len(rows) == 1:
            gap = h - 1 - int(rows[0])
            if 1 <= gap <= 5:
                bottom_entries.append((c, gap, int(grid[int(rows[0]), c])))
    add_candidate("bottom", bottom_entries)

    assert candidates, "legend not found"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, _side, mapping, mask = candidates[0]
    return mapping, mask


def _horizontal_gap(comp_a: dict, comp_b: dict, grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]] | None:
    a0, a1, a2, a3 = comp_a["bbox"]
    b0, b1, b2, b3 = comp_b["bbox"]
    if a0 != b0 or a2 != b2:
        return None
    if a3 < b1:
        left, right = comp_a, comp_b
    elif b3 < a1:
        left, right = comp_b, comp_a
    else:
        return None
    r0, _, r1, c1 = left["bbox"]
    _, c0, _, _ = right["bbox"]
    gap = c0 - c1 - 1
    if gap < 1:
        return None
    region = grid[r0 : r1 + 1, c1 + 1 : c0]
    if region.size == 0 or not np.all(region == bg):
        return None
    return gap, (r0, r1, c1 + 1, c0 - 1)


def _vertical_gap(comp_a: dict, comp_b: dict, grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]] | None:
    a0, a1, a2, a3 = comp_a["bbox"]
    b0, b1, b2, b3 = comp_b["bbox"]
    if a1 != b1 or a3 != b3:
        return None
    if a2 < b0:
        top, bottom = comp_a, comp_b
    elif b2 < a0:
        top, bottom = comp_b, comp_a
    else:
        return None
    _, c0, r1, c1 = top["bbox"]
    r0, _, _, _ = bottom["bbox"]
    gap = r0 - r1 - 1
    if gap < 1:
        return None
    region = grid[r1 + 1 : r0, c0 : c1 + 1]
    if region.size == 0 or not np.all(region == bg):
        return None
    return gap, (r1 + 1, r0 - 1, c0, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace background gaps between aligned same-color objects with bridge colors read from an edge legend.

    The legend maps gap widths 1..5 to colors. Each legend entry is a single colored
    marker placed that many cells away from one grid edge. Objects with equal row spans
    or equal column spans are bridged across pure-background gaps using the legend color.
    The legend itself is removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mapping, legend_mask = _detect_legend(g, bg)

    work = g.copy()
    work[legend_mask] = bg
    out = work.copy()

    comps = _find_components(work, bg)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    for color, group in by_color.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                hg = _horizontal_gap(group[i], group[j], work, bg)
                if hg is not None:
                    gap, (r0, r1, c0, c1) = hg
                    if gap in mapping:
                        out[r0 : r1 + 1, c0 : c1 + 1] = mapping[gap]
                vg = _vertical_gap(group[i], group[j], work, bg)
                if vg is not None:
                    gap, (r0, r1, c0, c1) = vg
                    if gap in mapping:
                        out[r0 : r1 + 1, c0 : c1 + 1] = mapping[gap]
    return out


def _paint_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top : top + height, left : left + width] = color


def _legend_positions(side: str, h: int, w: int) -> list[int]:
    if side in ("left", "right"):
        rows = list(range(1, h - 1))
        random.shuffle(rows)
        return sorted(rows[:5])
    cols = list(range(1, w - 1))
    random.shuffle(cols)
    return sorted(cols[:5])


def _place_legend(grid: np.ndarray, side: str, mapping: dict[int, int]) -> list[int]:
    h, w = grid.shape
    positions = _legend_positions(side, h, w)
    widths = [1, 2, 3, 4, 5]
    random.shuffle(widths)
    for pos, gap in zip(positions, widths, strict=False):
        color = mapping[gap]
        if side == "left":
            grid[pos, gap] = color
        elif side == "right":
            grid[pos, w - 1 - gap] = color
        elif side == "top":
            grid[gap, pos] = color
        else:
            grid[h - 1 - gap, pos] = color
    return positions


def _free(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return 0 <= r0 <= r1 < mask.shape[0] and 0 <= c0 <= c1 < mask.shape[1] and not mask[r0 : r1 + 1, c0 : c1 + 1].any()


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(14, 20)
    w = random.randint(14, 20)
    grid = np.full((h, w), BG, dtype=int)

    side = random.choice(["left", "right", "top", "bottom"])
    legend_colors = random.sample([c for c in ARC_COLORS if c != BG], 5)
    mapping = {gap: legend_colors[gap - 1] for gap in range(1, 6)}
    legend_positions = _place_legend(grid, side, mapping)

    blocked = np.zeros((h, w), dtype=bool)
    if side == "left":
        blocked[:, :7] = True
        for r in legend_positions:
            blocked[r, :] = True
    elif side == "right":
        blocked[:, w - 7 :] = True
        for r in legend_positions:
            blocked[r, :] = True
    elif side == "top":
        blocked[:7, :] = True
        for c in legend_positions:
            blocked[:, c] = True
    else:
        blocked[h - 7 :, :] = True
        for c in legend_positions:
            blocked[:, c] = True

    pair_count = random.randint(2, 4)
    available_colors = [c for c in ARC_COLORS if c not in set(legend_colors) | {BG}]
    random.shuffle(available_colors)
    if len(available_colors) < pair_count:
        return None

    mode_choices = ["horizontal", "vertical"] * 2
    random.shuffle(mode_choices)
    orientations = [mode_choices[i % len(mode_choices)] for i in range(pair_count)]

    for idx in range(pair_count):
        color = available_colors[idx]
        orientation = orientations[idx]
        placed = False
        for _ in range(120):
            if orientation == "horizontal":
                height = random.randint(1, 2)
                width_a = random.randint(1, 3)
                width_b = random.randint(1, 3)
                gap = random.randint(1, 5)
                row = random.randint(1, h - height - 2)
                total = width_a + gap + width_b
                col = random.randint(1, w - total - 2)
                a = (row, col, row + height - 1, col + width_a - 1)
                b = (row, col + width_a + gap, row + height - 1, col + total - 1)
                if not (_free(blocked, *a) and _free(blocked, *b)):
                    continue
                bridge = (row, a[3] + 1, row + height - 1, b[1] - 1)
                if not _free(blocked, bridge[0], bridge[1], bridge[2], bridge[3]):
                    continue
                _paint_rect(grid, a[0], a[1], height, width_a, color)
                _paint_rect(grid, b[0], b[1], height, width_b, color)
                _reserve(blocked, *a, pad=1)
                _reserve(blocked, *b, pad=1)
                _reserve(blocked, bridge[0], bridge[1], bridge[2], bridge[3], pad=0)
                placed = True
                break
            else:
                width = random.randint(1, 2)
                height_a = random.randint(1, 3)
                height_b = random.randint(1, 3)
                gap = random.randint(1, 5)
                col = random.randint(1, w - width - 2)
                total = height_a + gap + height_b
                row = random.randint(1, h - total - 2)
                a = (row, col, row + height_a - 1, col + width - 1)
                b = (row + height_a + gap, col, row + total - 1, col + width - 1)
                if not (_free(blocked, *a) and _free(blocked, *b)):
                    continue
                bridge = (a[2] + 1, col, b[0] - 1, col + width - 1)
                if not _free(blocked, bridge[0], bridge[1], bridge[2], bridge[3]):
                    continue
                _paint_rect(grid, a[0], a[1], height_a, width, color)
                _paint_rect(grid, b[0], b[1], height_b, width, color)
                _reserve(blocked, *a, pad=1)
                _reserve(blocked, *b, pad=1)
                _reserve(blocked, bridge[0], bridge[1], bridge[2], bridge[3], pad=0)
                placed = True
                break
        if not placed:
            return None

    # Keep the legend rows/cols sparse so detection is unambiguous.
    out = main(grid)
    if np.array_equal(out, grid):
        return None

    # Validate the intended structure after removing legend.
    work = grid.copy()
    _, legend_mask = _detect_legend(work, BG)
    work[legend_mask] = BG
    comps = _find_components(work, BG)
    by_color = Counter(int(comp["color"]) for comp in comps)
    if sorted(by_color.values()) != [2] * pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid bridge-legend task")
