import random
from collections import Counter, deque

import numpy as np


BG = 0
NORMAL_BAR = 5
HARD_BAR = 9
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _sprite_records(grid: np.ndarray, bg: int) -> list[dict]:
    mask = (grid != bg) & (grid != NORMAL_BAR) & (grid != HARD_BAR)
    sprites = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox(coords)
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each sprite must be monochrome"
        submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            submask[r - r0, c - c0] = True
        sprites.append(
            {
                "color": colors.pop(),
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": submask,
                "holes": _hole_count(submask),
            }
        )
    return sprites


def _bar_records(grid: np.ndarray) -> list[dict]:
    bars = []
    for color in (NORMAL_BAR, HARD_BAR):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            assert r0 == r1, "bars must be one cell tall"
            bars.append({"color": color, "row": r0, "c0": c0, "c1": c1})
    bars.sort(key=lambda item: (item["row"], item["c0"]))
    return bars


def _simulate_from_records(grid: np.ndarray, sprites: list[dict], bars: list[dict], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for sprite in sprites:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = bg

    h, w = out.shape
    for sprite in sorted(sprites, key=lambda item: item["bbox"][1]):
        r0, c0, r1, c1 = sprite["bbox"]
        mask = sprite["mask"]
        quota = int(sprite["holes"])
        sprite_h = mask.shape[0]
        landing_top = h - sprite_h

        relevant = [
            bar
            for bar in bars
            if bar["row"] > r1 and bar["c0"] <= c0 and c1 <= bar["c1"]
        ]
        relevant.sort(key=lambda item: item["row"])

        for bar in relevant:
            if bar["color"] == HARD_BAR:
                landing_top = bar["row"] - sprite_h
                break
            if quota > 0:
                out[bar["row"], c0 : c1 + 1] = bg
                quota -= 1
                continue
            landing_top = bar["row"] - sprite_h
            break

        patch = out[landing_top : landing_top + sprite_h, c0 : c0 + mask.shape[1]]
        patch[mask] = sprite["color"]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each sprite straight down, letting its enclosed-hole count determine how many normal bars it may punch through.

    Input structure:
    - The background is the dominant color.
    - Several monochrome sprites hang in the upper part of the grid; each sprite may contain 0 to 3 enclosed background holes.
    - Below them sit horizontal grey normal bars and dark-red hard bars; sprites occupy disjoint column ranges.

    Rule:
    - Erase each sprite from its starting position and let it fall vertically without changing shape.
    - A sprite may pass through its first k grey bars, where k is its enclosed-hole count; every passed grey bar is erased only across that sprite's own column span.
    - The sprite stops above the next grey bar after its quota is exhausted.
    - A dark-red hard bar always stops the sprite immediately, even if quota remains.
    - If no stopping bar exists below, the sprite falls to the bottom edge.

    Key insight:
    - Topology controls gravity: the number of sealed holes is a per-sprite penetration budget, hard bars override that budget, and the punched gaps are the visible evidence of which grey bars were truly crossed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _get_bg(g)
    sprites = _sprite_records(g, bg)
    bars = _bar_records(g)
    assert sprites, "expected at least one sprite"
    assert bars, "expected at least one bar"
    return _simulate_from_records(g, sprites, bars, bg)


def _add_top_protrusions(mask: np.ndarray) -> np.ndarray:
    extra_rows = random.randint(0, 2)
    if extra_rows == 0:
        return mask
    h, w = mask.shape
    out = np.zeros((h + extra_rows, w), dtype=bool)
    out[extra_rows:, :] = mask
    for _ in range(random.randint(1, min(3, w))):
        width = random.randint(1, min(2, w))
        start = random.randint(0, w - width)
        height = random.randint(1, extra_rows)
        out[extra_rows - height : extra_rows, start : start + width] = True
    return out


def _make_sprite_mask(holes: int) -> np.ndarray:
    if holes == 0:
        h = random.randint(3, 5)
        w = random.randint(3, 6)
        mask = np.ones((h, w), dtype=bool)
        mask = _add_top_protrusions(mask)
        assert _hole_count(mask) == 0
        return mask

    chamber_widths = [random.randint(1, 2) for _ in range(holes)]
    h = random.randint(4, 6)
    w = 2 + sum(chamber_widths) + (holes - 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    pos = 1
    for width in chamber_widths[:-1]:
        pos += width
        mask[:, pos] = True
        pos += 1
    mask = _add_top_protrusions(mask)
    assert _hole_count(mask) == holes
    return mask


def _lane_bar_plan(scenario: str, holes: int) -> list[int]:
    if scenario == "quota_stop":
        return [NORMAL_BAR] * (holes + 1)
    if scenario == "hard_stop":
        passed = random.randint(1, holes - 1)
        return [NORMAL_BAR] * passed + [HARD_BAR]
    assert scenario == "bottom_fall"
    passed = random.randint(1, holes - 1)
    return [NORMAL_BAR] * passed


def _place_rows(top: int, sprite_h: int, bar_count: int, H: int) -> list[int] | None:
    if bar_count == 0:
        return []
    rows = []
    row = top + sprite_h + random.randint(2, 4)
    rows.append(row)
    for _ in range(1, bar_count):
        row += random.randint(3, 5)
        rows.append(row)
    if rows[-1] > H - 3:
        return None
    return rows


def _render_expected(grid: np.ndarray, meta: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in meta:
        r0, c0 = item["start"]
        mask = item["mask"]
        out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = BG
        out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][~mask] = out[
            r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]
        ][~mask]

    for item in meta:
        top = item["landing_top"]
        left = item["start"][1]
        mask = item["mask"]
        out[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = item["color"]
        for row in item["passed_rows"]:
            c0 = item["start"][1]
            c1 = c0 + mask.shape[1] - 1
            out[row, c0 : c1 + 1] = BG
    return out


def _validate_generated(grid: np.ndarray, meta: list[dict]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    sprites = _sprite_records(grid, BG)
    if len(sprites) != len(meta):
        return False
    by_left = {item["start"][1]: item for item in meta}
    for sprite in sprites:
        left = sprite["bbox"][1]
        item = by_left.get(left)
        if item is None:
            return False
        if sprite["holes"] != item["holes"]:
            return False

    rows_seen = set()
    for item in meta:
        span = range(item["start"][1], item["start"][1] + item["mask"].shape[1])
        if any(c in rows_seen for c in span):
            return False
        rows_seen.update(span)

        if item["scenario"] == "hard_stop":
            if item["bars"][-1][1] != HARD_BAR:
                return False
            if not item["passed_rows"]:
                return False
            if len(item["passed_rows"]) >= item["holes"]:
                return False
        elif item["scenario"] == "quota_stop":
            if len(item["passed_rows"]) != item["holes"]:
                return False
            if len(item["bars"]) != item["holes"] + 1:
                return False
        else:
            if not item["passed_rows"]:
                return False
            if len(item["passed_rows"]) >= item["holes"]:
                return False
            if item["landing_top"] != grid.shape[0] - item["mask"].shape[0]:
                return False

    expected = _render_expected(grid, meta)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False
    if np.array_equal(actual, grid):
        return False

    for item in meta:
        lane_left, lane_right = item["lane"]
        sprite_left = item["start"][1]
        sprite_right = sprite_left + item["mask"].shape[1] - 1
        for row in item["passed_rows"]:
            if sprite_left > lane_left and not np.all(actual[row, lane_left:sprite_left] == NORMAL_BAR):
                return False
            if sprite_right < lane_right and not np.all(actual[row, sprite_right + 1 : lane_right + 1] == NORMAL_BAR):
                return False
            if not np.all(actual[row, sprite_left : sprite_right + 1] == BG):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    quota_holes = random.choice([0, 1, 2])
    hard_holes = random.choice([2, 3])
    bottom_holes = random.choice([2, 3])
    scenarios = [
        ("quota_stop", quota_holes),
        ("hard_stop", hard_holes),
        ("bottom_fall", bottom_holes),
    ]
    random.shuffle(scenarios)

    colors = random.sample(SPRITE_COLORS, len(scenarios))
    sprite_defs = []
    for color, (scenario, holes) in zip(colors, scenarios):
        mask = _make_sprite_mask(holes)
        sprite_defs.append(
            {
                "color": color,
                "scenario": scenario,
                "holes": holes,
                "mask": mask,
                "lane_width": mask.shape[1] + 2,
            }
        )

    gaps = [1 for _ in range(len(sprite_defs) - 1)]
    used = sum(item["lane_width"] for item in sprite_defs) + sum(gaps)
    min_w = used + 1
    if min_w > 30:
        return None
    extra_w = random.randint(0, 30 - min_w)
    left_margin = 1
    right_margin = 0
    for _ in range(extra_w):
        slot = random.randint(0, len(gaps) + 1)
        if slot == 0:
            left_margin += 1
        elif slot == len(gaps) + 1:
            right_margin += 1
        else:
            gaps[slot - 1] += 1
    W = left_margin + sum(item["lane_width"] for item in sprite_defs) + sum(gaps) + right_margin
    H = random.randint(28, 30)
    grid = np.full((H, W), BG, dtype=int)

    cursor = left_margin
    meta = []

    for idx, item in enumerate(sprite_defs):
        mask = item["mask"]
        lane_left = cursor
        lane_right = lane_left + item["lane_width"] - 1
        sprite_left = lane_left + 1
        sprite_top = random.randint(1, 2)

        bar_colors = _lane_bar_plan(item["scenario"], item["holes"])
        bar_rows = _place_rows(sprite_top, mask.shape[0], len(bar_colors), H)
        if bar_rows is None:
            return None

        grid[sprite_top : sprite_top + mask.shape[0], sprite_left : sprite_left + mask.shape[1]][mask] = item["color"]

        for row, color in zip(bar_rows, bar_colors):
            grid[row, lane_left : lane_right + 1] = color

        passed_rows = []
        quota = item["holes"]
        landing_top = H - mask.shape[0]
        for row, color in zip(bar_rows, bar_colors):
            if color == HARD_BAR:
                landing_top = row - mask.shape[0]
                break
            if quota > 0:
                passed_rows.append(row)
                quota -= 1
                continue
            landing_top = row - mask.shape[0]
            break

        meta.append(
            {
                "color": item["color"],
                "scenario": item["scenario"],
                "holes": item["holes"],
                "mask": mask,
                "lane": (lane_left, lane_right),
                "start": (sprite_top, sprite_left),
                "bars": list(zip(bar_rows, bar_colors)),
                "passed_rows": passed_rows,
                "landing_top": landing_top,
            }
        )

        if idx < len(sprite_defs) - 1:
            cursor = lane_right + 1 + gaps[idx]

    if not _validate_generated(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
