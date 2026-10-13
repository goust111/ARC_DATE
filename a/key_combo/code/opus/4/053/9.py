import random
from collections import Counter

import numpy as np


BG = 7
RED = 2
GREY = 5
WALL_COLORS = (1, 3, 4, 6, 8, 9)
VALID_BAND_WIDTHS = (7, 9)
GRID_H_RANGE = (11, 16)
GRID_W_RANGE = (11, 16)
MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    """Return the most common grid color."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _flip_band_color(color: int) -> int:
    if int(color) == RED:
        return GREY
    if int(color) == GREY:
        return RED
    raise ValueError(f"expected band color 2 or 5, got {color}")


def _parse_band(grid: np.ndarray) -> tuple[int, int, list[int]]:
    top = np.asarray(grid[0], dtype=int)
    nz = np.flatnonzero(top != BG)
    assert nz.size > 0, "top-row band missing"
    left = int(nz[0])
    right = int(nz[-1])
    assert np.all(top[left : right + 1] != BG), "top-row band must be contiguous"
    assert np.all(top[:left] == BG) and np.all(top[right + 1 :] == BG), "band must be unique"

    band = [int(v) for v in top[left : right + 1]]
    width = len(band)
    assert width in VALID_BAND_WIDTHS, "band width must be a supported odd size"
    for i, color in enumerate(band):
        assert color in (RED, GREY), "band must use only red and grey"
        if i:
            assert color != band[i - 1], "band must strictly alternate"
    return left, right, band


def _build_intended_map(shape: tuple[int, int], left: int, right: int, band: list[int]) -> dict[tuple[int, int], int]:
    height, width = map(int, shape)
    band_by_col = {left + i: int(color) for i, color in enumerate(band)}
    intended: dict[tuple[int, int], int] = {}

    for r in range(height):
        span_left = left + r
        span_right = right - r
        if span_left > span_right:
            break
        for c in range(span_left, span_right + 1):
            base = int(band_by_col[c - r])
            intended[(r, c)] = base if r % 2 == 0 else _flip_band_color(base)

    return intended


def _wall_direction(c: int, center: float) -> int:
    return 1 if int(c) <= center else -1


def _trace_injection(
    wall: tuple[int, int, int],
    intended: dict[tuple[int, int], int],
    center: float,
) -> tuple[int, list[tuple[int, int]]]:
    rw, cw, wall_color = map(int, wall)
    target_color = int(intended[(rw, cw)])
    dc = _wall_direction(cw, center)
    rr = rw + 1
    cc = cw + dc
    affected: list[tuple[int, int]] = []

    while (rr, cc) in intended:
        if int(intended[(rr, cc)]) == target_color:
            affected.append((rr, cc))
        rr += 1
        cc += dc

    return int(wall_color), affected


def _collect_walls(grid: np.ndarray, intended: dict[tuple[int, int], int]) -> list[tuple[int, int, int]]:
    walls: list[tuple[int, int, int]] = []
    rows, cols = np.argwhere(grid != BG).T
    for r, c in zip(rows, cols):
        rr = int(r)
        cc = int(c)
        color = int(grid[rr, cc])
        if rr == 0:
            continue
        assert color not in (RED, GREY), "walls must use non-band colors"
        assert (rr, cc) in intended, "wall must lie inside the wedge coverage"
        walls.append((rr, cc, color))

    assert 1 <= len(walls) <= 2, "expected one or two walls"
    assert len({color for _r, _c, color in walls}) == len(walls), "wall colors must be distinct"
    return sorted(walls, key=lambda item: (item[0], item[1]))


def _transform(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    bg = get_bg(g)
    assert bg == BG, "background must be orange"

    left, right, band = _parse_band(g)
    intended = _build_intended_map(g.shape, left, right, band)
    walls = _collect_walls(g, intended)
    center = (left + right) / 2.0

    out = np.full_like(g, BG)
    for (r, c), color in intended.items():
        out[r, c] = int(color)

    replacements: dict[tuple[int, int], int] = {}
    influence_sets: list[set[tuple[int, int]]] = []
    for wall in walls:
        wall_color, affected = _trace_injection(wall, intended, center)
        influence = set(affected)
        influence_sets.append(influence)
        for cell in affected:
            if cell not in {(wr, wc) for wr, wc, _color in walls}:
                replacements[cell] = int(wall_color)

    for (r, c), color in replacements.items():
        out[r, c] = int(color)

    for r, c, color in walls:
        out[r, c] = int(color)

    info = {
        "left": left,
        "right": right,
        "band": tuple(band),
        "intended": intended,
        "walls": tuple(walls),
        "center": center,
        "influence_sets": tuple(frozenset(s) for s in influence_sets),
        "replacements": dict(replacements),
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a shrinking red-grey wedge downward and let each wall inject a replacement color along one diagonal channel.

    Input structure:
    - The grid is an orange background with one contiguous odd-width top-row band made of strictly alternating red and grey cells.
    - One or two single-pixel walls of distinct non-band colors appear below the band inside the future wedge coverage.
    - The wedge naturally ends after shrinking to a point, leaving lower rows as background.

    Rule:
    - Recreate the shrinking wedge: row r fills columns [left+r, right-r], and cell (r,c) copies the top-row band color at column c-r, flipped between red and grey on every odd row.
    - A wall pixel keeps its own color instead of the wedge color that would normally land there.
    - From each wall, continue downward on the side-dependent diagonal: down-right when the wall is on or left of the wedge center, otherwise down-left; every later wedge cell on that chain whose intended wedge color matches the wall's intended color is permanently replaced by the wall color.

    Key insight:
    - The top band fixes a global alternating wedge pattern, but each wall does not block or bounce it; instead it replaces one of the two alternating color channels along a single future diagonal stripe.
    """
    out, _info = _transform(grid)
    return out


def _candidate_walls(grid_shape: tuple[int, int], left: int, right: int, band: list[int]) -> list[dict]:
    intended = _build_intended_map(grid_shape, left, right, band)
    center = (left + right) / 2.0
    walls: list[dict] = []
    wedge_rows = len({r for r, _c in intended})

    for r in range(1, wedge_rows):
        span_left = left + r
        span_right = right - r
        for c in range(span_left, span_right + 1):
            color = int(intended[(r, c)])
            dummy_color = WALL_COLORS[0]
            _wall_color, affected = _trace_injection((r, c, dummy_color), intended, center)
            if not affected:
                continue
            walls.append(
                {
                    "pos": (r, c),
                    "intended_color": color,
                    "side": "left" if c <= center else "right",
                    "diag_lr": c - r,
                    "diag_rl": c + r,
                    "affected": tuple(affected),
                }
            )

    return walls


def _choose_wall_specs(candidates: list[dict]) -> list[dict] | None:
    if not candidates:
        return None

    left_candidates = [cand for cand in candidates if cand["side"] == "left"]
    right_candidates = [cand for cand in candidates if cand["side"] == "right"]
    want_two = bool(left_candidates and right_candidates and random.random() < 0.65)

    def compatible(a: dict, b: dict) -> bool:
        if a["pos"] == b["pos"]:
            return False
        if a["diag_lr"] == b["diag_lr"]:
            return False
        if a["diag_rl"] == b["diag_rl"]:
            return False
        aset = set(a["affected"])
        bset = set(b["affected"])
        if aset & bset:
            return False
        if a["pos"] in bset or b["pos"] in aset:
            return False
        return True

    if want_two:
        left_pool = random.sample(left_candidates, len(left_candidates))
        right_pool = random.sample(right_candidates, len(right_candidates))
        for a in left_pool:
            for b in right_pool:
                if compatible(a, b):
                    return [a, b]

    return [random.choice(candidates)]


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    left, right, band = _parse_band(inp)
    intended = _build_intended_map(inp.shape, left, right, band)
    walls = _collect_walls(inp, intended)
    center = (left + right) / 2.0
    wall_positions = {(r, c) for r, c, _color in walls}

    assert np.array_equal(out, main(inp)), "main must be deterministic"
    assert np.count_nonzero(out != inp) > len(walls), "output must reflect propagation, not only preserve walls"

    combined_affected: set[tuple[int, int]] = set()
    for r, c, wall_color in walls:
        target_color = int(intended[(r, c)])
        _wcol, affected = _trace_injection((r, c, wall_color), intended, center)
        assert affected, "every generated wall must affect later cells"
        affected_set = set(affected)
        assert not (combined_affected & affected_set), "generated walls must not share affected cells"
        combined_affected |= affected_set
        assert int(out[r, c]) == int(wall_color), "wall pixel must remain unchanged"
        for rr, cc in affected:
            if (rr, cc) in wall_positions:
                continue
            if int(intended[(rr, cc)]) == target_color:
                assert int(out[rr, cc]) == int(wall_color), "matched future cells must inherit wall color"

    for (r, c), color in intended.items():
        if (r, c) in wall_positions:
            continue
        if (r, c) not in combined_affected:
            assert int(out[r, c]) == int(color), "unaffected wedge cells must keep the base wedge color"

    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if (r, c) not in intended and (r, c) not in wall_positions:
                assert int(out[r, c]) == BG, "outside the wedge everything stays background"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        band_width = random.choice(VALID_BAND_WIDTHS)
        if width < band_width + 2:
            continue

        left = random.randint(1, width - band_width - 1)
        right = left + band_width - 1
        band_start = random.choice((RED, GREY))
        band = [_flip_band_color(band_start) if i % 2 else band_start for i in range(band_width)]
        candidates = _candidate_walls((height, width), left, right, band)
        chosen = _choose_wall_specs(candidates)
        if not chosen:
            continue

        wall_colors = random.sample(WALL_COLORS, len(chosen))
        grid = np.full((height, width), BG, dtype=int)
        grid[0, left : right + 1] = np.array(band, dtype=int)

        for spec, color in zip(chosen, wall_colors):
            r, c = spec["pos"]
            grid[r, c] = int(color)

        out, info = _transform(grid)
        if len(info["replacements"]) < len(chosen):
            continue
        if len(chosen) == 2 and {spec["side"] for spec in chosen} != {"left", "right"}:
            continue
        if np.array_equal(grid, out):
            continue

        _validate_semantics(grid, out)
        return grid

    raise RuntimeError("failed to generate a valid puzzle 9 input")
