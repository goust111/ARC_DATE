from __future__ import annotations

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("arc_utils_shared", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"unable to load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


GROUND_SIDES = ("top", "bottom", "left", "right")
BLOCK_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GRID_WIDTH_RANGE = (8, 16)
BLOCK_COUNT_RANGE = (2, 5)
CLEAR_COUNT_CHOICES = (1, 1, 2, 2, 3)
EXTRA_HEIGHT_RANGE = (0, 4)
TOP_MARGIN_RANGE = (6, 11)
MAX_GENERATION_ATTEMPTS = 400


def _detect_ground(grid: np.ndarray) -> tuple[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.reshape(-1)))
    candidates: list[tuple[str, int]] = []

    if np.all(g[0, :] == g[0, 0]) and counts[int(g[0, 0])] == w:
        candidates.append(("top", int(g[0, 0])))
    if np.all(g[h - 1, :] == g[h - 1, 0]) and counts[int(g[h - 1, 0])] == w:
        candidates.append(("bottom", int(g[h - 1, 0])))
    if np.all(g[:, 0] == g[0, 0]) and counts[int(g[0, 0])] == h:
        candidates.append(("left", int(g[0, 0])))
    if np.all(g[:, w - 1] == g[0, w - 1]) and counts[int(g[0, w - 1])] == h:
        candidates.append(("right", int(g[0, w - 1])))

    assert len(candidates) == 1, "expected exactly one unique full-edge ground line"
    return candidates[0]


def _to_canonical(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "bottom":
        return np.array(grid, dtype=int, copy=True)
    if side == "top":
        return np.flipud(grid).copy()
    if side == "left":
        return np.rot90(grid, 1).copy()
    if side == "right":
        return np.rot90(grid, 3).copy()
    raise ValueError(f"unknown side: {side}")


def _from_canonical(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "bottom":
        return np.array(grid, dtype=int, copy=True)
    if side == "top":
        return np.flipud(grid).copy()
    if side == "left":
        return np.rot90(grid, 3).copy()
    if side == "right":
        return np.rot90(grid, 1).copy()
    raise ValueError(f"unknown side: {side}")


def _extract_blocks(grid: np.ndarray, bg: int, ground_color: int) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {int(bg), int(ground_color)}]
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=colors,
        return_format="dict",
    )
    assert comps, "expected at least one foreground block"

    blocks = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert int(comp["area"]) == height * width, "each block must be a solid rectangle"
        blocks.append(
            {
                "color": int(comp["color"]),
                "coords": np.array(comp["coords"], dtype=int),
                "bbox": (r0, c0, r1, c1),
                "height": height,
                "width": width,
            }
        )

    blocks.sort(key=lambda block: (block["bbox"][0], block["bbox"][1], block["color"]))
    return blocks


def _validate_perpendicular_separation(blocks: list[dict]) -> None:
    for i, left in enumerate(blocks):
        _r0, c0, _r1, c1 = left["bbox"]
        for right in blocks[i + 1 :]:
            _rr0, cc0, _rr1, cc1 = right["bbox"]
            assert c1 < cc0 or cc1 < c0, "blocks must not share columns in canonical gravity"


def _apply_gravity_bottom(grid: np.ndarray, bg: int, ground_color: int) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h, w), int(bg), dtype=int)
    out[h - 1, :] = int(ground_color)
    occupied = np.zeros((h, w), dtype=bool)
    occupied[h - 1, :] = True

    blocks = _extract_blocks(grid, bg, ground_color)
    _validate_perpendicular_separation(blocks)

    for block in sorted(blocks, key=lambda item: (item["bbox"][2], item["bbox"][1]), reverse=True):
        coords = block["coords"]
        delta = 0
        while True:
            trial = coords.copy()
            trial[:, 0] += delta + 1
            if np.any(trial[:, 0] >= h):
                break
            if occupied[trial[:, 0], trial[:, 1]].any():
                break
            delta += 1

        landed = coords.copy()
        landed[:, 0] += delta
        out[landed[:, 0], landed[:, 1]] = int(block["color"])
        occupied[landed[:, 0], landed[:, 1]] = True

    return out


def _clear_full_rows_bottom(grid: np.ndarray, bg: int, ground_color: int) -> np.ndarray:
    h, w = grid.shape
    full_rows = [r for r in range(h - 1) if np.all(grid[r, :] != int(bg))]
    if not full_rows:
        return np.array(grid, dtype=int, copy=True)

    keep_rows = [r for r in range(h - 1) if r not in set(full_rows)]
    out = np.full((h, w), int(bg), dtype=int)
    out[h - 1, :] = int(ground_color)
    start = h - 1 - len(keep_rows)
    for idx, src_r in enumerate(keep_rows):
        out[start + idx, :] = grid[src_r, :]
    return out


def _solve_canonical_bottom(grid: np.ndarray, ground_color: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert np.all(g[-1, :] == int(ground_color)), "canonical grid must have bottom ground"
    landed = _apply_gravity_bottom(g, bg, ground_color)
    return _clear_full_rows_bottom(landed, bg, ground_color)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Drop floating solid blocks toward the unique full-edge ground line, then erase every filled parallel layer and cascade the outer foreground inward.

    Input structure:
    - Exactly one grid edge is a single-color line spanning the whole edge, and that color appears nowhere else.
    - The remaining foreground consists of monochromatic solid rectangular blocks on a dominant background.
    - Gravity points toward the ground edge; in the axis perpendicular to gravity, distinct blocks do not share columns or rows.

    Rule:
    - Use the unique full-edge line as the ground and move every foreground block rigidly toward it until blocked by the ground or by already settled foreground.
    - After all blocks settle, inspect every layer parallel to the ground; any completely filled layer is erased.
    - When a layer is erased, all foreground farther from the ground shifts one step toward the ground while the ground and the nearer layers stay fixed; repeat until no filled layer remains.

    Key insight:
    - The full line is not a missing stripe to complete: it defines both the gravity direction and the fixed side relative to which row or column clears must be resolved.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    side, ground_color = _detect_ground(g)
    canonical = _to_canonical(g, side)
    solved = _solve_canonical_bottom(canonical, ground_color)
    return _from_canonical(solved, side)


def _random_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    cuts = sorted(random.sample(range(1, total), parts - 1))
    spans = []
    start = 0
    for cut in cuts + [total]:
        spans.append(cut - start)
        start = cut
    return spans


def _build_expected_bottom(height: int, widths: list[int], heights: list[int], colors: list[int], ground_color: int, bg: int) -> np.ndarray:
    clear_count = min(heights)
    out = np.full((height, sum(widths)), int(bg), dtype=int)
    out[height - 1, :] = int(ground_color)
    col = 0
    for width, block_height, color in zip(widths, heights, colors):
        remain = block_height - clear_count
        if remain > 0:
            out[height - 1 - remain : height - 1, col : col + width] = int(color)
        col += width
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    side = random.choice(GROUND_SIDES)
    bg = Color.BLACK

    width = random.randint(*GRID_WIDTH_RANGE)
    block_count = random.randint(BLOCK_COUNT_RANGE[0], min(BLOCK_COUNT_RANGE[1], width))
    widths = _random_partition(width, block_count)

    usable_colors = BLOCK_COLORS.copy()
    ground_color = random.choice(usable_colors)
    usable_colors.remove(ground_color)
    colors = random.sample(usable_colors, block_count)

    clear_count = random.choice(CLEAR_COUNT_CHOICES)
    extras = [random.randint(*EXTRA_HEIGHT_RANGE) for _ in range(block_count)]
    extras[random.randrange(block_count)] = 0
    if all(extra == 0 for extra in extras):
        extras[random.randrange(block_count)] = random.randint(1, EXTRA_HEIGHT_RANGE[1])
    heights = [clear_count + extra for extra in extras]

    max_height = max(heights)
    grid_height = max_height + random.randint(*TOP_MARGIN_RANGE)
    assert grid_height <= 30

    canonical = np.full((grid_height, width), int(bg), dtype=int)
    canonical[grid_height - 1, :] = int(ground_color)
    landing_tops = [grid_height - 1 - block_height for block_height in heights]

    top_rows = []
    for landing_top in landing_tops:
        top_rows.append(random.randint(0, landing_top))
    if all(top == landing for top, landing in zip(top_rows, landing_tops)):
        idx = random.randrange(block_count)
        if landing_tops[idx] > 0:
            top_rows[idx] = random.randint(0, landing_tops[idx] - 1)

    col = 0
    for width_i, height_i, top_i, color in zip(widths, heights, top_rows, colors):
        canonical[top_i : top_i + height_i, col : col + width_i] = int(color)
        col += width_i

    expected = _build_expected_bottom(grid_height, widths, heights, colors, ground_color, bg)
    return _from_canonical(canonical, side), _from_canonical(expected, side)


def _validate_generated_pair(inp: np.ndarray, out: np.ndarray) -> None:
    side, ground_color = _detect_ground(inp)
    cin = _to_canonical(inp, side)
    cout = _to_canonical(out, side)
    bg = get_bg(cin)

    assert np.all(cin[-1, :] == int(ground_color))
    assert np.all(cout[-1, :] == int(ground_color))
    assert get_bg(cout) == int(bg)

    in_blocks = _extract_blocks(cin, bg, ground_color)
    out_blocks = _extract_blocks(cout, bg, ground_color) if np.any(cout[:-1, :] != int(bg)) else []

    _validate_perpendicular_separation(in_blocks)
    assert len({block["color"] for block in in_blocks}) == len(in_blocks), "generator expects unique block colors"

    ordered = sorted(in_blocks, key=lambda block: block["bbox"][1])
    cursor = 0
    heights_by_color: dict[int, int] = {}
    moved_any = False
    for block in ordered:
        r0, c0, r1, c1 = block["bbox"]
        assert c0 == cursor, "blocks should partition the canonical width without gaps"
        cursor = c1 + 1
        heights_by_color[int(block["color"])] = int(block["height"])
        if r0 < cin.shape[0] - 1 - block["height"]:
            moved_any = True
    assert cursor == cin.shape[1], "blocks should cover the full width to force line clears"
    assert moved_any, "at least one block must visibly fall"

    clear_count = min(heights_by_color.values())
    assert clear_count >= 1, "each generated example must require at least one line clear"
    assert not any(np.all(cout[r, :] != int(bg)) for r in range(cout.shape[0] - 1)), "output must be stable after cascade"

    out_by_color = {int(block["color"]): block for block in out_blocks}
    for color, original_height in heights_by_color.items():
        remain = original_height - clear_count
        if remain == 0:
            assert color not in out_by_color, "fully cleared blocks must disappear"
            continue
        assert color in out_by_color, "surviving block missing from output"
        r0, c0, r1, c1 = out_by_color[color]["bbox"]
        assert r1 == cout.shape[0] - 2, "surviving blocks must rest against the ground"
        assert r1 - r0 + 1 == remain, "line-clear depth must subtract from block height"
        assert c1 - c0 + 1 == ordered[[block["color"] for block in ordered].index(color)]["width"]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, expected = _generate_case()
        try:
            actual = main(inp)
            if not np.array_equal(actual, expected):
                continue
            _validate_generated_pair(inp, actual)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input for ground-gravity line-clear puzzle")
