import importlib.util
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_shared", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BACKGROUND = Color.BLACK
FRAME_COLOR = Color.GREY
WALLPAPER_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

GRID_ROWS_RANGE = (20, 28)
GRID_COLS_RANGE = (20, 28)
ROW_PERIOD_RANGE = (3, 5)
COL_PERIOD_RANGE = (3, 5)
NONEMPTY_WINDOW_COUNT_RANGE = (4, 5)
NONEMPTY_SIZE_CAP = 6
EMPTY_SIZE_CAP = 7
PLACEMENT_MARGIN = 1
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_TRIES = 400


def _is_rectangular_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(component["area"]) != 2 * h + 2 * w - 4:
        return False
    coords = component["coords"]
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return True


def _extract_frames(grid: np.ndarray) -> tuple[int, int, list[dict], dict]:
    bg = int(get_bg(grid))
    candidates: list[tuple[int, list[dict], dict]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        comps = find_components(
            grid,
            bg=bg,
            colors=[color],
            connectivity=4,
            monochromatic=True,
            return_format="dict",
        )
        if len(comps) < 2 or not all(_is_rectangular_frame(comp) for comp in comps):
            continue
        empty_frames = []
        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0:
                break
            if np.all(interior == bg):
                empty_frames.append(comp)
        else:
            if len(empty_frames) == 1:
                candidates.append((color, comps, empty_frames[0]))
    assert len(candidates) == 1
    frame_color, frames, empty_frame = candidates[0]
    frames = sorted(frames, key=lambda comp: tuple(map(int, comp["bbox"])))
    return bg, int(frame_color), frames, empty_frame


def _window_samples(
    grid: np.ndarray, frames: list[dict], bg: int
) -> tuple[list[tuple[int, int, int]], dict, list[dict]]:
    samples: list[tuple[int, int, int]] = []
    empty_frame = None
    nonempty_frames = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if np.all(interior == bg):
            assert empty_frame is None
            empty_frame = frame
            continue
        nonempty_frames.append(frame)
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                samples.append((r, c, int(grid[r, c])))
    assert empty_frame is not None and nonempty_frames
    return samples, empty_frame, nonempty_frames


def _mapping_for_period(
    samples: list[tuple[int, int, int]], row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    mapping: dict[tuple[int, int], int] = {}
    for r, c, color in samples:
        key = (r % row_period, c % col_period)
        if key in mapping and mapping[key] != color:
            return None
        mapping[key] = color
    return mapping


def _consistent_periods(
    samples: list[tuple[int, int, int]], shape: tuple[int, int]
) -> list[tuple[int, int, dict[tuple[int, int], int]]]:
    height, width = map(int, shape)
    candidates = []
    for row_period in range(1, height + 1):
        for col_period in range(1, width + 1):
            mapping = _mapping_for_period(samples, row_period, col_period)
            if mapping is not None:
                candidates.append((row_period, col_period, mapping))
    return candidates


def _interior_keys(frame_bbox: tuple[int, int, int, int], row_period: int, col_period: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    return {
        (r % row_period, c % col_period)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    }


def _best_period(
    samples: list[tuple[int, int, int]], empty_frame: dict, shape: tuple[int, int]
) -> tuple[int, int, dict[tuple[int, int], int]]:
    candidates = _consistent_periods(samples, shape)
    assert candidates
    min_area = min(row_period * col_period for row_period, col_period, _ in candidates)
    best = [
        candidate
        for candidate in candidates
        if candidate[0] * candidate[1] == min_area
    ]
    assert len(best) == 1
    row_period, col_period, mapping = best[0]
    assert _interior_keys(empty_frame["bbox"], row_period, col_period) <= set(mapping)
    return int(row_period), int(col_period), mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the unique empty frame by extending the same globally phase-aligned periodic wallpaper seen through the other frames.

    Input structure:
    - The grid has a black background and several non-touching one-cell-thick rectangular frames of one frame color.
    - Exactly one frame has an all-black interior; every other frame interior shows a colored patch sampled from one shared wallpaper.
    - The wallpaper is anchored to absolute `(row, col)` coordinates, so different windows reveal different phase offsets of the same periodic tile.

    Rule:
    - Detect the rectangular frames and collect every interior cell from the non-empty windows as evidence.
    - Find the unique smallest row period and column period whose residue classes explain all visible samples with consistent colors.
    - Use that absolute-coordinate periodic rule to fill the empty frame interior cell by cell; keep the frame borders, other window contents, and the outside background unchanged.

    Key insight:
    - The visible patches are not independent motifs: they are fragments of one global wallpaper, so the missing window must be reconstructed by absolute phase alignment rather than by copying any single patch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, _frame_color, frames, _empty_frame = _extract_frames(g)
    samples, empty_frame, _nonempty_frames = _window_samples(g, frames, bg)
    row_period, col_period, mapping = _best_period(samples, empty_frame, g.shape)

    out = g.copy()
    r0, c0, r1, c1 = empty_frame["bbox"]
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            out[r, c] = mapping[(r % row_period, c % col_period)]
    return out


def _tile_has_smaller_period(tile: np.ndarray) -> bool:
    row_period, col_period = map(int, tile.shape)
    for r_small in range(1, row_period + 1):
        for c_small in range(1, col_period + 1):
            if (r_small, c_small) == (row_period, col_period):
                continue
            ok = True
            for r in range(row_period):
                for c in range(col_period):
                    if int(tile[r, c]) != int(tile[r % r_small, c % c_small]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _build_tile(row_period: int, col_period: int) -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        palette_size = random.randint(3, min(6, len(WALLPAPER_COLORS)))
        palette = random.sample(WALLPAPER_COLORS, palette_size)
        tile = np.array(
            [
                [random.choice(palette) for _ in range(col_period)]
                for _ in range(row_period)
            ],
            dtype=int,
        )
        if len(set(map(int, tile.ravel()))) < 3:
            continue
        if not _tile_has_smaller_period(tile):
            return tile
    raise RuntimeError("failed to build aperiodic tile")


def _wallpaper_patch(
    tile: np.ndarray, top: int, left: int, height: int, width: int
) -> np.ndarray:
    row_period, col_period = map(int, tile.shape)
    rows = np.arange(top, top + height, dtype=int) % row_period
    cols = np.arange(left, left + width, dtype=int) % col_period
    return tile[rows[:, None], cols[None, :]].copy()


def _choose_nonempty_size(kind: str, row_period: int, col_period: int) -> tuple[int, int]:
    if kind == "wide":
        height = random.randint(2, row_period - 1)
        width = random.randint(col_period, min(NONEMPTY_SIZE_CAP, col_period + 1))
    elif kind == "tall":
        height = random.randint(row_period, min(NONEMPTY_SIZE_CAP, row_period + 1))
        width = random.randint(2, col_period - 1)
    else:
        height = random.randint(2, row_period - 1)
        width = random.randint(2, col_period - 1)
    assert height < row_period or width < col_period
    return int(height), int(width)


def _separated(bboxes: list[tuple[int, int, int, int]], new_bbox: tuple[int, int, int, int]) -> bool:
    nr0, nc0, nr1, nc1 = new_bbox
    for br0, bc0, br1, bc1 in bboxes:
        row_gap = nr1 + 1 < br0 or br1 + 1 < nr0
        col_gap = nc1 + 1 < bc0 or bc1 + 1 < nc0
        if not (row_gap or col_gap):
            return False
    return True


def _place_windows(
    rows: int, cols: int, specs: list[dict]
) -> list[dict] | None:
    placed: list[dict | None] = [None] * len(specs)
    taken: list[tuple[int, int, int, int]] = []
    order = sorted(
        range(len(specs)),
        key=lambda idx: (specs[idx]["inner_h"] + 2) * (specs[idx]["inner_w"] + 2),
        reverse=True,
    )
    for idx in order:
        spec = specs[idx]
        outer_h = spec["inner_h"] + 2
        outer_w = spec["inner_w"] + 2
        candidates = [
            (r0, c0)
            for r0 in range(PLACEMENT_MARGIN, rows - outer_h - PLACEMENT_MARGIN + 1)
            for c0 in range(PLACEMENT_MARGIN, cols - outer_w - PLACEMENT_MARGIN + 1)
        ]
        random.shuffle(candidates)
        placed_here = None
        for r0, c0 in candidates[:MAX_PLACEMENT_TRIES]:
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if _separated(taken, bbox):
                placed_here = {
                    **spec,
                    "bbox": bbox,
                }
                taken.append(bbox)
                break
        if placed_here is None:
            return None
        placed[idx] = placed_here
    assert all(item is not None for item in placed)
    return [item for item in placed if item is not None]


def _render_input(tile: np.ndarray, placed_specs: list[dict], rows: int, cols: int) -> np.ndarray:
    grid = np.full((rows, cols), BACKGROUND, dtype=int)
    for spec in placed_specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        if spec["empty"]:
            continue
        patch = _wallpaper_patch(tile, r0 + 1, c0 + 1, spec["inner_h"], spec["inner_w"])
        grid[r0 + 1 : r1, c0 + 1 : c1] = patch
    return grid


def _residue_coverage(spec: dict, row_period: int, col_period: int) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = spec["bbox"]
    return {
        ((r0 + 1 + dr) % row_period, (c0 + 1 + dc) % col_period)
        for dr in range(spec["inner_h"])
        for dc in range(spec["inner_w"])
    }


def _patches_are_distinct(placed_specs: list[dict], tile: np.ndarray) -> bool:
    patches = []
    for spec in placed_specs:
        if spec["empty"]:
            continue
        r0, c0, _r1, _c1 = spec["bbox"]
        patch = _wallpaper_patch(tile, r0 + 1, c0 + 1, spec["inner_h"], spec["inner_w"])
        if len(set(map(int, patch.ravel()))) < 2:
            return False
        patches.append(patch)
    for i, patch_a in enumerate(patches):
        for patch_b in patches[i + 1 :]:
            if patch_a.shape == patch_b.shape and np.array_equal(patch_a, patch_b):
                return False
    return True


def _contains_subpatch(patch: np.ndarray, target: np.ndarray) -> bool:
    ph, pw = patch.shape
    th, tw = target.shape
    if th > ph or tw > pw:
        return False
    for r0 in range(ph - th + 1):
        for c0 in range(pw - tw + 1):
            if np.array_equal(patch[r0 : r0 + th, c0 : c0 + tw], target):
                return True
    return False


def _semantic_checks(tile: np.ndarray, placed_specs: list[dict], inp: np.ndarray) -> bool:
    row_period, col_period = map(int, tile.shape)
    empty_specs = [spec for spec in placed_specs if spec["empty"]]
    nonempty_specs = [spec for spec in placed_specs if not spec["empty"]]
    if len(empty_specs) != 1 or len(nonempty_specs) < 4:
        return False

    empty_spec = empty_specs[0]
    target = _wallpaper_patch(
        tile,
        empty_spec["bbox"][0] + 1,
        empty_spec["bbox"][1] + 1,
        empty_spec["inner_h"],
        empty_spec["inner_w"],
    )
    if len(set(map(int, target.ravel()))) < 2:
        return False

    seen = set()
    for spec in nonempty_specs:
        cov = _residue_coverage(spec, row_period, col_period)
        if len(cov) == row_period * col_period:
            return False
        seen |= cov
    if seen != {
        (r, c)
        for r in range(row_period)
        for c in range(col_period)
    }:
        return False

    if not _patches_are_distinct(placed_specs, tile):
        return False

    for spec in nonempty_specs:
        r0, c0, _r1, _c1 = spec["bbox"]
        patch = _wallpaper_patch(tile, r0 + 1, c0 + 1, spec["inner_h"], spec["inner_w"])
        if _contains_subpatch(patch, target):
            return False

    samples = []
    for spec in nonempty_specs:
        r0, c0, _r1, _c1 = spec["bbox"]
        patch = _wallpaper_patch(tile, r0 + 1, c0 + 1, spec["inner_h"], spec["inner_w"])
        for dr in range(spec["inner_h"]):
            for dc in range(spec["inner_w"]):
                samples.append((r0 + 1 + dr, c0 + 1 + dc, int(patch[dr, dc])))

    consistent = _consistent_periods(samples, inp.shape)
    min_area = min(row_p * col_p for row_p, col_p, _ in consistent)
    best = [(row_p, col_p) for row_p, col_p, _ in consistent if row_p * col_p == min_area]
    if best != [(row_period, col_period)]:
        return False

    mapping = _mapping_for_period(samples, row_period, col_period)
    if mapping is None:
        return False
    if set(mapping) != {
        (r, c)
        for r in range(row_period)
        for c in range(col_period)
    }:
        return False

    expected = inp.copy()
    er0, ec0, er1, ec1 = empty_spec["bbox"]
    expected[er0 + 1 : er1, ec0 + 1 : ec1] = target
    return np.array_equal(main(inp), expected)


def _attempt_generate() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    row_period = random.randint(*ROW_PERIOD_RANGE)
    col_period = random.randint(*COL_PERIOD_RANGE)
    tile = _build_tile(row_period, col_period)

    nonempty_count = random.randint(*NONEMPTY_WINDOW_COUNT_RANGE)
    kinds = ["wide", "tall"]
    extra_kinds = ["wide", "tall", "small"]
    while len(kinds) < nonempty_count:
        kinds.append(random.choice(extra_kinds))
    random.shuffle(kinds)

    specs = []
    for kind in kinds:
        inner_h, inner_w = _choose_nonempty_size(kind, row_period, col_period)
        specs.append({"empty": False, "inner_h": inner_h, "inner_w": inner_w})

    max_non_h = max(spec["inner_h"] for spec in specs)
    max_non_w = max(spec["inner_w"] for spec in specs)
    empty_h_lo = max(row_period, max_non_h + 1)
    empty_w_lo = max(col_period, max_non_w + 1)
    if empty_h_lo > EMPTY_SIZE_CAP or empty_w_lo > EMPTY_SIZE_CAP:
        return None
    empty_h = random.randint(empty_h_lo, EMPTY_SIZE_CAP)
    empty_w = random.randint(empty_w_lo, EMPTY_SIZE_CAP)
    specs.append({"empty": True, "inner_h": empty_h, "inner_w": empty_w})

    placed_specs = _place_windows(rows, cols, specs)
    if placed_specs is None:
        return None

    inp = _render_input(tile, placed_specs, rows, cols)
    if not _semantic_checks(tile, placed_specs, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid phase-window puzzle")
