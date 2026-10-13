from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np


def _load_arc_utils():
    try:
        from new.solvers.arc_utils import Color, crop_bbox, find_components

        return Color, crop_bbox, find_components
    except Exception:
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "new" / "solvers" / "arc_utils.py"
            if candidate.is_file():
                spec = importlib.util.spec_from_file_location("_arc_utils_fallback", candidate)
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.Color, module.crop_bbox, module.find_components
        raise


Color, crop_bbox, find_components = _load_arc_utils()

BG = Color.BLACK
BORDER_COLORS = (Color.BLUE, Color.RED)
PATTERN_COLORS = (Color.GREEN, Color.YELLOW)
TILE_SHAPES = ((2, 3), (3, 2), (3, 3))
TARGET_SIZE_BY_TILE = {
    2: (6, 8),
    3: (6, 9),
}
MIN_CANVAS = 20
MAX_CANVAS = 30
GEN_ATTEMPTS = 400
TARGET_SORT_MODES = {"colrow", "rowcol"}


def _is_complete_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 4 or w < 4:
        return False
    if int(component["area"]) != 2 * (h + w) - 4:
        return False
    for r_raw, c_raw in component["coords"]:
        r = int(r_raw)
        c = int(c_raw)
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return True


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for component in find_components(grid, bg=BG, connectivity=4, monochromatic=True):
        color = int(component["color"])
        if color == BG or not _is_complete_frame(component):
            continue
        r0, c0, r1, c1 = map(int, component["bbox"])
        inner_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": inner_bbox,
                "inner": crop_bbox(grid, inner_bbox),
                "inner_area": int((r1 - r0 - 1) * (c1 - c0 - 1)),
            }
        )
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1], frame["color"]))
    return frames


def _tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    return np.tile(tile, ((h + th - 1) // th, (w + tw - 1) // tw))[:h, :w].copy()


def _repair_inner(inner: np.ndarray, tile: np.ndarray, phase: tuple[int, int] = (0, 0)) -> np.ndarray:
    out = inner.copy()
    th, tw = tile.shape
    pr, pc = phase
    holes = np.argwhere(out == BG)
    for r_raw, c_raw in holes:
        r = int(r_raw)
        c = int(c_raw)
        out[r, c] = tile[(r + pr) % th, (c + pc) % tw]
    return out


def _assert_target_matches_tile(inner: np.ndarray, tile: np.ndarray) -> None:
    th, tw = tile.shape
    known = np.argwhere(inner != BG)
    for r_raw, c_raw in known:
        r = int(r_raw)
        c = int(c_raw)
        assert int(inner[r, c]) == int(tile[r % th, c % tw])


def _pair_frames_by_color(frames: list[dict]) -> list[dict]:
    by_color: dict[int, list[dict]] = {}
    for frame in frames:
        by_color.setdefault(int(frame["color"]), []).append(frame)

    pairs = []
    for color, group in by_color.items():
        assert len(group) == 2, "each border color must appear on exactly two frames"
        ordered = sorted(group, key=lambda frame: (frame["inner_area"], frame["bbox"][0], frame["bbox"][1]))
        source, target = ordered
        assert source["inner_area"] < target["inner_area"], "paired frames must have distinct sizes"
        tile = source["inner"]
        assert np.all(tile != BG), "the source tile must be complete"
        _assert_target_matches_tile(target["inner"], tile)
        pairs.append({"color": color, "source": source, "target": target})
    return pairs


def _stack_repaired(pairs: list[dict], *, sort_mode: str = "colrow", absolute_phase: bool = False) -> np.ndarray:
    assert sort_mode in TARGET_SORT_MODES
    parts = []
    for pair in pairs:
        target = pair["target"]
        tile = pair["source"]["inner"]
        if absolute_phase:
            phase = (
                int(target["inner_bbox"][0]) % tile.shape[0],
                int(target["inner_bbox"][1]) % tile.shape[1],
            )
        else:
            phase = (0, 0)
        repaired = _repair_inner(target["inner"], tile, phase=phase)
        parts.append((target["bbox"][0], target["bbox"][1], repaired))

    if sort_mode == "colrow":
        parts.sort(key=lambda item: (item[1], item[0]))
    else:
        parts.sort(key=lambda item: (item[0], item[1]))

    widths = {part.shape[1] for _r, _c, part in parts}
    assert len(widths) == 1, "stacked outputs must share a common width"
    return np.vstack([part for _r, _c, part in parts])


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the larger framed periodic patterns by borrowing the complete tile from the same-colored smaller frame.

    Input structure:
    - A black canvas contains several complete one-pixel-thick rectangular frames.
    - Frames come in same-colored pairs; one frame in each pair is small and contains a complete green/yellow tile, while the other is larger and contains the repeated tile with black gaps.
    - Frames do not overlap or touch, and the larger interiors all have widths that can be stacked vertically.

    Rule:
    - Detect every complete rectangular frame and group the frames by border color.
    - In each color group, treat the smaller interior as the source tile and the larger interior as the damaged target.
    - Fill each black target cell at local target coordinate `(r, c)` with the source-tile value at `(r mod tile_h, c mod tile_w)`, keeping non-black target cells unchanged.
    - Output only the repaired larger interiors, ordered by target frame position from left to right and then from top to bottom, stacked with no blank rows between them.

    Key insight:
    - The damaged pattern does not reveal its period on its own; the complete period tile is supplied externally by the same-colored partner frame, and the modulo alignment is relative to the target interior's own top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    assert frames, "expected at least one complete rectangular frame"
    pairs = _pair_frames_by_color(frames)
    return _stack_repaired(pairs, sort_mode="colrow", absolute_phase=False)


def _has_smaller_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if ph == h and pw == w:
                continue
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % ph, c % pw]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _has_shift_symmetry(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for dr in range(h):
        for dc in range(w):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(tile, dr, axis=0), dc, axis=1)
            if np.array_equal(tile, shifted):
                return True
    return False


def _random_tile(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    for _ in range(200):
        tile = np.where(rng.integers(0, 2, size=(h, w)) == 0, PATTERN_COLORS[0], PATTERN_COLORS[1]).astype(int)
        if len(set(map(int, tile.ravel()))) != 2:
            continue
        if _has_smaller_period(tile):
            continue
        if _has_shift_symmetry(tile):
            continue
        return tile
    raise RuntimeError("failed to sample a non-degenerate tile")


def _sample_tile_pair(rng: np.random.Generator, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    coords = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    for _ in range(300):
        tile_a = _random_tile(rng, shape)
        order = [coords[int(i)] for i in rng.permutation(len(coords))]
        for diff_r, diff_c in order:
            tile_b = tile_a.copy()
            tile_b[diff_r, diff_c] = PATTERN_COLORS[1] if tile_a[diff_r, diff_c] == PATTERN_COLORS[0] else PATTERN_COLORS[0]
            if len(set(map(int, tile_b.ravel()))) != 2:
                continue
            if _has_smaller_period(tile_b):
                continue
            if _has_shift_symmetry(tile_b):
                continue
            return tile_a, tile_b, (diff_r, diff_c)
    raise RuntimeError("failed to sample a valid tile pair")


def _has_self_period_candidate(damaged: np.ndarray) -> bool:
    h, w = damaged.shape
    known = np.argwhere(damaged != BG)
    assert len(known) > 0
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            tile = np.full((ph, pw), -1, dtype=int)
            ok = True
            for r_raw, c_raw in known:
                r = int(r_raw)
                c = int(c_raw)
                rr = r % ph
                cc = c % pw
                color = int(damaged[r, c])
                if tile[rr, cc] == -1:
                    tile[rr, cc] = color
                elif tile[rr, cc] != color:
                    ok = False
                    break
            if ok and np.all(tile != -1):
                return True
    return False


def _damage_target(
    rng: np.random.Generator,
    tile: np.ndarray,
    other_tile: np.ndarray,
    target_shape: tuple[int, int],
    diff_coord: tuple[int, int],
) -> np.ndarray | None:
    full = _tile_fill(target_shape, tile)
    h, w = target_shape
    th, tw = tile.shape
    diff_r, diff_c = diff_coord

    hole_mask = np.zeros(target_shape, dtype=bool)
    for r in range(h):
        for c in range(w):
            if r % th == diff_r and c % tw == diff_c:
                hole_mask[r, c] = True

    area = h * w
    stable_positions = [(r, c) for r in range(h) for c in range(w) if not hole_mask[r, c]]
    base_holes = int(hole_mask.sum())
    min_holes = max(base_holes, int(np.ceil(area * 0.18)))
    max_holes = min(area - 2, int(np.floor(area * 0.32)))
    if min_holes > max_holes:
        return None

    extra_min = min_holes - base_holes
    extra_max = min(len(stable_positions), max_holes - base_holes)
    if extra_min > extra_max:
        return None
    extra_count = int(rng.integers(extra_min, extra_max + 1))
    if extra_count:
        picks = rng.choice(len(stable_positions), size=extra_count, replace=False)
        for idx in np.atleast_1d(picks):
            r, c = stable_positions[int(idx)]
            hole_mask[r, c] = True

    damaged = full.copy()
    damaged[hole_mask] = BG
    if len(set(map(int, damaged[damaged != BG].ravel()))) != 2:
        return None

    wrong = _repair_inner(damaged, other_tile)
    if np.array_equal(wrong, full):
        return None
    if _has_self_period_candidate(damaged):
        return None
    return damaged


def _make_frame(border_color: int, inner: np.ndarray) -> np.ndarray:
    h, w = inner.shape
    frame = np.full((h + 2, w + 2), BG, dtype=int)
    frame[0, :] = border_color
    frame[-1, :] = border_color
    frame[:, 0] = border_color
    frame[:, -1] = border_color
    frame[1:-1, 1:-1] = inner
    return frame


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _build_layout(
    rng: np.random.Generator,
    large_top: np.ndarray,
    small_top: np.ndarray,
    large_bottom: np.ndarray,
    small_bottom: np.ndarray,
) -> np.ndarray | None:
    top_band_h = max(large_top.shape[0], small_top.shape[0])
    bottom_band_h = max(large_bottom.shape[0], small_bottom.shape[0])
    top_gap = int(rng.integers(2, 5))
    bottom_gap = int(rng.integers(2, 5))
    row_gap = int(rng.integers(2, 5))

    top_row_w = small_top.shape[1] + top_gap + large_top.shape[1]
    bottom_row_w = large_bottom.shape[1] + bottom_gap + small_bottom.shape[1]

    left_pad = int(rng.integers(1, 4))
    right_pad = int(rng.integers(1, 5))
    top_pad = int(rng.integers(1, 4))
    bottom_pad = int(rng.integers(1, 5))

    canvas_h = top_pad + top_band_h + row_gap + bottom_band_h + bottom_pad
    canvas_w = left_pad + max(top_row_w, bottom_row_w) + right_pad

    canvas_h = max(canvas_h, MIN_CANVAS)
    canvas_w = max(canvas_w, MIN_CANVAS)
    if canvas_h > MAX_CANVAS or canvas_w > MAX_CANVAS:
        return None

    grid = np.full((canvas_h, canvas_w), BG, dtype=int)

    top_row_slack = canvas_w - left_pad - right_pad - top_row_w
    bottom_row_slack = canvas_w - left_pad - right_pad - bottom_row_w
    top_shift = int(rng.integers(0, top_row_slack + 1))
    bottom_shift = int(rng.integers(0, bottom_row_slack + 1))

    top_r0 = top_pad
    bottom_r0 = top_pad + top_band_h + row_gap
    top_small_r = top_r0 + int(rng.integers(0, top_band_h - small_top.shape[0] + 1))
    top_large_r = top_r0 + int(rng.integers(0, top_band_h - large_top.shape[0] + 1))
    bottom_large_r = bottom_r0 + int(rng.integers(0, bottom_band_h - large_bottom.shape[0] + 1))
    bottom_small_r = bottom_r0 + int(rng.integers(0, bottom_band_h - small_bottom.shape[0] + 1))

    top_small_c = left_pad + top_shift
    top_large_c = top_small_c + small_top.shape[1] + top_gap
    bottom_large_c = left_pad + bottom_shift
    bottom_small_c = bottom_large_c + large_bottom.shape[1] + bottom_gap

    _paste(grid, small_top, top_small_r, top_small_c)
    _paste(grid, large_top, top_large_r, top_large_c)
    _paste(grid, large_bottom, bottom_large_r, bottom_large_c)
    _paste(grid, small_bottom, bottom_small_r, bottom_small_c)
    return grid


def _frame_center(frame: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = frame["bbox"]
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _nearest_pair_output(frames: list[dict]) -> np.ndarray | None:
    ordered = sorted(frames, key=lambda frame: (frame["inner_area"], frame["bbox"][0], frame["bbox"][1]))
    smalls = ordered[:2]
    larges = ordered[2:]
    if not smalls or not larges or max(frame["inner_area"] for frame in smalls) >= min(frame["inner_area"] for frame in larges):
        return None

    unused = list(smalls)
    pairs = []
    for large in sorted(larges, key=lambda frame: (frame["bbox"][0], frame["bbox"][1])):
        lr, lc = _frame_center(large)
        best_idx = None
        best_key = None
        for idx, small in enumerate(unused):
            sr, sc = _frame_center(small)
            key = (abs(lr - sr) + abs(lc - sc), small["bbox"][0], small["bbox"][1])
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        assert best_idx is not None
        source = unused.pop(best_idx)
        _assert_target_matches_tile(large["inner"], source["inner"])
        pairs.append({"source": source, "target": large})
    return _stack_repaired(pairs, sort_mode="colrow", absolute_phase=False)


def _row_major_output(frames: list[dict]) -> np.ndarray:
    return _stack_repaired(_pair_frames_by_color(frames), sort_mode="rowcol", absolute_phase=False)


def _absolute_phase_output(frames: list[dict]) -> np.ndarray:
    return _stack_repaired(_pair_frames_by_color(frames), sort_mode="colrow", absolute_phase=True)


def _semantic_checks(grid: np.ndarray) -> bool:
    frames = _extract_frames(grid)
    if len(frames) != 4:
        return False

    colors = sorted({int(frame["color"]) for frame in frames})
    if colors != sorted(BORDER_COLORS):
        return False

    try:
        pairs = _pair_frames_by_color(frames)
        expected = _stack_repaired(pairs, sort_mode="colrow", absolute_phase=False)
    except Exception:
        return False

    if not np.array_equal(main(grid), expected):
        return False

    repaired_parts = [_repair_inner(pair["target"]["inner"], pair["source"]["inner"]) for pair in pairs]
    if len({part.shape[1] for part in repaired_parts}) != 1:
        return False
    if np.array_equal(repaired_parts[0], repaired_parts[1]):
        return False

    for pair in pairs:
        source_tile = pair["source"]["inner"]
        if source_tile.shape not in TILE_SHAPES:
            return False
        if np.any(source_tile == BG):
            return False
        if not np.any(pair["target"]["inner"] == BG):
            return False
        if _has_self_period_candidate(pair["target"]["inner"]):
            return False

    try:
        if np.array_equal(_nearest_pair_output(frames), expected):
            return False
    except Exception:
        return False

    if np.array_equal(_row_major_output(frames), expected):
        return False
    if np.array_equal(_absolute_phase_output(frames), expected):
        return False
    return True


def _generate_one(rng: np.random.Generator) -> np.ndarray | None:
    tile_shape = TILE_SHAPES[int(rng.integers(0, len(TILE_SHAPES)))]
    tile_a, tile_b, diff_coord = _sample_tile_pair(rng, tile_shape)

    height_choices = TARGET_SIZE_BY_TILE[tile_shape[0]]
    width_choices = TARGET_SIZE_BY_TILE[tile_shape[1]]
    target_h_a = int(height_choices[int(rng.integers(0, len(height_choices)))])
    target_h_b = int(height_choices[int(rng.integers(0, len(height_choices)))])
    target_w = int(width_choices[int(rng.integers(0, len(width_choices)))])

    damaged_a = _damage_target(rng, tile_a, tile_b, (target_h_a, target_w), diff_coord)
    if damaged_a is None:
        return None
    damaged_b = _damage_target(rng, tile_b, tile_a, (target_h_b, target_w), diff_coord)
    if damaged_b is None:
        return None

    pair_colors = [int(BORDER_COLORS[int(i)]) for i in rng.permutation(len(BORDER_COLORS))]
    small_a = _make_frame(pair_colors[0], tile_a)
    large_a = _make_frame(pair_colors[0], damaged_a)
    small_b = _make_frame(pair_colors[1], tile_b)
    large_b = _make_frame(pair_colors[1], damaged_b)

    grid = _build_layout(
        rng,
        large_top=large_a,
        small_top=small_b,
        large_bottom=large_b,
        small_bottom=small_a,
    )
    if grid is None:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paired-frame periodic-repair input")
