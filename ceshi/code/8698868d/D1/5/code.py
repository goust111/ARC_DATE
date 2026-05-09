# Variant of 8698868d
# Dimension: D1 — 匹配条件：连通洞计数 (match_cond)
# Change: hole_counting — 精灵与框架各自内部的 4-连通背景色洞分量数目相等即配对... → 洞的空间排列匹配
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     before: frame/sprite are paired only by the number of enclosed 4-connected holes
#     after: frame/sprite are paired by the relative arrangement of those holes
#   concretization:
#     - Keep the original object family: large square frames in a tight grid, smaller square sprites scattered elsewhere.
#     - Each frame interior contains 2 or 3 isolated bg holes. Each sprite also contains 2 or 3 isolated bg holes.
#     - The key is no longer the raw count; it is the hole-layout signature formed from hole-centroid row/col ranks.
#     - Example:
#       frame A holes = {(top,left), (bottom,right)}  -> signature ((0,0),(1,1))
#       frame B holes = {(mid,left), (mid,right)}     -> signature ((0,0),(0,1))
#       Both have count=2, so the original solver would treat them as equivalent.
#       The variant solver must distinguish them and only pair by the rank signature.
#   complexity_calibration:
#     - Use a small catalog of 6 layout classes: 4 patterns with 2 holes, 2 patterns with 3 holes.
#     - N is still 2..6, but selected patterns are unique while hole counts intentionally repeat.
#     - This gives repeated evidence that "same count" is insufficient while keeping the visual vocabulary small enough
#       for humans to infer from a few examples.
#     - Frame sizes are restricted to values that still leave enough interior resolution for distinct layouts.
#   implicit_assumption_audit:
#     - Original frame feature extraction stored only "holes": that loses exactly the information this variant needs.
#       Replace it with a layout signature computed from enclosed-hole centroids.
#     - Original sprite processing also kept only hole counts; same replacement is required.
#     - Original matching loop used f["holes"] == s["holes"]; this is invalid once multiple arrangements share a count.
#       Matching must consume sprites by identical layout signature instead.
#     - Size grouping, frame-grid inference, recoloring, and output assembly remain valid because the variant changes
#       only the pairing criterion, not object sizes or nesting behavior.
#   placement_legality:
#     - Frames still form a tight rows x cols grid with no gaps.
#     - Sprites still have size frame_size - 2 and are placed in remaining bg area with padding, so they do not touch
#       frames or each other and remain recoverable by monochromatic connected components.
#     - The generator uses output-first validation: it constructs the intended arrangement pairing, then checks that
#       main(input) reproduces the expected output exactly.
#   visual_observability:
#     - Every sample contains repeated hole counts, usually multiple 2-hole patterns at once.
#     - Distinguishing features are visible in pixel space (diagonal vs anti-diagonal vs same-row vs same-col, etc.).
#     - No color-id ordering is used; colors only mark object identity.
#   cleanliness:
#     - Holes stay sparse and non-overlapping; output remains a clean frame grid with one nested sprite per slot.

from __future__ import annotations

from collections import Counter, defaultdict
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


BG = 8
FRAME_SIZE_RANGE = [7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
GEN_RETRIES = 400

PATTERN_CATALOG: tuple[dict[str, object], ...] = (
    {"name": "diag2", "anchors": (("top", "left"), ("bottom", "right"))},
    {"name": "anti2", "anchors": (("top", "right"), ("bottom", "left"))},
    {"name": "row2", "anchors": (("mid", "left"), ("mid", "right"))},
    {"name": "col2", "anchors": (("top", "mid"), ("bottom", "mid"))},
    {"name": "diag3", "anchors": (("top", "left"), ("mid", "mid"), ("bottom", "right"))},
    {"name": "l3", "anchors": (("top", "left"), ("bottom", "left"), ("bottom", "right"))},
)
COUNT2_PATTERNS = [p for p in PATTERN_CATALOG if len(p["anchors"]) == 2]
COUNT3_PATTERNS = [p for p in PATTERN_CATALOG if len(p["anchors"]) == 3]


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _compress_values(values: list[float], tol: float = 1e-6) -> list[float]:
    uniq: list[float] = []
    for v in sorted(float(x) for x in values):
        if not uniq or abs(float(v) - float(uniq[-1])) > tol:
            uniq.append(float(v))
    return uniq


def _rank_signature(centroids: list[tuple[float, float]]) -> tuple[tuple[int, int], ...]:
    if not centroids:
        return ()

    row_vals = _compress_values([float(r) for r, _c in centroids])
    col_vals = _compress_values([float(c) for _r, c in centroids])

    def _rank_of(value: float, uniq: list[float]) -> int:
        for idx, ref in enumerate(uniq):
            if abs(float(value) - float(ref)) <= 1e-6:
                return int(idx)
        raise ValueError("rank not found")

    return tuple(
        sorted(
            (_rank_of(float(r), row_vals), _rank_of(float(c), col_vals))
            for r, c in centroids
        )
    )


def _hole_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    lab, n = label_mask(mask.astype(bool), connectivity=4)
    cents: list[tuple[float, float]] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        cents.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))
    return _rank_signature(cents)


def _extract_objects(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        h = int(bbox_h(comp["bbox"]))
        w = int(bbox_w(comp["bbox"]))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * 0.5):
            square_regions.append((h, comp))
    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)
    return int(frame_size), int(sprite_size), sizes.get(frame_size, []), sizes.get(sprite_size, [])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by hole-layout signature, then nest each sprite inside its frame.

    The signature is the sorted tuple of (row-rank, col-rank) pairs of hole centroids.
    Two objects match iff their enclosed holes share the same relative arrangement.
    """
    g = np.array(grid, dtype=int)
    frame_size, _sprite_size, frames, sprites_raw = _extract_objects(g)
    assert frames

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            inner = region[1:-1, 1:-1]
            signature = _hole_signature(inner != int(frame_color))
            frame_grid.append(
                {
                    "color": int(frame_color),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "signature": signature,
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        signature = _hole_signature(sprite_region != int(sprite_color))
        sprites.append(
            {
                "color": int(sprite_color),
                "signature": signature,
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    sprites_by_sig: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for sprite in sprites:
        sprites_by_sig[sprite["signature"]].append(sprite)

    matches: dict[tuple[int, int], dict] = {}
    for frame in frame_grid:
        key = frame["signature"]
        if sprites_by_sig.get(key):
            matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprites_by_sig[key].pop(0)

    out = np.full((int(n_rows * frame_size), int(n_cols * frame_size)), int(BG), dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)
        sprite = matches.get((fr, fc))
        if sprite is None:
            continue
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content
    return out


def _anchor_map(size: int) -> dict[str, int]:
    return {"top": 1, "mid": int(size // 2), "bottom": int(size - 2), "left": 1, "right": int(size - 2)}


def _pattern_positions(size: int, pattern: dict[str, object]) -> list[tuple[int, int]]:
    anchors = _anchor_map(size)
    anchors["mid"] = int(size // 2)
    out: list[tuple[int, int]] = []
    for r_name, c_name in pattern["anchors"]:
        out.append((int(anchors[r_name]), int(anchors[c_name])))
    return out


def _make_patch(size: int, color: int, pattern: dict[str, object]) -> np.ndarray:
    patch = np.full((size, size), int(color), dtype=int)
    for r, c in _pattern_positions(size, pattern):
        patch[int(r), int(c)] = int(BG)
    return patch


def _choose_patterns(n: int) -> list[dict[str, object]]:
    if n <= 4:
        chosen = random.sample(COUNT2_PATTERNS, n)
    elif n == 5:
        chosen = COUNT2_PATTERNS[:] + random.sample(COUNT3_PATTERNS, 1)
    else:
        chosen = list(PATTERN_CATALOG)
    random.shuffle(chosen)
    return chosen


def _valid_frame_sizes(rows_grid: int, cols_grid: int) -> list[int]:
    valid: list[int] = []
    for frame_size in FRAME_SIZE_RANGE:
        sprite_size = frame_size - 2
        min_h = max(CANVAS_MIN, rows_grid * frame_size + sprite_size + SPRITE_PADDING)
        min_w = max(CANVAS_MIN, cols_grid * frame_size + sprite_size + SPRITE_PADDING)
        if min_h <= CANVAS_MAX and min_w <= CANVAS_MAX:
            valid.append(int(frame_size))
    return valid


def _build_expected_output(
    frames: list[tuple[str, np.ndarray]],
    sprites: list[tuple[str, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_name = {name: patch for name, patch in sprites}
    for idx, (name, _patch) in enumerate(frames):
        rr = (idx // cols_grid) * frame_size
        cc = (idx % cols_grid) * frame_size
        fc = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(fc)
        content = sprite_by_name[name].copy()
        content[content == BG] = int(fc)
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    valid_sizes = _valid_frame_sizes(rows_grid, cols_grid)
    if not valid_sizes:
        return None

    frame_size = random.choice(valid_sizes)
    sprite_size = int(frame_size - 2)
    patterns = _choose_patterns(n)

    grid_pixel_h = int(rows_grid * frame_size)
    grid_pixel_w = int(cols_grid * frame_size)
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    h_lo = max(min_h, CANVAS_MAX - 4)
    w_lo = max(min_w, CANVAS_MAX - 4)
    if h_lo > CANVAS_MAX or w_lo > CANVAS_MAX:
        return None
    canvas_h = random.randint(h_lo, CANVAS_MAX)
    canvas_w = random.randint(w_lo, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n)
    sprite_colors = random.sample(available, n)

    frame_patterns = patterns[:]
    sprite_patterns = patterns[:]
    random.shuffle(frame_patterns)
    random.shuffle(sprite_patterns)

    frames: list[tuple[str, np.ndarray]] = []
    for color, pattern in zip(frame_colors, frame_patterns):
        frames.append((str(pattern["name"]), _make_patch(frame_size, int(color), pattern)))

    sprites: list[tuple[str, np.ndarray]] = []
    for color, pattern in zip(sprite_colors, sprite_patterns):
        sprites.append((str(pattern["name"]), _make_patch(sprite_size, int(color), pattern)))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, (_name, frame_patch) in enumerate(frames):
        rr = r0 + (idx // cols_grid) * frame_size
        cc = c0 + (idx % cols_grid) * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _name, sprite_patch in sprites:
        try:
            sr, sc = random_free_location(
                canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, sr, sc, transparent=BG)

    expected = _build_expected_output(
        frames,
        sprites,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
    )
    try:
        out = main(canvas)
    except Exception:
        return None
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
