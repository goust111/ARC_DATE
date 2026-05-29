# Combo variant of 8698868d (index 6)
# combo_spec: D1/5+D8/0
# Name: 空间排列匹配多像素框架洞
#
# Design analysis
# ===============
# a) concretization + behavior trace
#    - Keep the original object family: a tight grid of large square frames plus scattered smaller square sprites.
#    - Original task paired a frame and a sprite only by the number of enclosed 4-connected holes.
#      Example: two 2-hole frames were interchangeable even if one was diagonal and one was same-row.
#    - This combo changes two linked steps:
#      1. Matching key becomes the row/column rank order of hole centroids.
#         Example:
#           diagonal 2-hole layout -> signature ((0,0),(1,1))
#           same-row 2-hole layout -> signature ((0,0),(0,1))
#      2. Frame holes are no longer forced to be single pixels; they are connected multi-pixel blobs.
#         Their shapes differ from sprite holes, so raw geometry is richer, but matching still depends only on
#         centroid arrangement. That forces the solver to extract component centroids rather than count cells.
# b) complexity calibration
#    - Use a catalog of 6 arrangement classes: 4 with two holes and 2 with three holes.
#    - For N=2..4, all chosen patterns have the same hole count (=2), so counting is provably insufficient.
#    - For N=5..6, repeated two-hole layouts still dominate and at least one three-hole layout is added.
#    - Hole shapes are restricted to a small set of connected blobs, which keeps examples readable while making
#      frame interiors visibly different from the original single-pixel-hole family.
# c) implicit assumption audit
#    - Original `main()` stored only `num_holes` for frames/sprites; that discards the needed arrangement signal.
#      Replace it with a centroid-rank signature.
#    - Original frame generator assumed isolated single-pixel holes. This combo breaks that assumption, so frame
#      construction must explicitly emit connected multi-pixel components and keep them mutually separated.
#    - Matching by `f["holes"] == s["holes"]` is invalid once multiple layouts share the same count.
#      Matching must consume sprites by exact signature.
#    - Grid inference by largest square size, border-majority frame color, and final nesting/recoloring still hold.
# d) placement legality
#    - Frames are pasted as a gapless rows x cols grid.
#    - Sprites are placed only in remaining background using `random_free_location(..., padding=1)`, so they do not
#      overlap frames or each other and remain recoverable as separate monochromatic components.
#    - Output-first validation is retained: every generated input is accepted only if `main(input)` equals the
#      intended packed output exactly.
# e) visual observability
#    - Human-solvable evidence comes from repeated hole counts with visibly different layouts in the same sample.
#      A solver that only counts holes cannot explain why one 2-hole sprite goes to the diagonal frame while another
#      2-hole sprite goes to the same-row frame.
#    - No invisible metadata such as color ordering is used. Only visible hole placement matters.
# f) cleanliness
#    - Hole components are sparse, non-touching, and stay away from one another enough to remain visually separable.
#    - Output remains the same tidy packed frame grid as the base family; only the pairing rule and frame hole style
#      are changed.
# g) inferability of the change
#    - Colors merely identify objects. The only cross-object relation is spatial arrangement of holes.
#    - Because frame holes use multi-pixel blobs while sprites use a mix of single- and multi-pixel holes, examples
#      show that exact shape is not the key; the transferable feature is the centroid layout.

from __future__ import annotations

from collections import Counter, defaultdict
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


BG = 8
FRAME_SIZE = 8
SPRITE_SIZE = FRAME_SIZE - 2
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
GEN_RETRIES = 400

# Each pattern stores the intended arrangement class. Frame holes are always multi-pixel components.
PATTERN_CATALOG: tuple[dict[str, object], ...] = (
    {
        "name": "diag2",
        "count": 2,
        "frame_components": (
            ((1, 1), (1, 2), (2, 1)),
            ((5, 5), (5, 6), (6, 6)),
        ),
        "sprite_components": (
            ((1, 1),),
            ((4, 4), (4, 3)),
        ),
    },
    {
        "name": "anti2",
        "count": 2,
        "frame_components": (
            ((1, 5), (1, 6), (2, 6)),
            ((5, 1), (6, 1), (6, 2)),
        ),
        "sprite_components": (
            ((1, 4), (2, 4)),
            ((4, 1),),
        ),
    },
    {
        "name": "row2",
        "count": 2,
        "frame_components": (
            ((3, 1), (4, 1)),
            ((3, 5), (4, 5)),
        ),
        "sprite_components": (
            ((2, 1),),
            ((2, 4),),
        ),
    },
    {
        "name": "col2",
        "count": 2,
        "frame_components": (
            ((1, 3), (1, 4)),
            ((6, 3), (6, 4)),
        ),
        "sprite_components": (
            ((1, 2), (1, 3)),
            ((4, 2), (4, 3)),
        ),
    },
    {
        "name": "diag3",
        "count": 3,
        "frame_components": (
            ((1, 1), (1, 2)),
            ((3, 3), (3, 4), (4, 3), (4, 4)),
            ((5, 5), (6, 5)),
        ),
        "sprite_components": (
            ((1, 1),),
            ((2, 2), (3, 2)),
            ((4, 4),),
        ),
    },
    {
        "name": "l3",
        "count": 3,
        "frame_components": (
            ((1, 1), (1, 2)),
            ((6, 1), (6, 2)),
            ((6, 5), (6, 6)),
        ),
        "sprite_components": (
            ((1, 1),),
            ((4, 1),),
            ((4, 4),),
        ),
    },
)

COUNT2_PATTERNS = [p for p in PATTERN_CATALOG if int(p["count"]) == 2]

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _compress_values(values: list[float], tol: float = 1e-6) -> list[float]:
    uniq: list[float] = []
    for value in sorted(float(v) for v in values):
        if not uniq or abs(float(value) - float(uniq[-1])) > tol:
            uniq.append(float(value))
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
    centroids: list[tuple[float, float]] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        centroids.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))
    return _rank_signature(centroids)


def _validate_pattern_catalog() -> None:
    signatures: dict[tuple[tuple[int, int], ...], str] = {}
    for pattern in PATTERN_CATALOG:
        frame_patch = _make_patch(FRAME_SIZE, 1, pattern["frame_components"])
        sprite_patch = _make_patch(SPRITE_SIZE, 2, pattern["sprite_components"])
        frame_sig = _hole_signature(frame_patch == int(BG))
        sprite_sig = _hole_signature(sprite_patch == int(BG))
        assert frame_sig == sprite_sig, f"pattern {pattern['name']} frame/sprite signature mismatch"
        assert frame_sig not in signatures, (
            f"pattern {pattern['name']} duplicates signature of {signatures[frame_sig]}"
        )
        signatures[frame_sig] = str(pattern["name"])


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

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
    assert int(sprite_size) == int(frame_size - 2)
    return int(frame_size), sizes.get(frame_size, []), sizes.get(sprite_size, [])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered sprites to grid frames by hole-arrangement signature, then pack the matched sprites into the frames.

    Input structure:
    - Background is azure (8). Large same-size square frames form a tight rows x cols grid, while smaller same-size
      square sprites are scattered elsewhere on the background.
    - Every frame and sprite is monochromatic except for enclosed hole regions. In this combo, frame holes can be
      multi-pixel 4-connected blobs instead of isolated single pixels.

    Rule:
    - For each frame interior and each sprite, find the 4-connected hole components and compute the rank-order pattern
      of their centroids: which holes are above/below and left/right of the others.
    - Pair each frame with the sprite having the same centroid-rank signature, not merely the same number of holes.
    - Output the tight frame grid only, placing the matched sprite inside each frame border and recoloring the sprite's
      hole pixels to the surrounding frame color.

    Key insight:
    - Hole shape and hole area can vary, especially because frame holes are multi-pixel, but the transferable feature
      is the relative arrangement of hole components. Count alone is intentionally ambiguous in these instances.
    """
    g = np.array(grid, dtype=int)
    frame_size, frames, sprites_raw = _extract_objects(g)
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
        sig = frame["signature"]
        assert sprites_by_sig.get(sig)
        matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprites_by_sig[sig].pop(0)

    out = np.full((int(n_rows * frame_size), int(n_cols * frame_size)), int(BG), dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        sprite = matches[(fr, fc)]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


def _make_patch(size: int, color: int, components: tuple[tuple[tuple[int, int], ...], ...]) -> np.ndarray:
    patch = np.full((size, size), int(color), dtype=int)
    for comp in components:
        for r, c in comp:
            patch[int(r), int(c)] = int(BG)
    return patch


_validate_pattern_catalog()


def _choose_patterns(n: int) -> list[dict[str, object]]:
    if n <= len(COUNT2_PATTERNS):
        chosen = random.sample(COUNT2_PATTERNS, n)
    else:
        chosen = random.sample(list(PATTERN_CATALOG), n)
    random.shuffle(chosen)
    return chosen


def _assign_sprite_colors(patterns: list[dict[str, object]]) -> dict[str, int]:
    # The original solver consumes sprites in ascending color order within each hole-count bucket.
    # Deliberately rotate the pattern order inside each count bucket so count-only matching is wrong.
    grouped: dict[int, list[str]] = defaultdict(list)
    for pattern in patterns:
        grouped[int(pattern["count"])].append(str(pattern["name"]))

    adversarial_order: list[str] = []
    for count in sorted(grouped):
        names = grouped[count]
        if len(names) > 1:
            names = names[1:] + names[:1]
        adversarial_order.extend(names)

    colors = sorted(random.sample([c for c in range(10) if c != BG], len(patterns)))
    return {name: int(color) for name, color in zip(adversarial_order, colors)}


def _build_expected_output(
    frames: list[tuple[str, np.ndarray]],
    sprites: list[tuple[str, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
) -> np.ndarray:
    expected = np.full((rows_grid * FRAME_SIZE, cols_grid * FRAME_SIZE), BG, dtype=int)
    sprite_by_name = {name: patch for name, patch in sprites}
    for idx, (name, _patch) in enumerate(frames):
        rr = (idx // cols_grid) * FRAME_SIZE
        cc = (idx % cols_grid) * FRAME_SIZE
        frame_color = int(frame_colors[idx])
        expected[rr : rr + FRAME_SIZE, cc : cc + FRAME_SIZE] = int(frame_color)
        content = sprite_by_name[name].copy()
        content[content == BG] = int(frame_color)
        expected[rr + 1 : rr + 1 + SPRITE_SIZE, cc + 1 : cc + 1 + SPRITE_SIZE] = content
    return expected


def _count_multi_pixel_frame_holes(frames: list[tuple[str, np.ndarray]]) -> int:
    total = 0
    for _name, patch in frames:
        lab, n = label_mask(patch == int(BG), connectivity=4)
        for cid in range(1, int(n) + 1):
            if int(np.sum(lab == int(cid))) > 1:
                total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])

    grid_pixel_h = int(rows_grid * FRAME_SIZE)
    grid_pixel_w = int(cols_grid * FRAME_SIZE)
    min_h = max(CANVAS_MIN, grid_pixel_h + SPRITE_SIZE + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + SPRITE_SIZE + SPRITE_PADDING)
    h_lo = max(min_h, CANVAS_MAX - 4)
    w_lo = max(min_w, CANVAS_MAX - 4)
    if h_lo > CANVAS_MAX or w_lo > CANVAS_MAX:
        return None

    canvas_h = random.randint(h_lo, CANVAS_MAX)
    canvas_w = random.randint(w_lo, CANVAS_MAX)
    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n)
    patterns = _choose_patterns(n)
    sprite_color_by_name = _assign_sprite_colors(patterns)

    frames: list[tuple[str, np.ndarray]] = []
    for color, pattern in zip(frame_colors, patterns):
        frames.append(
            (
                str(pattern["name"]),
                _make_patch(FRAME_SIZE, int(color), pattern["frame_components"]),
            )
        )

    sprites: list[tuple[str, np.ndarray]] = []
    sprite_patterns = patterns[:]
    random.shuffle(sprite_patterns)
    for pattern in sprite_patterns:
        sprites.append(
            (
                str(pattern["name"]),
                _make_patch(
                    SPRITE_SIZE,
                    int(sprite_color_by_name[str(pattern["name"])]),
                    pattern["sprite_components"],
                ),
            )
        )

    if _count_multi_pixel_frame_holes(frames) < n:
        return None

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, (_name, frame_patch) in enumerate(frames):
        rr = r0 + (idx // cols_grid) * FRAME_SIZE
        cc = c0 + (idx % cols_grid) * FRAME_SIZE
        canvas[rr : rr + FRAME_SIZE, cc : cc + FRAME_SIZE] = frame_patch

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

    expected = _build_expected_output(frames, sprites, frame_colors, rows_grid, cols_grid)
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
