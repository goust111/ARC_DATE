# Combo variant of 8698868d (index 40)
# combo_spec: D1/5+D12/0
# Name: 空间排列匹配+八连通
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     before: frame and sprite were paired by the number of 4-connected holes.
#     after: they are paired by the row/column rank signature of 8-connected hole-component centroids.
#   concretization:
#     - Keep the original family: large square frames in a tight grid and smaller square sprites scattered elsewhere.
#     - Replace single-pixel holes with "diagonal mini-blocks": each logical hole anchor is a 2x2 block containing
#       exactly one diagonal, so it is 2 holes under 4-connectivity but 1 hole under 8-connectivity.
#     - A logical pattern is defined only by which anchor blocks are occupied: diag2, anti2, row2, col2.
#     - Example trace:
#         frame pattern = diag2 using blocks TL+BR, with "\" in TL and "/" in BR
#         sprite pattern = diag2 using the same TL+BR blocks, but "/" in TL and "\" in BR
#         under 4-connectivity both objects expose 4 isolated pixels with different fine-grained rank signatures;
#         under 8-connectivity they both collapse to 2 hole components whose centroids are TL and BR,
#         so both signatures become ((0,0),(1,1)) and they match.
#   complexity_calibration:
#     - All logical patterns have the same 8-connected hole count (=2), so counting is useless.
#     - The catalog is small (4 layouts) and visually distinct (same-row, same-col, main diagonal, anti-diagonal).
#     - Inputs use N in {2,4}; N=4 samples especially give repeated evidence that arrangement, not count, drives pairing.
#   implicit_assumption_audit:
#     - Original code stored only hole counts for frames and sprites; that discards the needed spatial information.
#       Replace this with centroid-rank signatures.
#     - Original hole extraction used connectivity=4; that would treat each diagonal mini-block as two holes and
#       often break the intended pairing because frame and sprite may use different diagonal orientations inside a block.
#     - Size grouping and frame-grid recovery still work, but the detectable square-size range must allow larger
#       9x9 or 10x10 frames because the 8-connectivity effect needs enough interior room for separated 2x2 blocks.
#   placement_legality:
#     - Frames remain a tight visible grid with no gaps. Sprites remain non-overlapping monochromatic components.
#     - Hole blocks are placed strictly inside the solid border, and distinct blocks are separated by at least one
#       empty row/column so different 8-components never merge accidentally.
#     - The generator validates each sample against the fused solver and also rejects samples where a 4-connected
#       arrangement solver would produce the same output, preventing degeneration.
#   visual_observability:
#     - Humans can see the four coarse layouts directly from the positions of the diagonal mini-blocks.
#     - The 8-connectivity twist is visible because each logical hole "looks like two diagonal pixels" yet must be
#       treated as one merged hole region; the within-block slash direction is irrelevant for matching.
#   cleanliness:
#     - Objects remain sparse, aligned, and non-overlapping; output is still a compact frame grid with no background.
from __future__ import annotations

from collections import Counter, defaultdict
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


BG = 8
FRAME_SIZE_RANGE = [9, 10]
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
GEN_RETRIES = 500

PATTERN_CATALOG: tuple[dict[str, object], ...] = (
    {"name": "diag2", "blocks": ("tl", "br")},
    {"name": "anti2", "blocks": ("tr", "bl")},
    {"name": "row2", "blocks": ("tl", "tr")},
    {"name": "col2", "blocks": ("tl", "bl")},
)

LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {
    2: [(1, 2), (2, 1)],
    4: [(2, 2)],
}
VALID_NS = [2, 4]


def _compress_values(values: list[float], tol: float = 1e-6) -> list[float]:
    uniq: list[float] = []
    for value in sorted(float(v) for v in values):
        if not uniq or abs(float(value) - float(uniq[-1])) > tol:
            uniq.append(float(value))
    return uniq


def _rank_signature(centroids: list[tuple[float, float]]) -> tuple[tuple[int, int], ...]:
    if not centroids:
        return ()

    rows = _compress_values([float(r) for r, _c in centroids])
    cols = _compress_values([float(c) for _r, c in centroids])

    def _rank_of(value: float, refs: list[float]) -> int:
        for idx, ref in enumerate(refs):
            if abs(float(value) - float(ref)) <= 1e-6:
                return int(idx)
        raise ValueError("rank not found")

    return tuple(
        sorted(
            (_rank_of(float(r), rows), _rank_of(float(c), cols))
            for r, c in centroids
        )
    )


def _hole_signature(mask: np.ndarray, connectivity: int) -> tuple[tuple[int, int], ...]:
    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    centroids: list[tuple[float, float]] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        centroids.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))
    return _rank_signature(centroids)


def _extract_objects(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        size_h = int(bbox_h(comp["bbox"]))
        size_w = int(bbox_w(comp["bbox"]))
        area = int(comp["area"])
        if size_h == size_w and 6 <= size_h <= 12 and area >= int(size_h * size_h * 0.5):
            square_regions.append((int(size_h), comp))
    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)
    return int(frame_size), int(sprite_size), sizes.get(frame_size, []), sizes.get(sprite_size, [])


def _solve_with_connectivity(grid: np.ndarray, hole_connectivity: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    frame_size, _sprite_size, frames, sprites_raw = _extract_objects(g)
    assert frames

    frame_positions = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            src_r = int(min_r + ri * frame_size)
            src_c = int(min_c + ci * frame_size)
            region = g[src_r : src_r + frame_size, src_c : src_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            inner = region[1:-1, 1:-1]
            signature = _hole_signature(inner != int(frame_color), connectivity=int(hole_connectivity))
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
        signature = _hole_signature(sprite_region != int(sprite_color), connectivity=int(hole_connectivity))
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered sprites to grid frames by 8-connected hole-layout signature, then nest them into the frame grid.

    Input structure:
    - bg = azure (8). Large square frames form a tight grid; smaller square sprites of another size are scattered elsewhere.
    - Each frame and sprite is monochromatic except for bg-colored interior hole pixels.
    - Logical holes are drawn as tiny diagonal pairs inside separated 2x2 anchor blocks, so one visible cluster may be
      split under 4-connectivity but merged under 8-connectivity.

    Rule:
    - Recover each frame slot and each sprite patch.
    - For every object, label its bg hole regions with 8-connectivity and compute the row/column-rank signature of the
      resulting hole-component centroids.
    - Pair each frame with the sprite sharing that same centroid-rank arrangement, regardless of the diagonal slash
      direction inside each tiny block.
    - Insert the matched sprite into the frame interior and recolor the sprite's hole pixels to the frame color.

    Key insight:
    - The discriminative feature is the relative arrangement of 8-connected hole components.
      Counting holes fails, and 4-connected hole extraction is too fine-grained because it treats each diagonal mini-block
      as two separate holes even though the intended logical hole is the merged 8-connected component.
    """
    return _solve_with_connectivity(grid, hole_connectivity=8)


def _block_origins(size: int) -> dict[str, tuple[int, int]]:
    far = int(size - 3)
    return {
        "tl": (1, 1),
        "tr": (1, far),
        "bl": (far, 1),
        "br": (far, far),
    }


def _paint_diag_block(patch: np.ndarray, top: int, left: int, orientation: str) -> None:
    if str(orientation) == "\\":
        patch[int(top), int(left)] = int(BG)
        patch[int(top + 1), int(left + 1)] = int(BG)
    else:
        patch[int(top), int(left + 1)] = int(BG)
        patch[int(top + 1), int(left)] = int(BG)


def _make_patch(size: int, color: int, blocks: tuple[str, ...], orientations: dict[str, str]) -> np.ndarray:
    patch = np.full((size, size), int(color), dtype=int)
    for block_name in blocks:
        rr, cc = _block_origins(size)[str(block_name)]
        _paint_diag_block(patch, int(rr), int(cc), str(orientations[str(block_name)]))
    return patch


def _choose_patterns(n: int) -> list[dict[str, object]]:
    chosen = random.sample(list(PATTERN_CATALOG), int(n))
    random.shuffle(chosen)
    return chosen


def _valid_frame_sizes(rows_grid: int, cols_grid: int) -> list[int]:
    valid: list[int] = []
    for frame_size in FRAME_SIZE_RANGE:
        sprite_size = int(frame_size - 2)
        min_h = max(CANVAS_MIN, rows_grid * frame_size + sprite_size + SPRITE_PADDING)
        min_w = max(CANVAS_MIN, cols_grid * frame_size + sprite_size + SPRITE_PADDING)
        if min_h <= CANVAS_MAX and min_w <= CANVAS_MAX:
            valid.append(int(frame_size))
    return valid


def _random_orientations(blocks: tuple[str, ...]) -> dict[str, str]:
    return {str(name): random.choice(["\\", "/"]) for name in blocks}


def _flip_one_orientation(orientations: dict[str, str], block_name: str) -> dict[str, str]:
    out = dict(orientations)
    out[str(block_name)] = "/" if str(out[str(block_name)]) == "\\" else "\\"
    return out


def _hole_signature_from_patch(patch: np.ndarray, color: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    return _hole_signature(patch != int(color), connectivity=int(connectivity))


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
    for idx, (name, _frame_patch) in enumerate(frames):
        rr = (idx // cols_grid) * frame_size
        cc = (idx % cols_grid) * frame_size
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)
        content = sprite_by_name[name].copy()
        content[content == BG] = int(frame_color)
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(VALID_NS)
    rows_grid, cols_grid = random.choice(LAYOUTS_BY_N[int(n)])
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
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    # Keep all objects visually distinct so the hole-layout clue is easy to read.
    chosen_colors = random.sample(available, 2 * n)
    frame_colors = chosen_colors[:n]
    sprite_colors = chosen_colors[n:]

    frames: list[tuple[str, np.ndarray]] = []
    sprites: list[tuple[str, np.ndarray]] = []
    changed_under_4 = 0

    for idx, pattern in enumerate(patterns):
        name = str(pattern["name"])
        blocks = tuple(str(b) for b in pattern["blocks"])

        frame_orients = _random_orientations(blocks)
        sprite_orients = _random_orientations(blocks)
        if sprite_orients == frame_orients:
            sprite_orients = _flip_one_orientation(sprite_orients, blocks[0])

        frame_patch = _make_patch(frame_size, int(frame_colors[idx]), blocks, frame_orients)
        sprite_patch = _make_patch(sprite_size, int(sprite_colors[idx]), blocks, sprite_orients)

        sig8_frame = _hole_signature_from_patch(frame_patch[1:-1, 1:-1], int(frame_colors[idx]), connectivity=8)
        sig8_sprite = _hole_signature_from_patch(sprite_patch, int(sprite_colors[idx]), connectivity=8)
        if sig8_frame != sig8_sprite:
            return None

        sig4_frame = _hole_signature_from_patch(frame_patch[1:-1, 1:-1], int(frame_colors[idx]), connectivity=4)
        sig4_sprite = _hole_signature_from_patch(sprite_patch, int(sprite_colors[idx]), connectivity=4)
        if sig4_frame != sig4_sprite:
            changed_under_4 += 1

        frames.append((name, frame_patch))
        sprites.append((name, sprite_patch))

    if changed_under_4 == 0:
        return None

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
        out8 = main(canvas)
        out4 = _solve_with_connectivity(canvas, hole_connectivity=4)
    except Exception:
        return None

    if out8.shape != expected.shape or not np.array_equal(out8, expected):
        return None
    if np.array_equal(out4, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
