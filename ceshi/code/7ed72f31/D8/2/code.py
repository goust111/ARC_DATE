# Variant of 7ed72f31
# Dimension: D8 — 标记几何形态的判别逻辑 (marker_axis)
# Change: 红色像素的几何形态决定反射类型：单个像素=点反射中心，同行排列=水平轴，同列排列=垂直轴... → 十字形标记定义四折对称

from __future__ import annotations

"""
design_analysis:
  concretization:
    embedded_cross:
      marker_shape: exact 5-pixel plus sign centered at one cell
      input_layout: |
        A monochrome 4-connected payload sits in exactly one quadrant around the plus.
        The payload never uses the center row/center col, but it must include the corner
        cell nearest the center so that the payload and plus remain one 4-connected component.
      output_layout: |
        The payload is copied into the other three quadrants: mirror across the horizontal
        axis, mirror across the vertical axis, and point reflection around the same center.
        Together with the original quadrant this forms a fourfold symmetric sprite.
    isolated_marker:
      unchanged: a single red pixel stays 4-disconnected and point-reflects each adjacent
      8-connected monochrome component exactly as in the source task.
  example_trace:
    old_logic: |
      Mixed component with a horizontal red line only produced one opposite-side mirror.
    new_logic: |
      Suppose the red pixels are {(3,3),(2,3),(4,3),(3,2),(3,4)} and the payload cells are
      {(2,2),(1,2)} in the top-left quadrant. The solver now paints:
      - horizontal reflection: {(4,2),(5,2)}
      - vertical reflection: {(2,4),(1,4)}
      - point reflection: {(4,4),(5,4)}
      The original cells remain, yielding four quadrants around the plus.
    delta: |
      The control variable is no longer "row vs col vs point"; it is "plus-shaped marker",
      which simultaneously induces both axis reflections plus the 180-degree copy.
  complexity_calibration:
    - The embedded case is visually stronger than the source task because one input payload
      expands into three new copies instead of one.
    - To keep the rule learnable, the cross is always the same exact 5-pixel shape; arm
      length is not another latent degree of freedom.
    - Every generated grid contains at least one embedded cross, and larger grids often have
      two or three. This prevents degeneration into isolated-point-only samples.
    - Quadrant payload masks are rejected if they are symmetric under vertical, horizontal,
      or 180-degree flip inside their own bbox; this makes the three new copies observable.
  implicit_assumptions_review:
    main:
      - Non-background components are still labeled with 4-connectivity.
      - For a plus marker to share a component with the payload, the payload must include the
        corner cell nearest the center. The generator enforces this.
      - The solver assumes embedded components use exactly one plus marker and contain only
        the original quadrant payload in the input; otherwise reflecting every non-red cell
        would duplicate already mirrored data. The generator enforces one-quadrant input.
      - The solver ignores ambiguous red geometries. This is safe because the generator only
        emits exact plus signs or isolated single red pixels.
      - Reflections must land on in-bounds background cells. The generator constructs full
        output sprites first, places them without overlap, then erases only the reflected
        quadrants to derive the input.
    generator:
      - Embedded sprites use a local patch of size (2h+1) x (2w+1); each quadrant is disjoint,
        so the three generated copies never overlap the original or the red cross.
      - Placement uses the full output footprint, so later solver writes land on cells that
        were background in the input and remain inside the global grid.
  placement_legality:
    - Sprites are built output-first, then placed with 8-neighbor padding to avoid mergers.
    - The input is obtained by erasing only the reflected quadrants of embedded sprites.
    - Isolated markers remain 4-disconnected from all colored cells, so they do not get
      mistaken for embedded crosses.
  visual_observability:
    - A human can see the exact same cross shape paired with one occupied quadrant in input
      and all four quadrants in output across multiple examples.
    - Because the payload masks are asymmetric and copied, the new D8 behavior is visible
      even from two or three training pairs.
  cleanliness_and_color:
    - Each sprite is monochrome, matching the source task.
    - Only 3-4 non-background colors are used per grid, so color remains a grouping cue
      rather than a hidden code.
    - Marker red is reserved for the cross or isolated point only.
"""

import pathlib
import random
import sys

import numpy as np
from scipy import ndimage

ROOT = pathlib.Path(__file__).resolve().parents[5]
SOLVERS_DIR = ROOT / "solvers"
if str(SOLVERS_DIR) not in sys.path:
    sys.path.append(str(SOLVERS_DIR))

from arc_utils import STRUCT8, get_bg, label_mask


MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 18, 28

# Marker counts. Fourfold embedded sprites are larger than the source-task half-mirrors,
# so the embedded count is capped slightly lower to keep placement and bg coverage healthy.
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 5
MAX_EMBEDDED, MAX_ISOLATED = 3, 3

# Embedded quadrant payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 4
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 7
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.45, 0.75

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 8
ISOLATED_BBOX_MAX = 4

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 80
GEN_RETRIES = 400

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_QUADRANTS = ("tl", "tr", "bl", "br")


def _plus_center(reds: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(reds) != 5:
        return None
    s = {(int(r), int(c)) for r, c in reds}
    for r, c in s:
        needed = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        if s == needed:
            return int(r), int(c)
    return None


def _random_connected_mask(
    h: int,
    w: int,
    target_pixels: int,
    *,
    rng: random.Random,
    must_include: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Grow a random 4-connected mask with an optional forced seed cell."""
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    if must_include is None:
        start = (rng.randrange(h), rng.randrange(w))
    else:
        start = tuple(map(int, must_include))
        if not (0 <= start[0] < h and 0 <= start[1] < w):
            return None
    mask[start] = True
    count = 1
    while count < target_pixels:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        r, c = rng.choice(frontier)
        mask[r, c] = True
        count += 1
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        return None
    return mask


def _is_asymmetric(mask: np.ndarray) -> bool:
    return not (
        np.array_equal(mask, mask[::-1])
        or np.array_equal(mask, mask[:, ::-1])
        or np.array_equal(mask, mask[::-1, ::-1])
    )


def _refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, p)
    return int(2 * pr - r), int(2 * pc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    D8 variant of 7ed72f31.

    - Embedded red plus signs define both a horizontal and vertical reflection axis.
      The payload in one quadrant is copied into the other three quadrants.
    - Isolated single red pixels still point-reflect adjacent monochrome components.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    # --- 1) Embedded plus-sign markers create three mirrored copies ---
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        center = _plus_center(reds)
        if center is None:
            continue

        for (r, c), v in zip(coords, vals):
            v = int(v)
            if v == int(MARK):
                continue
            for rr, cc in (
                _refl_row(center[0], int(r), int(c)),
                _refl_col(center[1], int(r), int(c)),
                _refl_point(center, int(r), int(c)),
            ):
                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    # --- 2) Isolated red pixels keep the original point-reflection behavior ---
    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

    return out


def _quadrant_corner(h: int, w: int, quadrant: str) -> tuple[int, int]:
    return {
        "tl": (h - 1, w - 1),
        "tr": (h - 1, 0),
        "bl": (0, w - 1),
        "br": (0, 0),
    }[quadrant]


def _place_quadrant_cells(mask: np.ndarray, quadrant: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if quadrant == "tl":
            rr, cc = r, c
        elif quadrant == "tr":
            rr, cc = r, w + 1 + c
        elif quadrant == "bl":
            rr, cc = h + 1 + r, c
        else:
            rr, cc = h + 1 + r, w + 1 + c
        cells.append((int(rr), int(cc)))
    return cells


def _build_embedded_sprite(rng: random.Random) -> dict | None:
    """Build an output sprite whose embedded marker is a 5-pixel red plus sign."""
    for _ in range(MASK_RETRIES):
        ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
        pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
        max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
        min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
        if min_px > max_px:
            continue

        quadrant = rng.choice(_QUADRANTS)
        pmask = _random_connected_mask(
            ph,
            pw,
            rng.randint(min_px, max_px),
            rng=rng,
            must_include=_quadrant_corner(ph, pw, quadrant),
        )
        if pmask is None or not _is_asymmetric(pmask):
            continue

        H, W = 2 * ph + 1, 2 * pw + 1
        center = (ph, pw)
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)

        payload_cells = _place_quadrant_cells(pmask, quadrant)
        for r, c in payload_cells:
            payload[int(r), int(c)] = True

        for r, c in payload_cells:
            for rr, cc in (
                _refl_row(center[0], int(r), int(c)),
                _refl_col(center[1], int(r), int(c)),
                _refl_point(center, int(r), int(c)),
            ):
                mirror[int(rr), int(cc)] = True

        red_pixels = [
            center,
            (center[0] - 1, center[1]),
            (center[0] + 1, center[1]),
            (center[0], center[1] - 1),
            (center[0], center[1] + 1),
        ]

        if (payload & mirror).any():
            continue
        bad = False
        for rr, cc in red_pixels:
            if payload[int(rr), int(cc)] or mirror[int(rr), int(cc)]:
                bad = True
                break
        if bad:
            continue

        return {
            "payload": payload,
            "mirror": mirror,
            "red_pixels": red_pixels,
            "kind": "embedded_cross",
        }
    return None


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    H, W = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
            return True
    return False


def _build_isolated_sprite(rng: random.Random) -> dict | None:
    """Unchanged source-task isolated marker builder."""
    n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
    pmask = None
    for _ in range(MASK_RETRIES):
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        pmask = (
            np.ones((ph, pw), dtype=bool)
            if n_px == ph * pw
            else _random_connected_mask(ph, pw, n_px, rng=rng)
        )
        if pmask is not None:
            break
    if pmask is None:
        return None

    pcoords = list(zip(*np.where(pmask)))
    rng.shuffle(pcoords)

    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = pr + dr, pc + dc
            refl = [(2 * red_r - r, 2 * red_c - c) for r, c in pcoords]
            all_pts = pcoords + refl + [(red_r, red_c)]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            H = max(all_rs) - minr + 1
            W = max(all_cs) - minc + 1

            payload = np.zeros((H, W), dtype=bool)
            mirror = np.zeros((H, W), dtype=bool)
            for r, c in pcoords:
                payload[r - minr, c - minc] = True
            for r, c in refl:
                mirror[r - minr, c - minc] = True
            red_pos = (red_r - minr, red_c - minc)

            if payload[red_pos] or mirror[red_pos]:
                continue
            if (payload & mirror).any():
                continue
            if _is_4connected_to(payload | mirror, *red_pos):
                continue
            return {
                "payload": payload,
                "mirror": mirror,
                "red_pixels": [red_pos],
                "kind": "isolated_point",
            }
    return None


def _random_free_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    bg: int,
    rng: random.Random,
    *,
    padding: int = PLACEMENT_PADDING,
):
    H, W = canvas.shape
    h, w = patch_mask.shape
    if h > H or w > W:
        return None
    occ = canvas != bg
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not (occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["payload"].shape
    colored = spr["payload"] | spr["mirror"]
    canvas[r0 : r0 + h, c0 : c0 + w][colored] = spr["color"]
    for rr, cc in spr["red_pixels"]:
        canvas[r0 + rr, c0 + cc] = MARK


def _erase_mirror(canvas: np.ndarray, spr: dict, bg: int) -> None:
    r0, c0 = spr["r0"], spr["c0"]
    h, w = spr["mirror"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["mirror"]] = bg


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c != MARK and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(
        max(1, total_markers - MAX_ISOLATED),
        min(MAX_EMBEDDED, total_markers),
    )
    n_isolated = total_markers - n_embedded

    n_colors = rng.randint(
        PALETTE_COLORS_MIN,
        min(PALETTE_COLORS_MAX, len(palette), total_markers),
    )
    chosen_colors = rng.sample(palette, n_colors)

    sprites: list[dict] = []
    for builder, count in ((_build_embedded_sprite, n_embedded), (_build_isolated_sprite, n_isolated)):
        for _ in range(count):
            s = builder(rng)
            if s is None:
                return None
            sprites.append(s)

    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, c in zip(sprites, assignment):
        spr["color"] = c

    output = np.full((N, N), bg, dtype=int)
    rng.shuffle(sprites)
    for spr in sprites:
        footprint = spr["payload"] | spr["mirror"]
        for r, c in spr["red_pixels"]:
            footprint[r, c] = True
        loc = _random_free_location(output, footprint, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr["r0"], spr["c0"] = r0, c0
        _stamp_sprite(output, spr, r0, c0)

    inp = output.copy()
    for spr in sprites:
        _erase_mirror(inp, spr, bg)

    if (inp == bg).sum() / (N * N) < BG_COVERAGE_MIN:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
