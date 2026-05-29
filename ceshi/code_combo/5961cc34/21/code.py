# Combo variant of 5961cc34 (index 21)
# combo_spec: D11/2+D13/0
# Name: 对称排布+裁剪到活跃区域

from __future__ import annotations

# Design analysis for combo D11/2 + D13/0
# a) Concrete behavior trace:
#    The base task launches one upward ray from the bottom marker, activates the
#    first blue blob it touches, then performs BFS over the blobs' green-marked
#    ports and keeps the full canvas size. This combo changes two linked stages.
#    First, the generator no longer places the cascade as an arbitrary sparse
#    path: it uses a vertically mirrored layout around the grid's middle seam.
#    A central root blob now has two side ports, so one activation fans out into
#    mirrored left/right branches; those branches climb to a mirrored top pair
#    and connect inward, producing a visibly symmetric red motif. Second, after
#    the full red network is rendered, the output is cropped to the minimal box
#    containing red pixels. On a concrete example, the input can contain
#    root(center) -> pair(left/right) -> pair(top-left/top-right); the original
#    task would keep all outer margins, while this combo returns only the red
#    active rectangle spanning that symmetric cascade.
# b) Complexity calibration:
#    A purely cosmetic "mirror-looking" placement would be too weak, and a fully
#    free symmetric search would be noisy and fragile. The chosen template keeps
#    one stable source of symmetry in every sample: a multi-port center blob
#    creates mirrored BFS branches that would not exist in the original
#    one-port generator. Shape sizes and gaps still vary, so the pattern is not
#    a single memorized picture, but the family stays narrow enough to remain
#    legible from 2-4 ARC examples.
# c) Hidden assumptions audit:
#    The original solver logic already supports multiple ports per blob because
#    it collects every green-adjacent direction in `seen_dirs`; only the old
#    generator implicitly assumed "one edge per blob". This combo intentionally
#    uses that latent capability for the center blob, so BFS over multiple
#    outgoing ports becomes semantically important rather than incidental.
#    Another original assumption was "output shape equals input shape"; that is
#    now false, so the new solver must render on the full grid first and crop
#    only at the end. The old generator also leaned on the assert
#    `len(blobs) - len(activated) <= 1`; symmetric branching does not need
#    distractors, so the new solver drops that generator-specific constraint.
# d) Placement legality:
#    The generator computes every blob position from port geometry, then rejects
#    layouts where any footprint is out of bounds, 8-neighbor-adjacent to a
#    different blob, or blocks an unintended ray lane. Start-to-root, root-to-
#    side, side-to-top, and top-to-top bridge segments are all checked on the
#    actual input grid before acceptance.
# e) Visual observability:
#    Each accepted sample has two mandatory visible signals: there are mirrored
#    blob pairs around the center seam, and the produced output is strictly
#    narrower and top-cropped relative to the input. A human can therefore infer
#    both "use the symmetric cascade" and "crop to the red active area" from a
#    small set of examples.
# f) Visual cleanliness:
#    The symmetric branches are arranged in three horizontal bands
#    (root / side pair / top pair), so rays mostly occupy separate corridors.
#    No distractors are added, which avoids ambiguous overlaps between red rays
#    and untouched blue/green objects inside the crop box.
# g) Inferability of the change:
#    No hidden ordering by color id is used. The same structural cues as in the
#    original task still drive activation (blue bodies, green ports, red rays),
#    while symmetry and cropping are both geometric effects that are directly
#    visible in the pixel grids.

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

# --- Shapes ---
TURN_SHAPES = ((3, 4), (5, 6))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
}

# --- Generator bounds ---
H_MIN, H_MAX = 17, 27
W_MIN, W_MAX = 18, 28  # even widths only; symmetry axis is the middle seam
START_H_MIN, START_H_MAX = 2, 3
TOP_MARGIN_MIN = 1
SIDE_MARGIN_MIN = 1
H_GAP_MIN, H_GAP_MAX = 1, 3
V_GAP_MIN, V_GAP_MAX = 1, 4
ROOT_GAP_MIN, ROOT_GAP_MAX = 3, 5
GEN_RETRIES = 400


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _offset(coords: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in coords]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _mirror_left_col(W: int, c0: int, w: int) -> int:
    return int(W - c0 - w)


def _crop_to_red(grid: np.ndarray, red: int) -> np.ndarray:
    pts = np.argwhere(grid == int(red))
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = map(int, d)
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        guard = H + W + 10
        while (r, c) != (er, ec):
            r += dr
            c += dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((int(r), int(c)))
    return cells


def _paint_ray_to_hits(
    out: np.ndarray,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    hits: list[tuple[int, int]],
) -> None:
    cells = _path_to_entries(starts, d, hits, *out.shape)
    assert cells is not None
    paint_coords(out, cells, RED)


def _make_blob(size: tuple[int, int], port_dirs: tuple[tuple[int, int], ...]) -> dict:
    h, w = size
    mask = _shape_mask(h, w)
    greens: set[tuple[int, int]] = set()
    ports_local: list[dict] = []
    for d in port_dirs:
        edge = _edge_offsets(mask, d)
        assert edge
        for gr, gc in edge:
            greens.add((int(gr), int(gc)))
        port_pixels: list[tuple[int, int]] = []
        for gr, gc in edge:
            br, bc = gr - d[0], gc - d[1]
            assert 0 <= br < h and 0 <= bc < w and mask[br, bc]
            assert (int(br), int(bc)) not in greens
            port_pixels.append((int(br), int(bc)))
        ports_local.append({"dir": d, "pixels": _sort_on_ray(port_pixels, d)})

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in greens:
        patch[r, c] = GREEN

    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in greens]
    return {
        "size": size,
        "mask": mask,
        "patch": patch,
        "full": full,
        "blue": blue,
        "ports_local": ports_local,
    }


def _detect_start(
    g: np.ndarray, bg: int
) -> tuple[int, int, int, list[tuple[int, int]]]:
    H, _W = g.shape
    start_candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if c0 != c1 or (r1 - r0 + 1) < 2 or r1 != H - 1:
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            if r0 - 1 < 0:
                continue
            cap = int(g[r0 - 1, c0])
            if cap in (int(bg), int(col)):
                continue
            start_candidates.append(
                (int(col), int(cap), int(c0), [(int(rr), int(c0)) for rr in range(r0, r1 + 1)])
            )
    assert len(start_candidates) == 1
    return start_candidates[0]


def _detect_blob_colors(
    g: np.ndarray, bg: int, red_col: int, yellow_col: int
) -> tuple[int, int]:
    H, W = g.shape
    counts = Counter(map(int, g.flatten()))
    other_cols = [
        (int(cnt), int(col))
        for col, cnt in counts.items()
        if int(col) not in (int(bg), int(red_col), int(yellow_col))
    ]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        touches_blue = False
        for r, c in pts:
            rr, cc = int(r), int(c)
            for dr, dc in DIRS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(blue_col):
                    touches_blue = True
                    break
            if touches_blue:
                break
        if touches_blue:
            green_candidates.append(int(col))
    assert len(green_candidates) == 1
    return blue_col, int(green_candidates[0])


def _render_full_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    red_col, yellow_col, start_col, start_seg = _detect_start(g, bg)
    blue_col, green_col = _detect_blob_colors(g, bg, red_col, yellow_col)

    out = np.full_like(g, int(bg))
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in blob["pts"]:
            for ddr, ddc in DIRS:
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in blob["pts"]
                if 0 <= int(r + ddr) < H
                and 0 <= int(c + ddc) < W
                and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})

    activated: set[int] = set()
    queue: deque[int] = deque()

    def activate_blob_at(r: int, c: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert int(top_r) > 0 and int(g[int(top_r) - 1, int(start_col)]) == int(yellow_col)
    out[int(top_r) - 1, int(start_col)] = int(red_col)
    for r in range(int(top_r) - 1, -1, -1):
        out[int(r), int(start_col)] = int(red_col)
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col))
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
                    break

    while queue:
        bi = int(queue.popleft())
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a symmetric red cascade from blue blobs and crop the result to the active area.

    Input structure:
    - The background is the dominant color. Blue diamond-like blobs have adjacent green edge markers that define one or more emission ports.
    - A single launcher remains at the bottom: a short red stem touching the last row with one yellow cap directly above it.

    Rule:
    - Fire the thin start ray upward, activate any blue blob it first reaches, and then run BFS over every green-indicated port of each activated blob.
    - Each port emits a straight red ray of its own thickness until the ray exits the grid or reaches another blue blob, which then activates in turn.
    - After the full red network is painted, return only the smallest rectangle that contains all red pixels.

    Key insight:
    - Green cells still encode the outgoing directions, but the fused variant makes symmetric multi-branch layouts common, so one activated center blob can legitimately spawn mirrored sub-cascades. The final shape is recovered by simulating the whole propagation first and cropping only after the red support is known.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_col, _yellow_col, _start_col, _start_seg = _detect_start(g, bg)
    full = _render_full_output(g)
    return _crop_to_red(full, red_col)


def _fits_with_padding(placed: list[dict], candidate: dict, H: int, W: int) -> bool:
    cand_full = _offset(candidate["blob"]["full"], candidate["r0"], candidate["c0"])
    if not _in_bounds(cand_full, H, W):
        return False
    occ = np.zeros((H, W), dtype=bool)
    for item in placed:
        paint_coords(occ, _offset(item["blob"]["full"], item["r0"], item["c0"]), True)
    occ = _dilate8(occ)
    return not any(bool(occ[r, c]) for r, c in cand_full)


def _place_blob(inp: np.ndarray, blob: dict, r0: int, c0: int) -> None:
    paste(inp, blob["patch"], int(r0), int(c0), transparent=BG)


def _global_port_pixels(blob: dict, r0: int, c0: int, d: tuple[int, int]) -> list[tuple[int, int]]:
    for port in blob["ports_local"]:
        if port["dir"] == d:
            return _sort_on_ray(_offset(port["pixels"], r0, c0), d)
    raise ValueError("missing port")


def _manual_expected(
    H: int,
    W: int,
    cap_r: int,
    start_col: int,
    root: dict,
    root_pos: tuple[int, int],
    left_a: dict,
    left_a_pos: tuple[int, int],
    right_a: dict,
    right_a_pos: tuple[int, int],
    left_b: dict,
    left_b_pos: tuple[int, int],
    right_b: dict,
    right_b_pos: tuple[int, int],
    hits: dict,
) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)

    for r in range(cap_r + 1, H):
        out[r, start_col] = RED
    out[cap_r, start_col] = RED

    all_blue_sets = [
        _offset(root["blue"], *root_pos),
        _offset(left_a["blue"], *left_a_pos),
        _offset(right_a["blue"], *right_a_pos),
        _offset(left_b["blue"], *left_b_pos),
        _offset(right_b["blue"], *right_b_pos),
    ]
    for blue_pts in all_blue_sets:
        paint_coords(out, blue_pts, RED)

    _paint_ray_to_hits(
        out,
        [(int(cap_r), int(start_col))],
        UP,
        [hits["root_start"]],
    )
    _paint_ray_to_hits(
        out,
        _global_port_pixels(root, *root_pos, LEFT),
        LEFT,
        [hits["left_a"]],
    )
    _paint_ray_to_hits(
        out,
        _global_port_pixels(root, *root_pos, RIGHT),
        RIGHT,
        [hits["right_a"]],
    )
    _paint_ray_to_hits(
        out,
        _global_port_pixels(left_a, *left_a_pos, UP),
        UP,
        hits["left_b"],
    )
    _paint_ray_to_hits(
        out,
        _global_port_pixels(right_a, *right_a_pos, UP),
        UP,
        hits["right_b"],
    )
    _paint_ray_to_hits(
        out,
        _global_port_pixels(left_b, *left_b_pos, RIGHT),
        RIGHT,
        [hits["right_b_bridge"]],
    )
    _paint_ray_to_hits(
        out,
        _global_port_pixels(right_b, *right_b_pos, LEFT),
        LEFT,
        [hits["left_b_bridge"]],
    )
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randrange(W_MIN + (W_MIN % 2), W_MAX + 1, 2)

    root = _make_blob(random.choice(TURN_SHAPES), (LEFT, RIGHT))
    side = _make_blob(random.choice(TURN_SHAPES), (UP,))
    top_left = _make_blob(random.choice(TURN_SHAPES), (RIGHT,))
    top_right = _make_blob(top_left["size"], (LEFT,))

    _root_h, root_w = root["size"]
    _side_h, side_w = side["size"]
    _top_h, top_w = top_left["size"]

    root_bottom = _edge_offsets(root["mask"], DOWN)
    root_start_col_local = min(c for _r, c in root_bottom)
    root_bottom_row = root_bottom[0][0]
    root_left_green_col = min(c for _r, c in _edge_offsets(root["mask"], LEFT))
    root_left_port_row = _global_port_pixels(root, 0, 0, LEFT)[0][0]

    side_right_entry = _edge_offsets(side["mask"], RIGHT)
    side_left_entry = _edge_offsets(side["mask"], LEFT)
    assert len(side_right_entry) == 1 and len(side_left_entry) == 1
    side_right_row, side_right_col = map(int, side_right_entry[0])
    side_left_row, side_left_col = map(int, side_left_entry[0])
    side_up_pixels_local = _global_port_pixels(side, 0, 0, UP)
    side_up_row = side_up_pixels_local[0][0]
    side_up_cols = [int(c) for _r, c in side_up_pixels_local]

    top_bottom = _edge_offsets(top_left["mask"], DOWN)
    top_bottom_row = top_bottom[0][0]
    top_bottom_cols = [int(c) for _r, c in top_bottom]
    top_right_entry = _edge_offsets(top_left["mask"], RIGHT)
    top_left_entry = _edge_offsets(top_right["mask"], LEFT)
    assert len(top_right_entry) == 1 and len(top_left_entry) == 1
    top_right_hit_row, top_right_hit_col = map(int, top_right_entry[0])
    top_left_hit_row, top_left_hit_col = map(int, top_left_entry[0])

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    root_gap = random.randint(ROOT_GAP_MIN, ROOT_GAP_MAX)

    root_c0 = W // 2 - root_w // 2
    root_r0 = cap_r - root_gap - root_bottom_row
    start_col = root_c0 + root_start_col_local

    if root_r0 < 0:
        return None

    hgap = random.randint(H_GAP_MIN, H_GAP_MAX)
    left_a_entry_col_global = root_c0 + root_left_green_col - 1 - hgap
    left_a_c0 = left_a_entry_col_global - side_right_col
    right_a_c0 = _mirror_left_col(W, left_a_c0, side_w)
    left_a_r0 = root_r0 + root_left_port_row - side_right_row
    right_a_r0 = left_a_r0

    vgap = random.randint(V_GAP_MIN, V_GAP_MAX)
    left_b_c0 = left_a_c0 + side_up_cols[0] - top_bottom_cols[0]
    right_b_c0 = _mirror_left_col(W, left_b_c0, top_w)
    left_b_r0 = left_a_r0 + side_up_row - 1 - vgap - top_bottom_row
    right_b_r0 = left_b_r0

    placements = [
        {"blob": root, "r0": root_r0, "c0": root_c0},
        {"blob": side, "r0": left_a_r0, "c0": left_a_c0},
        {"blob": side, "r0": right_a_r0, "c0": right_a_c0},
        {"blob": top_left, "r0": left_b_r0, "c0": left_b_c0},
        {"blob": top_right, "r0": right_b_r0, "c0": right_b_c0},
    ]

    if left_b_r0 < TOP_MARGIN_MIN:
        return None
    if min(left_a_c0, left_b_c0) < SIDE_MARGIN_MIN:
        return None
    if max(right_a_c0 + side_w, right_b_c0 + top_w) > W - SIDE_MARGIN_MIN:
        return None

    accepted: list[dict] = []
    for item in placements:
        if not _fits_with_padding(accepted, item, H, W):
            return None
        accepted.append(item)

    inp = np.full((H, W), BG, dtype=int)
    inp[cap_r, start_col] = YELLOW
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
    for item in placements:
        _place_blob(inp, item["blob"], item["r0"], item["c0"])

    hits = {
        "root_start": (int(root_r0 + root_bottom_row), int(start_col)),
        "left_a": (int(left_a_r0 + side_right_row), int(left_a_c0 + side_right_col)),
        "right_a": (int(right_a_r0 + side_left_row), int(right_a_c0 + side_left_col)),
        "left_b": [
            (int(left_b_r0 + top_bottom_row), int(left_b_c0 + c))
            for c in top_bottom_cols
        ],
        "right_b": [
            (int(right_b_r0 + top_bottom_row), int(right_b_c0 + c))
            for c in top_bottom_cols
        ],
        "right_b_bridge": (
            int(right_b_r0 + top_left_hit_row),
            int(right_b_c0 + top_left_hit_col),
        ),
        "left_b_bridge": (
            int(left_b_r0 + top_right_hit_row),
            int(left_b_c0 + top_right_hit_col),
        ),
    }

    expected_full = _manual_expected(
        H,
        W,
        cap_r,
        start_col,
        root,
        (root_r0, root_c0),
        side,
        (left_a_r0, left_a_c0),
        side,
        (right_a_r0, right_a_c0),
        top_left,
        (left_b_r0, left_b_c0),
        top_right,
        (right_b_r0, right_b_c0),
        hits,
    )
    expected = _crop_to_red(expected_full, RED)

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    if solved.shape[1] >= W or solved.shape[0] >= H:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
