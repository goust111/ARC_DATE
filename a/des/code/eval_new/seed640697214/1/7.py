import random
from collections import Counter

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
INWARD = {
    "TL": (1, 1),
    "TR": (1, -1),
    "BL": (-1, 1),
    "BR": (-1, -1),
}
OUTWARD = {corner: (-vr, -hc) for corner, (vr, hc) in INWARD.items()}
CORNER_OFFSETS = {
    "TL": (0, 0),
    "TR": (0, -1),
    "BL": (-1, 0),
    "BR": (-1, -1),
}
_GEN_COUNTER = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _classify_l(coords: np.ndarray) -> dict | None:
    if coords.shape != (3, 2):
        return None
    pts = [tuple(map(int, pt)) for pt in coords]
    ptset = set(pts)
    elbow = None
    for r, c in pts:
        deg = sum((nr, nc) in ptset for nr, nc in _neighbors4(r, c))
        if deg == 2:
            elbow = (r, c)
            break
    if elbow is None:
        return None
    arms = []
    er, ec = elbow
    for r, c in pts:
        if (r, c) == elbow:
            continue
        dr, dc = r - er, c - ec
        if abs(dr) + abs(dc) != 1:
            return None
        arms.append((dr, dc))
    if len(arms) != 2:
        return None
    vertical = [dr for dr, dc in arms if dc == 0]
    horizontal = [dc for dr, dc in arms if dr == 0]
    if len(vertical) != 1 or len(horizontal) != 1:
        return None
    return {
        "coords": pts,
        "elbow": elbow,
        "orient": (int(vertical[0]), int(horizontal[0])),
    }


def _target_rect(singletons: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if len(singletons) != 4:
        return None
    rows = sorted({r for r, _ in singletons})
    cols = sorted({c for _, c in singletons})
    if len(rows) != 2 or len(cols) != 2:
        return None
    expect = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    if set(singletons) != expect:
        return None
    return rows[0], cols[0], rows[1], cols[1]


def _diag_roles(elbows: list[tuple[int, int]]) -> dict[tuple[int, int], str] | None:
    rows = sorted({r for r, _ in elbows})
    cols = sorted({c for _, c in elbows})
    if len(rows) != 2 or len(cols) != 2:
        return None
    ptset = set(elbows)
    tlbr = {(rows[0], cols[0]), (rows[1], cols[1])}
    trbl = {(rows[0], cols[1]), (rows[1], cols[0])}
    if ptset == tlbr:
        return {(rows[0], cols[0]): "TL", (rows[1], cols[1]): "BR"}
    if ptset == trbl:
        return {(rows[0], cols[1]): "TR", (rows[1], cols[0]): "BL"}
    return None


def _decode_transform(l_infos: list[dict], roles: dict[tuple[int, int], str]) -> str:
    by_role = {roles[info["elbow"]]: info for info in l_infos}
    pair = set(by_role)
    if all(by_role[corner]["orient"] == INWARD[corner] for corner in pair):
        return "id"
    if all(by_role[corner]["orient"] == OUTWARD[corner] for corner in pair):
        return "rot180"

    top_corner = "TL" if "TL" in pair else "TR"
    bottom_corner = "BL" if "BL" in pair else "BR"
    top = by_role[top_corner]["orient"]
    bottom = by_role[bottom_corner]["orient"]
    if top[0] == 1 and bottom[0] == -1 and top[1] == bottom[1]:
        return "hflip"

    left_corner = "TL" if "TL" in pair else "BL"
    right_corner = "TR" if "TR" in pair else "BR"
    left = by_role[left_corner]["orient"]
    right = by_role[right_corner]["orient"]
    if left[1] == 1 and right[1] == -1 and left[0] == right[0]:
        return "vflip"

    raise ValueError("unrecognized L orientation pairing")


def _apply_transform(patch: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return patch.copy()
    if name == "hflip":
        return np.fliplr(patch)
    if name == "vflip":
        return np.flipud(patch)
    if name == "rot180":
        return np.rot90(patch, 2)
    raise ValueError(name)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each L-bracketed source pattern into the matching corner-marked target, using the L orientations to choose the geometric transform.

    Input structure:
    - The background is black.
    - Each active marker color appears as exactly two 3-pixel L-shaped corner markers and four singleton corner markers of the same color.
    - The two L markers sit on opposite corners of a source rectangle that contains a small pattern drawn in other colors.
    - The four singleton markers sit on the corners of an empty target rectangle with the same size as the source rectangle.

    Rule:
    - For each marker color, identify its two L components and its four singleton target corners.
    - Use the two L elbows as the diagonal corners of the source rectangle, then erase the source-color L pixels from that rectangle to isolate the pattern.
    - Decode the transform from the L arm directions: inward-facing corners mean copy as-is; both Ls facing outward mean rotate 180 degrees; matching left/right openings mean horizontal flip; matching up/down openings mean vertical flip.
    - Paste the transformed pattern into the target rectangle while keeping the target corner markers unchanged.

    Key insight:
    - The marker color pairs source and target, while the two L orientations encode which symmetry of the source rectangle should be applied before copying.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        comps = _components(g == color)
        l_infos = []
        singletons = []
        for comp in comps:
            if comp.shape[0] == 1:
                singletons.append(tuple(map(int, comp[0])))
                continue
            info = _classify_l(comp)
            if info is not None:
                l_infos.append(info)
        if len(l_infos) != 2 or len(singletons) != 4:
            continue

        target = _target_rect(singletons)
        if target is None:
            continue
        elbows = [info["elbow"] for info in l_infos]
        roles = _diag_roles(elbows)
        if roles is None:
            continue

        sr0 = min(r for r, _ in elbows)
        sc0 = min(c for _, c in elbows)
        sr1 = max(r for r, _ in elbows)
        sc1 = max(c for _, c in elbows)
        tr0, tc0, tr1, tc1 = target
        if (sr1 - sr0, sc1 - sc0) != (tr1 - tr0, tc1 - tc0):
            continue

        patch = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        for info in l_infos:
            for r, c in info["coords"]:
                if sr0 <= r <= sr1 and sc0 <= c <= sc1:
                    patch[r - sr0, c - sc0] = bg

        transform = _decode_transform(l_infos, roles)
        stamped = _apply_transform(patch, transform)
        block = out[tr0 : tr1 + 1, tc0 : tc1 + 1]
        mask = stamped != bg
        block[mask] = stamped[mask]
        for r, c in singletons:
            out[r, c] = color

    return out


def _shape_variants(mask: np.ndarray) -> set[bytes]:
    return {
        mask.tobytes(),
        np.fliplr(mask).tobytes(),
        np.flipud(mask).tobytes(),
        np.rot90(mask, 2).tobytes(),
    }


def _random_mask(inner_h: int, inner_w: int, rng: random.Random) -> np.ndarray | None:
    area = inner_h * inner_w
    target = rng.randint(3, min(area - 1, max(4, area - 2)))
    for _ in range(120):
        mask = np.zeros((inner_h, inner_w), dtype=bool)
        start = (rng.randrange(inner_h), rng.randrange(inner_w))
        cells = {start}
        frontier = {start}
        while len(cells) < target and frontier:
            cr, cc = rng.choice(tuple(frontier))
            nbrs = []
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove((cr, cc))
                continue
            nxt = rng.choice(nbrs)
            cells.add(nxt)
            frontier.add(nxt)
        if len(cells) != target:
            continue
        for r, c in cells:
            mask[r, c] = True
        if len(_shape_variants(mask)) != 4:
            continue
        return mask
    return None


def _pattern_patch(h: int, w: int, color: int, rng: random.Random) -> np.ndarray | None:
    inner_h = h - 2
    inner_w = w - 2
    mask = _random_mask(inner_h, inner_w, rng)
    if mask is None:
        return None
    patch = np.zeros((h, w), dtype=int)
    patch[1 : h - 1, 1 : w - 1][mask] = color
    return patch


def _sample_l_pair(transform: str, rng: random.Random) -> list[tuple[str, tuple[int, int]]]:
    diag = rng.choice(("TLBR", "TRBL"))
    if diag == "TLBR":
        if transform == "id":
            return [("TL", INWARD["TL"]), ("BR", INWARD["BR"])]
        if transform == "rot180":
            return [("TL", OUTWARD["TL"]), ("BR", OUTWARD["BR"])]
        if transform == "hflip":
            hsign = rng.choice((-1, 1))
            return [("TL", (1, hsign)), ("BR", (-1, hsign))]
        vsign = rng.choice((-1, 1))
        return [("TL", (vsign, 1)), ("BR", (vsign, -1))]

    if transform == "id":
        return [("TR", INWARD["TR"]), ("BL", INWARD["BL"])]
    if transform == "rot180":
        return [("TR", OUTWARD["TR"]), ("BL", OUTWARD["BL"])]
    if transform == "hflip":
        hsign = rng.choice((-1, 1))
        return [("TR", (1, hsign)), ("BL", (-1, hsign))]
    vsign = rng.choice((-1, 1))
    return [("TR", (vsign, -1)), ("BL", (vsign, 1))]


def _corner_position(role: str, h: int, w: int) -> tuple[int, int]:
    if role == "TL":
        return 0, 0
    if role == "TR":
        return 0, w - 1
    if role == "BL":
        return h - 1, 0
    return h - 1, w - 1


def _l_cells(role: str, orient: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    er, ec = _corner_position(role, h, w)
    vr, hc = orient
    return [(er, ec), (er + vr, ec), (er, ec + hc)]


def _source_extents(spec: dict) -> tuple[int, int, int, int]:
    cells = []
    h = spec["h"]
    w = spec["w"]
    for role, orient in spec["l_pair"]:
        cells.extend(_l_cells(role, orient, h, w))
    rows = [r for r, _ in cells] + [0, h - 1]
    cols = [c for _, c in cells] + [0, w - 1]
    return min(rows), max(rows), min(cols), max(cols)


def _reserve_ok(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = occ.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return not occ[r0 : r1 + 1, c0 : c1 + 1].any()


def _reserve(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int):
    occ[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_rect(
    occ: np.ndarray,
    top_range: tuple[int, int],
    left_range: tuple[int, int],
    footprint: tuple[int, int, int, int],
    rng: random.Random,
) -> tuple[int, int] | None:
    min_r, max_r, min_c, max_c = footprint
    lo_r, hi_r = top_range
    lo_c, hi_c = left_range
    if lo_r > hi_r or lo_c > hi_c:
        return None
    for _ in range(160):
        top = rng.randint(lo_r, hi_r)
        left = rng.randint(lo_c, hi_c)
        rr0 = top + min_r - 1
        rr1 = top + max_r + 1
        cc0 = left + min_c - 1
        cc1 = left + max_c + 1
        if _reserve_ok(occ, rr0, cc0, rr1, cc1):
            _reserve(occ, rr0, cc0, rr1, cc1)
            return top, left
    return None


def _draw_object(grid: np.ndarray, spec: dict):
    h = spec["h"]
    w = spec["w"]
    sr, sc = spec["source_pos"]
    tr, tc = spec["target_pos"]

    patch = spec["patch"]
    mask = patch != BG
    grid[sr : sr + h, sc : sc + w][mask] = patch[mask]

    for role, orient in spec["l_pair"]:
        for rr, cc in _l_cells(role, orient, h, w):
            grid[sr + rr, sc + cc] = spec["marker_color"]

    for rr in (0, h - 1):
        for cc in (0, w - 1):
            grid[tr + rr, tc + cc] = spec["marker_color"]


def _make_spec(transform: str, marker_color: int, pattern_color: int, rng: random.Random) -> dict | None:
    h = rng.randint(5, 8)
    w = rng.randint(5, 8)
    patch = _pattern_patch(h, w, pattern_color, rng)
    if patch is None:
        return None
    l_pair = _sample_l_pair(transform, rng)
    return {
        "transform": transform,
        "marker_color": marker_color,
        "pattern_color": pattern_color,
        "h": h,
        "w": w,
        "patch": patch,
        "l_pair": l_pair,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    global _GEN_COUNTER

    count = rng.randint(2, 4)
    height = rng.randint(max(18, count * 6), 30)
    width = rng.randint(max(18, count * 6), 30)
    if 2 * count > len(COLORS):
        return None

    color_sample = rng.sample(COLORS, 2 * count)
    marker_colors = color_sample[:count]
    pattern_colors = color_sample[count:]

    class_cycle = ["id", "hflip", "vflip", "rot180"]
    transforms = [class_cycle[(_GEN_COUNTER + i) % 4] for i in range(count)]
    rng.shuffle(transforms)
    _GEN_COUNTER += 1

    specs = []
    for idx in range(count):
        spec = _make_spec(transforms[idx], marker_colors[idx], pattern_colors[idx], rng)
        if spec is None:
            return None
        specs.append(spec)

    occ = np.zeros((height, width), dtype=bool)
    for spec in sorted(specs, key=lambda item: -(item["h"] * item["w"])):
        source_fp = _source_extents(spec)
        target_fp = (0, spec["h"] - 1, 0, spec["w"] - 1)

        src = _place_rect(
            occ,
            (-source_fp[0], height - 1 - source_fp[1]),
            (-source_fp[2], width - 1 - source_fp[3]),
            source_fp,
            rng,
        )
        if src is None:
            return None
        spec["source_pos"] = src

        tgt = _place_rect(
            occ,
            (0, height - spec["h"]),
            (0, width - spec["w"]),
            target_fp,
            rng,
        )
        if tgt is None:
            return None
        spec["target_pos"] = tgt

    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        _draw_object(grid, spec)

    solved = main(grid)
    if solved.shape != grid.shape or np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
