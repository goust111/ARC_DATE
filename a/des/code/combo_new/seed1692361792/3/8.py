import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNER_DIRS = {
    "tl": (1, 1),
    "tr": (1, -1),
    "bl": (-1, 1),
    "br": (-1, -1),
}

BG = 0
PALETTE_LEN = 5
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _count_holes(coords: np.ndarray) -> int:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True

    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _find_corner_direction(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    corners = {
        (0, 0): CORNER_DIRS["tl"],
        (0, w - 1): CORNER_DIRS["tr"],
        (h - 1, 0): CORNER_DIRS["bl"],
        (h - 1, w - 1): CORNER_DIRS["br"],
    }
    active = [(rc, d) for rc, d in corners.items() if int(grid[rc]) != bg]
    assert len(active) == 1, "expected exactly one non-background corner marker"
    return active[0]


def _find_palette(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int], list[int], set[tuple[int, int]]]] = []
    for row in (0, h - 1):
        arr = grid[row]
        c = 0
        while c < w:
            if int(arr[c]) == bg:
                c += 1
                continue
            start = c
            while c < w and int(arr[c]) != bg:
                c += 1
            segment = list(map(int, arr[start:c]))
            if len(segment) == PALETTE_LEN and len(set(segment)) == PALETTE_LEN:
                cells = {(row, cc) for cc in range(start, c)}
                candidates.append(((row, start), segment, cells))
    assert len(candidates) == 1, "expected exactly one 5-color edge palette"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each foreground object diagonally into a color-selected shadow whose shift equals the object's area.

    Input structure:
    - The background is black.
    - Exactly one corner contains a 1-pixel direction marker; its corner chooses one diagonal shadow direction.
    - The remaining foreground contains several separated monochrome 4-connected objects, all in the same base color, and a 5-cell edge palette row.
    - Objects may contain enclosed background holes, and the palette colors from left to right encode hole counts 0 through 4.

    Rule:
    - Read the marker corner to get the diagonal step: top-left -> (+1,+1), top-right -> (+1,-1), bottom-left -> (-1,+1), bottom-right -> (-1,-1).
    - For each foreground object, count its enclosed background holes and use that count to index the palette color.
    - Let the object's translation distance equal its pixel count, then stamp a shifted copy of the object's foreground shape in the chosen diagonal direction.
    - Keep the original scene unchanged and place it above any shadow pixels where overlaps occur.

    Key insight:
    The task combines four independent cues: corner marker -> direction, enclosed-hole count -> palette index, object area -> translation distance, and shape copying -> shadow geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    (_marker_rc, (dr, dc)) = _find_corner_direction(g, bg)
    (palette_pos, palette, palette_cells) = _find_palette(g, bg)

    special = np.zeros(g.shape, dtype=bool)
    special[list(zip(*palette_cells))] = True
    special[_find_corner_direction(g, bg)[0]] = True

    object_mask = (g != bg) & ~special
    object_colors = sorted(set(map(int, g[object_mask].tolist())))
    assert len(object_colors) == 1, "objects must share one color"

    shadow = np.full(g.shape, bg, dtype=int)
    for comp in _mask_components(object_mask):
        holes = _count_holes(comp)
        assert 0 <= holes < len(palette), "hole count must fit the palette"
        color = int(palette[holes])
        distance = int(comp.shape[0])
        shifted = comp + np.array([dr * distance, dc * distance], dtype=int)
        for rr, cc in shifted:
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                shadow[int(rr), int(cc)] = color

    out = shadow
    out[g != bg] = g[g != bg]
    return out


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _shape_hole_count(mask: np.ndarray) -> int:
    return _count_holes(np.argwhere(mask))


def _dilate8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _make_object_mask(hole_count: int) -> np.ndarray:
    dims: list[tuple[int, int]] = []
    for h in range(3, 7):
        for w in range(3, 7):
            interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
            parity_max = max(
                sum(1 for r, c in interior if (r + c) % 2 == parity) for parity in (0, 1)
            )
            if parity_max < hole_count:
                continue
            area = h * w - hole_count
            if 4 <= area <= 21:
                dims.append((h, w))

    for _ in range(200):
        h, w = random.choice(dims)
        mask = np.ones((h, w), dtype=bool)
        if hole_count:
            buckets = [
                [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if (r + c) % 2 == parity]
                for parity in (0, 1)
            ]
            bucket = max(buckets, key=len)
            for r, c in random.sample(bucket, hole_count):
                mask[r, c] = False

        pad = np.pad(mask, 1)
        outer_candidates: list[tuple[int, int]] = []
        ph, pw = pad.shape
        for r in range(ph):
            for c in range(pw):
                if pad[r, c]:
                    continue
                if 1 <= r <= h and 1 <= c <= w:
                    continue
                if any(pad[nr, nc] for nr, nc in _neighbors4(r, c, ph, pw)):
                    outer_candidates.append((r, c))

        random.shuffle(outer_candidates)
        for r, c in outer_candidates[: random.randint(0, 2 if hole_count < 4 else 1)]:
            pad[r, c] = True

        mask = _crop_bool(pad)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        if _shape_hole_count(mask) == hole_count and 4 <= int(mask.sum()) <= 22:
            return np.asarray(mask, dtype=bool)

    raise RuntimeError("failed to build object mask")


def _place_objects(
    h: int,
    w: int,
    objects: list[dict],
    reserved: set[tuple[int, int]],
) -> list[dict] | None:
    blocked = _dilate8(reserved, h, w)
    occupied: set[tuple[int, int]] = set(reserved)
    placed: list[dict] = []

    for obj in sorted(objects, key=lambda item: -item["area"]):
        mask = obj["mask"]
        mh, mw = mask.shape
        rel = np.argwhere(mask)
        shift = int(obj["area"])
        max_r = h - mh - shift - 1
        max_c = w - mw - shift
        if max_r < 1 or max_c < 1:
            return None

        candidates: list[tuple[int, int, set[tuple[int, int]], set[tuple[int, int]]]] = []
        for r0 in range(1, max_r + 1):
            for c0 in range(1, max_c + 1):
                src = {(int(r0 + r), int(c0 + c)) for r, c in rel}
                shadow = {(int(r0 + r + shift), int(c0 + c + shift)) for r, c in rel}
                if src & occupied or shadow & occupied:
                    continue
                if src & blocked or shadow & blocked:
                    continue
                candidates.append((r0, c0, src, shadow))

        if not candidates:
            return None

        r0, c0, src, shadow = random.choice(candidates)
        placed.append(
            {
                "hole_count": obj["hole_count"],
                "mask": mask,
                "area": obj["area"],
                "row": r0,
                "col": c0,
                "src": src,
                "shadow": shadow,
            }
        )
        used = src | shadow
        occupied |= used
        blocked |= _dilate8(used, h, w)

    return placed


def _mirror_for_corner(grid: np.ndarray, corner: str) -> np.ndarray:
    out = grid
    if corner in {"tr", "br"}:
        out = np.fliplr(out)
    if corner in {"bl", "br"}:
        out = np.flipud(out)
    return np.ascontiguousarray(out)


def _gen_attempt() -> np.ndarray | None:
    object_count = random.choices([2, 3, 4], weights=[4, 4, 2], k=1)[0]
    hole_counts = random.sample(range(PALETTE_LEN), object_count)
    objects = []
    for holes in hole_counts:
        mask = _make_object_mask(holes)
        objects.append({"hole_count": holes, "mask": mask, "area": int(mask.sum())})

    min_h = max(int(obj["mask"].shape[0] + obj["area"] + 2) for obj in objects)
    min_w = max(int(obj["mask"].shape[1] + obj["area"] + 1) for obj in objects)
    min_h = max(min_h, 22)
    min_w = max(min_w, 12)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(max(min_w, 8), 30)

    color_pool = list(range(1, 10))
    object_color = random.choice(color_pool)
    color_pool.remove(object_color)
    marker_color = random.choice(color_pool)
    color_pool.remove(marker_color)
    palette = random.sample(color_pool, PALETTE_LEN)

    palette_start = random.randint(1, w - PALETTE_LEN - 1)
    reserved = {(0, 0)}
    reserved |= {(h - 1, palette_start + i) for i in range(PALETTE_LEN)}

    placed = _place_objects(h, w, objects, reserved)
    if placed is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[0, 0] = marker_color
    grid[h - 1, palette_start : palette_start + PALETTE_LEN] = np.array(palette, dtype=int)
    for item in placed:
        rr, cc = np.nonzero(item["mask"])
        grid[item["row"] + rr, item["col"] + cc] = object_color

    corner = random.choice(["tl", "tr", "bl", "br"])
    grid = _mirror_for_corner(grid, corner)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
