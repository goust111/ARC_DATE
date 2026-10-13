from __future__ import annotations

from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

KEY_COLUMN = np.array([BLUE, GREEN, MAGENTA], dtype=int)
FILL_POOL = [YELLOW, GRAY, ORANGE, AZURE, BROWN]
FRAME_POOL = [YELLOW, GRAY, ORANGE, AZURE, BROWN]
SIDES = ("top", "bottom", "left", "right")

RNG = np.random.default_rng()
MAX_RETRIES = 400
OBJECT_RETRIES = 250


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _symmetry_type(mask: np.ndarray, *, require_unique: bool = True) -> str | None:
    horizontal = np.array_equal(mask, np.flipud(mask))
    vertical = np.array_equal(mask, np.fliplr(mask))
    diagonal = False
    if mask.shape[0] == mask.shape[1]:
        anti = np.flipud(np.fliplr(mask)).T
        diagonal = np.array_equal(mask, mask.T) or np.array_equal(mask, anti)
    kinds = [
        name
        for name, ok in (
            ("horizontal", horizontal),
            ("vertical", vertical),
            ("diagonal", diagonal),
        )
        if ok
    ]
    if require_unique:
        assert len(kinds) == 1, "red pattern must have exactly one symmetry class"
    return kinds[0] if len(kinds) == 1 else None


def _enumerate_pattern_bank() -> dict[str, list[np.ndarray]]:
    bank = {"horizontal": [], "vertical": [], "diagonal": []}
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for h in range(2, 4):
        for w in range(2, 4):
            cells = [(r, c) for r in range(h) for c in range(w)]
            for n in range(3, min(5, h * w) + 1):
                for choice in combinations(cells, n):
                    mask = np.zeros((h, w), dtype=bool)
                    for r, c in choice:
                        mask[r, c] = True
                    if len(_components(mask)) != 1:
                        continue
                    tight = _tight_mask(mask)
                    key = (tuple(map(int, tight.shape)), tight.tobytes())
                    if key in seen:
                        continue
                    kind = _symmetry_type(tight, require_unique=False)
                    if kind is None:
                        continue
                    seen.add(key)
                    bank[kind].append(tight)
    for kind, masks in bank.items():
        assert masks, f"missing pattern bank for {kind}"
        for mask in masks:
            assert _symmetry_type(mask) == kind
    return bank


PATTERN_BANK = _enumerate_pattern_bank()


def _find_legend(grid: np.ndarray) -> tuple[dict[str, int], np.ndarray]:
    h, w = grid.shape
    candidates = []
    for r in range(h - 2):
        for c in range(w - 1):
            if not (r == 0 or c == 0 or r + 3 == h or c + 2 == w):
                continue
            block = grid[r : r + 3, c : c + 2]
            if np.array_equal(block[:, 0], KEY_COLUMN) and np.all(block[:, 1] != BLACK):
                candidates.append((r, c, block[:, 1].copy()))
    assert len(candidates) == 1, "expected exactly one edge legend"
    r, c, fills = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[r : r + 3, c : c + 2] = True
    mapping = {
        "horizontal": int(fills[0]),
        "vertical": int(fills[1]),
        "diagonal": int(fills[2]),
    }
    return mapping, mask


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _parse_frame(coords: np.ndarray) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame too small"
    cells = {(int(r), int(c)) for r, c in coords}
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    assert cells <= border, "frame component must lie on rectangle border"
    missing = list(border - cells)
    assert len(missing) == 1, "frame must have exactly one gap"
    gap_r, gap_c = map(int, missing[0])
    assert not (
        gap_r in (r0, r1) and gap_c in (c0, c1)
    ), "gap must lie on a single side, not a corner"
    if gap_r == r0:
        direction = (-1, 0)
    elif gap_r == r1:
        direction = (1, 0)
    elif gap_c == c0:
        direction = (0, -1)
    else:
        direction = (0, 1)
    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (gap_r, gap_c),
        "direction": direction,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up a fill color from the edge legend using the interior red pattern's
    symmetry class, then spray a ray outward from the frame gap whose length is
    the red pattern's pixel count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, legend_mask = _find_legend(g)

    red_components = _components(g == RED)
    assert red_components, "expected at least one red pattern"

    frame_mask = (g != BLACK) & (g != RED) & (~legend_mask)
    frame_components = _components(frame_mask)
    assert frame_components, "expected at least one frame"

    out = g.copy()
    used_red_ids: set[int] = set()

    for frame_coords in sorted(frame_components, key=lambda comp: _bbox(comp)):
        frame = _parse_frame(frame_coords)
        r0, c0, r1, c1 = frame["bbox"]
        inner_red_ids = []
        for idx, red_coords in enumerate(red_components):
            rr0, cc0, rr1, cc1 = _bbox(red_coords)
            if rr0 > r0 and rr1 < r1 and cc0 > c0 and cc1 < c1:
                inner_red_ids.append(idx)
        assert len(inner_red_ids) == 1, "each frame must contain exactly one red pattern"

        red_id = int(inner_red_ids[0])
        assert red_id not in used_red_ids, "red pattern assigned to multiple frames"
        used_red_ids.add(red_id)

        red_mask = _mask_from_coords(red_components[red_id])
        sym = _symmetry_type(red_mask)
        fill = int(mapping[sym])
        gap_r, gap_c = frame["gap"]
        dr, dc = frame["direction"]

        for step in range(int(red_mask.sum())):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = fill

    assert len(used_red_ids) == len(red_components), "every red pattern must belong to a frame"
    return out


def _mark_halo(reserved: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                reserved[rr, cc] = True


def _draw_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _legend_block(fill_map: dict[str, int]) -> np.ndarray:
    return np.array(
        [
            [BLUE, fill_map["horizontal"]],
            [GREEN, fill_map["vertical"]],
            [MAGENTA, fill_map["diagonal"]],
        ],
        dtype=int,
    )


def _place_legend(
    inp: np.ndarray, out: np.ndarray, reserved: np.ndarray, fill_map: dict[str, int]
) -> bool:
    h, w = inp.shape
    edge = str(RNG.choice(SIDES))
    if edge in ("top", "bottom"):
        if w < 4:
            return False
        r0 = 0 if edge == "top" else h - 3
        c0 = int(RNG.integers(1, w - 2))
    else:
        if h < 4:
            return False
        r0 = int(RNG.integers(1, h - 3))
        c0 = 0 if edge == "left" else w - 2

    block = _legend_block(fill_map)
    inp[r0 : r0 + 3, c0 : c0 + 2] = block
    out[r0 : r0 + 3, c0 : c0 + 2] = block
    cells = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 2)}
    _mark_halo(reserved, cells)
    return True


def _sample_pattern(sym: str) -> np.ndarray:
    masks = PATTERN_BANK[sym]
    return masks[int(RNG.integers(0, len(masks)))]


def _place_object(
    inp: np.ndarray, out: np.ndarray, reserved: np.ndarray, sym: str, fill_map: dict[str, int]
) -> bool:
    h, w = inp.shape
    fill = int(fill_map[sym])

    for _ in range(OBJECT_RETRIES):
        pattern = _sample_pattern(sym)
        ph, pw = map(int, pattern.shape)
        ray_len = int(pattern.sum())

        outer_h = int(RNG.integers(ph + 4, ph + 7))
        outer_w = int(RNG.integers(pw + 4, pw + 7))
        side = str(RNG.choice(SIDES))

        r_lo, r_hi = 0, h - outer_h
        c_lo, c_hi = 0, w - outer_w
        if side == "top":
            r_lo = ray_len - 1
        elif side == "bottom":
            r_hi = h - outer_h - (ray_len - 1)
        elif side == "left":
            c_lo = ray_len - 1
        else:
            c_hi = w - outer_w - (ray_len - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        r0 = int(RNG.integers(r_lo, r_hi + 1))
        c0 = int(RNG.integers(c_lo, c_hi + 1))

        if side in ("top", "bottom"):
            gap_offset = int(RNG.integers(1, outer_w - 1))
            gap = (r0 if side == "top" else r0 + outer_h - 1, c0 + gap_offset)
            direction = (-1, 0) if side == "top" else (1, 0)
        else:
            gap_offset = int(RNG.integers(1, outer_h - 1))
            gap = (r0 + gap_offset, c0 if side == "left" else c0 + outer_w - 1)
            direction = (0, -1) if side == "left" else (0, 1)

        inner_h = outer_h - 2
        inner_w = outer_w - 2
        pr = int(RNG.integers(1, inner_h - ph))
        pc = int(RNG.integers(1, inner_w - pw))

        frame_cells = {
            (r, c)
            for r in range(r0, r0 + outer_h)
            for c in range(c0, c0 + outer_w)
            if r in (r0, r0 + outer_h - 1) or c in (c0, c0 + outer_w - 1)
        }
        frame_cells.remove(gap)

        red_cells = {
            (r0 + 1 + pr + rr, c0 + 1 + pc + cc)
            for rr in range(ph)
            for cc in range(pw)
            if pattern[rr, cc]
        }

        ray_cells = set()
        for step in range(ray_len):
            rr = gap[0] + direction[0] * step
            cc = gap[1] + direction[1] * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            ray_cells.add((rr, cc))

        output_cells = frame_cells | red_cells | ray_cells
        if any(reserved[r, c] for r, c in output_cells):
            continue

        frame_color_choices = [color for color in FRAME_POOL if color != fill]
        frame_color = int(frame_color_choices[int(RNG.integers(0, len(frame_color_choices)))])

        _draw_cells(inp, frame_cells, frame_color)
        _draw_cells(out, frame_cells, frame_color)
        _draw_cells(inp, red_cells, RED)
        _draw_cells(out, red_cells, RED)
        _draw_cells(out, ray_cells, fill)
        _mark_halo(reserved, output_cells)
        return True

    return False


def _generate_attempt() -> np.ndarray | None:
    h = int(RNG.integers(22, 31))
    w = int(RNG.integers(22, 31))
    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    fills = [int(color) for color in RNG.choice(np.array(FILL_POOL, dtype=int), size=3, replace=False)]
    fill_map = {
        "horizontal": fills[0],
        "vertical": fills[1],
        "diagonal": fills[2],
    }
    if not _place_legend(inp, out, reserved, fill_map):
        return None

    n_frames = int(RNG.integers(3, 5))
    syms = ["horizontal", "vertical", "diagonal"]
    while len(syms) < n_frames:
        syms.append(str(RNG.choice(["horizontal", "vertical", "diagonal"])))
    RNG.shuffle(syms)

    for sym in syms:
        if not _place_object(inp, out, reserved, sym, fill_map):
            return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
