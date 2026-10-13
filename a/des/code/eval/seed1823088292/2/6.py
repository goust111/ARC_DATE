from __future__ import annotations

import numpy as np


BG = 0
MAROON = 9
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _crop_non_bg(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    assert coords.size > 0, "expected at least one non-background pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _parse_frame(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    comp = set(coords)
    border = set(_border_cells(r0, c0, r1, c1))
    if comp != border:
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
        return None

    return {"color": int(color), "bbox": (int(r0), int(c0), int(r1), int(c1))}


def _parse_envelope(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    comp = set(coords)
    border = _border_cells(r0, c0, r1, c1)
    stamp_pos: tuple[int, int] | None = None
    stamp_color: int | None = None
    expected_maroon = set()
    for cell in border:
        v = int(grid[cell])
        if v == MAROON:
            expected_maroon.add(cell)
            continue
        if v == BG:
            return None
        if stamp_pos is not None:
            return None
        stamp_pos = cell
        stamp_color = v
    if stamp_pos is None or stamp_color is None:
        return None
    if comp != expected_maroon:
        return None

    lining = grid[r0 + 1 : r1, c0 + 1 : c1]
    if lining.shape[0] < 3 or lining.shape[1] < 3:
        return None
    if np.any(lining[0, :] != BG) or np.any(lining[-1, :] != BG):
        return None
    if np.any(lining[:, 0] != BG) or np.any(lining[:, -1] != BG):
        return None

    core = grid[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1]
    if core.size == 0 or not np.any(core != BG):
        return None
    pattern = _crop_non_bg(core)

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "stamp_color": int(stamp_color),
        "pattern": pattern,
    }


def _paste_transparent(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BG) -> None:
    h, w = patch.shape
    block = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    block[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each maroon "envelope", read its stamp color, and place the enclosed pattern
    at the center of the matching hollow rectangle fence of the same color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    envelopes: list[dict] = []
    for comp in _components(g == MAROON):
        envelope = _parse_envelope(g, comp)
        if envelope is not None:
            envelopes.append(envelope)

    envelope_boxes = [env["bbox"] for env in envelopes]

    frames: dict[int, dict] = {}
    for color in map(int, np.unique(g)):
        if color in (BG, MAROON):
            continue
        for comp in _components(g == color):
            frame = _parse_frame(g, comp, color)
            if frame is None:
                continue
            r0, c0, r1, c1 = frame["bbox"]
            if any(er0 <= r0 <= r1 <= er1 and ec0 <= c0 <= c1 <= ec1 for er0, ec0, er1, ec1 in envelope_boxes):
                continue
            prev = frames.get(int(color))
            if prev is None:
                frames[int(color)] = frame
                continue
            pr0, pc0, pr1, pc1 = prev["bbox"]
            prev_area = (pr1 - pr0 + 1) * (pc1 - pc0 + 1)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area > prev_area:
                frames[int(color)] = frame

    for env in envelopes:
        r0, c0, r1, c1 = env["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for env in envelopes:
        frame = frames[env["stamp_color"]]
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        pattern = env["pattern"]
        ph, pw = map(int, pattern.shape)
        if ph > inner_h or pw > inner_w:
            raise ValueError("pattern does not fit inside matching fence")
        dst_r = r0 + 1 + (inner_h - ph) // 2
        dst_c = c0 + 1 + (inner_w - pw) // 2
        _paste_transparent(out, pattern, dst_r, dst_c, transparent=BG)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
PAIR_COUNT_RANGE = (2, 4)
PATTERN_DIM_RANGE = (2, 6)
PATTERN_COLOR_COUNT_RANGE = (2, 4)
PADDING = 1
GEN_ATTEMPTS = 300


def _grow_mask(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        pick = list(frontier)
        assert pick, "frontier should not be empty while the bbox still has free cells"
        cells.add(pick[int(rng.integers(0, len(pick)))])

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_pattern(rng: np.random.Generator, color_pool: list[int]) -> np.ndarray:
    h = int(rng.integers(*PATTERN_DIM_RANGE))
    w = int(rng.integers(*PATTERN_DIM_RANGE))
    area_lo = max(3, (h * w + 2) // 3)
    area = int(rng.integers(area_lo, h * w + 1))
    mask = _grow_mask(rng, h, w, area)

    patch = np.full((h, w), BG, dtype=int)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    rng.shuffle(cells)

    color_count = int(rng.integers(PATTERN_COLOR_COUNT_RANGE[0], min(PATTERN_COLOR_COUNT_RANGE[1], len(cells), len(color_pool)) + 1))
    colors = [int(x) for x in rng.choice(np.array(color_pool, dtype=int), size=color_count, replace=False)]
    for idx, (r, c) in enumerate(cells):
        if idx < color_count:
            color = colors[idx]
        else:
            color = int(colors[int(rng.integers(0, color_count))])
        patch[r, c] = color

    patch = _crop_non_bg(patch)
    if bool(rng.integers(0, 2)):
        patch = np.fliplr(patch)
    if bool(rng.integers(0, 2)):
        patch = np.flipud(patch)
    patch = np.rot90(patch, int(rng.integers(0, 4)))
    return patch.copy()


def _frame_patch(color: int, inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _envelope_patch(pattern: np.ndarray, stamp_color: int, rng: np.random.Generator) -> np.ndarray:
    ph, pw = map(int, pattern.shape)
    patch = np.full((ph + 4, pw + 4), BG, dtype=int)
    patch[0, :] = MAROON
    patch[-1, :] = MAROON
    patch[:, 0] = MAROON
    patch[:, -1] = MAROON
    patch[2 : 2 + ph, 2 : 2 + pw] = pattern

    border = _border_cells(0, 0, ph + 3, pw + 3)
    sr, sc = border[int(rng.integers(0, len(border)))]
    patch[sr, sc] = int(stamp_color)
    return patch


def _choose_inner_size(
    rng: np.random.Generator, ph: int, pw: int, exact_center: bool
) -> tuple[int, int]:
    slack_h = 2 * int(rng.integers(0, 4))
    slack_w = 2 * int(rng.integers(0, 4))
    if not exact_center:
        if bool(rng.integers(0, 2)):
            slack_h += 1
        else:
            slack_w += 1
    return ph + slack_h, pw + slack_w


def _find_slot(
    rng: np.random.Generator, occupied: np.ndarray, h: int, w: int
) -> tuple[int, int] | None:
    H, W = occupied.shape
    spots: list[tuple[int, int]] = []
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            rr0 = r0 - PADDING
            cc0 = c0 - PADDING
            rr1 = r0 + h + PADDING
            cc1 = c0 + w + PADDING
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            spots.append((r0, c0))
    if not spots:
        return None
    return spots[int(rng.integers(0, len(spots)))]


def _place_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    pair_count = int(rng.integers(*PAIR_COUNT_RANGE))
    fence_colors = [int(x) for x in rng.choice(np.arange(1, 9, dtype=int), size=pair_count, replace=False)]
    pattern_color_pool = [color for color in range(1, 9) if color not in fence_colors]

    pairs: list[dict] = []
    for fence_color in fence_colors:
        pattern = _make_pattern(rng, pattern_color_pool)
        ph, pw = map(int, pattern.shape)
        exact_center = bool(rng.integers(0, 2))
        inner_h, inner_w = _choose_inner_size(rng, ph, pw, exact_center)
        fence_patch = _frame_patch(fence_color, inner_h, inner_w)
        envelope_patch = _envelope_patch(pattern, fence_color, rng)
        pairs.append(
            {
                "fence_color": int(fence_color),
                "pattern": pattern,
                "fence_patch": fence_patch,
                "envelope_patch": envelope_patch,
            }
        )

    H = int(rng.integers(*GRID_H_RANGE))
    W = int(rng.integers(*GRID_W_RANGE))
    inp = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True

    objects: list[dict] = []
    for idx, pair in enumerate(pairs):
        objects.append({"kind": "fence", "pair": idx, "patch": pair["fence_patch"]})
        objects.append({"kind": "envelope", "pair": idx, "patch": pair["envelope_patch"]})
    rng.shuffle(objects)
    objects.sort(key=lambda obj: obj["patch"].shape[0] * obj["patch"].shape[1], reverse=True)

    placements: dict[tuple[str, int], tuple[int, int]] = {}
    for obj in objects:
        patch = obj["patch"]
        h, w = map(int, patch.shape)
        slot = _find_slot(rng, occupied, h, w)
        if slot is None:
            return None
        r0, c0 = slot
        inp[r0 : r0 + h, c0 : c0 + w] = patch
        _place_rect(occupied, r0, c0, h, w)
        placements[(obj["kind"], obj["pair"])] = (int(r0), int(c0))

    expected = inp.copy()
    for idx, pair in enumerate(pairs):
        er, ec = placements[("envelope", idx)]
        eh, ew = pair["envelope_patch"].shape
        expected[er : er + eh, ec : ec + ew] = BG

        fr, fc = placements[("fence", idx)]
        fh, fw = pair["fence_patch"].shape
        pattern = pair["pattern"]
        ph, pw = pattern.shape
        dst_r = fr + 1 + (fh - 2 - ph) // 2
        dst_c = fc + 1 + (fw - 2 - pw) // 2
        _paste_transparent(expected, pattern, dst_r, dst_c, transparent=BG)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
