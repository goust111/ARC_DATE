import random
from collections import defaultdict

import numpy as np


BG = 0
CANVAS_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_info(coords: np.ndarray, color: int) -> dict | None:
    if coords.shape[0] < 8:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    border = _border_mask(height, width)
    local = np.zeros((height, width), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if np.any(local & ~border):
        return None

    missing = np.argwhere(border & ~local)
    if missing.shape[0] > 1:
        return None

    gap_side = None
    if missing.shape[0] == 1:
        mr = int(missing[0, 0] + r0)
        mc = int(missing[0, 1] + c0)
        if (mr in (r0, r1)) and (mc in (c0, c1)):
            return None
        if mr == r0:
            if width % 2 == 0 or mc != (c0 + c1) // 2:
                return None
            gap_side = "top"
        elif mr == r1:
            if width % 2 == 0 or mc != (c0 + c1) // 2:
                return None
            gap_side = "bottom"
        elif mc == c0:
            if height % 2 == 0 or mr != (r0 + r1) // 2:
                return None
            gap_side = "left"
        elif mc == c1:
            if height % 2 == 0 or mr != (r0 + r1) // 2:
                return None
            gap_side = "right"
        else:
            return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_side": gap_side,
    }


def _detect_frames(grid: np.ndarray) -> dict[int, list[dict]]:
    frames_by_color: dict[int, list[dict]] = defaultdict(list)
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components(grid == color):
            info = _frame_info(comp, color)
            if info is not None:
                frames_by_color[int(color)].append(info)
    return frames_by_color


def _flip_inner(inner: np.ndarray, gap_side: str) -> np.ndarray:
    if gap_side in ("top", "bottom"):
        return np.flipud(inner)
    return np.fliplr(inner)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match source frames to gapped target frames by border color and mirror the source pattern into the target.

    Input structure:
    - The background is black.
    - Several one-pixel-thick colored rectangular frames appear on the grid.
    - For each border color, one frame is a complete source frame whose interior contains a black-and-one-color asymmetric pattern, and one frame is a target frame whose interior is black and whose border has a single centered one-pixel gap on one side.

    Rule:
    - Detect rectangular frame components and pair one complete frame with one gapped frame of the same border color.
    - Read the target gap side as the mirror instruction: a gap on the top or bottom means flip the source interior vertically, while a gap on the left or right means flip the source interior horizontally.
    - Copy that flipped interior into the target frame interior, leaving every source frame and every target border cell unchanged.

    Key insight:
    - The border color identifies which source and target belong together, and the target border's missing pixel encodes the axis of reflection rather than being part of the pattern itself.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    pair_count = 0

    for color, frames in _detect_frames(g).items():
        sources = [frame for frame in frames if frame["gap_side"] is None]
        targets = [frame for frame in frames if frame["gap_side"] is not None]
        if len(sources) != 1 or len(targets) != 1:
            continue

        src = sources[0]
        tgt = targets[0]
        sr0, sc0, sr1, sc1 = src["bbox"]
        tr0, tc0, tr1, tc1 = tgt["bbox"]
        if (sr1 - sr0) != (tr1 - tr0) or (sc1 - sc0) != (tc1 - tc0):
            continue

        inner = g[sr0 + 1 : sr1, sc0 + 1 : sc1]
        out[tr0 + 1 : tr1, tc0 + 1 : tc1] = _flip_inner(inner, tgt["gap_side"])
        pair_count += 1

    assert pair_count > 0, "expected at least one source/target frame pair"
    return out


def _mask_has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any((~mask) & (~outside)))


def _random_pattern(inner_h: int, inner_w: int, rng: random.Random) -> np.ndarray:
    area = inner_h * inner_w
    target_min = max(3, area // 4)
    target_max = max(target_min, min(area - 1, (2 * area) // 3))

    for _ in range(200):
        target = rng.randint(target_min, target_max)
        start = (rng.randrange(inner_h), rng.randrange(inner_w))
        cells = {start}
        frontier = set(_neighbors4(start[0], start[1], inner_h, inner_w))
        while len(cells) < target and frontier:
            nr, nc = rng.choice(tuple(frontier))
            frontier.remove((nr, nc))
            if (nr, nc) in cells:
                continue
            cells.add((nr, nc))
            for nxt in _neighbors4(nr, nc, inner_h, inner_w):
                if nxt not in cells:
                    frontier.add(nxt)

        if len(cells) != target:
            continue

        mask = np.zeros((inner_h, inner_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if _mask_has_hole(mask):
            continue
        return mask

    raise RuntimeError("failed to generate an asymmetric pattern")


def _choose_inner_size(side: str, n_pairs: int, rng: random.Random) -> tuple[int, int]:
    max_dim = {2: 7, 3: 6, 4: 5}[n_pairs]
    inner_h = rng.randint(3, max_dim)
    inner_w = rng.randint(3, max_dim)
    if side in ("top", "bottom") and inner_w % 2 == 0:
        inner_w += 1 if inner_w < max_dim else -1
    if side in ("left", "right") and inner_h % 2 == 0:
        inner_h += 1 if inner_h < max_dim else -1
    inner_h = max(3, inner_h)
    inner_w = max(3, inner_w)
    return inner_h, inner_w


def _make_source_patch(border_color: int, pattern_color: int, pattern_mask: np.ndarray) -> np.ndarray:
    inner_h, inner_w = pattern_mask.shape
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    patch[1:-1, 1:-1][pattern_mask] = pattern_color
    return patch


def _make_target_patch(border_color: int, inner_h: int, inner_w: int, side: str) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    if side == "top":
        patch[0, (inner_w + 2) // 2] = BG
    elif side == "bottom":
        patch[-1, (inner_w + 2) // 2] = BG
    elif side == "left":
        patch[(inner_h + 2) // 2, 0] = BG
    else:
        patch[(inner_h + 2) // 2, -1] = BG
    return patch


def _candidate_positions(occupied: np.ndarray, patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = patch_shape
    out = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            out.append((r0, c0))
    return out


def _crop_with_padding(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(inp != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    pad_top = rng.randint(1, min(4, int(r0)))
    pad_left = rng.randint(1, min(4, int(c0)))
    pad_bottom = rng.randint(1, min(4, int(inp.shape[0] - 1 - r1)))
    pad_right = rng.randint(1, min(4, int(inp.shape[1] - 1 - c1)))
    rr0 = int(r0 - pad_top)
    cc0 = int(c0 - pad_left)
    rr1 = int(r1 + pad_bottom + 1)
    cc1 = int(c1 + pad_right + 1)
    return inp[rr0:rr1, cc0:cc1].copy(), out[rr0:rr1, cc0:cc1].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.choice([2, 3, 4])
    palette = list(range(1, 10))
    rng.shuffle(palette)
    border_colors = palette[:n_pairs]
    pattern_colors = palette[n_pairs : 2 * n_pairs]
    sides = rng.sample(SIDES, n_pairs)

    pair_data = []
    objects = []
    for idx in range(n_pairs):
        side = sides[idx]
        inner_h, inner_w = _choose_inner_size(side, n_pairs, rng)
        pattern_mask = _random_pattern(inner_h, inner_w, rng)
        source_patch = _make_source_patch(border_colors[idx], pattern_colors[idx], pattern_mask)
        target_patch = _make_target_patch(border_colors[idx], inner_h, inner_w, side)
        source_inner = np.where(pattern_mask, pattern_colors[idx], BG).astype(int)
        pair_data.append(
            {
                "side": side,
                "source_inner": source_inner,
                "inner_shape": (inner_h, inner_w),
                "target_pos": None,
            }
        )
        objects.append(
            {
                "pair_id": idx,
                "kind": "source",
                "patch": source_patch,
                "area": int(source_patch.shape[0] * source_patch.shape[1]),
            }
        )
        objects.append(
            {
                "pair_id": idx,
                "kind": "target",
                "patch": target_patch,
                "area": int(target_patch.shape[0] * target_patch.shape[1]),
            }
        )

    rng.shuffle(objects)
    objects.sort(key=lambda item: -item["area"])

    inp = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    occupied = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    for obj in objects:
        patch = obj["patch"]
        candidates = _candidate_positions(occupied, patch.shape)
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        ph, pw = patch.shape
        inp[r0 : r0 + ph, c0 : c0 + pw] = np.where(patch != BG, patch, inp[r0 : r0 + ph, c0 : c0 + pw])
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch != BG
        if obj["kind"] == "target":
            pair_data[obj["pair_id"]]["target_pos"] = (r0, c0)

    out = inp.copy()
    for info in pair_data:
        if info["target_pos"] is None:
            return None
        r0, c0 = info["target_pos"]
        inner_h, inner_w = info["inner_shape"]
        flipped = _flip_inner(info["source_inner"], info["side"])
        out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = flipped

    cropped_inp, cropped_out = _crop_with_padding(inp, out, rng)
    if max(cropped_inp.shape) > 30 or max(cropped_out.shape) > 30:
        return None
    try:
        solved = main(cropped_inp)
    except Exception:
        return None
    if not np.array_equal(solved, cropped_out):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid instance")
