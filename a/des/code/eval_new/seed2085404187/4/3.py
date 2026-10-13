from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MASK = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _signature(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = bbox
    return (
        grid[r0 - 1, c0 : c1 + 1].copy(),
        grid[r1 + 1, c0 : c1 + 1].copy(),
        grid[r0 : r1 + 1, c0 - 1].copy(),
        grid[r0 : r1 + 1, c1 + 1].copy(),
    )


def _full_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _matches_signature(patch: np.ndarray, sig: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> bool:
    top, bottom, left, right = sig
    return (
        patch.shape == (len(left), len(top))
        and np.array_equal(patch[0, :], top)
        and np.array_equal(patch[-1, :], bottom)
        and np.array_equal(patch[:, 0], left)
        and np.array_equal(patch[:, -1], right)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the grey-masked rectangle by moving in the only scattered fragment whose full border colors continue the surrounding picture.

    Input structure:
    - The background is black.
    - One large multicolor picture contains a solid grey rectangular masked area inside it.
    - Several smaller multicolor fragments are scattered away from the large picture; they are separate 4-connected non-black components.

    Rule:
    - Find the grey rectangle and read the colors immediately above, below, left, and right of it as a border signature.
    - Among the scattered fragments, keep only the one whose occupied shape is the same full rectangle and whose four edge color sequences exactly match that signature.
    - Remove every scattered fragment from the canvas and paste the unique matching fragment into the masked rectangle, replacing all grey cells.

    Key insight:
    - Shape alone is not enough: the correct fragment is identified by contextual color continuity across all four sides of the mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mask_comps = _components(g == MASK)
    assert len(mask_comps) == 1, "expected one grey mask"
    mask_comp = mask_comps[0]
    assert _full_rectangle(mask_comp), "mask must be a full rectangle"
    r0, c0, r1, c1 = mask_comp["bbox"]
    h, w = g.shape
    assert 0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "mask must be interior"
    sig = _signature(g, mask_comp["bbox"])

    occupied = (g != BG) & (g != MASK)
    comps = _components(occupied)
    assert len(comps) >= 2, "expected one large picture and at least one fragment"
    picture = max(comps, key=lambda comp: int(comp["area"]))
    fragments = [comp for comp in comps if comp is not picture]
    assert fragments, "expected scattered fragments"

    matches: list[dict] = []
    for comp in fragments:
        if not _full_rectangle(comp):
            continue
        fr0, fc0, fr1, fc1 = comp["bbox"]
        patch = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
        if _matches_signature(patch, sig):
            matches.append(comp)

    assert len(matches) == 1, "expected a unique matching fragment"
    match = matches[0]
    mr0, mc0, mr1, mc1 = match["bbox"]
    piece = g[mr0 : mr1 + 1, mc0 : mc1 + 1].copy()

    out = g.copy()
    for comp in fragments:
        for rr, cc in comp["coords"]:
            out[rr, cc] = BG
    out[r0 : r1 + 1, c0 : c1 + 1] = piece
    return out


GEN_RETRIES = 400


def _sample_piece(height: int, width: int, colors: list[int]) -> np.ndarray:
    min_distinct = 2 if height * width > 1 else 1
    for _ in range(200):
        mode = random.choice(("linear", "bands", "noise"))
        piece = np.empty((height, width), dtype=int)
        if mode == "linear":
            a = random.randint(1, 3)
            b = random.randint(1, 3)
            shift = random.randint(0, len(colors) - 1)
            for r in range(height):
                for c in range(width):
                    piece[r, c] = colors[(a * r + b * c + shift) % len(colors)]
        elif mode == "bands":
            row_keys = [random.randrange(len(colors)) for _ in range(height)]
            col_keys = [random.randrange(len(colors)) for _ in range(width)]
            for r in range(height):
                for c in range(width):
                    piece[r, c] = colors[(row_keys[r] + col_keys[c]) % len(colors)]
        else:
            for r in range(height):
                for c in range(width):
                    piece[r, c] = random.choice(colors)

        border = np.concatenate((piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]))
        if len(set(int(v) for v in piece.ravel())) < min_distinct:
            continue
        if len(set(int(v) for v in border)) < min_distinct:
            continue
        return piece
    raise RuntimeError("failed to sample a varied piece")


def _paint_picture(
    picture_mask: np.ndarray,
    core_origin: tuple[int, int],
    mask_bbox: tuple[int, int, int, int],
    piece: np.ndarray,
    colors: list[int],
) -> np.ndarray:
    h, w = picture_mask.shape
    pr0, pc0 = core_origin
    picture = np.full((h, w), BG, dtype=int)
    mode = random.choice(("diag", "checker", "stripes"))
    row_stride = random.randint(1, 3)
    col_stride = random.randint(1, 3)
    shift = random.randint(0, len(colors) - 1)

    for r in range(h):
        for c in range(w):
            if not bool(picture_mask[r, c]):
                continue
            rr = r - pr0
            cc = c - pc0
            if mode == "diag":
                idx = rr + 2 * cc + shift
            elif mode == "checker":
                idx = (rr % 2) + 2 * (cc % 2) + shift
            else:
                idx = rr // row_stride + cc // col_stride + shift
            picture[r, c] = colors[idx % len(colors)]

    r0, c0, r1, c1 = mask_bbox
    picture[r0 : r1 + 1, c0 : c1 + 1] = piece
    picture[r0 - 1, c0 : c1 + 1] = piece[0, :]
    picture[r1 + 1, c0 : c1 + 1] = piece[-1, :]
    picture[r0 : r1 + 1, c0 - 1] = piece[:, 0]
    picture[r0 : r1 + 1, c1 + 1] = piece[:, -1]
    return picture


def _add_protrusions(picture_mask: np.ndarray, core_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = core_bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    for side in random.sample(("top", "bottom", "left", "right"), random.randint(0, 2)):
        if side == "top":
            max_depth = min(3, r0 - 1)
            if max_depth < 1:
                continue
            depth = random.randint(1, max_depth)
            span = random.randint(2, max(2, width - 1))
            start = random.randint(c0, c1 - span + 1)
            picture_mask[r0 - depth : r0, start : start + span] = True
        elif side == "bottom":
            max_depth = min(3, picture_mask.shape[0] - r1 - 2)
            if max_depth < 1:
                continue
            depth = random.randint(1, max_depth)
            span = random.randint(2, max(2, width - 1))
            start = random.randint(c0, c1 - span + 1)
            picture_mask[r1 + 1 : r1 + depth + 1, start : start + span] = True
        elif side == "left":
            max_depth = min(3, c0 - 1)
            if max_depth < 1:
                continue
            depth = random.randint(1, max_depth)
            span = random.randint(2, max(2, height - 1))
            start = random.randint(r0, r1 - span + 1)
            picture_mask[start : start + span, c0 - depth : c0] = True
        else:
            max_depth = min(3, picture_mask.shape[1] - c1 - 2)
            if max_depth < 1:
                continue
            depth = random.randint(1, max_depth)
            span = random.randint(2, max(2, height - 1))
            start = random.randint(r0, r1 - span + 1)
            picture_mask[start : start + span, c1 + 1 : c1 + depth + 1] = True


def _mutate_boundary(piece: np.ndarray, colors: list[int]) -> np.ndarray:
    out = piece.copy()
    height, width = out.shape
    border_cells = {(0, c) for c in range(width)}
    border_cells |= {(height - 1, c) for c in range(width)}
    border_cells |= {(r, 0) for r in range(height)}
    border_cells |= {(r, width - 1) for r in range(height)}
    for r, c in random.sample(list(border_cells), random.randint(1, min(3, len(border_cells)))):
        choices = [color for color in colors if color != int(out[r, c])]
        out[r, c] = random.choice(choices)
    return out


def _make_irregular(piece: np.ndarray) -> np.ndarray:
    height, width = piece.shape
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
    ]
    if not candidates:
        candidates = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(100):
        out = piece.copy()
        r, c = random.choice(candidates)
        out[r, c] = BG
        mask = out != BG
        if mask.any() and not mask.all() and _is_connected(mask):
            return out
    raise RuntimeError("failed to build an irregular distractor")


def _random_distractor(colors: list[int], max_height: int, max_width: int) -> np.ndarray:
    height = random.randint(2, max_height)
    width = random.randint(2, max_width)
    patch = _sample_piece(height, width, colors)
    if random.random() < 0.5:
        patch = _make_irregular(patch)
    elif height * width >= 4:
        patch = _mutate_boundary(patch, colors)
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for dr, dc in DIRS8:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    if not mask.any():
        return
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _scatter_fragments(
    base_grid: np.ndarray,
    picture_mask: np.ndarray,
    patches: list[np.ndarray],
) -> np.ndarray | None:
    grid = base_grid.copy()
    h, w = grid.shape
    max_dim = max(max(patch.shape) for patch in patches)
    band = min(max(5, max_dim + 1), max(5, min(h, w) // 2))
    blocked = _dilate8(picture_mask)

    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1],
        reverse=True,
    )
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        positions: list[tuple[int, int]] = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                in_edge_band = r + ph <= band or r >= h - band or c + pw <= band or c >= w - band
                if not in_edge_band:
                    continue
                if blocked[r : r + ph, c : c + pw].any():
                    continue
                positions.append((r, c))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _place_patch(grid, patch, r0, c0)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc1 = min(w, c0 + pw + 1)
        blocked[rr0:rr1, cc0:cc1] = True
    return grid


def _bg_is_unique_mode(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[BG]) > max(int(counts[color]) for color in range(10) if color != BG)


def _gen_attempt() -> np.ndarray | None:
    core_h = random.randint(8, 12)
    core_w = random.randint(8, 12)
    grid_h = random.randint(max(22, core_h + 8), 30)
    grid_w = random.randint(max(22, core_w + 8), 30)

    pr0 = random.randint(4, grid_h - core_h - 4)
    pc0 = random.randint(4, grid_w - core_w - 4)
    core_bbox = (pr0, pc0, pr0 + core_h - 1, pc0 + core_w - 1)

    picture_mask = np.zeros((grid_h, grid_w), dtype=bool)
    picture_mask[pr0 : pr0 + core_h, pc0 : pc0 + core_w] = True
    _add_protrusions(picture_mask, core_bbox)

    mask_h = random.randint(2, min(5, core_h - 2))
    mask_w = random.randint(2, min(5, core_w - 2))
    mask_r0 = random.randint(pr0 + 1, pr0 + core_h - mask_h - 1)
    mask_c0 = random.randint(pc0 + 1, pc0 + core_w - mask_w - 1)
    mask_bbox = (mask_r0, mask_c0, mask_r0 + mask_h - 1, mask_c0 + mask_w - 1)

    scene_colors = random.sample(list(PALETTE), random.randint(3, 5))
    piece = _sample_piece(mask_h, mask_w, scene_colors)
    out = _paint_picture(picture_mask, (pr0, pc0), mask_bbox, piece, scene_colors)

    inp = out.copy()
    inp[mask_r0 : mask_r0 + mask_h, mask_c0 : mask_c0 + mask_w] = MASK

    patches = [
        piece.copy(),
        _mutate_boundary(piece, scene_colors),
        _make_irregular(piece),
    ]
    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        patches.append(_random_distractor(scene_colors, 6, 6))
    random.shuffle(patches)

    inp = _scatter_fragments(inp, picture_mask | (inp == MASK), patches)
    if inp is None:
        return None
    if not _bg_is_unique_mode(inp):
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
