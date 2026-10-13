from __future__ import annotations

import numpy as np


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    assert len(order) >= 1
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]])
    return int(values[order[0]])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _shape_dims(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _classify_edge(seq: np.ndarray) -> tuple[str, tuple[int, int] | None]:
    if np.all(seq):
        return "full", None
    if not bool(seq[0]) or not bool(seq[-1]):
        return "invalid", None
    gap = np.flatnonzero(~seq)
    if gap.size < 2:
        return "invalid", None
    if np.any(np.diff(gap) != 1):
        return "invalid", None
    return "gap", (int(gap[0]), int(gap[-1]))


def _analyze_frame_component(
    grid: np.ndarray, color: int, bg: int, coords: np.ndarray
) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    comp_mask = np.zeros((h, w), dtype=bool)
    comp_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if np.any(comp_mask & ~border_mask):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior != bg):
        return None

    edges = {
        "top": grid[r0, c0 : c1 + 1] == color,
        "bottom": grid[r1, c0 : c1 + 1] == color,
        "left": grid[r0 : r1 + 1, c0] == color,
        "right": grid[r0 : r1 + 1, c1] == color,
    }
    status = {side: _classify_edge(seq) for side, seq in edges.items()}
    gap_sides = [side for side, (kind, _run) in status.items() if kind == "gap"]
    if len(gap_sides) != 1:
        return None
    if any(kind == "invalid" for kind, _run in status.values()):
        return None

    side = gap_sides[0]
    gap0, gap1 = status[side][1]
    if side in ("top", "bottom"):
        gap_abs = (c0 + gap0, c0 + gap1)
        delta = (1, 0) if side == "top" else (-1, 0)
    else:
        gap_abs = (r0 + gap0, r0 + gap1)
        delta = (0, 1) if side == "left" else (0, -1)

    return {
        "color": int(color),
        "coords": coords.copy(),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "gap_abs": gap_abs,
        "delta": delta,
    }


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, dict], dict[int, list[np.ndarray]]]:
    bg = _majority_color(grid)
    frames: dict[int, dict] = {}
    fragments: dict[int, list[np.ndarray]] = {}
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components(grid == color)
        frame = None
        extras: list[np.ndarray] = []
        for comp in comps:
            info = _analyze_frame_component(grid, color, bg, comp)
            if info is None:
                extras.append(comp)
            else:
                assert frame is None
                frame = info
        assert frame is not None
        assert extras
        frames[color] = frame
        fragments[color] = extras
    assert frames
    return bg, frames, fragments


def _fragment_sort_key(coords: np.ndarray, frame: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["side"]
    rows = coords[:, 0]
    cols = coords[:, 1]
    if side == "top":
        dist = int(r0 - rows.max())
        return dist, int(cols.min()), int(cols.max()), int(coords.shape[0])
    if side == "bottom":
        dist = int(rows.min() - r1)
        return dist, int(cols.min()), int(cols.max()), int(coords.shape[0])
    if side == "left":
        dist = int(c0 - cols.max())
        return dist, int(rows.min()), int(rows.max()), int(coords.shape[0])
    dist = int(cols.min() - c1)
    return dist, int(rows.min()), int(rows.max()), int(coords.shape[0])


def _slide_until_blocked(
    coords: np.ndarray, delta: tuple[int, int], occupied: np.ndarray
) -> np.ndarray:
    h, w = occupied.shape
    dr, dc = delta
    cur = coords.copy()
    while True:
        nxt = cur.copy()
        nxt[:, 0] += dr
        nxt[:, 1] += dc
        rows = nxt[:, 0]
        cols = nxt[:, 1]
        if rows.min() < 0 or rows.max() >= h or cols.min() < 0 or cols.max() >= w:
            break
        if np.any(occupied[rows, cols]):
            break
        cur = nxt
    return cur


def _inside_frame_interior(coords: np.ndarray, frame: dict) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(rows.min() > r0 and rows.max() < r1 and cols.min() > c0 and cols.max() < c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered fragments by color and slide each group into the frame of the same color.

    Input structure
    - Azure background.
    - Each non-background color has one hollow rectangular frame with one continuous side gap.
    - The same color also appears as several small solid fragments outside that frame.

    Rule
    - Detect the frame and its gap side for each color.
    - Sort same-color fragments by distance to the gap, nearest first.
    - Remove the fragments from their original locations.
    - Slide them straight inward through the gap until blocked by the opposite inner wall or by
      already stacked fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, frames, fragments = _parse_scene(g)

    out = np.full_like(g, bg)
    occupied = np.zeros_like(g, dtype=bool)
    for frame in frames.values():
        rows = frame["coords"][:, 0]
        cols = frame["coords"][:, 1]
        out[rows, cols] = frame["color"]
        occupied[rows, cols] = True

    for color, frame in frames.items():
        ordered = sorted(fragments[color], key=lambda comp: _fragment_sort_key(comp, frame))
        for comp in ordered:
            assert _fragment_sort_key(comp, frame)[0] > 0
            placed = _slide_until_blocked(comp, frame["delta"], occupied)
            assert _inside_frame_interior(placed, frame)
            rows = placed[:, 0]
            cols = placed[:, 1]
            out[rows, cols] = color
            occupied[rows, cols] = True

    return out


# --------------- generator ---------------

BG = 8
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
SIDES = ("top", "bottom", "left", "right")
WORK_H = 30
WORK_W = 30
PAD = 1
GEN_ATTEMPTS = 400
LOCAL_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 250


def _grow_connected_cells(
    rng: np.random.Generator, h: int, w: int, area: int
) -> np.ndarray:
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        choice = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(choice)
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    coords[:, 0] -= r0
    coords[:, 1] -= c0
    return coords


def _random_piece(rng: np.random.Generator, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(50):
        h = int(rng.integers(1, max_h + 1))
        w = int(rng.integers(1, max_w + 1))
        if h * w < 2:
            continue
        lo = max(2, (h * w + 1) // 2)
        area = int(rng.integers(lo, h * w + 1))
        coords = _grow_connected_cells(rng, h, w, area)
        if coords.shape[0] >= 2:
            return coords
    return np.array([[0, 0], [0, 1]], dtype=int)


def _frame_mask(
    h: int, w: int, side: str, gap_start: int, gap_end: int
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if side == "top":
        mask[0, gap_start : gap_end + 1] = False
    elif side == "bottom":
        mask[-1, gap_start : gap_end + 1] = False
    elif side == "left":
        mask[gap_start : gap_end + 1, 0] = False
    else:
        mask[gap_start : gap_end + 1, -1] = False
    return mask


def _inside_rect(
    coords: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(rows.min() >= r0 and rows.max() <= r1 and cols.min() >= c0 and cols.max() <= c1)


def _build_local_scene(
    rng: np.random.Generator, color: int, side: str
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(LOCAL_ATTEMPTS):
        outer_h = int(rng.integers(7, 13))
        outer_w = int(rng.integers(7, 13))
        side_len = outer_w if side in ("top", "bottom") else outer_h
        max_gap = min(5, side_len - 2)
        if max_gap < 2:
            continue
        gap_len = int(rng.integers(2, max_gap + 1))
        gap_start = int(rng.integers(1, side_len - gap_len))
        gap_end = gap_start + gap_len - 1
        n_pieces = int(rng.integers(2, 5))

        pieces: list[dict] = []
        dist = int(rng.integers(1, 3))
        valid = True
        for _piece_idx in range(n_pieces):
            if side in ("top", "bottom"):
                piece = _random_piece(rng, min(3, outer_h - 2), gap_len)
                ph, pw = _shape_dims(piece)
                if pw > gap_len:
                    valid = False
                    break
                local_c = int(rng.integers(gap_start, gap_end - pw + 2))
                pieces.append(
                    {
                        "coords": piece,
                        "h": ph,
                        "w": pw,
                        "start_dist": dist,
                        "local_c": local_c,
                    }
                )
                dist += ph + int(rng.integers(1, 3))
            else:
                piece = _random_piece(rng, gap_len, min(3, outer_w - 2))
                ph, pw = _shape_dims(piece)
                if ph > gap_len:
                    valid = False
                    break
                local_r = int(rng.integers(gap_start, gap_end - ph + 2))
                pieces.append(
                    {
                        "coords": piece,
                        "h": ph,
                        "w": pw,
                        "start_dist": dist,
                        "local_r": local_r,
                    }
                )
                dist += pw + int(rng.integers(1, 3))
        if not valid:
            continue

        if side in ("top", "bottom"):
            outward = max(piece["start_dist"] + piece["h"] - 1 for piece in pieces)
            reserve_h = outer_h + outward
            reserve_w = outer_w
            frame_r0 = outward if side == "top" else 0
            frame_c0 = 0
        else:
            outward = max(piece["start_dist"] + piece["w"] - 1 for piece in pieces)
            reserve_h = outer_h
            reserve_w = outer_w + outward
            frame_r0 = 0
            frame_c0 = outward if side == "left" else 0

        if reserve_h > WORK_H - 2 or reserve_w > WORK_W - 2:
            continue

        frame = np.full((reserve_h, reserve_w), BG, dtype=int)
        frame_mask = _frame_mask(outer_h, outer_w, side, gap_start, gap_end)
        frame_patch_rows, frame_patch_cols = np.where(frame_mask)
        frame[frame_r0 + frame_patch_rows, frame_c0 + frame_patch_cols] = color

        local_in = frame.copy()
        local_out = frame.copy()
        occupied = local_out != BG

        ok = True
        for piece in pieces:
            rel = piece["coords"]
            if side == "top":
                start_r = frame_r0 - piece["start_dist"] - (piece["h"] - 1)
                start_c = frame_c0 + piece["local_c"]
            elif side == "bottom":
                start_r = frame_r0 + outer_h - 1 + piece["start_dist"]
                start_c = frame_c0 + piece["local_c"]
            elif side == "left":
                start_r = frame_r0 + piece["local_r"]
                start_c = frame_c0 - piece["start_dist"] - (piece["w"] - 1)
            else:
                start_r = frame_r0 + piece["local_r"]
                start_c = frame_c0 + outer_w - 1 + piece["start_dist"]

            start = rel + np.array([start_r, start_c], dtype=int)
            rows = start[:, 0]
            cols = start[:, 1]
            if not _inside_rect(start, 0, 0, reserve_h - 1, reserve_w - 1):
                ok = False
                break
            if np.any(local_in[rows, cols] != BG):
                ok = False
                break
            local_in[rows, cols] = color

            delta = {
                "top": (1, 0),
                "bottom": (-1, 0),
                "left": (0, 1),
                "right": (0, -1),
            }[side]
            placed = _slide_until_blocked(start, delta, occupied)
            if not _inside_rect(
                placed,
                frame_r0 + 1,
                frame_c0 + 1,
                frame_r0 + outer_h - 2,
                frame_c0 + outer_w - 2,
            ):
                ok = False
                break
            prow = placed[:, 0]
            pcol = placed[:, 1]
            local_out[prow, pcol] = color
            occupied[prow, pcol] = True

        if not ok:
            continue
        if np.array_equal(local_in, local_out):
            continue
        return local_in, local_out
    return None


def _place_patches(
    rng: np.random.Generator, patches: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray] | None:
    occ = np.zeros((WORK_H, WORK_W), dtype=bool)
    full_in = np.full((WORK_H, WORK_W), BG, dtype=int)
    full_out = np.full((WORK_H, WORK_W), BG, dtype=int)

    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx][0].shape[0] * patches[idx][0].shape[1],
        reverse=True,
    )
    positions: dict[int, tuple[int, int]] = {}

    for idx in order:
        patch_in, _patch_out = patches[idx]
        ph, pw = patch_in.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(0, WORK_H - ph + 1))
            c0 = int(rng.integers(0, WORK_W - pw + 1))
            rr0 = max(0, r0 - PAD)
            cc0 = max(0, c0 - PAD)
            rr1 = min(WORK_H, r0 + ph + PAD)
            cc1 = min(WORK_W, c0 + pw + PAD)
            if np.any(occ[rr0:rr1, cc0:cc1]):
                continue
            occ[rr0:rr1, cc0:cc1] = True
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    for idx, (patch_in, patch_out) in enumerate(patches):
        r0, c0 = positions[idx]
        in_mask = patch_in != BG
        out_mask = patch_out != BG
        full_in[r0 : r0 + patch_in.shape[0], c0 : c0 + patch_in.shape[1]][in_mask] = patch_in[in_mask]
        full_out[r0 : r0 + patch_out.shape[0], c0 : c0 + patch_out.shape[1]][out_mask] = patch_out[out_mask]

    return full_in, full_out


def _crop_with_margin(
    rng: np.random.Generator, inp: np.ndarray, out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_extra = int(rng.integers(0, min(3, int(r0)) + 1))
    left_extra = int(rng.integers(0, min(3, int(c0)) + 1))
    bottom_extra = int(rng.integers(0, min(3, WORK_H - 1 - int(r1)) + 1))
    right_extra = int(rng.integers(0, min(3, WORK_W - 1 - int(c1)) + 1))
    rr0 = int(r0) - top_extra
    cc0 = int(c0) - left_extra
    rr1 = int(r1) + bottom_extra
    cc1 = int(c1) + right_extra
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1], out[rr0 : rr1 + 1, cc0 : cc1 + 1]


def _generated_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = int(rng.integers(2, 4))
    colors = [int(v) for v in rng.choice(PALETTE, size=n_frames, replace=False)]
    directions = [SIDES[i] for i in rng.permutation(len(SIDES))[:n_frames]]

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for color, side in zip(colors, directions):
        local = _build_local_scene(rng, color, side)
        if local is None:
            return None
        patches.append(local)

    placed = _place_patches(rng, patches)
    if placed is None:
        return None
    full_in, full_out = placed
    inp, out = _crop_with_margin(rng, full_in, full_out)

    bg = _majority_color(inp)
    if bg != BG:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        pair = _generated_pair(rng)
        if pair is not None:
            return pair[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
