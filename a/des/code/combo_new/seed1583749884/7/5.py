from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
PALETTE = tuple(color for color in range(10) if color not in (BG, SEP))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
MAX_MASK_ATTEMPTS = 200


def _full_separator_col(grid: np.ndarray, bg: int) -> tuple[int, int]:
    cols = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color != bg and bool(np.all(grid[:, c] == color)):
            cols.append((c, color))
    assert len(cols) == 1, "expected exactly one full non-background divider column"
    return cols[0]


def _connected_components(panel: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(panel[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            patch = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                }
            )

    return components


def _sender_port_row(patch: np.ndarray, bg: int) -> int:
    rows = np.flatnonzero(patch[:, -1] != bg)
    assert rows.size == 1, "sender must have a unique divider-facing port pixel"
    return int(rows[0])


def _stamp(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= top and top + h <= out.shape[0], "patch rows out of bounds"
    assert 0 <= left and left + w <= out.shape[1], "patch cols out of bounds"
    target = out[top : top + h, left : left + w]
    mask = patch != bg
    assert not bool(np.any((target != bg) & mask)), "projected senders may not overlap"
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each left-panel sender and stamp it just beyond its same-colored receiver.

    Input structure:
    - The background is black, and one full grey vertical divider splits the grid into a sender panel on the left and a receiver panel on the right.
    - The left panel contains separate monochrome 4-connected sender objects; the right panel contains same-colored single-pixel receiver markers.
    - Every sender color appears exactly once on each side, and each sender's tight bounding box has exactly one occupied cell on the divider-facing edge.

    Rule:
    - Keep all senders and the grey divider unchanged.
    - For each color pair, crop the sender to its tight bounding box and mirror that crop horizontally.
    - Use the mirrored counterpart of the sender's unique divider-facing edge cell as the anchor, and place it immediately to the right of the matching receiver marker.
    - Clear the receiver markers themselves and paint the mirrored sender copies in the right panel without overlaps.

    Key insight:
    - Color chooses the sender/receiver pairing, while the sender's unique divider-facing tip determines how the mirrored copy is anchored on the far side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_col, sep_color = _full_separator_col(g, BG)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    senders = _connected_components(left, BG)
    assert senders, "expected at least one sender"

    sender_by_color: dict[int, dict[str, object]] = {}
    for sender in senders:
        color = int(sender["color"])
        assert color not in sender_by_color, "sender colors must be unique"
        sender_by_color[color] = {
            "patch": np.asarray(sender["patch"], dtype=int),
            "port_row": _sender_port_row(np.asarray(sender["patch"], dtype=int), BG),
        }

    receivers: dict[int, tuple[int, int]] = {}
    for r_raw, c_raw in np.argwhere(right != BG):
        r = int(r_raw)
        c = int(c_raw)
        color = int(right[r, c])
        assert color not in receivers, "receiver colors must be unique"
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < right.shape[0] and 0 <= nc < right.shape[1]:
                assert int(right[nr, nc]) == BG, "receivers must be isolated single pixels"
        receivers[color] = (r, c)

    assert set(sender_by_color) == set(receivers), "sender and receiver colors must match one-to-one"

    out = g.copy()
    for r, c in receivers.values():
        out[r, sep_col + 1 + c] = BG

    for color, sender in sender_by_color.items():
        receiver_r, receiver_c = receivers[color]
        patch = np.fliplr(np.asarray(sender["patch"], dtype=int))
        port_row = int(sender["port_row"])
        assert int(np.count_nonzero(patch[:, 0] != BG)) == 1, "mirrored sender must keep one left-edge anchor"
        top = receiver_r - port_row
        left_col = sep_col + 2 + receiver_c
        assert left_col > sep_col and sep_color == SEP
        _stamp(out, patch, top, left_col, BG)

    return out


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _grow_sender_mask(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(MAX_MASK_ATTEMPTS):
        h = random.randint(3, max_h)
        w = random.randint(3, max_w)
        anchor_r = random.randrange(h)
        start_r = random.randrange(h)

        mask = np.zeros((h, w), dtype=bool)
        target_c = w - 2
        r = start_r
        c = 0
        mask[r, c] = True
        while c < target_c or r != anchor_r:
            moves = []
            if c < target_c:
                moves.append((r, c + 1))
            if r < anchor_r:
                moves.append((r + 1, c))
            if r > anchor_r:
                moves.append((r - 1, c))
            r, c = random.choice(moves)
            mask[r, c] = True
        mask[anchor_r, w - 1] = True

        min_area = max(5, int(mask.sum()))
        max_area = min(h * w - 1, max(min_area + 2, min_area + (h * w) // 2))
        target_area = random.randint(min_area, max_area)
        while int(mask.sum()) < target_area:
            frontier = set()
            for rr_raw, cc_raw in np.argwhere(mask):
                rr = int(rr_raw)
                cc = int(cc_raw)
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if mask[nr, nc] or nc == w - 1:
                        continue
                    frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        tight = _tight_mask(mask)
        if tight.shape[0] < 2 or tight.shape[1] < 3:
            continue
        if np.array_equal(tight, np.fliplr(tight)):
            continue
        if int(np.count_nonzero(tight[:, -1])) != 1:
            continue
        return tight
    return None


def _try_place_rects(
    panel_h: int, panel_w: int, shapes: list[tuple[int, int]], margin: int
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((panel_h, panel_w), dtype=bool)
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)
    placed: list[tuple[int, int] | None] = [None] * len(shapes)

    for idx in order:
        h, w = shapes[idx]
        positions = [(r, c) for r in range(panel_h - h + 1) for c in range(panel_w - w + 1)]
        random.shuffle(positions)
        found = False
        for r0, c0 in positions:
            rr0 = max(0, r0 - margin)
            rr1 = min(panel_h, r0 + h + margin)
            cc0 = max(0, c0 - margin)
            cc1 = min(panel_w, c0 + w + margin)
            if bool(occupied[rr0:rr1, cc0:cc1].any()):
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            placed[idx] = (r0, c0)
            found = True
            break
        if not found:
            return None

    return [tuple(pos) for pos in placed if pos is not None]


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _build_expected(
    inp: np.ndarray,
    sep_col: int,
    receiver_specs: list[tuple[int, int, int]],
    output_specs: list[tuple[np.ndarray, int, int]],
) -> np.ndarray:
    out = inp.copy()
    for _color, rr, rc in receiver_specs:
        out[rr, rc] = BG
    for patch, top, left in output_specs:
        _stamp(out, patch, top, left, BG)
    return out


def _gen_attempt() -> np.ndarray | None:
    sender_count = random.randint(2, 4)
    panel_w = random.randint(max(9, 7 + sender_count), 14)
    grid_h = random.randint(max(10, 5 + 2 * sender_count), 24)
    grid_w = 2 * panel_w + 1

    max_mask_h = min(7, grid_h - 2)
    max_mask_w = min(6, panel_w - 2)
    if max_mask_h < 3 or max_mask_w < 3:
        return None

    colors = random.sample(PALETTE, sender_count)
    sender_patches: list[np.ndarray] = []
    port_rows: list[int] = []
    for color in colors:
        mask = _grow_sender_mask(max_mask_h, max_mask_w)
        if mask is None:
            return None
        sender_patches.append(_mask_to_patch(mask, color))
        port_rows.append(_sender_port_row(sender_patches[-1], BG))

    left_shapes = [tuple(map(int, patch.shape)) for patch in sender_patches]
    left_positions = _try_place_rects(grid_h, panel_w, left_shapes, margin=1)
    if left_positions is None:
        return None

    footprint_shapes = [(int(patch.shape[0]), int(patch.shape[1] + 1)) for patch in sender_patches]
    right_positions = _try_place_rects(grid_h, panel_w, footprint_shapes, margin=1)
    if right_positions is None:
        return None

    sep_col = panel_w
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[:, sep_col] = SEP

    output_specs: list[tuple[np.ndarray, int, int]] = []
    receiver_specs: list[tuple[int, int, int]] = []

    for idx, color in enumerate(colors):
        patch = sender_patches[idx]
        left_r0, left_c0 = left_positions[idx]
        sender_slice = inp[left_r0 : left_r0 + patch.shape[0], left_c0 : left_c0 + patch.shape[1]]
        sender_slice[patch != BG] = color

        footprint_r0, footprint_c0 = right_positions[idx]
        receiver_r = footprint_r0 + port_rows[idx]
        receiver_c = sep_col + 1 + footprint_c0
        inp[receiver_r, receiver_c] = color

        mirrored = np.fliplr(patch)
        output_specs.append((mirrored, footprint_r0, sep_col + 2 + footprint_c0))
        receiver_specs.append((color, receiver_r, receiver_c))

    expected = _build_expected(inp, sep_col, receiver_specs, output_specs)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sender-receiver mirror task")
