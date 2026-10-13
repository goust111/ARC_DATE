from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
GRAY = 5

SLOT = 15
GRID_SIZE = SLOT * 2
USABLE = SLOT - 1
SIDES = ("left", "right", "top", "bottom")
H_SIDES = ("left", "right")
V_SIDES = ("top", "bottom")
MASK_DIMS = ((3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (5, 3))
GEN_RETRY_LIMIT = 400

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_shape(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _paste_mask(
    grid: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray, color: int
) -> None:
    r0, c0, r1, c1 = box
    assert mask.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1][mask] = int(color)


def _find_components(grid: np.ndarray, color: int, connectivity: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == int(color)
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(int(x) for x in coords[0])}
    q = deque([tuple(int(x) for x in coords[0])])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _same_bbox(mask: np.ndarray, shape: tuple[int, int]) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return _bbox(coords) == (0, 0, shape[0] - 1, shape[1] - 1)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any()
        and mask[-1].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _mirror(mask: np.ndarray, side: str) -> np.ndarray:
    if side in H_SIDES:
        return np.fliplr(mask)
    if side in V_SIDES:
        return np.flipud(mask)
    raise ValueError(f"unknown side: {side}")


def _is_hollow_red_frame(grid: np.ndarray, comp: dict) -> bool:
    box = comp["bbox"]
    h, w = _bbox_shape(box)
    if h < 5 or w < 5:
        return False
    patch = _crop(grid, box)
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(
        np.all(patch[border] == RED)
        and not np.any(patch[1:-1, 1:-1] == RED)
        and int(comp["area"]) == int(border.sum())
    )


def _pair_side_and_gap(
    frame_box: tuple[int, int, int, int],
    inside_box: tuple[int, int, int, int],
    outer_box: tuple[int, int, int, int],
) -> tuple[str, int] | None:
    fr0, fc0, fr1, fc1 = frame_box
    ir0, ic0, ir1, ic1 = inside_box
    or0, oc0, or1, oc1 = outer_box
    ih, iw = _bbox_shape(inside_box)
    oh, ow = _bbox_shape(outer_box)
    if (ih, iw) != (oh, ow):
        return None
    if (or0, or1) == (ir0, ir1) and oc1 < fc0 and ic1 == fc1 - 1:
        return "left", int(fc0 - oc1)
    if (or0, or1) == (ir0, ir1) and oc0 > fc1 and ic0 == fc0 + 1:
        return "right", int(oc0 - fc1)
    if (oc0, oc1) == (ic0, ic1) and or1 < fr0 and ir1 == fr1 - 1:
        return "top", int(fr0 - or1)
    if (oc0, oc1) == (ic0, ic1) and or0 > fr1 and ir0 == fr0 + 1:
        return "bottom", int(or0 - fr1)
    return None


def _parse_scene(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frame_comps = [comp for comp in _find_components(g, RED, 4) if _is_hollow_red_frame(g, comp)]
    assert frame_comps, "expected at least one hollow red frame"

    gray_comps = _find_components(g, GRAY, 8)
    assert gray_comps, "expected gray sprites"

    inside_for_frame: dict[tuple[int, int, int, int], dict] = {}
    outer_components: list[dict] = []
    for comp in gray_comps:
        box = comp["bbox"]
        holders = []
        for frame in frame_comps:
            fr0, fc0, fr1, fc1 = frame["bbox"]
            r0, c0, r1, c1 = box
            if fr0 < r0 <= r1 < fr1 and fc0 < c0 <= c1 < fc1:
                holders.append(frame)
        if holders:
            assert len(holders) == 1, "gray component must belong to exactly one frame interior"
            frame_box = holders[0]["bbox"]
            assert frame_box not in inside_for_frame, "each frame must have one inner sprite"
            inside_for_frame[frame_box] = comp
        else:
            outer_components.append(comp)

    assigned: dict[tuple[int, int, int, int], tuple[dict, str]] = {}
    for outer_comp in outer_components:
        matches: list[tuple[tuple[int, int, int, int], str, int]] = []
        for frame in frame_comps:
            frame_box = frame["bbox"]
            inner_comp = inside_for_frame[frame_box]
            matched = _pair_side_and_gap(frame_box, inner_comp["bbox"], outer_comp["bbox"])
            if matched is not None:
                side, gap = matched
                matches.append((frame_box, side, gap))
        assert matches, "outer sprite must match a frame"
        best_gap = min(gap for _frame_box, _side, gap in matches)
        best = [(frame_box, side) for frame_box, side, gap in matches if gap == best_gap]
        assert len(best) == 1, "outer-to-frame pairing must be unique"
        frame_box, side = best[0]
        assert frame_box not in assigned, "frame received multiple outer sprites"
        assigned[frame_box] = (outer_comp, side)

    assert len(assigned) == len(frame_comps), "each frame must receive one outer sprite"

    pairs: list[dict] = []
    for frame in sorted(frame_comps, key=lambda item: item["bbox"]):
        frame_box = frame["bbox"]
        assert frame_box in inside_for_frame, "each frame needs an inner sprite"
        assert frame_box in assigned, "frame missing its outer sprite"
        outer_comp, side = assigned[frame_box]
        pairs.append(
            {
                "frame_box": frame_box,
                "inner_box": inside_for_frame[frame_box]["bbox"],
                "outer_box": outer_comp["bbox"],
                "side": side,
            }
        )

    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each sprite's true shape from complementary mirrored fragments and fill the missing pixels yellow on both sides.

    Input structure:
    - The input is a black grid containing one or more hollow red rectangular frames.
    - For each frame, one gray sprite lies outside the frame on one side, while a gray mirrored counterpart lies inside the frame on the opposite side.
    - The outside sprite and the inside counterpart share the same bounding-box size, but some gray pixels are missing on each side.
    - At least one frame is already complete on both sides and acts as an unchanged distractor.

    Rule:
    - For each frame, infer the mirror direction from whether the outside sprite sits left/right or above/below the frame.
    - Mirror the inner sprite back into the outer sprite's orientation and take the union of the two masks to obtain the true shape.
    - Color every pixel that is absent on a side but present in the true shape yellow on that side; keep all existing gray pixels, red frame pixels, and background unchanged.

    Key insight:
    - The full sprite never appears directly in incomplete cases; it is only recoverable by aligning the outer fragment with the mirrored inner fragment in the frame-specific reference direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pairs = _parse_scene(g)
    out = g.copy()
    for pair in pairs:
        outer_patch = _crop(g, pair["outer_box"]) == GRAY
        inner_patch = _crop(g, pair["inner_box"]) == GRAY

        mirrored_inner = _mirror(inner_patch, pair["side"])
        truth = outer_patch | mirrored_inner

        outer_missing = truth & ~outer_patch
        inner_truth = _mirror(truth, pair["side"])
        inner_missing = inner_truth & ~inner_patch

        _paste_mask(out, pair["outer_box"], outer_missing, YELLOW)
        _paste_mask(out, pair["inner_box"], inner_missing, YELLOW)

    return out


def _enumerate_valid_masks() -> dict[str, list[np.ndarray]]:
    pools = {"h": [], "v": []}
    for h, w in MASK_DIMS:
        cells = h * w
        for bits in range(1, 1 << cells):
            count = bits.bit_count()
            if count < 6 or count > cells - 2:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for idx in range(cells):
                if bits & (1 << idx):
                    mask[idx // w, idx % w] = True
            if not _touches_all_sides(mask) or not _mask_connected(mask):
                continue
            if not np.array_equal(mask, np.fliplr(mask)):
                pools["h"].append(mask.copy())
            if not np.array_equal(mask, np.flipud(mask)):
                pools["v"].append(mask.copy())
    return pools


MASK_POOL = _enumerate_valid_masks()


def _sample_incomplete_views(mask: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    truth = mask.copy()
    coords = [tuple(int(x) for x in p) for p in np.argwhere(truth)]
    shape = truth.shape
    limit = min(3, len(coords) - 2)
    if limit < 1:
        return None

    for _ in range(400):
        random.shuffle(coords)
        a_size = random.randint(1, limit)
        b_size = random.randint(1, limit)
        a_cells = coords[:a_size]
        remaining = coords[a_size:]
        if len(remaining) < b_size:
            continue
        b_cells = remaining[:b_size]

        outer = truth.copy()
        inner_outer = truth.copy()
        for r, c in a_cells:
            outer[r, c] = False
        for r, c in b_cells:
            inner_outer[r, c] = False

        if not (_mask_connected(outer) and _mask_connected(inner_outer)):
            continue
        if not (_same_bbox(outer, shape) and _same_bbox(inner_outer, shape)):
            continue
        inner_actual = _mirror(inner_outer, side)
        wrong_union = outer | inner_actual
        if np.array_equal(wrong_union, truth):
            continue
        return outer, inner_outer
    return None


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED


def _build_slot_item(
    slot_index: int,
    side: str,
    complete: bool,
) -> dict | None:
    slot_r0 = (slot_index // 2) * SLOT
    slot_c0 = (slot_index % 2) * SLOT
    axis = "h" if side in H_SIDES else "v"
    candidates = MASK_POOL[axis][:]
    random.shuffle(candidates)

    for truth in candidates:
        sh, sw = truth.shape
        if side in H_SIDES:
            width_options = []
            for gap in (1, 2):
                for frame_w in (sw + 4, sw + 5):
                    if sw + gap + frame_w <= USABLE:
                        width_options.append((gap, frame_w))
            if not width_options:
                continue
            gap, frame_w = random.choice(width_options)
            frame_h = random.randint(sh + 2, min(USABLE, sh + 5))
            total_w = sw + gap + frame_w
            left_margin = random.randint(0, USABLE - total_w)
            top_margin = random.randint(0, USABLE - frame_h)
            frame_r0 = slot_r0 + top_margin
            row_off = random.randint(0, frame_h - 2 - sh)
            if side == "left":
                outer_c0 = slot_c0 + left_margin
                frame_c0 = outer_c0 + sw + gap
                inner_c0 = frame_c0 + frame_w - 1 - sw
            else:
                frame_c0 = slot_c0 + left_margin
                outer_c0 = frame_c0 + frame_w + gap
                inner_c0 = frame_c0 + 1
            outer_r0 = frame_r0 + 1 + row_off
            inner_r0 = outer_r0
        else:
            height_options = []
            for gap in (1, 2):
                for frame_h in (sh + 4, sh + 5):
                    if sh + gap + frame_h <= USABLE:
                        height_options.append((gap, frame_h))
            if not height_options:
                continue
            gap, frame_h = random.choice(height_options)
            frame_w = random.randint(sw + 2, min(USABLE, sw + 5))
            total_h = sh + gap + frame_h
            top_margin = random.randint(0, USABLE - total_h)
            left_margin = random.randint(0, USABLE - frame_w)
            frame_c0 = slot_c0 + left_margin
            col_off = random.randint(0, frame_w - 2 - sw)
            if side == "top":
                outer_r0 = slot_r0 + top_margin
                frame_r0 = outer_r0 + sh + gap
                inner_r0 = frame_r0 + frame_h - 1 - sh
            else:
                frame_r0 = slot_r0 + top_margin
                outer_r0 = frame_r0 + frame_h + gap
                inner_r0 = frame_r0 + 1
            outer_c0 = frame_c0 + 1 + col_off
            inner_c0 = outer_c0

        frame_box = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)
        outer_box = (outer_r0, outer_c0, outer_r0 + sh - 1, outer_c0 + sw - 1)
        inner_box = (inner_r0, inner_c0, inner_r0 + sh - 1, inner_c0 + sw - 1)

        if complete:
            outer_visible = truth.copy()
            inner_outer = truth.copy()
        else:
            sampled = _sample_incomplete_views(truth, side)
            if sampled is None:
                continue
            outer_visible, inner_outer = sampled

        inner_visible = _mirror(inner_outer, side)
        return {
            "side": side,
            "complete": complete,
            "frame_box": frame_box,
            "outer_box": outer_box,
            "inner_box": inner_box,
            "truth": truth.copy(),
            "outer_visible": outer_visible,
            "inner_visible": inner_visible,
        }

    return None


def _render_scene(items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)

    for item in items:
        _draw_frame(inp, item["frame_box"])
        _draw_frame(out, item["frame_box"])
        _paste_mask(inp, item["outer_box"], item["outer_visible"], GRAY)
        _paste_mask(inp, item["inner_box"], item["inner_visible"], GRAY)
        _paste_mask(out, item["outer_box"], item["outer_visible"], GRAY)
        _paste_mask(out, item["inner_box"], item["inner_visible"], GRAY)

        truth = item["truth"]
        outer_missing = truth & ~item["outer_visible"]
        inner_truth = _mirror(truth, item["side"])
        inner_missing = inner_truth & ~item["inner_visible"]
        _paste_mask(out, item["outer_box"], outer_missing, YELLOW)
        _paste_mask(out, item["inner_box"], inner_missing, YELLOW)

    return inp, out


def _validate_scene(items: list[dict], inp: np.ndarray, expected: np.ndarray) -> bool:
    complete_count = sum(1 for item in items if item["complete"])
    if complete_count != 1:
        return False

    if not any((not item["complete"]) and item["side"] in H_SIDES for item in items):
        return False
    if not any((not item["complete"]) and item["side"] in V_SIDES for item in items):
        return False

    for item in items:
        truth = item["truth"]
        outer_visible = item["outer_visible"]
        inner_visible = item["inner_visible"]
        inner_outer = _mirror(inner_visible, item["side"])

        if item["complete"]:
            if not (
                np.array_equal(outer_visible, truth)
                and np.array_equal(inner_outer, truth)
            ):
                return False
            continue

        if np.array_equal(outer_visible, truth) or np.array_equal(inner_outer, truth):
            return False
        if not (
            _mask_connected(outer_visible)
            and _mask_connected(inner_outer)
            and _same_bbox(outer_visible, truth.shape)
            and _same_bbox(inner_outer, truth.shape)
        ):
            return False
        if not np.any(truth & ~outer_visible):
            return False
        if not np.any(truth & ~inner_outer):
            return False
        if np.array_equal(outer_visible | inner_outer, outer_visible):
            return False
        if np.array_equal(outer_visible | inner_outer, inner_outer):
            return False
        if np.array_equal(outer_visible | inner_visible, truth):
            return False

    try:
        parsed = sorted(_parse_scene(inp), key=lambda item: item["frame_box"])
    except AssertionError:
        return False
    latent = sorted(
        (
            {
                "frame_box": item["frame_box"],
                "inner_box": item["inner_box"],
                "outer_box": item["outer_box"],
                "side": item["side"],
            }
            for item in items
        ),
        key=lambda item: item["frame_box"],
    )
    if parsed != latent:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice((3, 4))
    slot_indices = random.sample(range(4), frame_count)

    roles = [
        {"side": random.choice(H_SIDES), "complete": False},
        {"side": random.choice(V_SIDES), "complete": False},
        {"side": random.choice(SIDES), "complete": True},
    ]
    if frame_count == 4:
        roles.append({"side": random.choice(SIDES), "complete": False})
    random.shuffle(roles)

    items: list[dict] = []
    for slot_index, role in zip(slot_indices, roles):
        item = _build_slot_item(slot_index, role["side"], role["complete"])
        if item is None:
            return None
        items.append(item)

    inp, expected = _render_scene(items)
    if not _validate_scene(items, inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-complement scene")
