from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
DIVIDER = 5
STAMP_COLORS = (1, 2, 3, 4)
SIGNATURE_COLORS = (6, 7, 8, 9)
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _find_divider(grid: np.ndarray) -> tuple[int, int]:
    h, _w = grid.shape
    hits: list[tuple[int, int]] = []
    for r in range(h - 1):
        if np.all(grid[r] == grid[r, 0]) and np.all(grid[r + 1] == grid[r, 0]) and int(grid[r, 0]) != BG:
            hits.append((int(r), int(grid[r, 0])))
    assert len(hits) == 1, "expected one two-row divider"
    return hits[0]


def _extract_stamps(upper: np.ndarray) -> list[dict[str, object]]:
    stamps: list[dict[str, object]] = []
    for coords in _components(upper != BG):
        r0, c0, r1, c1 = _bbox(coords)
        patch = upper[r0 : r1 + 1, c0 : c1 + 1].copy()
        counts = Counter(int(v) for v in patch.reshape(-1) if int(v) != BG)
        signature_colors = [color for color, count in counts.items() if count == 1]
        assert len(signature_colors) == 1, "each stamp must have one signature color"
        signature_color = int(signature_colors[0])
        signature_pos = np.argwhere(patch == signature_color)
        assert len(signature_pos) == 1
        sig_r, sig_c = map(int, signature_pos[0])
        stamps.append(
            {
                "patch": patch,
                "signature_color": signature_color,
                "signature_pos": (sig_r, sig_c),
            }
        )
    assert stamps, "expected at least one stamp"
    return stamps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recognize stamps above a two-row divider and stamp them onto the lower canvas.

    Each stamp has exactly one signature color that appears once inside that stamp and nowhere else
    in the stamp area. The matching color anchor on the lower canvas tells where to align the stamp:
    the signature pixel in the stamp must land on the anchor. Stamps are applied in anchor order,
    top-to-bottom then left-to-right. Background pixels inside a stamp are transparent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider_row, _divider_color = _find_divider(g)
    upper = g[:divider_row]
    lower = g[divider_row + 2 :]
    assert upper.size and lower.size

    stamps = _extract_stamps(upper)
    placed: list[tuple[tuple[int, int], dict[str, object]]] = []
    for stamp in stamps:
        color = int(stamp["signature_color"])
        anchors = np.argwhere(lower == color)
        assert len(anchors) == 1, "each signature color must map to one anchor"
        placed.append(((int(anchors[0, 0]), int(anchors[0, 1])), stamp))

    placed.sort(key=lambda item: item[0])
    out = lower.copy()
    h, w = lower.shape
    for (anchor_r, anchor_c), stamp in placed:
        patch = np.asarray(stamp["patch"], dtype=int)
        sig_r, sig_c = stamp["signature_pos"]
        top = int(anchor_r - sig_r)
        left = int(anchor_c - sig_c)
        ph, pw = patch.shape
        assert 0 <= top and 0 <= left and top + ph <= h and left + pw <= w
        target = out[top : top + ph, left : left + pw]
        mask = patch != BG
        target[mask] = patch[mask]
    return out


def _non_singleton_non_bg(patch: np.ndarray) -> bool:
    counts = Counter(int(v) for v in patch.reshape(-1) if int(v) != BG)
    return all(count >= 2 for count in counts.values())


def _motif_cells(h: int, w: int) -> list[tuple[int, int]]:
    styles = ["row", "col", "corners", "block", "pair"]
    while True:
        style = random.choice(styles)
        if style == "row":
            r = random.randrange(h)
            cells = [(r, c) for c in range(w)]
        elif style == "col":
            c = random.randrange(w)
            cells = [(r, c) for r in range(h)]
        elif style == "corners":
            cells = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        elif style == "block":
            if h >= 2 and w >= 2:
                r = random.randrange(h - 1)
                c = random.randrange(w - 1)
                cells = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
            elif h >= 2:
                r = random.randrange(h - 1)
                cells = [(r, 0), (r + 1, 0)]
            else:
                c = random.randrange(w - 1)
                cells = [(0, c), (0, c + 1)]
        else:
            if random.random() < 0.5:
                r = random.randrange(h)
                c0 = random.randrange(w - 1)
                cells = [(r, c0), (r, c0 + 1)]
            else:
                c = random.randrange(w)
                r0 = random.randrange(h - 1)
                cells = [(r0, c), (r0 + 1, c)]
        if 2 <= len(cells) <= max(2, h * w - 3):
            return cells


def _make_stamp(signature_color: int, force_hole: bool) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(200):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        base = random.choice(STAMP_COLORS)
        patch = np.full((h, w), base, dtype=int)

        accent1 = random.choice([c for c in STAMP_COLORS if c != base])
        for r, c in _motif_cells(h, w):
            patch[r, c] = accent1

        if random.random() < 0.4:
            accent_choices = [c for c in STAMP_COLORS if c not in {base, accent1}]
            if accent_choices:
                accent2 = random.choice(accent_choices)
                cells2 = [(r, c) for r, c in _motif_cells(h, w) if int(patch[r, c]) == base]
                if len(cells2) >= 2:
                    for r, c in cells2:
                        patch[r, c] = accent2

        hole_budget = random.randint(1 if force_hole else 0, 2 if min(h, w) >= 4 else 1)
        for _ in range(hole_budget):
            base_cells = [
                (r, c)
                for r in range(1, h - 1)
                for c in range(1, w - 1)
                if int(patch[r, c]) == base
            ]
            random.shuffle(base_cells)
            for r, c in base_cells:
                candidate = patch.copy()
                candidate[r, c] = BG
                if _is_connected(candidate != BG) and _non_singleton_non_bg(candidate):
                    patch = candidate
                    break

        base_cells = [(r, c) for r in range(h) for c in range(w) if int(patch[r, c]) == base]
        if len(base_cells) < 3:
            continue
        sig_r, sig_c = random.choice(base_cells)
        patch[sig_r, sig_c] = signature_color

        counts = Counter(int(v) for v in patch.reshape(-1) if int(v) != BG)
        if counts[signature_color] != 1:
            continue
        singletons = [color for color, count in counts.items() if count == 1]
        if singletons != [signature_color]:
            continue
        if not _is_connected(patch != BG):
            continue
        return patch, (sig_r, sig_c)
    raise RuntimeError("failed to build stamp")


def _box_from_pos(pos: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = pos
    h, w = shape
    return int(r), int(c), int(r + h - 1), int(c + w - 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _sample_free_position(
    shape: tuple[int, int],
    sig_pos: tuple[int, int],
    canvas_h: int,
    canvas_w: int,
    existing_boxes: list[tuple[int, int, int, int]],
    used_anchors: set[tuple[int, int]],
    gap: int,
) -> tuple[int, int] | None:
    h, w = shape
    for _ in range(200):
        top = random.randint(0, canvas_h - h)
        left = random.randint(0, canvas_w - w)
        box = _box_from_pos((top, left), shape)
        if any(_boxes_overlap(box, other, gap=gap) for other in existing_boxes):
            continue
        anchor = (int(top + sig_pos[0]), int(left + sig_pos[1]))
        if anchor in used_anchors:
            continue
        return top, left
    return None


def _sample_overlap_position(
    shape: tuple[int, int],
    sig_pos: tuple[int, int],
    target_box: tuple[int, int, int, int],
    target_pos: tuple[int, int],
    canvas_h: int,
    canvas_w: int,
    used_anchors: set[tuple[int, int]],
    forbidden: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    r_lo = max(0, target_box[0] - h + 1)
    r_hi = min(canvas_h - h, target_box[2])
    c_lo = max(0, target_box[1] - w + 1)
    c_hi = min(canvas_w - w, target_box[3])
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(250):
        top = random.randint(r_lo, r_hi)
        left = random.randint(c_lo, c_hi)
        if (top, left) == target_pos:
            continue
        box = _box_from_pos((top, left), shape)
        overlap = _intersection_area(box, target_box)
        if overlap == 0 or overlap >= min(shape[0] * shape[1], (target_box[2] - target_box[0] + 1) * (target_box[3] - target_box[1] + 1)):
            continue
        if any(_boxes_overlap(box, other, gap=0) for other in forbidden):
            continue
        anchor = (int(top + sig_pos[0]), int(left + sig_pos[1]))
        if anchor in used_anchors:
            continue
        return top, left
    return None


def _overlap_pairs(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(boxes[i], boxes[j], gap=0):
                pairs.append((i, j))
    return pairs


def _place_on_canvas(stamps: list[dict[str, object]], mode: str) -> tuple[int, int, list[tuple[int, int]]] | None:
    max_h = max(int(np.asarray(stamp["patch"]).shape[0]) for stamp in stamps)
    max_w = max(int(np.asarray(stamp["patch"]).shape[1]) for stamp in stamps)
    n = len(stamps)

    for _ in range(120):
        if mode == "none":
            canvas_h = random.randint(max(8, max_h + 3), min(16, max_h + n + 7))
            canvas_w = random.randint(max(12, max_w + 6), min(24, max_w + 4 * n + 6))
        elif mode == "pair":
            canvas_h = random.randint(max(8, max_h + 2), min(15, max_h + n + 5))
            canvas_w = random.randint(max(11, max_w + 4), min(22, max_w + 3 * n + 5))
        else:
            canvas_h = random.randint(max(8, max_h + 2), min(14, max_h + n + 4))
            canvas_w = random.randint(max(10, max_w + 3), min(20, max_w + 2 * n + 5))

        positions: list[tuple[int, int] | None] = [None] * n
        boxes: list[tuple[int, int, int, int] | None] = [None] * n
        used_anchors: set[tuple[int, int]] = set()

        order = list(range(n))
        random.shuffle(order)

        if mode == "none":
            ok = True
            for idx in order:
                patch = np.asarray(stamps[idx]["patch"])
                pos = _sample_free_position(
                    patch.shape,
                    stamps[idx]["signature_pos"],
                    canvas_h,
                    canvas_w,
                    [box for box in boxes if box is not None],
                    used_anchors,
                    gap=1,
                )
                if pos is None:
                    ok = False
                    break
                positions[idx] = pos
                boxes[idx] = _box_from_pos(pos, patch.shape)
                used_anchors.add((pos[0] + stamps[idx]["signature_pos"][0], pos[1] + stamps[idx]["signature_pos"][1]))
            if not ok:
                continue

        elif mode == "pair":
            a, b = order[:2]
            for idx in (a,):
                patch = np.asarray(stamps[idx]["patch"])
                pos = _sample_free_position(
                    patch.shape,
                    stamps[idx]["signature_pos"],
                    canvas_h,
                    canvas_w,
                    [],
                    used_anchors,
                    gap=0,
                )
                if pos is None:
                    break
                positions[idx] = pos
                boxes[idx] = _box_from_pos(pos, patch.shape)
                used_anchors.add((pos[0] + stamps[idx]["signature_pos"][0], pos[1] + stamps[idx]["signature_pos"][1]))
            if positions[a] is None or boxes[a] is None:
                continue

            patch_b = np.asarray(stamps[b]["patch"])
            pos_b = _sample_overlap_position(
                patch_b.shape,
                stamps[b]["signature_pos"],
                boxes[a],
                positions[a],
                canvas_h,
                canvas_w,
                used_anchors,
                [],
            )
            if pos_b is None:
                continue
            positions[b] = pos_b
            boxes[b] = _box_from_pos(pos_b, patch_b.shape)
            used_anchors.add((pos_b[0] + stamps[b]["signature_pos"][0], pos_b[1] + stamps[b]["signature_pos"][1]))

            ok = True
            for idx in order[2:]:
                patch = np.asarray(stamps[idx]["patch"])
                pos = _sample_free_position(
                    patch.shape,
                    stamps[idx]["signature_pos"],
                    canvas_h,
                    canvas_w,
                    [box for box in boxes if box is not None],
                    used_anchors,
                    gap=1,
                )
                if pos is None:
                    ok = False
                    break
                positions[idx] = pos
                boxes[idx] = _box_from_pos(pos, patch.shape)
                used_anchors.add((pos[0] + stamps[idx]["signature_pos"][0], pos[1] + stamps[idx]["signature_pos"][1]))
            if not ok:
                continue

        else:
            trio = order[:3]
            a, b, c = trio

            patch_a = np.asarray(stamps[a]["patch"])
            pos_a = _sample_free_position(
                patch_a.shape,
                stamps[a]["signature_pos"],
                canvas_h,
                canvas_w,
                [],
                used_anchors,
                gap=0,
            )
            if pos_a is None:
                continue
            positions[a] = pos_a
            boxes[a] = _box_from_pos(pos_a, patch_a.shape)
            used_anchors.add((pos_a[0] + stamps[a]["signature_pos"][0], pos_a[1] + stamps[a]["signature_pos"][1]))

            patch_b = np.asarray(stamps[b]["patch"])
            pos_b = _sample_overlap_position(
                patch_b.shape,
                stamps[b]["signature_pos"],
                boxes[a],
                positions[a],
                canvas_h,
                canvas_w,
                used_anchors,
                [],
            )
            if pos_b is None:
                continue
            positions[b] = pos_b
            boxes[b] = _box_from_pos(pos_b, patch_b.shape)
            used_anchors.add((pos_b[0] + stamps[b]["signature_pos"][0], pos_b[1] + stamps[b]["signature_pos"][1]))

            patch_c = np.asarray(stamps[c]["patch"])
            pos_c = _sample_overlap_position(
                patch_c.shape,
                stamps[c]["signature_pos"],
                boxes[b],
                positions[b],
                canvas_h,
                canvas_w,
                used_anchors,
                [boxes[a]],
            )
            if pos_c is None:
                continue
            positions[c] = pos_c
            boxes[c] = _box_from_pos(pos_c, patch_c.shape)
            used_anchors.add((pos_c[0] + stamps[c]["signature_pos"][0], pos_c[1] + stamps[c]["signature_pos"][1]))

            ok = True
            for idx in order[3:]:
                patch = np.asarray(stamps[idx]["patch"])
                pos = _sample_free_position(
                    patch.shape,
                    stamps[idx]["signature_pos"],
                    canvas_h,
                    canvas_w,
                    [box for box in boxes if box is not None],
                    used_anchors,
                    gap=1,
                )
                if pos is None:
                    ok = False
                    break
                positions[idx] = pos
                boxes[idx] = _box_from_pos(pos, patch.shape)
                used_anchors.add((pos[0] + stamps[idx]["signature_pos"][0], pos[1] + stamps[idx]["signature_pos"][1]))
            if not ok:
                continue

        final_boxes = [box for box in boxes if box is not None]
        assert len(final_boxes) == n
        pairs = _overlap_pairs(final_boxes)
        if mode == "none" and pairs:
            continue
        if mode == "pair" and len(pairs) != 1:
            continue
        if mode == "chain":
            covered = {idx for pair in pairs for idx in pair}
            if len(pairs) < 2 or len(covered) < 3:
                continue

        return canvas_h, canvas_w, [pos for pos in positions if pos is not None]

    return None


def _place_upper(stamps: list[dict[str, object]], width: int) -> tuple[int, list[tuple[int, int]]] | None:
    max_h = max(int(np.asarray(stamp["patch"]).shape[0]) for stamp in stamps)
    for _ in range(120):
        upper_h = random.randint(max(7, max_h + 1), min(12, max_h + 6))
        positions: list[tuple[int, int] | None] = [None] * len(stamps)
        boxes: list[tuple[int, int, int, int]] = []
        order = list(range(len(stamps)))
        random.shuffle(order)
        ok = True
        for idx in order:
            patch = np.asarray(stamps[idx]["patch"])
            pos = _sample_free_position(
                patch.shape,
                (0, 0),
                upper_h,
                width,
                boxes,
                set(),
                gap=1,
            )
            if pos is None:
                ok = False
                break
            positions[idx] = pos
            boxes.append(_box_from_pos(pos, patch.shape))
        if ok:
            return upper_h, [pos for pos in positions if pos is not None]
    return None


def _render_canvas(
    stamps: list[dict[str, object]],
    canvas_h: int,
    canvas_w: int,
    positions: list[tuple[int, int]],
) -> np.ndarray:
    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    order: list[tuple[tuple[int, int], int]] = []
    for idx, (top, left) in enumerate(positions):
        sig_r, sig_c = stamps[idx]["signature_pos"]
        anchor = (int(top + sig_r), int(left + sig_c))
        canvas[anchor] = int(stamps[idx]["signature_color"])
        order.append((anchor, idx))

    order.sort(key=lambda item: item[0])
    for _anchor, idx in order:
        top, left = positions[idx]
        patch = np.asarray(stamps[idx]["patch"], dtype=int)
        mask = patch != BG
        region = canvas[top : top + patch.shape[0], left : left + patch.shape[1]]
        region[mask] = patch[mask]
    return canvas


def _assemble_input(
    stamps: list[dict[str, object]],
    upper_h: int,
    width: int,
    canvas_h: int,
    lower_positions: list[tuple[int, int]],
    upper_positions: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((upper_h + 2 + canvas_h, width), BG, dtype=int)
    for stamp, (top, left) in zip(stamps, upper_positions):
        patch = np.asarray(stamp["patch"], dtype=int)
        grid[top : top + patch.shape[0], left : left + patch.shape[1]] = patch

    grid[upper_h : upper_h + 2, :] = DIVIDER

    lower = grid[upper_h + 2 :]
    for stamp, (top, left) in zip(stamps, lower_positions):
        sig_r, sig_c = stamp["signature_pos"]
        lower[top + sig_r, left + sig_c] = int(stamp["signature_color"])
    return grid


def _scene_attempt() -> np.ndarray | None:
    n_stamps = random.choices([2, 3, 4], weights=[1, 2, 2])[0]
    mode_choices = ["none", "pair"] if n_stamps == 2 else ["none", "pair", "chain"]
    overlap_mode = random.choice(mode_choices)

    force_hole_scene = random.random() < 0.5
    hole_idx = random.randrange(n_stamps) if force_hole_scene else -1

    signature_palette = random.sample(list(SIGNATURE_COLORS), n_stamps)
    stamps: list[dict[str, object]] = []
    for i, signature_color in enumerate(signature_palette):
        patch, sig_pos = _make_stamp(signature_color, force_hole=i == hole_idx)
        stamps.append(
            {
                "patch": patch,
                "signature_color": int(signature_color),
                "signature_pos": sig_pos,
            }
        )

    placed = _place_on_canvas(stamps, overlap_mode)
    if placed is None:
        return None
    canvas_h, canvas_w, lower_positions = placed

    width = canvas_w
    upper = _place_upper(stamps, width)
    while upper is None and width < 28:
        width += 1
        upper = _place_upper(stamps, width)
    if upper is None:
        return None
    upper_h, upper_positions = upper

    inp = _assemble_input(stamps, upper_h, width, canvas_h, lower_positions, upper_positions)
    expected = _render_canvas(stamps, canvas_h, width, lower_positions)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _scene_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-and-canvas puzzle")
