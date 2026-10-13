import random
from collections import deque

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "right", "bottom", "left")
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _marker_side(marker_pos: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = marker_pos
    h, w = shape
    sides: list[str] = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, (marker_pos, shape, sides)
    return sides[0]


def _find_template(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, int, tuple[int, int, int, int]] | None = None

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                np.all(patch[0, :] == color)
                and np.all(patch[-1, :] == color)
                and np.all(patch[:, 0] == color)
                and np.all(patch[:, -1] == color)
            ):
                continue
            if not np.any(patch == bg):
                continue
            area = len(comp)
            bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            cand = (area, bbox_area, (r0, c0, r1, c1))
            if best is None or cand[:2] > best[:2]:
                best = cand

    assert best is not None
    area, _bbox_area, bbox = best
    assert area > 0
    return int(grid[bbox[0], bbox[1]]), bbox


def _find_slots(
    grid: np.ndarray,
    bg: int,
    template_color: int,
    template_bbox: tuple[int, int, int, int],
) -> list[dict]:
    h, w = grid.shape
    tr0, tc0, tr1, tc1 = template_bbox
    slots: list[dict] = []

    for comp in _components(grid == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if not (tr0 < r0 <= r1 < tr1 and tc0 < c0 <= c1 < tc1):
            continue
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if r0 - 1 < tr0 or c0 - 1 < tc0 or r1 + 1 > tr1 or c1 + 1 > tc1:
            continue

        marker: tuple[str, int] | None = None
        for cc in range(c0, c1 + 1):
            val = int(grid[r0 - 1, cc])
            if val != template_color:
                assert marker is None
                marker = ("top", val)
        for cc in range(c0, c1 + 1):
            val = int(grid[r1 + 1, cc])
            if val != template_color:
                assert marker is None
                marker = ("bottom", val)
        for rr in range(r0, r1 + 1):
            val = int(grid[rr, c0 - 1])
            if val != template_color:
                assert marker is None
                marker = ("left", val)
        for rr in range(r0, r1 + 1):
            val = int(grid[rr, c1 + 1])
            if val != template_color:
                assert marker is None
                marker = ("right", val)

        assert marker is not None
        side, marker_color = marker
        slots.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
                "side": side,
                "marker_color": marker_color,
            }
        )

    slots.sort(key=lambda item: item["bbox"])
    return slots


def _find_pieces(
    grid: np.ndarray,
    bg: int,
    template_bbox: tuple[int, int, int, int],
) -> list[dict]:
    tr0, tc0, tr1, tc1 = template_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[tr0 : tr1 + 1, tc0 : tc1 + 1] = False
    pieces: list[dict] = []

    for comp in _components((grid != bg) & outside):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert not np.any(patch == bg)
        colors, counts = np.unique(patch, return_counts=True)
        assert len(colors) == 2
        marker_idx = int(np.argmin(counts))
        assert int(counts[marker_idx]) == 1
        marker_color = int(colors[marker_idx])
        base_color = int(colors[1 - marker_idx])
        marker_pos = tuple(map(int, np.argwhere(patch == marker_color)[0]))
        pieces.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": patch.shape,
                "patch": patch,
                "base_color": base_color,
                "marker_color": marker_color,
                "side": _marker_side(marker_pos, patch.shape),
            }
        )

    pieces.sort(key=lambda item: item["bbox"])
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """Embed each outside rectangle into the uniquely size-matched slot and rotate it to align marker sides."""
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    template_color, template_bbox = _find_template(g, bg)
    slots = _find_slots(g, bg, template_color, template_bbox)
    pieces = _find_pieces(g, bg, template_bbox)

    assert slots and len(slots) == len(pieces)

    slot_by_size: dict[tuple[int, int], dict] = {}
    for slot in slots:
        key = tuple(sorted(slot["shape"]))
        assert key not in slot_by_size
        slot_by_size[key] = slot

    out = g.copy()
    for piece in pieces:
        r0, c0, r1, c1 = piece["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    used_slots: set[tuple[int, int, int, int]] = set()
    for piece in pieces:
        key = tuple(sorted(piece["shape"]))
        assert key in slot_by_size
        slot = slot_by_size[key]
        slot_key = slot["bbox"]
        assert slot_key not in used_slots
        used_slots.add(slot_key)

        aligned: np.ndarray | None = None
        for k in range(4):
            rotated = np.rot90(piece["patch"], k)
            if rotated.shape != slot["shape"]:
                continue
            marker_pos = tuple(map(int, np.argwhere(rotated == piece["marker_color"])[0]))
            if _marker_side(marker_pos, rotated.shape) != slot["side"]:
                continue
            aligned = rotated.copy()
            aligned[marker_pos] = slot["marker_color"]
            break

        assert aligned is not None
        r0, c0, r1, c1 = slot["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = aligned

    return out


def _piece_marker_coord(h: int, w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(1, w - 2)
    if side == "bottom":
        return h - 1, random.randint(1, w - 2)
    if side == "left":
        return random.randint(1, h - 2), 0
    if side == "right":
        return random.randint(1, h - 2), w - 1
    raise AssertionError(side)


def _sample_sides(count: int) -> list[str]:
    for _ in range(30):
        sides = [random.choice(SIDES) for _ in range(count)]
        if len(set(sides)) >= min(2, count):
            return sides
    return [random.choice(SIDES) for _ in range(count)]


def _sample_slot_shapes(count: int) -> list[tuple[int, int]]:
    classes = [(a, b) for a in range(3, 7) for b in range(a, 7)]
    picked = random.sample(classes, count)
    shapes: list[tuple[int, int]] = []
    for a, b in picked:
        if a == b or random.random() < 0.5:
            shapes.append((a, b))
        else:
            shapes.append((b, a))
    return shapes


def _place_slots(template_shape: tuple[int, int], slot_shapes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    th, tw = template_shape
    order = sorted(range(len(slot_shapes)), key=lambda i: slot_shapes[i][0] * slot_shapes[i][1], reverse=True)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(slot_shapes)
    occupied = np.zeros((th, tw), dtype=bool)

    for idx in order:
        h, w = slot_shapes[idx]
        choices: list[tuple[int, int]] = []
        for r0 in range(1, th - h):
            for c0 in range(1, tw - w):
                if np.any(occupied[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1]):
                    continue
                choices.append((r0, c0))
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        placed[idx] = (r0, c0, r0 + h - 1, c0 + w - 1)
        occupied[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True

    return [box for box in placed if box is not None]


def _place_pieces(
    canvas_shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    tr0, tc0, tr1, tc1 = template_bbox
    occupied = np.zeros((h, w), dtype=bool)
    occupied[max(0, tr0 - 1) : min(h, tr1 + 2), max(0, tc0 - 1) : min(w, tc1 + 2)] = True

    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    placed: list[tuple[int, int] | None] = [None] * len(patches)

    for idx in order:
        ph, pw = patches[idx].shape
        choices: list[tuple[int, int]] = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r0 + ph + 1)
                cc1 = min(w, c0 + pw + 1)
                if np.any(occupied[rr0:rr1, cc0:cc1]):
                    continue
                choices.append((r0, c0))
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        placed[idx] = (r0, c0)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc1 = min(w, c0 + pw + 1)
        occupied[rr0:rr1, cc0:cc1] = True

    return [pos for pos in placed if pos is not None]


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = random.randint(2, 4)
    slot_shapes = _sample_slot_shapes(piece_count)
    sides = _sample_sides(piece_count)

    max_h = max(h for h, _ in slot_shapes)
    max_w = max(w for _, w in slot_shapes)
    expanded_area = sum((h + 2) * (w + 2) for h, w in slot_shapes)
    template_low = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(expanded_area))) + 1)
    if template_low > 21:
        return None
    template_h = random.randint(template_low, min(21, template_low + 4))
    template_w = random.randint(template_low, min(21, template_low + 4))
    slot_boxes = _place_slots((template_h, template_w), slot_shapes)
    if slot_boxes is None:
        return None

    if piece_count > len(PALETTE):
        return None
    base_colors = random.sample(PALETTE, piece_count)
    remaining = [c for c in PALETTE if c not in base_colors]
    if len(remaining) < piece_count:
        return None
    slot_marker_colors = random.sample(remaining, piece_count)
    piece_marker_colors = []
    for base_color in base_colors:
        choices = [c for c in PALETTE if c != base_color]
        piece_marker_colors.append(random.choice(choices))

    template = np.full((template_h, template_w), TEMPLATE_COLOR, dtype=int)
    final_fills: list[np.ndarray] = []
    piece_patches: list[np.ndarray] = []

    for idx, (box, side, base_color, slot_marker_color, piece_marker_color) in enumerate(
        zip(slot_boxes, sides, base_colors, slot_marker_colors, piece_marker_colors)
    ):
        r0, c0, r1, c1 = box
        slot_h = r1 - r0 + 1
        slot_w = c1 - c0 + 1
        template[r0 : r1 + 1, c0 : c1 + 1] = BG

        if side == "top":
            mr = r0 - 1
            mc = random.randint(c0, c1)
        elif side == "bottom":
            mr = r1 + 1
            mc = random.randint(c0, c1)
        elif side == "left":
            mr = random.randint(r0, r1)
            mc = c0 - 1
        else:
            mr = random.randint(r0, r1)
            mc = c1 + 1
        template[mr, mc] = slot_marker_color

        fill = np.full((slot_h, slot_w), base_color, dtype=int)
        pr, pc = _piece_marker_coord(slot_h, slot_w, side)
        fill[pr, pc] = slot_marker_color
        final_fills.append(fill)

        piece = np.full((slot_h, slot_w), base_color, dtype=int)
        piece[pr, pc] = piece_marker_color
        piece_patches.append(np.rot90(piece, random.randrange(4)))

    canvas_low_h = max(template_h + 8, 22)
    canvas_low_w = max(template_w + 8, 22)
    canvas_h = random.randint(canvas_low_h, min(30, canvas_low_h + 5))
    canvas_w = random.randint(canvas_low_w, min(30, canvas_low_w + 5))
    if canvas_h <= template_h + 2 or canvas_w <= template_w + 2:
        return None

    template_r0 = random.randint(1, canvas_h - template_h - 1)
    template_c0 = random.randint(1, canvas_w - template_w - 1)
    template_bbox = (template_r0, template_c0, template_r0 + template_h - 1, template_c0 + template_w - 1)

    piece_positions = _place_pieces((canvas_h, canvas_w), template_bbox, piece_patches)
    if piece_positions is None:
        return None

    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    inp[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = template

    for patch, (r0, c0) in zip(piece_patches, piece_positions):
        ph, pw = patch.shape
        inp[r0 : r0 + ph, c0 : c0 + pw] = patch

    expected = inp.copy()
    for patch, (r0, c0) in zip(piece_patches, piece_positions):
        ph, pw = patch.shape
        expected[r0 : r0 + ph, c0 : c0 + pw] = BG

    for fill, box in zip(final_fills, slot_boxes):
        r0, c0, r1, c1 = box
        expected[
            template_r0 + r0 : template_r0 + r1 + 1,
            template_c0 + c0 : template_c0 + c1 + 1,
        ] = fill

    if _most_common_color(inp) != BG:
        return None
    if np.array_equal(inp, expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_attempt()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
