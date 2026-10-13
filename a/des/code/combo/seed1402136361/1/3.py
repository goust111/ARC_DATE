from __future__ import annotations

import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_MAX_ATTEMPTS = 400
GEN_POSITION_ATTEMPTS = 250
GEN_SHAPE_ATTEMPTS = 200
GEN_HEIGHT_RANGE = (9, 16)
GEN_WIDTH_RANGE = (13, 20)
GEN_TARGET_PAIR_COUNT = 4
GEN_DUPLICATE_COLOR_PROB = 0.4


def _most_common_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    colors, counts = np.unique(flat, return_counts=True)
    return int(colors[np.argmax(counts)])


def _control_pairs(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    row = [int(v) for v in g[0]]
    control = [color for color in row if color != bg]
    assert control and len(control) % 2 == 0, "top control row must contain an even number of non-bg cells"
    assert len(control) == len(set(control)), "control colors must be unique"
    return [(control[i], control[i + 1]) for i in range(0, len(control), 2)]


def _components(canvas: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = canvas.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(canvas[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(canvas[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            objects.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return objects


def _tight_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _rotation_offset(h: int, w: int, turns: int) -> tuple[int, int]:
    if turns % 2 == 0:
        return 0, 0
    assert h % 2 == w % 2, "90/270 degree rotation around the bbox center needs equal bbox parity"
    return (h - w) // 2, (w - h) // 2


def _rotated_placement(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    turns: int,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, int, int]:
    turns %= 4
    rotated = np.rot90(mask, -turns)
    r0, c0, _r1, _c1 = bbox
    h, w = mask.shape
    rh, rw = rotated.shape
    dr0, dc0 = _rotation_offset(h, w, turns)
    nr0 = r0 + dr0
    nc0 = c0 + dc0
    H, W = canvas_shape
    assert 0 <= nr0 and 0 <= nc0 and nr0 + rh <= H and nc0 + rw <= W
    return rotated, int(nr0), int(nc0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read color pairs from the top control row and rotate matching foreground objects on the canvas.

    Input structure
    - The dominant background color fills the main canvas.
    - The top row contains a sequence of unique control colors; consecutive pairs encode instructions.
    - Below the control row, the main canvas contains separated monochrome 4-connected objects.

    Rule
    - In each control pair, the first color selects all objects of that color.
    - The pair index k from left to right determines a clockwise rotation of k x 90 degrees.
    - Rotate each matching object around the center of its tight bounding box.
    - Remove the control row from the output.

    Key insight
    - The two colors in each pair jointly encode the action: the first color identifies the object color,
      while the pair's position in the sequence encodes the rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2
    canvas = g[1:, :]
    bg = _most_common_color(canvas)
    pairs = _control_pairs(g, bg)

    target_to_turns: dict[int, int] = {}
    seen_codes: set[int] = set()
    for index, (target, code) in enumerate(pairs, start=1):
        assert target != bg and code != bg
        assert code not in seen_codes
        seen_codes.add(code)
        assert target not in target_to_turns
        target_to_turns[target] = index % 4

    canvas_colors = set(map(int, np.unique(canvas))) - {bg}
    assert canvas_colors <= set(target_to_turns), "every canvas color must be keyed by the control row"
    assert seen_codes.isdisjoint(canvas_colors), "second colors act only as rotation codes"

    out = np.full_like(canvas, bg)
    for comp in _components(canvas, bg):
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        bbox = tuple(map(int, comp["bbox"]))
        mask = _tight_mask(coords, bbox)
        rotated, nr0, nc0 = _rotated_placement(mask, bbox, target_to_turns[color], canvas.shape)

        for rr, cc in np.argwhere(rotated):
            ar = nr0 + int(rr)
            ac = nc0 + int(cc)
            assert out[ar, ac] == bg, "rotated objects must remain disjoint"
            out[ar, ac] = color

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _rotation_visible(mask: np.ndarray, turns: int) -> bool:
    turns %= 4
    if turns == 0:
        return True
    rotated = np.rot90(mask, -turns)
    return mask.shape != rotated.shape or not np.array_equal(mask, rotated)


def _random_connected_mask() -> np.ndarray:
    box_h = random.randint(3, 6)
    box_w_choices = [w for w in range(3, 7) if w % 2 == box_h % 2]
    box_w = random.choice(box_w_choices)
    area = random.randint(4, min(box_h * box_w - 1, 10))

    mask = np.zeros((box_h, box_w), dtype=bool)
    start = (random.randrange(box_h), random.randrange(box_w))
    mask[start] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        mask[random.choice(tuple(frontier))] = True

    return _crop_mask(mask)


def _random_object_mask(turns: int) -> np.ndarray:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        mask = _random_connected_mask()
        h, w = mask.shape
        if min(h, w) < 2:
            continue
        if h % 2 != w % 2:
            continue
        if int(mask.sum()) < 4:
            continue
        if not _rotation_visible(mask, turns):
            continue
        return mask
    raise RuntimeError("failed to generate a usable object mask")


def _candidate_ranges(
    canvas_shape: tuple[int, int],
    mask_shape: tuple[int, int],
    turns: int,
) -> tuple[range, range]:
    H, W = canvas_shape
    h, w = mask_shape
    rotated = np.rot90(np.zeros((h, w), dtype=bool), -turns)
    rh, rw = rotated.shape
    dr0, dc0 = _rotation_offset(h, w, turns)

    r_min = max(0, -dr0)
    r_max = min(H - h, H - rh - dr0)
    c_min = max(0, -dc0)
    c_max = min(W - w, W - rw - dc0)
    assert r_min <= r_max and c_min <= c_max
    return range(r_min, r_max + 1), range(c_min, c_max + 1)


def _place_specs(
    canvas_shape: tuple[int, int],
    specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    H, W = canvas_shape
    input_occ = np.zeros((H, W), dtype=bool)
    output_occ = np.zeros((H, W), dtype=bool)
    placed: list[dict[str, object]] = []

    order = sorted(
        specs,
        key=lambda spec: max(int(np.asarray(spec["mask"]).sum()), int(np.asarray(spec["rotated_mask"]).sum())),
        reverse=True,
    )

    for spec in order:
        mask = np.asarray(spec["mask"], dtype=bool)
        rotated_mask = np.asarray(spec["rotated_mask"], dtype=bool)
        h, w = mask.shape
        rh, rw = rotated_mask.shape
        dr0 = int(spec["dr0"])
        dc0 = int(spec["dc0"])
        row_range, col_range = _candidate_ranges(canvas_shape, mask.shape, int(spec["turns"]))

        success = False
        for _ in range(GEN_POSITION_ATTEMPTS):
            r0 = random.choice(tuple(row_range))
            c0 = random.choice(tuple(col_range))
            orig_abs = np.zeros((H, W), dtype=bool)
            rot_abs = np.zeros((H, W), dtype=bool)
            orig_abs[r0 : r0 + h, c0 : c0 + w] = mask
            rr0 = r0 + dr0
            cc0 = c0 + dc0
            rot_abs[rr0 : rr0 + rh, cc0 : cc0 + rw] = rotated_mask

            if bool((orig_abs & _dilate8(input_occ)).any()):
                continue
            if bool((rot_abs & _dilate8(output_occ)).any()):
                continue

            input_occ |= orig_abs
            output_occ |= rot_abs
            placed.append(
                {
                    "color": int(spec["color"]),
                    "turns": int(spec["turns"]),
                    "mask": mask,
                    "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                }
            )
            success = True
            break

        if not success:
            return None

    return placed


def _object_specs(target_to_turns: dict[int, int]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for color, turns in target_to_turns.items():
        count = 2 if random.random() < GEN_DUPLICATE_COLOR_PROB else 1
        for _ in range(count):
            mask = _random_object_mask(turns)
            rotated_mask = np.rot90(mask, -turns)
            dr0, dc0 = _rotation_offset(mask.shape[0], mask.shape[1], turns)
            specs.append(
                {
                    "color": int(color),
                    "turns": int(turns),
                    "mask": mask,
                    "rotated_mask": rotated_mask,
                    "dr0": int(dr0),
                    "dc0": int(dc0),
                }
            )
    return specs


def _render_canvas(canvas_shape: tuple[int, int], placed: list[dict[str, object]], bg: int) -> np.ndarray:
    H, W = canvas_shape
    canvas = np.full((H, W), bg, dtype=int)
    for obj in placed:
        color = int(obj["color"])
        mask = np.asarray(obj["mask"], dtype=bool)
        r0, c0, _r1, _c1 = map(int, obj["bbox"])
        patch = canvas[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        assert not bool((patch != bg)[mask].any())
        patch[mask] = color
    return canvas


def _control_row(width: int, ordered_pairs: list[tuple[int, int]], bg: int) -> np.ndarray:
    colors = [color for pair in ordered_pairs for color in pair]
    row = np.full(width, bg, dtype=int)
    start = random.randint(0, width - len(colors))
    row[start : start + len(colors)] = np.array(colors, dtype=int)
    return row


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GEN_HEIGHT_RANGE)
    W = random.randint(*GEN_WIDTH_RANGE)

    palette = list(range(10))
    bg = random.choice(palette)
    usable = [color for color in palette if color != bg]
    chosen = random.sample(usable, 2 * GEN_TARGET_PAIR_COUNT)
    target_colors = chosen[:GEN_TARGET_PAIR_COUNT]
    code_colors = chosen[GEN_TARGET_PAIR_COUNT:]
    random.shuffle(target_colors)

    ordered_pairs = [(target_colors[i], code_colors[i]) for i in range(GEN_TARGET_PAIR_COUNT)]
    target_to_turns = {target: (i + 1) % 4 for i, (target, _code) in enumerate(ordered_pairs)}

    try:
        specs = _object_specs(target_to_turns)
    except RuntimeError:
        return None

    placed = _place_specs((H, W), specs)
    if placed is None:
        return None

    canvas = _render_canvas((H, W), placed, bg)
    row = _control_row(W, ordered_pairs, bg)
    grid = np.vstack([row[None, :], canvas])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != canvas.shape:
        return None
    if not all(np.any(canvas == target) for target in target_colors):
        return None

    # Each non-identity branch must visibly change at least one object.
    for target, turns in target_to_turns.items():
        if turns == 0:
            continue
        before = canvas == target
        after = out == target
        if np.array_equal(before, after):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
