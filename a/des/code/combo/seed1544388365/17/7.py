from __future__ import annotations

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS = ("up", "down", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == int(bg):
                continue

            color = int(g[r0, c0])
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in _NEIGHBORS4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if seen[rr, cc] or int(g[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    stack.append((rr, cc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r1, r2 = min(rows), max(rows)
            c1, c2 = min(cols), max(cols)
            mask = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=bool)
            for r, c in coords:
                mask[int(r - r1), int(c - c1)] = True

            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r1), int(c1), int(r2), int(c2)),
                    "mask": mask,
                    "area": int(len(coords)),
                }
            )

    return comps


def _container_mask(shape: tuple[int, int], direction: str) -> np.ndarray | None:
    h, w = map(int, shape)
    if direction in ("up", "down") and (h < 2 or w < 3):
        return None
    if direction in ("left", "right") and (h < 3 or w < 2):
        return None

    mask = np.zeros((h, w), dtype=bool)
    if direction == "up":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif direction == "down":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif direction == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    else:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    return mask


def _container_spec(mask: np.ndarray) -> dict | None:
    h, w = map(int, mask.shape)
    matches: list[dict] = []

    for direction in _DIRS:
        expected = _container_mask((h, w), direction)
        if expected is None or not np.array_equal(mask, expected):
            continue

        if direction == "up":
            matches.append(
                {
                    "direction": direction,
                    "slot_top": 0,
                    "slot_left": 1,
                    "slot_h": int(h - 1),
                    "slot_w": int(w - 2),
                }
            )
        elif direction == "down":
            matches.append(
                {
                    "direction": direction,
                    "slot_top": 1,
                    "slot_left": 1,
                    "slot_h": int(h - 1),
                    "slot_w": int(w - 2),
                }
            )
        elif direction == "left":
            matches.append(
                {
                    "direction": direction,
                    "slot_top": 1,
                    "slot_left": 0,
                    "slot_h": int(h - 2),
                    "slot_w": int(w - 1),
                }
            )
        else:
            matches.append(
                {
                    "direction": direction,
                    "slot_top": 1,
                    "slot_left": 1,
                    "slot_h": int(h - 2),
                    "slot_w": int(w - 1),
                }
            )

    if len(matches) != 1:
        return None
    return matches[0]


def _match_pairs(pieces: list[dict], containers: list[dict]) -> dict[int, int]:
    candidates: dict[int, list[int]] = {}
    for ci, container in enumerate(containers):
        slot_h = int(container["slot_h"])
        slot_w = int(container["slot_w"])
        slot_area = int(slot_h * slot_w)
        fit: list[int] = []
        for pi, piece in enumerate(pieces):
            piece_h, piece_w = map(int, piece["mask"].shape)
            if int(piece["area"]) != slot_area:
                continue
            if (piece_h, piece_w) == (slot_h, slot_w) or (piece_w, piece_h) == (slot_h, slot_w):
                fit.append(int(pi))
        assert fit, "every container must have at least one matching piece"
        candidates[int(ci)] = fit

    order = sorted(
        range(len(containers)),
        key=lambda ci: (
            len(candidates[int(ci)]),
            int(containers[int(ci)]["slot_h"]) * int(containers[int(ci)]["slot_w"]),
            int(containers[int(ci)]["bbox"][0]),
            int(containers[int(ci)]["bbox"][1]),
        ),
    )

    assign: dict[int, int] = {}
    used: set[int] = set()

    def search(k: int) -> bool:
        if k == len(order):
            return True
        ci = int(order[k])
        for pi in candidates[ci]:
            if int(pi) in used:
                continue
            used.add(int(pi))
            assign[ci] = int(pi)
            if search(k + 1):
                return True
            used.remove(int(pi))
            del assign[ci]
        return False

    assert search(0), "piece-container matching must be solvable"
    return assign


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group solid colored rectangles into matching open-frame containers by area and orientation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    pieces: list[dict] = []
    containers: list[dict] = []

    for comp in _components(g, bg):
        spec = _container_spec(comp["mask"])
        if spec is not None:
            r0, c0, _r1, _c1 = map(int, comp["bbox"])
            containers.append(
                {
                    "color": int(comp["color"]),
                    "bbox": tuple(map(int, comp["bbox"])),
                    "direction": spec["direction"],
                    "slot_h": int(spec["slot_h"]),
                    "slot_w": int(spec["slot_w"]),
                    "slot_r0": int(r0 + spec["slot_top"]),
                    "slot_c0": int(c0 + spec["slot_left"]),
                }
            )
            continue

        if bool(comp["mask"].all()):
            pieces.append(comp)
            continue

        raise AssertionError("foreground components must be either solid rectangles or open containers")

    assert containers, "at least one container is required"
    assert len(pieces) == len(containers), "pieces and containers must pair one-to-one"

    assign = _match_pairs(pieces, containers)
    out = g.copy()

    for piece in pieces:
        for r, c in piece["coords"]:
            out[int(r), int(c)] = int(bg)

    for ci, pi in assign.items():
        container = containers[int(ci)]
        piece = pieces[int(pi)]
        r0 = int(container["slot_r0"])
        c0 = int(container["slot_c0"])
        h = int(container["slot_h"])
        w = int(container["slot_w"])
        out[r0 : r0 + h, c0 : c0 + w] = int(piece["color"])

    return out


# --------------- generator ---------------

_GEN_CONTAINER_COLOR = 8
_GEN_SLOT_SIDE_MIN = 2
_GEN_SLOT_SIDE_MAX = 5
_GEN_OBJECT_COUNT_RANGE = (2, 6)
_GEN_EDGE_MARGIN = 1
_GEN_PADDING = 1
_GEN_CANVAS_MIN = 16
_GEN_CANVAS_EXTRA = 8
_GEN_PLACE_TRIES = 300
_GEN_ATTEMPTS = 400


def _build_container_patch(
    slot_h: int, slot_w: int, direction: str, bg: int, container_color: int
) -> tuple[np.ndarray, tuple[int, int]]:
    if direction in ("up", "down"):
        h = int(slot_h + 1)
        w = int(slot_w + 2)
    else:
        h = int(slot_h + 2)
        w = int(slot_w + 1)

    mask = _container_mask((h, w), direction)
    assert mask is not None
    patch = np.full((h, w), int(bg), dtype=int)
    patch[mask] = int(container_color)

    if direction == "up":
        slot_origin = (0, 1)
    elif direction == "down":
        slot_origin = (1, 1)
    elif direction == "left":
        slot_origin = (1, 0)
    else:
        slot_origin = (1, 1)
    return patch, slot_origin


def _build_piece_patch(height: int, width: int, bg: int, color: int) -> np.ndarray:
    patch = np.full((int(height), int(width)), int(bg), dtype=int)
    patch[:, :] = int(color)
    return patch


def _can_place_rect(
    occupied: np.ndarray, top: int, left: int, height: int, width: int, padding: int
) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, int(top - padding))
    r1 = min(h, int(top + height + padding))
    c0 = max(0, int(left - padding))
    c1 = min(w, int(left + width + padding))
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_rectangles(
    rng: np.random.Generator, canvas_shape: tuple[int, int], rectangles: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    h, w = map(int, canvas_shape)
    occupied = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(rectangles)
    order = sorted(
        range(len(rectangles)),
        key=lambda idx: (
            -(rectangles[int(idx)][0] * rectangles[int(idx)][1]),
            -max(rectangles[int(idx)]),
        ),
    )

    for idx in order:
        rh, rw = map(int, rectangles[int(idx)])
        if h - rh - 2 * _GEN_EDGE_MARGIN < 0 or w - rw - 2 * _GEN_EDGE_MARGIN < 0:
            return None

        placed = False
        for _ in range(_GEN_PLACE_TRIES):
            top = int(rng.integers(_GEN_EDGE_MARGIN, h - rh - _GEN_EDGE_MARGIN + 1))
            left = int(rng.integers(_GEN_EDGE_MARGIN, w - rw - _GEN_EDGE_MARGIN + 1))
            if not _can_place_rect(occupied, top, left, rh, rw, _GEN_PADDING):
                continue
            occupied[top : top + rh, left : left + rw] = True
            positions[int(idx)] = (int(top), int(left))
            placed = True
            break
        if not placed:
            return None

    return [pos for pos in positions if pos is not None]


def _sample_slot_sizes(
    rng: np.random.Generator, count: int
) -> list[tuple[int, int]] | None:
    candidates = [
        (int(h), int(w))
        for h in range(_GEN_SLOT_SIDE_MIN, _GEN_SLOT_SIDE_MAX + 1)
        for w in range(_GEN_SLOT_SIDE_MIN, _GEN_SLOT_SIDE_MAX + 1)
    ]

    for _ in range(50):
        picked: list[tuple[int, int]] = []
        used_areas: set[int] = set()
        for idx in rng.permutation(len(candidates)):
            slot_h, slot_w = candidates[int(idx)]
            area = int(slot_h * slot_w)
            if area in used_areas:
                continue
            picked.append((slot_h, slot_w))
            used_areas.add(area)
            if len(picked) == count:
                break
        if len(picked) == count and any(h != w for h, w in picked):
            return picked
    return None


def _sample_directions(rng: np.random.Generator, count: int) -> list[str]:
    dirs = [str(_DIRS[int(i)]) for i in rng.permutation(len(_DIRS))]
    while len(dirs) < count:
        dirs.append(str(_DIRS[int(rng.integers(0, len(_DIRS)))]))
    picked = dirs[:count]
    rng.shuffle(picked)
    return picked


def _expected_output(inp: np.ndarray, pairs: list[dict], bg: int) -> np.ndarray:
    out = np.array(inp, copy=True)
    for pair in pairs:
        pr, pc = map(int, pair["piece_pos"])
        ph, pw = map(int, pair["piece_patch"].shape)
        out[pr : pr + ph, pc : pc + pw] = int(bg)

    for pair in pairs:
        cr, cc = map(int, pair["container_pos"])
        sr, sc = map(int, pair["slot_origin"])
        sh = int(pair["slot_h"])
        sw = int(pair["slot_w"])
        out[cr + sr : cr + sr + sh, cc + sc : cc + sc + sw] = int(pair["piece_color"])
    return out


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c != _GEN_CONTAINER_COLOR]
    bg = int(bg_choices[int(rng.integers(0, len(bg_choices)))])
    pair_count = int(rng.integers(_GEN_OBJECT_COUNT_RANGE[0], _GEN_OBJECT_COUNT_RANGE[1]))
    slot_sizes = _sample_slot_sizes(rng, pair_count)
    if slot_sizes is None:
        return None

    piece_colors_pool = [c for c in range(10) if c not in (bg, _GEN_CONTAINER_COLOR)]
    if len(piece_colors_pool) < pair_count:
        return None
    piece_colors = [
        int(piece_colors_pool[int(i)])
        for i in rng.choice(len(piece_colors_pool), size=pair_count, replace=False)
    ]
    directions = _sample_directions(rng, pair_count)

    rotate_flags = [False] * pair_count
    rectangular = [i for i, (h, w) in enumerate(slot_sizes) if h != w]
    for idx in rectangular:
        rotate_flags[int(idx)] = bool(rng.integers(0, 2))
    if len(rectangular) >= 2:
        if all(rotate_flags[int(idx)] for idx in rectangular):
            rotate_flags[int(rectangular[0])] = False
        if not any(rotate_flags[int(idx)] for idx in rectangular):
            rotate_flags[int(rectangular[0])] = True

    pairs: list[dict] = []
    rectangles: list[tuple[int, int]] = []

    for idx in range(pair_count):
        slot_h, slot_w = map(int, slot_sizes[idx])
        direction = str(directions[idx])
        piece_color = int(piece_colors[idx])
        container_patch, slot_origin = _build_container_patch(
            slot_h, slot_w, direction, bg, _GEN_CONTAINER_COLOR
        )

        if rotate_flags[idx]:
            piece_h, piece_w = int(slot_w), int(slot_h)
        else:
            piece_h, piece_w = int(slot_h), int(slot_w)
        piece_patch = _build_piece_patch(piece_h, piece_w, bg, piece_color)

        pairs.append(
            {
                "slot_h": slot_h,
                "slot_w": slot_w,
                "direction": direction,
                "piece_color": piece_color,
                "container_patch": container_patch,
                "piece_patch": piece_patch,
                "slot_origin": slot_origin,
            }
        )
        rectangles.append(tuple(map(int, container_patch.shape)))
        rectangles.append(tuple(map(int, piece_patch.shape)))

    reserved = sum((h + 2 * _GEN_PADDING) * (w + 2 * _GEN_PADDING) for h, w in rectangles)
    max_h = max(h for h, _ in rectangles)
    max_w = max(w for _, w in rectangles)
    base = int(np.ceil(np.sqrt(float(reserved) * 2.4)))
    canvas_h = max(_GEN_CANVAS_MIN, max_h + 2 * _GEN_EDGE_MARGIN + 1, base)
    canvas_w = max(_GEN_CANVAS_MIN, max_w + 2 * _GEN_EDGE_MARGIN + 1, base)
    canvas_h += int(rng.integers(0, _GEN_CANVAS_EXTRA + 1))
    canvas_w += int(rng.integers(0, _GEN_CANVAS_EXTRA + 1))

    positions = _place_rectangles(rng, (canvas_h, canvas_w), rectangles)
    if positions is None:
        return None

    inp = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    for idx, pair in enumerate(pairs):
        container_pos = positions[2 * idx]
        piece_pos = positions[2 * idx + 1]
        pair["container_pos"] = container_pos
        pair["piece_pos"] = piece_pos

        cr, cc = map(int, container_pos)
        pr, pc = map(int, piece_pos)
        cp = pair["container_patch"]
        pp = pair["piece_patch"]
        inp[cr : cr + cp.shape[0], cc : cc + cp.shape[1]] = cp
        inp[pr : pr + pp.shape[0], pc : pc + pp.shape[1]] = pp

    expected = _expected_output(inp, pairs, bg)
    if int(np.count_nonzero(inp != bg)) >= int(np.count_nonzero(inp == bg)):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_ATTEMPTS} attempts")
