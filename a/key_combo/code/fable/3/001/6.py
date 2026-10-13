from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))

GENERATION_ATTEMPTS = 700
SPRITE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120

_CHAIN_QUEUE: list[int] = []


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []

    for sr in range(height):
        for sc in range(width):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue

            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ORTHOGONAL:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _is_connected(mask: np.ndarray) -> bool:
    return bool(np.any(mask)) and len(_components4(mask)) == 1


def _object_body_connected(g: np.ndarray, coords: np.ndarray, body_color: int) -> bool:
    mask = np.zeros(g.shape, dtype=bool)
    body = g[coords[:, 0], coords[:, 1]] == int(body_color)
    mask[coords[body, 0], coords[body, 1]] = True
    return _is_connected(mask)


def _parse_objects(g: np.ndarray) -> tuple[list[dict[str, object]], dict[int, int]]:
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert not np.any(g < 0), "colors must be non-negative"

    components = _components4(g != BACKGROUND)
    assert components, "expected at least one foreground object"

    objects: list[dict[str, object]] = []
    mapping: dict[int, int] = {}
    for coords in components:
        values = g[coords[:, 0], coords[:, 1]]
        colors, counts = np.unique(values, return_counts=True)
        colors = colors.astype(int)
        counts = counts.astype(int)

        marker: int | None = None
        if len(colors) == 1:
            body = int(colors[0])
            area = int(counts[0])
            assert area >= 5, "unmarked objects must have area at least five"
        elif len(colors) == 2:
            singletons = [int(color) for color, count in zip(colors, counts) if int(count) == 1]
            assert len(singletons) == 1, "a marked object must contain exactly one embedded marker pixel"
            marker = singletons[0]
            body = int(colors[0] if int(colors[1]) == marker else colors[1])
            body_count = int(counts[0] if int(colors[0]) == body else counts[1])
            assert body_count >= 5, "marked object body must have area at least five"

            marker_coords = coords[values == marker]
            assert marker_coords.shape == (1, 2), "marker must be a singleton"
            mr, mc = map(int, marker_coords[0])
            height, width = map(int, g.shape)
            for dr, dc in ORTHOGONAL:
                nr = mr + dr
                nc = mc + dc
                assert 0 <= nr < height and 0 <= nc < width, "embedded marker must be internal"
                assert int(g[nr, nc]) == body, "embedded marker must be 4-surrounded by the object color"

            assert _object_body_connected(g, coords, body), "object body must be 4-connected around its marker"
            assert body not in mapping, "each object color may define at most one mapping entry"
            mapping[int(body)] = int(marker)
        else:
            raise AssertionError("each object may contain only its body color plus at most one marker color")

        objects.append({"coords": coords, "color": int(body), "marker": marker})

    for start in mapping:
        seen: set[int] = set()
        color = int(start)
        while color in mapping:
            assert color not in seen, "mapping chain must be acyclic"
            seen.add(color)
            color = int(mapping[color])

    return objects, mapping


def _terminal_color(color: int, mapping: dict[int, int]) -> int:
    seen: set[int] = set()
    cur = int(color)
    while cur in mapping:
        assert cur not in seen, "mapping chain must be acyclic"
        seen.add(cur)
        cur = int(mapping[cur])
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every foreground object by following embedded color-map entries to terminal colors.

    Input structure:
    - A black background contains several separate 4-connected foreground objects.
    - Each object has one body color and area at least five; some objects have exactly
      one internal singleton marker pixel of a different non-black color.
    - The marked objects define an acyclic color map from object body color to marker
      color, and each body color appears as a map key at most once.

    Rule:
    - Read each marked object as the mapping entry body-color -> embedded-marker-color.
    - For every foreground object, including unmarked ones, start from its body color
      and repeatedly follow the map while the current color is a key.
    - Repaint every cell of that whole object, including its embedded marker if present,
      with the terminal color reached by the chain.

    Key insight:
    - There is no separate legend: the marked objects are both drawable objects and map
      entries, and a marker color may itself be another object's body color, so the
      recoloring must follow the chain to its final non-key color.
    """
    g = np.asarray(grid, dtype=int)
    objects, mapping = _parse_objects(g)

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        assert isinstance(coords, np.ndarray)
        terminal = _terminal_color(int(obj["color"]), mapping)
        out[coords[:, 0], coords[:, 1]] = int(terminal)
    return out


def _next_chain_len() -> int:
    global _CHAIN_QUEUE
    if not _CHAIN_QUEUE:
        _CHAIN_QUEUE = [2, 3, 4]
        random.shuffle(_CHAIN_QUEUE)
    return int(_CHAIN_QUEUE.pop())


def _random_object_mask(marked: bool) -> tuple[np.ndarray, tuple[int, int] | None] | None:
    for _ in range(SPRITE_ATTEMPTS):
        if marked:
            height = random.randint(3, 5)
            width = random.randint(3, 6)
        else:
            height = random.randint(2, 5)
            width = random.randint(3, 6)
            if height * width < 5:
                continue

        mask = np.ones((height, width), dtype=bool)
        marker: tuple[int, int] | None = None
        reserved: set[tuple[int, int]] = set()
        if marked:
            marker = (random.randint(1, height - 2), random.randint(1, width - 2))
            reserved.add(marker)
            for dr, dc in ORTHOGONAL:
                reserved.add((marker[0] + dr, marker[1] + dc))

        border = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r in (0, height - 1) or c in (0, width - 1)) and (r, c) not in reserved
        ]
        random.shuffle(border)
        minimum = 6 if marked else 5
        remove_limit = min(len(border), height * width - minimum, 3)
        if remove_limit > 0:
            for r, c in border[: random.randint(0, remove_limit)]:
                mask[r, c] = False

        if int(mask.sum()) < minimum or not _is_connected(mask):
            continue
        if marked:
            assert marker is not None
            mr, mc = marker
            if not bool(mask[mr, mc]):
                continue
            body_mask = mask.copy()
            body_mask[mr, mc] = False
            if int(body_mask.sum()) < 5 or not _is_connected(body_mask):
                continue
            if not all(bool(body_mask[mr + dr, mc + dc]) for dr, dc in ORTHOGONAL):
                continue
        return mask, marker
    return None


def _candidate_positions(reserved: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = map(int, reserved.shape)
    obj_h, obj_w = map(int, mask.shape)
    positions: list[tuple[int, int]] = []

    for r in range(height - obj_h + 1):
        for c in range(width - obj_w + 1):
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(height, r + obj_h + 1)
            c1 = min(width, c + obj_w + 1)
            if np.any(reserved[r0:r1, c0:c1]):
                continue
            positions.append((r, c))

    random.shuffle(positions)
    return positions


def _place_objects(objects: list[dict[str, object]], shape: tuple[int, int]) -> bool:
    reserved = np.zeros(shape, dtype=bool)
    order = list(range(len(objects)))
    random.shuffle(order)
    order.sort(key=lambda i: int(np.prod(objects[i]["mask"].shape)), reverse=True)  # type: ignore[index,union-attr]

    for index in order:
        mask = objects[index]["mask"]
        assert isinstance(mask, np.ndarray)
        positions = _candidate_positions(reserved, mask)
        if not positions:
            return False
        r, c = random.choice(positions)
        h, w = map(int, mask.shape)
        reserved[r : r + h, c : c + w] = True
        objects[index]["r"] = int(r)
        objects[index]["c"] = int(c)
    return True


def _render(objects: list[dict[str, object]], mapping: dict[int, int], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    expected = np.zeros(shape, dtype=int)

    for obj in objects:
        mask = obj["mask"]
        assert isinstance(mask, np.ndarray)
        r0 = int(obj["r"])
        c0 = int(obj["c"])
        body = int(obj["color"])
        marker = obj["marker"]

        rel = np.argwhere(mask)
        rr = r0 + rel[:, 0]
        cc = c0 + rel[:, 1]
        inp[rr, cc] = body
        if marker is not None:
            mr, mc = obj["marker_rel"]  # type: ignore[misc]
            inp[r0 + int(mr), c0 + int(mc)] = int(marker)

        expected[rr, cc] = _terminal_color(body, mapping)
    return inp, expected


def _make_object_plan(chain_len: int) -> tuple[list[dict[str, object]], dict[int, int], list[int]] | None:
    chain = random.sample(COLORS, chain_len + 1)
    mapping = {int(chain[i]): int(chain[i + 1]) for i in range(chain_len)}

    objects: list[dict[str, object]] = []
    for key in chain[:-1]:
        objects.append({"color": int(key), "marker": int(mapping[int(key)]), "role": "entry"})

    duplicate_keys = [int(chain[0]), int(random.choice(chain[:-1]))]
    for key in duplicate_keys:
        objects.append({"color": key, "marker": None, "role": "unmarked_key"})

    objects.append({"color": int(chain[-1]), "marker": None, "role": "terminal"})

    available = [int(color) for color in COLORS if int(color) not in set(chain)]
    inert_count = random.randint(1, min(2, len(available)))
    for color in random.sample(available, inert_count):
        objects.append({"color": int(color), "marker": None, "role": "inert"})

    for obj in objects:
        shape = _random_object_mask(obj["marker"] is not None)
        if shape is None:
            return None
        mask, marker_rel = shape
        obj["mask"] = mask
        obj["marker_rel"] = marker_rel

    random.shuffle(objects)
    return objects, mapping, chain


def _semantic_candidate_is_valid(inp: np.ndarray, expected: np.ndarray, mapping: dict[int, int]) -> bool:
    try:
        objects, inferred = _parse_objects(inp)
        actual = main(inp)
    except AssertionError:
        return False

    if inferred != mapping or not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    has_chained_entry = False
    has_unmarked_key = False
    has_unmarked_nonkey = False
    for obj in objects:
        color = int(obj["color"])
        marker = obj["marker"]
        terminal = _terminal_color(color, mapping)
        if marker is not None and int(marker) in mapping and terminal != int(marker):
            has_chained_entry = True
        if marker is None and color in mapping and terminal != color:
            has_unmarked_key = True
        if marker is None and color not in mapping:
            has_unmarked_nonkey = True

    output_colors = set(map(int, expected.ravel())) - {BACKGROUND}
    return has_chained_entry and has_unmarked_key and has_unmarked_nonkey and len(output_colors) >= 2


def _gen_attempt(chain_len: int) -> np.ndarray | None:
    plan = _make_object_plan(chain_len)
    if plan is None:
        return None
    objects, mapping, _chain = plan

    count = len(objects)
    min_side = max(18, 2 * count + 7)
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)

    for _ in range(PLACEMENT_ATTEMPTS):
        trial_objects = [dict(obj) for obj in objects]
        if not _place_objects(trial_objects, (height, width)):
            continue
        inp, expected = _render(trial_objects, mapping, (height, width))
        if _semantic_candidate_is_valid(inp, expected, mapping):
            return inp
    return None


def generate_input() -> np.ndarray:
    preferred_chain_len = _next_chain_len()
    for attempt in range(GENERATION_ATTEMPTS):
        chain_len = preferred_chain_len if attempt < GENERATION_ATTEMPTS // 2 else random.randint(2, 4)
        grid = _gen_attempt(chain_len)
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid embedded color-chain recoloring input")
